"""
Sequential OT-GaussVLAD with BoQ-style token refinement.

The architecture mirrors the core BoQ pattern:

    x0 -> TransformerEncoder_1 -> x1 -> distribution head 1 -> out1
    x1 -> TransformerEncoder_2 -> x2 -> distribution head 2 -> out2
    concat(out1, out2) -> BoQ-style query-axis projection -> descriptor

Instead of learnable attention queries, each stage uses an independent set of
learnable Gaussian distribution parameters and OT assignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def log_sinkhorn_iterations(
    scores: torch.Tensor,
    log_mu: torch.Tensor,
    log_nu: torch.Tensor,
    iters: int,
) -> torch.Tensor:
    """Perform Sinkhorn normalization in log-space."""
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(scores + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(scores + u.unsqueeze(2), dim=1)
    return scores + u.unsqueeze(2) + v.unsqueeze(1)


class SequentialOTGaussVLADLayer(nn.Module):
    """
    BoQ-style sequential distribution-query OT-GaussVLAD.

    Each stage refines tokens with a TransformerEncoderLayer, reads the refined
    distribution with a stage-specific Gaussian OT codebook, and returns a
    per-cluster residual. Stage/cluster residuals are treated as distribution
    queries and projected along the query axis, matching the BoQ fusion pattern.
    """

    def __init__(
        self,
        in_channels=768,
        num_clusters=64,
        cluster_dim=64,
        hidden_channels=768,
        num_stages=2,
        nheads=8,
        transformer_ffn_ratio=4,
        descriptor_dim=8192,
        row_dim=64,
        score_mode="gmm",
        include_pi_in_scores=False,
        sinkhorn_iters=3,
        dustbin_mass=0.05,
        dropout=0.0,
        transformer_dropout=0.0,
        mass_preserving=True,
        mass_power=1.0,
        token_mass_mode="learned",
        token_mass_hidden_channels=None,
        fusion_dropout=0.0,
        eps=1e-6,
    ):
        super().__init__()

        if score_mode not in {"gmm", "attention"}:
            raise ValueError("score_mode must be either 'gmm' or 'attention'.")
        if token_mass_mode not in {"uniform", "learned"}:
            raise ValueError("token_mass_mode must be either 'uniform' or 'learned'.")
        if int(num_stages) < 1:
            raise ValueError("num_stages must be at least 1.")
        if not 0.0 < dustbin_mass < 1.0:
            raise ValueError("dustbin_mass must be in (0, 1).")
        if self._valid_nheads(cluster_dim, nheads) is False:
            raise ValueError("cluster_dim must be divisible by nheads.")

        self.in_channels = int(in_channels)
        self.num_clusters = int(num_clusters)
        self.cluster_dim = int(cluster_dim)
        self.hidden_channels = int(hidden_channels)
        self.num_stages = int(num_stages)
        self.nheads = int(nheads)
        self.score_mode = score_mode
        self.include_pi_in_scores = bool(include_pi_in_scores)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.dustbin_mass = float(dustbin_mass)
        self.mass_preserving = bool(mass_preserving)
        self.mass_power = float(mass_power)
        self.token_mass_mode = token_mass_mode
        self.eps = float(eps)

        local_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.local_features = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=1),
            local_dropout,
            nn.ReLU(),
            nn.Conv2d(self.hidden_channels, self.cluster_dim, kernel_size=1),
        )
        self.token_norm = nn.LayerNorm(self.cluster_dim)

        self.encoders = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.cluster_dim,
                    nhead=self.nheads,
                    dim_feedforward=int(transformer_ffn_ratio * self.cluster_dim),
                    dropout=transformer_dropout,
                    batch_first=True,
                )
                for _ in range(self.num_stages)
            ]
        )

        token_mass_hidden_channels = (
            max(32, self.cluster_dim) if token_mass_hidden_channels is None else int(token_mass_hidden_channels)
        )
        if self.token_mass_mode == "learned":
            token_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.token_mass_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(self.cluster_dim, token_mass_hidden_channels, kernel_size=1),
                        token_dropout,
                        nn.ReLU(),
                        nn.Conv1d(token_mass_hidden_channels, 1, kernel_size=1),
                    )
                    for _ in range(self.num_stages)
                ]
            )
            self.token_mass_scale = nn.Parameter(torch.ones(self.num_stages))
        else:
            self.token_mass_heads = None
            self.register_parameter("token_mass_scale", None)

        self.mu = nn.Parameter(torch.empty(self.num_stages, self.num_clusters, self.cluster_dim))
        nn.init.xavier_uniform_(self.mu)
        self.log_sigma = nn.Parameter(torch.zeros(self.num_stages, self.num_clusters, self.cluster_dim))
        self.log_alpha = nn.Parameter(torch.zeros(self.num_stages, self.num_clusters))
        self.attention_scale = nn.Parameter(torch.full((self.num_stages,), 10.0))
        self.dust_bin = nn.Parameter(torch.ones(self.num_stages))

        self.cluster_residual_dim = 2 * self.cluster_dim
        self.raw_dim_per_stage = self.num_clusters * self.cluster_residual_dim
        self.query_axis_dim = self.num_stages * self.num_clusters
        self.row_dim = int(row_dim)
        self.output_dim = int(descriptor_dim)
        expected_output_dim = self.cluster_residual_dim * self.row_dim
        if self.output_dim != expected_output_dim:
            raise ValueError(
                "BoQ-style query-axis fusion outputs cluster_residual_dim * row_dim. "
                f"descriptor_dim must be {expected_output_dim}, got {self.output_dim}."
            )

        fusion_dropout_layer = nn.Dropout(fusion_dropout) if fusion_dropout > 0 else nn.Identity()
        self.query_fusion = nn.Sequential(
            fusion_dropout_layer,
            nn.Linear(self.query_axis_dim, self.row_dim),
        )

    @staticmethod
    def _valid_nheads(cluster_dim: int, nheads: int) -> bool:
        return int(nheads) > 0 and int(cluster_dim) % int(nheads) == 0

    @staticmethod
    def _split_backbone_output(x):
        if isinstance(x, tuple):
            return x[0]
        return x

    def _extract_local_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local_features(x).flatten(2).transpose(1, 2)
        return self.token_norm(x)

    def _sigma(self, stage_idx: int) -> torch.Tensor:
        return torch.exp(self.log_sigma[stage_idx]).clamp_min(self.eps)

    def _compute_token_mass(self, tokens: torch.Tensor, stage_idx: int) -> torch.Tensor:
        b, n, _ = tokens.shape
        if self.token_mass_mode == "uniform":
            return tokens.new_full((b, n), 1.0 / max(n, 1))

        tokens_chw = tokens.transpose(1, 2).contiguous()
        token_logits = self.token_mass_heads[stage_idx](tokens_chw).squeeze(1)
        scale = F.softplus(self.token_mass_scale[stage_idx]) + self.eps
        return F.softmax(scale * token_logits, dim=-1).clamp_min(self.eps)

    def _gmm_scores(self, tokens: torch.Tensor, stage_idx: int) -> torch.Tensor:
        mu = self.mu[stage_idx]
        log_sigma = self.log_sigma[stage_idx]
        inv_sigma_sq = torch.exp(-2.0 * log_sigma)
        quadratic = torch.einsum("bnd,kd->bnk", tokens.square(), inv_sigma_sq)
        quadratic = quadratic - 2.0 * torch.einsum("bnd,kd->bnk", tokens, mu * inv_sigma_sq)
        quadratic = quadratic + (mu.square() * inv_sigma_sq).sum(dim=-1).view(1, 1, -1)
        log_sigma_sum = log_sigma.sum(dim=-1).view(1, 1, -1)
        logits = -0.5 * quadratic - log_sigma_sum
        if self.include_pi_in_scores:
            logits = logits + F.log_softmax(self.log_alpha[stage_idx], dim=0).view(1, 1, -1)
        return logits.transpose(1, 2).contiguous()

    def _attention_scores(self, tokens: torch.Tensor, stage_idx: int) -> torch.Tensor:
        centers = F.normalize(self.mu[stage_idx], p=2, dim=-1)
        norm_tokens = F.normalize(tokens.transpose(1, 2), p=2, dim=1)
        scale = F.softplus(self.attention_scale[stage_idx]) + self.eps
        return scale * torch.einsum("kd,bdn->bkn", centers, norm_tokens)

    def _transport(self, logits: torch.Tensor, token_mass: torch.Tensor, stage_idx: int) -> torch.Tensor:
        b, _, n = logits.shape
        dustbin = self.dust_bin[stage_idx].to(dtype=logits.dtype).expand(b, 1, n)
        scores = torch.cat([logits, dustbin], dim=1)

        cluster_mass = ((1.0 - self.dustbin_mass) * F.softmax(self.log_alpha[stage_idx], dim=0))
        cluster_mass = cluster_mass.to(dtype=logits.dtype).unsqueeze(0).expand(b, -1)
        dustbin_mass = logits.new_full((b, 1), self.dustbin_mass)
        source_mass = torch.cat([cluster_mass, dustbin_mass], dim=1)

        target_mass = token_mass / token_mass.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        transport = log_sinkhorn_iterations(
            scores,
            torch.log(source_mass.clamp_min(self.eps)),
            torch.log(target_mass.to(dtype=logits.dtype).clamp_min(self.eps)),
            self.sinkhorn_iters,
        )
        return torch.exp(transport[:, :-1, :])

    def _stage_descriptor(self, tokens: torch.Tensor, stage_idx: int) -> torch.Tensor:
        token_mass = self._compute_token_mass(tokens, stage_idx)
        if self.score_mode == "gmm":
            logits = self._gmm_scores(tokens, stage_idx)
        else:
            logits = self._attention_scores(tokens, stage_idx)

        assignments = self._transport(logits, token_mass, stage_idx)
        cluster_mass = assignments.sum(dim=-1, keepdim=True)
        gamma = assignments / cluster_mass.clamp_min(self.eps)

        tokens_chw = tokens.transpose(1, 2).contiguous()
        mu_hat = torch.einsum("bkn,bdn->bkd", gamma, tokens_chw)
        diff = tokens.unsqueeze(1) - mu_hat.unsqueeze(2)
        var_hat = torch.einsum("bkn,bknd->bkd", gamma, diff.square()).clamp_min(self.eps)
        sigma_hat = torch.sqrt(var_hat)

        mean_shift = mu_hat - self.mu[stage_idx].unsqueeze(0)
        sigma_shift = sigma_hat - self._sigma(stage_idx).unsqueeze(0)

        if self.mass_preserving:
            mass_weight = cluster_mass * (self.num_clusters / max(1.0 - self.dustbin_mass, self.eps))
            mass_weight = mass_weight.clamp_min(self.eps).pow(self.mass_power)
            mean_shift = mean_shift * mass_weight
            sigma_shift = sigma_shift * mass_weight

        return torch.cat([mean_shift, sigma_shift], dim=-1)

    def forward(self, x):
        x = self._split_backbone_output(x)
        tokens = self._extract_local_tokens(x)

        stage_descriptors = []
        for stage_idx, encoder in enumerate(self.encoders):
            tokens = encoder(tokens)
            stage_descriptors.append(self._stage_descriptor(tokens, stage_idx))

        # [B, S, K, 2D] -> [B, 2D, S*K] -> Linear(S*K, row_dim), as in BoQ.
        query_axis = torch.stack(stage_descriptors, dim=1).permute(0, 3, 1, 2).flatten(2)
        fused = self.query_fusion(query_axis)
        descriptor = fused.reshape(tokens.size(0), -1)
        return F.normalize(descriptor, p=2, dim=-1)
