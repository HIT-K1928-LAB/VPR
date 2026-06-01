"""
Multi-head OT-GaussVLAD with cluster-wise fusion.

This BoQ-inspired variant runs multiple OT transport heads in parallel over the
same local tokens. Each head produces a per-cluster Gaussian residual. For each
cluster, head residuals are concatenated and fused by a shared small projection,
mirroring BoQ-style structured projection without a huge global linear layer.

Default setup:
    num_heads = 2
    head 0: static source mass
    head 1: image-conditioned source mass
    per-head descriptor = 8192
    per-cluster fused residual = 128
    final descriptor = 8192
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_sigmoid(value: float) -> float:
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError("Sigmoid target must be in (0, 1).")
    return math.log(value / (1.0 - value))


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


class MultiHeadOTGaussVLADClusterFusionLayer(nn.Module):
    """
    Parallel multi-head OT-GaussVLAD with shared cluster-wise fusion.

    head_mode:
        static:
            all heads use static cluster prior and fixed dustbin mass.
        adaptive:
            all heads use image-conditioned cluster prior and dustbin mass.
        static_and_adaptive:
            head 0 is static, remaining heads are image-conditioned.
    """

    def __init__(
        self,
        in_channels=768,
        num_clusters=64,
        cluster_dim=64,
        hidden_channels=768,
        num_heads=2,
        share_local_features=True,
        head_mode="static_and_adaptive",
        descriptor_dim=8192,
        score_mode="gmm",
        include_pi_in_scores=False,
        sinkhorn_iters=3,
        dustbin_mass=0.05,
        dustbin_mass_min=0.01,
        dustbin_mass_max=0.30,
        dropout=0.0,
        mass_preserving=True,
        mass_power=1.0,
        token_mass_mode="learned",
        token_mass_hidden_channels=None,
        context_hidden_channels=None,
        prior_temperature_init=1.0,
        prior_residual_weight_init=0.0,
        fusion_hidden_dim=None,
        fusion_dropout=0.0,
        eps=1e-6,
    ):
        super().__init__()

        if score_mode not in {"gmm", "attention"}:
            raise ValueError("score_mode must be either 'gmm' or 'attention'.")
        if token_mass_mode not in {"uniform", "learned"}:
            raise ValueError("token_mass_mode must be either 'uniform' or 'learned'.")
        if head_mode not in {"static", "adaptive", "static_and_adaptive"}:
            raise ValueError("head_mode must be one of: static, adaptive, static_and_adaptive.")
        if not share_local_features:
            raise ValueError("share_local_features=False is intentionally not enabled in this lightweight version.")
        if int(num_heads) < 1:
            raise ValueError("num_heads must be at least 1.")
        if not 0.0 < dustbin_mass < 1.0:
            raise ValueError("dustbin_mass must be in (0, 1).")
        if not 0.0 < dustbin_mass_min < dustbin_mass_max < 1.0:
            raise ValueError("dustbin_mass_min and dustbin_mass_max must satisfy 0 < min < max < 1.")

        self.in_channels = int(in_channels)
        self.num_clusters = int(num_clusters)
        self.cluster_dim = int(cluster_dim)
        self.hidden_channels = int(hidden_channels)
        self.num_heads = int(num_heads)
        self.share_local_features = bool(share_local_features)
        self.head_mode = head_mode
        self.score_mode = score_mode
        self.include_pi_in_scores = bool(include_pi_in_scores)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.dustbin_mass = float(dustbin_mass)
        self.dustbin_mass_min = float(dustbin_mass_min)
        self.dustbin_mass_max = float(dustbin_mass_max)
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

        token_mass_hidden_channels = (
            max(32, self.cluster_dim) if token_mass_hidden_channels is None else int(token_mass_hidden_channels)
        )
        if self.token_mass_mode == "learned":
            token_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.token_mass_head = nn.Sequential(
                nn.Conv1d(self.cluster_dim, token_mass_hidden_channels, kernel_size=1),
                token_dropout,
                nn.ReLU(),
                nn.Conv1d(token_mass_hidden_channels, 1, kernel_size=1),
            )
            self.token_mass_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.token_mass_head = None
            self.register_parameter("token_mass_scale", None)

        self.mu = nn.Parameter(torch.empty(self.num_heads, self.num_clusters, self.cluster_dim))
        nn.init.xavier_uniform_(self.mu)
        self.log_sigma = nn.Parameter(torch.zeros(self.num_heads, self.num_clusters, self.cluster_dim))
        self.log_alpha = nn.Parameter(torch.zeros(self.num_heads, self.num_clusters))
        self.attention_scale = nn.Parameter(torch.full((self.num_heads,), 10.0))
        self.dust_bin = nn.Parameter(torch.ones(self.num_heads))

        context_hidden_channels = (
            max(64, self.cluster_dim) if context_hidden_channels is None else int(context_hidden_channels)
        )
        context_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.context_head = nn.Sequential(
            nn.Linear(2 * self.cluster_dim, context_hidden_channels),
            nn.ReLU(),
            context_dropout,
        )
        self.prior_head = nn.Linear(context_hidden_channels, self.num_heads * self.num_clusters)
        self.dustbin_head = nn.Linear(context_hidden_channels, self.num_heads)
        self.prior_temperature = nn.Parameter(torch.full((self.num_heads,), float(prior_temperature_init)))
        self.prior_residual_weight = nn.Parameter(torch.full((self.num_heads,), float(prior_residual_weight_init)))

        normalized_dustbin = (self.dustbin_mass - self.dustbin_mass_min) / (
            self.dustbin_mass_max - self.dustbin_mass_min
        )
        self.dustbin_bias = nn.Parameter(torch.full((self.num_heads,), inverse_sigmoid(normalized_dustbin)))

        self.cluster_residual_dim = 2 * self.cluster_dim
        self.raw_dim_per_head = self.num_clusters * self.cluster_residual_dim
        self.concat_dim = self.num_heads * self.raw_dim_per_head
        self.output_dim = int(descriptor_dim)
        expected_output_dim = self.num_clusters * self.cluster_residual_dim
        if self.output_dim != expected_output_dim:
            raise ValueError(
                "Cluster-wise fusion keeps the OT-GaussVLAD descriptor shape fixed. "
                f"descriptor_dim must be {expected_output_dim}, got {self.output_dim}."
            )

        fusion_input_dim = self.num_heads * self.cluster_residual_dim
        fusion_dropout_layer = nn.Dropout(fusion_dropout) if fusion_dropout > 0 else nn.Identity()
        if fusion_hidden_dim is None:
            self.cluster_fusion = nn.Sequential(
                nn.LayerNorm(fusion_input_dim),
                fusion_dropout_layer,
                nn.Linear(fusion_input_dim, self.cluster_residual_dim),
            )
        else:
            fusion_hidden_dim = int(fusion_hidden_dim)
            self.cluster_fusion = nn.Sequential(
                nn.LayerNorm(fusion_input_dim),
                nn.Linear(fusion_input_dim, fusion_hidden_dim),
                nn.ReLU(),
                fusion_dropout_layer,
                nn.Linear(fusion_hidden_dim, self.cluster_residual_dim),
            )

    @staticmethod
    def _split_backbone_output(x):
        if isinstance(x, tuple):
            return x[0]
        return x

    def _extract_local_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local_features(x).flatten(2).transpose(1, 2)
        x = self.token_norm(x)
        return x.transpose(1, 2).contiguous()

    def _sigma(self, head_idx: int) -> torch.Tensor:
        return torch.exp(self.log_sigma[head_idx]).clamp_min(self.eps)

    def _compute_token_mass(self, local: torch.Tensor) -> torch.Tensor:
        b, _, n = local.shape
        if self.token_mass_mode == "uniform":
            return local.new_full((b, n), 1.0 / max(n, 1))

        token_logits = self.token_mass_head(local).squeeze(1)
        scale = F.softplus(self.token_mass_scale) + self.eps
        return F.softmax(scale * token_logits, dim=-1).clamp_min(self.eps)

    def _context(self, local: torch.Tensor) -> torch.Tensor:
        mean_context = local.mean(dim=-1)
        max_context = local.amax(dim=-1)
        return torch.cat([mean_context, max_context], dim=1)

    def _head_is_adaptive(self, head_idx: int) -> bool:
        if self.head_mode == "adaptive":
            return True
        if self.head_mode == "static":
            return False
        return head_idx > 0

    def _adaptive_context_outputs(self, local: torch.Tensor):
        context = self.context_head(self._context(local))
        prior_logits = self.prior_head(context).view(local.size(0), self.num_heads, self.num_clusters)
        dustbin_logits = self.dustbin_head(context)
        return prior_logits, dustbin_logits

    def _source_mass(
        self,
        head_idx: int,
        batch_size: int,
        local: torch.Tensor,
        dynamic_prior_logits: torch.Tensor,
        dynamic_dustbin_logits: torch.Tensor,
    ):
        static_logits = self.log_alpha[head_idx].unsqueeze(0)
        if self._head_is_adaptive(head_idx):
            residual_weight = torch.tanh(self.prior_residual_weight[head_idx])
            temperature = F.softplus(self.prior_temperature[head_idx]) + self.eps
            cluster_prior = F.softmax(
                (static_logits + residual_weight * dynamic_prior_logits[:, head_idx]) / temperature,
                dim=-1,
            )
            dustbin_unit = torch.sigmoid(dynamic_dustbin_logits[:, head_idx] + self.dustbin_bias[head_idx])
            dustbin_mass = self.dustbin_mass_min + (self.dustbin_mass_max - self.dustbin_mass_min) * dustbin_unit
        else:
            cluster_prior = F.softmax(static_logits, dim=-1).expand(batch_size, -1)
            dustbin_mass = local.new_full((batch_size,), self.dustbin_mass)

        cluster_mass = (1.0 - dustbin_mass).unsqueeze(-1) * cluster_prior
        source_mass = torch.cat([cluster_mass, dustbin_mass.unsqueeze(-1)], dim=1)
        return source_mass, dustbin_mass

    def _gmm_scores(self, local: torch.Tensor, head_idx: int) -> torch.Tensor:
        x_tokens = local.transpose(1, 2)
        mu = self.mu[head_idx]
        log_sigma = self.log_sigma[head_idx]
        inv_sigma_sq = torch.exp(-2.0 * log_sigma)
        quadratic = torch.einsum("bnd,kd->bnk", x_tokens.square(), inv_sigma_sq)
        quadratic = quadratic - 2.0 * torch.einsum("bnd,kd->bnk", x_tokens, mu * inv_sigma_sq)
        quadratic = quadratic + (mu.square() * inv_sigma_sq).sum(dim=-1).view(1, 1, -1)
        log_sigma_sum = log_sigma.sum(dim=-1).view(1, 1, -1)
        logits = -0.5 * quadratic - log_sigma_sum
        if self.include_pi_in_scores:
            logits = logits + F.log_softmax(self.log_alpha[head_idx], dim=0).view(1, 1, -1)
        return logits.transpose(1, 2).contiguous()

    def _attention_scores(self, local: torch.Tensor, head_idx: int) -> torch.Tensor:
        centers = F.normalize(self.mu[head_idx], p=2, dim=-1)
        tokens = F.normalize(local, p=2, dim=1)
        scale = F.softplus(self.attention_scale[head_idx]) + self.eps
        return scale * torch.einsum("kd,bdn->bkn", centers, tokens)

    def _transport(
        self,
        logits: torch.Tensor,
        token_mass: torch.Tensor,
        source_mass: torch.Tensor,
        head_idx: int,
    ) -> torch.Tensor:
        b, _, n = logits.shape
        dustbin = self.dust_bin[head_idx].to(dtype=logits.dtype).expand(b, 1, n)
        scores = torch.cat([logits, dustbin], dim=1)

        target_mass = token_mass / token_mass.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        transport = log_sinkhorn_iterations(
            scores,
            torch.log(source_mass.to(dtype=logits.dtype).clamp_min(self.eps)),
            torch.log(target_mass.to(dtype=logits.dtype).clamp_min(self.eps)),
            self.sinkhorn_iters,
        )
        return torch.exp(transport[:, :-1, :])

    def _head_descriptor(
        self,
        local: torch.Tensor,
        token_mass: torch.Tensor,
        head_idx: int,
        dynamic_prior_logits: torch.Tensor,
        dynamic_dustbin_logits: torch.Tensor,
    ) -> torch.Tensor:
        source_mass, dustbin_mass = self._source_mass(
            head_idx,
            local.size(0),
            local,
            dynamic_prior_logits,
            dynamic_dustbin_logits,
        )

        if self.score_mode == "gmm":
            logits = self._gmm_scores(local, head_idx)
        else:
            logits = self._attention_scores(local, head_idx)

        assignments = self._transport(logits, token_mass, source_mass, head_idx)
        cluster_mass = assignments.sum(dim=-1, keepdim=True)
        gamma = assignments / cluster_mass.clamp_min(self.eps)

        mu_hat = torch.einsum("bkn,bdn->bkd", gamma, local)
        diff = local.transpose(1, 2).unsqueeze(1) - mu_hat.unsqueeze(2)
        var_hat = torch.einsum("bkn,bknd->bkd", gamma, diff.square()).clamp_min(self.eps)
        sigma_hat = torch.sqrt(var_hat)

        mean_shift = mu_hat - self.mu[head_idx].unsqueeze(0)
        sigma_shift = sigma_hat - self._sigma(head_idx).unsqueeze(0)

        if self.mass_preserving:
            normalizer = self.num_clusters / (1.0 - dustbin_mass).clamp_min(self.eps)
            mass_weight = cluster_mass * normalizer.view(-1, 1, 1)
            mass_weight = mass_weight.clamp_min(self.eps).pow(self.mass_power)
            mean_shift = mean_shift * mass_weight
            sigma_shift = sigma_shift * mass_weight

        return torch.cat([mean_shift, sigma_shift], dim=-1)

    def forward(self, x):
        x = self._split_backbone_output(x)
        local = self._extract_local_tokens(x)
        token_mass = self._compute_token_mass(local)
        dynamic_prior_logits, dynamic_dustbin_logits = self._adaptive_context_outputs(local)

        head_descriptors = [
            self._head_descriptor(local, token_mass, head_idx, dynamic_prior_logits, dynamic_dustbin_logits)
            for head_idx in range(self.num_heads)
        ]
        # [B, H, K, 2D] -> [B, K, H * 2D], then shared fusion per cluster.
        per_cluster = torch.stack(head_descriptors, dim=1).permute(0, 2, 1, 3).flatten(2)
        fused = self.cluster_fusion(per_cluster)
        descriptor = fused.reshape(local.size(0), -1)
        return F.normalize(descriptor, p=2, dim=-1)
