"""
OT-GaussVLAD with a Cross-query Similarity side branch.

The OT branch keeps the Gaussian residual descriptor from OT-GaussVLAD v2.
The CS branch follows the QAA idea: feature queries read local tokens, an
independent reference query codebook supplies a fixed frame, and their
cross-query similarity is fused back into the Gaussian residual descriptor.
"""

import math

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


def inverse_sigmoid(value: float) -> float:
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError("Sigmoid target must be in (0, 1).")
    return math.log(value / (1.0 - value))


class QuerySelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_queries: int,
        num_heads: int,
        dropout: float = 0.0,
        self_attn: bool = True,
        out_norm: bool = True,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.queries = nn.Parameter(torch.empty(1, num_queries, embed_dim))
        nn.init.xavier_uniform_(self.queries)
        self.self_attn_flag = bool(self_attn)
        self.out_norm = bool(out_norm)

        if self.self_attn_flag:
            self.self_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self) -> torch.Tensor:
        q = self.queries
        if self.self_attn_flag:
            q = q + self.self_attn(q, q, q, need_weights=False)[0]
        if self.out_norm:
            q = self.norm(q)
        return q


class OTGaussVLADCSLayer(nn.Module):
    """
    OT-GaussVLAD with a QAA-style Cross-query Similarity side branch.

    OT descriptor:
        Sinkhorn assignment over Gaussian prototypes, then [mu_hat - mu,
        sigma_hat - sigma].
    CS descriptor:
        learned feature queries attend to local tokens, independent reference
        queries form a codebook, and F_ref^T P_img gives a similarity matrix.
    Fusion:
        default cluster_fusion keeps the final descriptor shape identical to
        OT-GaussVLAD v2 while initializing as OT + a small CS residual.
    """

    def __init__(
        self,
        in_channels=768,
        num_clusters=64,
        cluster_dim=64,
        hidden_channels=768,
        descriptor_dim=None,
        score_mode="gmm",
        include_pi_in_scores=False,
        sinkhorn_iters=3,
        dustbin_mass=0.05,
        dropout=0.0,
        mass_preserving=True,
        mass_power=1.0,
        token_mass_mode="learned",
        token_mass_hidden_channels=None,
        cs_num_queries=256,
        cs_feature_dim=None,
        cs_reference_dim=None,
        cs_feature_nheads=8,
        cs_reference_nheads=8,
        cs_self_attn=True,
        cs_attn_dropout=0.0,
        cs_intra_norm=True,
        cs_scale_mode="sqrt",
        fusion_mode="cluster_fusion",
        fusion_dropout=0.0,
        cs_fusion_init=0.1,
        cs_gate_init=0.1,
        eps=1e-6,
    ):
        super().__init__()

        if score_mode not in {"gmm", "attention"}:
            raise ValueError("score_mode must be either 'gmm' or 'attention'.")
        if token_mass_mode not in {"uniform", "learned"}:
            raise ValueError("token_mass_mode must be either 'uniform' or 'learned'.")
        if cs_scale_mode not in {"none", "sqrt"}:
            raise ValueError("cs_scale_mode must be either 'none' or 'sqrt'.")
        if fusion_mode not in {"cluster_fusion", "residual_gated", "concat"}:
            raise ValueError("fusion_mode must be one of: cluster_fusion, residual_gated, concat.")
        if not 0.0 < dustbin_mass < 1.0:
            raise ValueError("dustbin_mass must be in (0, 1).")

        self.in_channels = int(in_channels)
        self.num_clusters = int(num_clusters)
        self.cluster_dim = int(cluster_dim)
        self.hidden_channels = int(hidden_channels)
        self.score_mode = score_mode
        self.include_pi_in_scores = bool(include_pi_in_scores)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.dustbin_mass = float(dustbin_mass)
        self.mass_preserving = bool(mass_preserving)
        self.mass_power = float(mass_power)
        self.token_mass_mode = token_mass_mode
        self.cs_num_queries = int(cs_num_queries)
        self.cs_intra_norm = bool(cs_intra_norm)
        self.cs_scale_mode = cs_scale_mode
        self.fusion_mode = fusion_mode
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

        self.mu = nn.Parameter(torch.empty(self.num_clusters, self.cluster_dim))
        nn.init.xavier_uniform_(self.mu)
        self.log_sigma = nn.Parameter(torch.zeros(self.num_clusters, self.cluster_dim))
        self.log_alpha = nn.Parameter(torch.zeros(self.num_clusters))

        self.attention_scale = nn.Parameter(torch.tensor(10.0))
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

        self.raw_cluster_dim = 2 * self.cluster_dim
        self.raw_dim = self.num_clusters * self.raw_cluster_dim

        self.cs_feature_dim = self.num_clusters if cs_feature_dim is None else int(cs_feature_dim)
        self.cs_reference_dim = self.raw_cluster_dim if cs_reference_dim is None else int(cs_reference_dim)
        self.cs_dim = self.cs_feature_dim * self.cs_reference_dim

        if self.cluster_dim % int(cs_feature_nheads) != 0:
            raise ValueError("cluster_dim must be divisible by cs_feature_nheads.")
        self.cs_feature_queries = QuerySelfAttention(
            embed_dim=self.cluster_dim,
            num_queries=self.cs_num_queries,
            num_heads=int(cs_feature_nheads),
            dropout=cs_attn_dropout,
            self_attn=cs_self_attn,
            out_norm=True,
        )
        self.cs_reference_queries = QuerySelfAttention(
            embed_dim=self.cs_reference_dim,
            num_queries=self.cs_num_queries,
            num_heads=int(cs_reference_nheads),
            dropout=cs_attn_dropout,
            self_attn=cs_self_attn,
            out_norm=True,
        )
        self.cs_cross_attn = nn.MultiheadAttention(
            embed_dim=self.cluster_dim,
            num_heads=int(cs_feature_nheads),
            dropout=cs_attn_dropout,
            batch_first=True,
        )
        self.cs_cross_norm = nn.LayerNorm(self.cluster_dim)
        self.cs_feature_proj = nn.Linear(self.cluster_dim, self.cs_feature_dim)
        self.cs_feature_norm = nn.LayerNorm(self.cs_feature_dim)

        if self.fusion_mode in {"cluster_fusion", "residual_gated"}:
            if self.cs_feature_dim != self.num_clusters or self.cs_reference_dim != self.raw_cluster_dim:
                raise ValueError(
                    "cluster_fusion and residual_gated require cs_feature_dim=num_clusters "
                    "and cs_reference_dim=2*cluster_dim."
                )
            expected_dim = self.raw_dim
        else:
            expected_dim = self.raw_dim + self.cs_dim

        if descriptor_dim is not None and int(descriptor_dim) != expected_dim:
            raise ValueError(f"descriptor_dim must be omitted or equal to {expected_dim}, got {descriptor_dim}.")
        self.output_dim = expected_dim

        fusion_dropout_layer = nn.Dropout(fusion_dropout) if fusion_dropout > 0 else nn.Identity()
        if self.fusion_mode == "cluster_fusion":
            self.cluster_fusion_dropout = fusion_dropout_layer
            self.cluster_fusion = nn.Linear(2 * self.raw_cluster_dim, self.raw_cluster_dim)
            self._init_cluster_fusion(float(cs_fusion_init))
        elif self.fusion_mode == "residual_gated":
            self.cs_gate_logit = nn.Parameter(torch.tensor(inverse_sigmoid(float(cs_gate_init))))
        else:
            self.cluster_fusion_dropout = fusion_dropout_layer

    def _init_cluster_fusion(self, cs_fusion_init: float) -> None:
        with torch.no_grad():
            self.cluster_fusion.weight.zero_()
            self.cluster_fusion.bias.zero_()
            eye = torch.eye(self.raw_cluster_dim)
            self.cluster_fusion.weight[:, : self.raw_cluster_dim].copy_(eye)
            self.cluster_fusion.weight[:, self.raw_cluster_dim :].copy_(cs_fusion_init * eye)

    @staticmethod
    def _split_backbone_output(x):
        if isinstance(x, tuple):
            return x[0]
        return x

    def _extract_local_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local_features(x).flatten(2).transpose(1, 2)  # [B, N, D]
        x = self.token_norm(x)
        return x.transpose(1, 2).contiguous()  # [B, D, N]

    def _cluster_prior(self) -> torch.Tensor:
        return F.softmax(self.log_alpha, dim=0)

    def _sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma).clamp_min(self.eps)

    def _compute_token_mass(self, local: torch.Tensor) -> torch.Tensor:
        b, _, n = local.shape
        if self.token_mass_mode == "uniform":
            return local.new_full((b, n), 1.0 / max(n, 1))

        token_logits = self.token_mass_head(local).squeeze(1)  # [B, N]
        scale = F.softplus(self.token_mass_scale) + self.eps
        token_mass = F.softmax(scale * token_logits, dim=-1)
        return token_mass.clamp_min(self.eps)

    def _gmm_scores(self, x: torch.Tensor) -> torch.Tensor:
        x_tokens = x.transpose(1, 2)  # [B, N, D]
        inv_sigma_sq = torch.exp(-2.0 * self.log_sigma)  # [K, D]
        quadratic = torch.einsum("bnd,kd->bnk", x_tokens.square(), inv_sigma_sq)
        quadratic = quadratic - 2.0 * torch.einsum("bnd,kd->bnk", x_tokens, self.mu * inv_sigma_sq)
        quadratic = quadratic + (self.mu.square() * inv_sigma_sq).sum(dim=-1).view(1, 1, -1)
        log_sigma_sum = self.log_sigma.sum(dim=-1).view(1, 1, -1)
        logits = -0.5 * quadratic - log_sigma_sum
        if self.include_pi_in_scores:
            logits = logits + F.log_softmax(self.log_alpha, dim=0).view(1, 1, -1)
        return logits.transpose(1, 2).contiguous()

    def _attention_scores(self, x: torch.Tensor) -> torch.Tensor:
        centers = F.normalize(self.mu, p=2, dim=-1)
        tokens = F.normalize(x, p=2, dim=1)
        scale = F.softplus(self.attention_scale) + self.eps
        return scale * torch.einsum("kd,bdn->bkn", centers, tokens)

    def _transport(self, logits: torch.Tensor, token_mass: torch.Tensor) -> torch.Tensor:
        b, _, n = logits.shape
        dustbin = self.dust_bin.to(dtype=logits.dtype).expand(b, 1, n)
        scores = torch.cat([logits, dustbin], dim=1)  # [B, K+1, N]

        cluster_mass = ((1.0 - self.dustbin_mass) * self._cluster_prior()).to(dtype=logits.dtype)
        cluster_mass = cluster_mass.unsqueeze(0).expand(b, -1)  # [B, K]
        dustbin_mass = logits.new_full((b, 1), self.dustbin_mass)
        source_mass = torch.cat([cluster_mass, dustbin_mass], dim=1)  # [B, K+1]

        target_mass = token_mass / token_mass.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        transport = log_sinkhorn_iterations(
            scores,
            torch.log(source_mass.clamp_min(self.eps)),
            torch.log(target_mass.clamp_min(self.eps)),
            self.sinkhorn_iters,
        )
        return torch.exp(transport[:, :-1, :])

    def _compute_ot_residual(self, local: torch.Tensor, token_mass: torch.Tensor) -> torch.Tensor:
        if self.score_mode == "gmm":
            logits = self._gmm_scores(local)
        else:
            logits = self._attention_scores(local)

        assignments = self._transport(logits, token_mass)  # [B, K, N]
        cluster_mass = assignments.sum(dim=-1, keepdim=True)  # [B, K, 1]
        gamma = assignments / cluster_mass.clamp_min(self.eps)

        mu_hat = torch.einsum("bkn,bdn->bkd", gamma, local)
        diff = local.transpose(1, 2).unsqueeze(1) - mu_hat.unsqueeze(2)  # [B, K, N, D]
        var_hat = torch.einsum("bkn,bknd->bkd", gamma, diff.square()).clamp_min(self.eps)
        sigma_hat = torch.sqrt(var_hat)

        mean_shift = mu_hat - self.mu.unsqueeze(0)
        sigma_shift = sigma_hat - self._sigma().unsqueeze(0)

        if self.mass_preserving:
            mass_weight = cluster_mass * (self.num_clusters / max(1.0 - self.dustbin_mass, self.eps))
            mass_weight = mass_weight.clamp_min(self.eps).pow(self.mass_power)
            mean_shift = mean_shift * mass_weight
            sigma_shift = sigma_shift * mass_weight

        return torch.cat([mean_shift, sigma_shift], dim=-1)

    def _compute_cs_residual(self, local: torch.Tensor) -> torch.Tensor:
        b = local.size(0)
        local_tokens = local.transpose(1, 2).contiguous()  # [B, N, D]

        feature_queries = self.cs_feature_queries().expand(b, -1, -1)
        query_features = self.cs_cross_attn(feature_queries, local_tokens, local_tokens, need_weights=False)[0]
        query_features = self.cs_cross_norm(query_features)
        query_features = self.cs_feature_norm(self.cs_feature_proj(query_features))  # [B, Q, Cf]

        reference_codebook = self.cs_reference_queries().squeeze(0)  # [Q, Cr]
        cs = torch.einsum("qr,bqf->brf", reference_codebook, query_features)
        if self.cs_scale_mode == "sqrt":
            cs = cs / math.sqrt(max(self.cs_num_queries, 1))
        if self.cs_intra_norm:
            cs = F.normalize(cs, p=2, dim=1)

        if self.fusion_mode in {"cluster_fusion", "residual_gated"}:
            return cs.transpose(1, 2).contiguous()  # [B, K, 2D]
        return cs.reshape(b, -1)

    def _fuse(self, ot_residual: torch.Tensor, cs_residual: torch.Tensor) -> torch.Tensor:
        if self.fusion_mode == "cluster_fusion":
            fused = torch.cat([ot_residual, cs_residual], dim=-1)
            fused = self.cluster_fusion_dropout(fused)
            return self.cluster_fusion(fused).reshape(ot_residual.size(0), -1)
        if self.fusion_mode == "residual_gated":
            gate = torch.sigmoid(self.cs_gate_logit)
            return (ot_residual + gate * cs_residual).reshape(ot_residual.size(0), -1)

        ot_descriptor = ot_residual.reshape(ot_residual.size(0), -1)
        cs_descriptor = self.cluster_fusion_dropout(cs_residual)
        return torch.cat([ot_descriptor, cs_descriptor], dim=-1)

    def forward(self, x):
        x = self._split_backbone_output(x)
        local = self._extract_local_tokens(x)  # [B, D, N]
        token_mass = self._compute_token_mass(local)  # [B, N]

        ot_residual = self._compute_ot_residual(local, token_mass)
        cs_residual = self._compute_cs_residual(local)
        descriptor = self._fuse(ot_residual, cs_residual)
        return F.normalize(descriptor, p=2, dim=-1)
