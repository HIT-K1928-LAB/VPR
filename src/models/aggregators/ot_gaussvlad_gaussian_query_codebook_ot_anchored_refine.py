"""
OT-GaussVLAD with a self-attentive Gaussian codebook and OT-anchored
image-side Gaussian refinement.

This implements scheme B on top of OTGaussVLADGaussianQueryCodebookLayer:

    local tokens -> OT assignment -> mu_hat_ot / sigma_hat_ot
    Gaussian-conditioned cluster queries cross-attend local tokens
    -> small residual corrections for mu_hat / log_sigma_hat
    -> [mu_hat - mu, sigma_hat - sigma].

The refinement is deliberately anchored to the OT statistics instead of
replacing them. The correction heads are zero-initialized and scaled by small
learnable gates, so the layer starts close to the base Gaussian query codebook
model and can learn image-side Gaussian corrections only when useful.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ot_gaussvlad_gaussian_query_codebook import (
    OTGaussVLADGaussianQueryCodebookLayer,
    inverse_sigmoid,
)

__all__ = ["OTGaussVLADGaussianQueryCodebookOTAnchoredRefineLayer"]


class GaussianEstimateRefiner(nn.Module):
    """
    Gaussian-conditioned cluster queries refine OT-estimated image moments.

    Query source:
        reference Gaussian codebook [mu_k, log_sigma_k]

    Memory:
        projected local image tokens

    Output:
        per-cluster delta_mu_hat and delta_log_sigma_hat.
    """

    def __init__(
        self,
        cluster_dim: int,
        refine_dim=None,
        num_heads: int = 4,
        dropout: float = 0.0,
        mlp_ratio: float = 2.0,
        use_ffn: bool = True,
        delta_tanh: bool = True,
        mu_scale_init: float = 0.05,
        sigma_scale_init: float = 0.05,
        max_residual_scale: float = 1.0,
    ):
        super().__init__()
        self.cluster_dim = int(cluster_dim)
        self.refine_dim = self.cluster_dim if refine_dim is None else int(refine_dim)
        self.use_ffn = bool(use_ffn)
        self.delta_tanh = bool(delta_tanh)
        self.max_residual_scale = float(max_residual_scale)

        if self.refine_dim % int(num_heads) != 0:
            raise ValueError("refine_dim must be divisible by num_heads.")
        if self.max_residual_scale <= 0:
            raise ValueError("max_residual_scale must be positive.")
        if not 0.0 < float(mu_scale_init) < self.max_residual_scale:
            raise ValueError("mu_scale_init must be in (0, max_residual_scale).")
        if not 0.0 < float(sigma_scale_init) < self.max_residual_scale:
            raise ValueError("sigma_scale_init must be in (0, max_residual_scale).")

        self.gaussian_query_norm = nn.LayerNorm(2 * self.cluster_dim)
        self.gaussian_query_proj = nn.Sequential(
            nn.Linear(2 * self.cluster_dim, self.refine_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(self.refine_dim, self.refine_dim),
        )

        self.token_norm = nn.LayerNorm(self.cluster_dim)
        self.token_proj = nn.Linear(self.cluster_dim, self.refine_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.refine_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(self.refine_dim)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        if self.use_ffn:
            hidden_dim = max(self.refine_dim, int(round(self.refine_dim * float(mlp_ratio))))
            self.ffn_norm = nn.LayerNorm(self.refine_dim)
            self.ffn = nn.Sequential(
                nn.Linear(self.refine_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, self.refine_dim),
                nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            )
        else:
            self.ffn_norm = None
            self.ffn = None

        self.delta_mu_head = nn.Linear(self.refine_dim, self.cluster_dim)
        self.delta_log_sigma_head = nn.Linear(self.refine_dim, self.cluster_dim)
        nn.init.zeros_(self.delta_mu_head.weight)
        nn.init.zeros_(self.delta_mu_head.bias)
        nn.init.zeros_(self.delta_log_sigma_head.weight)
        nn.init.zeros_(self.delta_log_sigma_head.bias)

        mu_ratio = float(mu_scale_init) / self.max_residual_scale
        sigma_ratio = float(sigma_scale_init) / self.max_residual_scale
        self.mu_scale_logit = nn.Parameter(torch.tensor(inverse_sigmoid(mu_ratio)))
        self.sigma_scale_logit = nn.Parameter(torch.tensor(inverse_sigmoid(sigma_ratio)))

    def _scales(self):
        mu_scale = self.max_residual_scale * torch.sigmoid(self.mu_scale_logit)
        sigma_scale = self.max_residual_scale * torch.sigmoid(self.sigma_scale_logit)
        return mu_scale, sigma_scale

    def forward(self, local: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor):
        b = local.size(0)
        gaussian = torch.cat([mu, log_sigma], dim=-1)  # [K, 2D]
        queries = self.gaussian_query_proj(self.gaussian_query_norm(gaussian))
        queries = queries.unsqueeze(0).expand(b, -1, -1)  # [B, K, C]

        tokens = local.transpose(1, 2).contiguous()  # [B, N, D]
        memory = self.token_proj(self.token_norm(tokens))  # [B, N, C]

        attended = self.cross_attn(queries, memory, memory, need_weights=False)[0]
        features = self.cross_norm(queries + self.dropout(attended))
        if self.use_ffn:
            features = features + self.ffn(self.ffn_norm(features))

        delta_mu = self.delta_mu_head(features)
        delta_log_sigma = self.delta_log_sigma_head(features)
        if self.delta_tanh:
            delta_mu = torch.tanh(delta_mu)
            delta_log_sigma = torch.tanh(delta_log_sigma)

        mu_scale, sigma_scale = self._scales()
        delta_mu = delta_mu * mu_scale.to(dtype=delta_mu.dtype, device=delta_mu.device)
        delta_log_sigma = delta_log_sigma * sigma_scale.to(
            dtype=delta_log_sigma.dtype, device=delta_log_sigma.device
        )
        return delta_mu, delta_log_sigma


class OTGaussVLADGaussianQueryCodebookOTAnchoredRefineLayer(OTGaussVLADGaussianQueryCodebookLayer):
    """
    Scheme B: OT-anchored Gaussian estimate refinement.

    The base OT branch computes reliable image-conditioned Gaussian estimates:
        mu_hat_ot, sigma_hat_ot.

    A Gaussian-conditioned cross-attention branch predicts small corrections:
        mu_hat = mu_hat_ot + delta_mu_hat
        log_sigma_hat = log(sigma_hat_ot) + delta_log_sigma_hat.

    The descriptor remains the standard Gaussian residual flattened over the
    fixed cluster order, so output_dim is unchanged.
    """

    def __init__(
        self,
        *args,
        estimate_refine_dim=None,
        estimate_refine_nheads=4,
        estimate_refine_dropout=0.0,
        estimate_refine_mlp_ratio=2.0,
        estimate_refine_use_ffn=True,
        estimate_refine_delta_tanh=True,
        estimate_refine_mu_scale_init=0.05,
        estimate_refine_sigma_scale_init=0.05,
        estimate_refine_max_residual_scale=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.estimate_refiner = GaussianEstimateRefiner(
            cluster_dim=self.cluster_dim,
            refine_dim=estimate_refine_dim,
            num_heads=int(estimate_refine_nheads),
            dropout=float(estimate_refine_dropout),
            mlp_ratio=float(estimate_refine_mlp_ratio),
            use_ffn=bool(estimate_refine_use_ffn),
            delta_tanh=bool(estimate_refine_delta_tanh),
            mu_scale_init=float(estimate_refine_mu_scale_init),
            sigma_scale_init=float(estimate_refine_sigma_scale_init),
            max_residual_scale=float(estimate_refine_max_residual_scale),
        )

    def _scores_with_codebook(self, local: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        if self.score_mode == "gmm":
            x_tokens = local.transpose(1, 2)  # [B, N, D]
            inv_sigma_sq = torch.exp(-2.0 * log_sigma)  # [K, D]
            quadratic = torch.einsum("bnd,kd->bnk", x_tokens.square(), inv_sigma_sq)
            quadratic = quadratic - 2.0 * torch.einsum("bnd,kd->bnk", x_tokens, mu * inv_sigma_sq)
            quadratic = quadratic + (mu.square() * inv_sigma_sq).sum(dim=-1).view(1, 1, -1)
            log_sigma_sum = log_sigma.sum(dim=-1).view(1, 1, -1)
            logits = -0.5 * quadratic - log_sigma_sum
            if self.include_pi_in_scores:
                logits = logits + F.log_softmax(self.log_alpha, dim=0).view(1, 1, -1)
            return logits.transpose(1, 2).contiguous()

        centers = F.normalize(mu, p=2, dim=-1)
        tokens = F.normalize(local, p=2, dim=1)
        scale = F.softplus(self.attention_scale) + self.eps
        return scale * torch.einsum("kd,bdn->bkn", centers, tokens)

    def forward(self, x):
        x = self._split_backbone_output(x)
        local = self._extract_local_tokens(x)  # [B, D, N]
        token_mass = self._compute_token_mass(local)  # [B, N]
        mu, log_sigma = self._gaussian_codebook()

        logits = self._scores_with_codebook(local, mu, log_sigma)
        assignments = self._transport(logits, token_mass)  # [B, K, N]
        cluster_mass = assignments.sum(dim=-1, keepdim=True)  # [B, K, 1]
        gamma = assignments / cluster_mass.clamp_min(self.eps)

        mu_hat_ot = torch.einsum("bkn,bdn->bkd", gamma, local)
        second_moment = torch.einsum("bkn,bdn->bkd", gamma, local.square())
        var_hat = (second_moment - mu_hat_ot.square()).clamp_min(self.eps)
        sigma_hat_ot = torch.sqrt(var_hat)

        delta_mu, delta_log_sigma = self.estimate_refiner(local, mu, log_sigma)
        mu_hat = mu_hat_ot + delta_mu
        log_sigma_hat = torch.log(sigma_hat_ot.clamp_min(self.eps)) + delta_log_sigma
        sigma_hat = torch.exp(log_sigma_hat).clamp_min(self.eps)

        mean_shift = mu_hat - mu.unsqueeze(0)
        sigma_shift = sigma_hat - torch.exp(log_sigma).clamp_min(self.eps).unsqueeze(0)

        if self.mass_preserving:
            mass_weight = cluster_mass * (self.num_clusters / max(1.0 - self.dustbin_mass, self.eps))
            mass_weight = mass_weight.clamp_min(self.eps).pow(self.mass_power)
            mean_shift = mean_shift * mass_weight
            sigma_shift = sigma_shift * mass_weight

        residual = torch.cat([mean_shift, sigma_shift], dim=-1)
        descriptor = residual.reshape(local.size(0), -1)
        return F.normalize(descriptor, p=2, dim=-1)
