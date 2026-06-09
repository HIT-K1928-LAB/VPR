"""
OT-GaussVLAD with a self-attentive Gaussian query codebook.

This conservative variant keeps only the OT/Gaussian residual branch from
OT-GaussVLAD and makes the Gaussian prototypes slightly adaptive through a
global query codebook:

    gaussian_queries -> self-attention -> bounded residuals for mu/log_sigma.

The final descriptor is still the standard [mu_hat - mu, sigma_hat - sigma]
residual flattened over clusters.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ot_gaussvlad import OTGaussVLADDiagCovLayer

__all__ = ["OTGaussVLADGaussianQueryCodebookLayer"]


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


class OTGaussVLADGaussianQueryCodebookLayer(OTGaussVLADDiagCovLayer):
    """
    OT-GaussVLAD with a self-attentive Gaussian codebook.

    The OT branch is unchanged in spirit:
        - local token projection
        - Gaussian score or attention score
        - Sinkhorn transport
        - [mu_hat - mu, sigma_hat - sigma]

    Only the prototype parameters become query-structured:
        base mu/log_sigma + bounded residuals from gaussian_queries.
    """

    def __init__(
        self,
        *args,
        gaussian_query_dim=None,
        gaussian_query_nheads=8,
        gaussian_self_attn=True,
        gaussian_attn_dropout=0.0,
        gaussian_mu_scale_init=0.05,
        gaussian_sigma_scale_init=0.05,
        gaussian_max_residual_scale=1.0,
        gaussian_residual_tanh=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.gaussian_query_dim = self.cluster_dim if gaussian_query_dim is None else int(gaussian_query_dim)
        self.gaussian_max_residual_scale = float(gaussian_max_residual_scale)
        self.gaussian_residual_tanh = bool(gaussian_residual_tanh)

        if self.gaussian_query_dim % int(gaussian_query_nheads) != 0:
            raise ValueError("gaussian_query_dim must be divisible by gaussian_query_nheads.")
        if self.gaussian_max_residual_scale <= 0:
            raise ValueError("gaussian_max_residual_scale must be positive.")
        if not 0.0 < float(gaussian_mu_scale_init) < self.gaussian_max_residual_scale:
            raise ValueError("gaussian_mu_scale_init must be in (0, gaussian_max_residual_scale).")
        if not 0.0 < float(gaussian_sigma_scale_init) < self.gaussian_max_residual_scale:
            raise ValueError("gaussian_sigma_scale_init must be in (0, gaussian_max_residual_scale).")

        self.gaussian_queries = QuerySelfAttention(
            embed_dim=self.gaussian_query_dim,
            num_queries=self.num_clusters,
            num_heads=int(gaussian_query_nheads),
            dropout=float(gaussian_attn_dropout),
            self_attn=bool(gaussian_self_attn),
            out_norm=True,
        )
        self.gaussian_mu_head = nn.Linear(self.gaussian_query_dim, self.cluster_dim)
        self.gaussian_log_sigma_head = nn.Linear(self.gaussian_query_dim, self.cluster_dim)
        nn.init.zeros_(self.gaussian_mu_head.weight)
        nn.init.zeros_(self.gaussian_mu_head.bias)
        nn.init.zeros_(self.gaussian_log_sigma_head.weight)
        nn.init.zeros_(self.gaussian_log_sigma_head.bias)

        mu_ratio = float(gaussian_mu_scale_init) / self.gaussian_max_residual_scale
        sigma_ratio = float(gaussian_sigma_scale_init) / self.gaussian_max_residual_scale
        self.gaussian_mu_scale_logit = nn.Parameter(torch.tensor(inverse_sigmoid(mu_ratio)))
        self.gaussian_sigma_scale_logit = nn.Parameter(torch.tensor(inverse_sigmoid(sigma_ratio)))

    def _gaussian_scales(self):
        mu_scale = self.gaussian_max_residual_scale * torch.sigmoid(self.gaussian_mu_scale_logit)
        sigma_scale = self.gaussian_max_residual_scale * torch.sigmoid(self.gaussian_sigma_scale_logit)
        return mu_scale, sigma_scale

    def _gaussian_codebook(self):
        slots = self.gaussian_queries().squeeze(0)  # [K, Cg]
        delta_mu = self.gaussian_mu_head(slots)
        delta_log_sigma = self.gaussian_log_sigma_head(slots)

        if self.gaussian_residual_tanh:
            delta_mu = torch.tanh(delta_mu)
            delta_log_sigma = torch.tanh(delta_log_sigma)

        mu_scale, sigma_scale = self._gaussian_scales()
        mu = self.mu + mu_scale.to(dtype=self.mu.dtype, device=self.mu.device) * delta_mu.to(
            dtype=self.mu.dtype, device=self.mu.device
        )
        log_sigma = self.log_sigma + sigma_scale.to(dtype=self.log_sigma.dtype, device=self.log_sigma.device) * delta_log_sigma.to(
            dtype=self.log_sigma.dtype, device=self.log_sigma.device
        )
        return mu, log_sigma

    def _sigma(self) -> torch.Tensor:
        _, log_sigma = self._gaussian_codebook()
        return torch.exp(log_sigma).clamp_min(self.eps)

    def _gmm_scores(self, x: torch.Tensor) -> torch.Tensor:
        mu, log_sigma = self._gaussian_codebook()

        x_tokens = x.transpose(1, 2)  # [B, N, D]
        inv_sigma_sq = torch.exp(-2.0 * log_sigma)  # [K, D]
        quadratic = torch.einsum("bnd,kd->bnk", x_tokens.square(), inv_sigma_sq)
        quadratic = quadratic - 2.0 * torch.einsum("bnd,kd->bnk", x_tokens, mu * inv_sigma_sq)
        quadratic = quadratic + (mu.square() * inv_sigma_sq).sum(dim=-1).view(1, 1, -1)
        log_sigma_sum = log_sigma.sum(dim=-1).view(1, 1, -1)
        logits = -0.5 * quadratic - log_sigma_sum
        if self.include_pi_in_scores:
            logits = logits + F.log_softmax(self.log_alpha, dim=0).view(1, 1, -1)
        return logits.transpose(1, 2).contiguous()

    def _attention_scores(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self._gaussian_codebook()
        centers = F.normalize(mu, p=2, dim=-1)
        tokens = F.normalize(x, p=2, dim=1)
        scale = F.softplus(self.attention_scale) + self.eps
        return scale * torch.einsum("kd,bdn->bkn", centers, tokens)

    def forward(self, x):
        x = self._split_backbone_output(x)
        local = self._extract_local_tokens(x)  # [B, D, N]
        token_mass = self._compute_token_mass(local)  # [B, N]
        mu, log_sigma = self._gaussian_codebook()

        if self.score_mode == "gmm":
            logits = self._gmm_scores(local)
        else:
            logits = self._attention_scores(local)

        assignments = self._transport(logits, token_mass)  # [B, K, N]
        cluster_mass = assignments.sum(dim=-1, keepdim=True)  # [B, K, 1]
        gamma = assignments / cluster_mass.clamp_min(self.eps)

        mu_hat = torch.einsum("bkn,bdn->bkd", gamma, local)
        second_moment = torch.einsum("bkn,bdn->bkd", gamma, local.square())
        var_hat = (second_moment - mu_hat.square()).clamp_min(self.eps)
        sigma_hat = torch.sqrt(var_hat)

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
