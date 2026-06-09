"""
OT-GaussVLAD + QAA-style CS with a self-attentive Gaussian codebook.

This conservative variant keeps the best-performing CS branch and cluster
fusion path from OTGaussVLADCSLayer, while replacing the independent Gaussian
prototype parameters used by the OT branch with a globally shared,
self-attentive query-generated codebook:

    Gaussian queries -> self-attention -> bounded residuals for mu/log_sigma.

The codebook is still global, not image-conditioned, so descriptors remain in a
stable shared coordinate system for retrieval.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ot_gaussvlad_cs import OTGaussVLADCSLayer, QuerySelfAttention, inverse_sigmoid


class OTGaussVLADCSQueryCodebookLayer(OTGaussVLADCSLayer):
    """
    OT-GaussVLAD-CS with query-structured Gaussian prototypes.

    The parent layer already provides:
        - local token projection
        - token-mass estimation
        - Sinkhorn transport
        - QAA-style CS branch
        - cluster-aligned fusion

    This subclass only changes how the Gaussian OT branch obtains mu and
    log_sigma. Instead of using each prototype as an isolated parameter, a
    self-attentive Gaussian query set produces small residual refinements over
    the base mu/log_sigma parameters.
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
        mu = self.mu + mu_scale.to(dtype=self.mu.dtype) * delta_mu.to(dtype=self.mu.dtype)
        log_sigma = self.log_sigma + sigma_scale.to(dtype=self.log_sigma.dtype) * delta_log_sigma.to(dtype=self.log_sigma.dtype)
        return mu, log_sigma

    def _sigma(self) -> torch.Tensor:
        _, log_sigma = self._gaussian_codebook()
        return torch.exp(log_sigma).clamp_min(self.eps)

    def _gmm_scores(self, x: torch.Tensor, mu=None, log_sigma=None) -> torch.Tensor:
        if mu is None or log_sigma is None:
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

    def _attention_scores(self, x: torch.Tensor, mu=None) -> torch.Tensor:
        if mu is None:
            mu, _ = self._gaussian_codebook()

        centers = F.normalize(mu, p=2, dim=-1)
        tokens = F.normalize(x, p=2, dim=1)
        scale = F.softplus(self.attention_scale) + self.eps
        return scale * torch.einsum("kd,bdn->bkn", centers, tokens)

    def _compute_ot_residual(self, local: torch.Tensor, token_mass: torch.Tensor) -> torch.Tensor:
        mu, log_sigma = self._gaussian_codebook()
        if self.score_mode == "gmm":
            logits = self._gmm_scores(local, mu, log_sigma)
        else:
            logits = self._attention_scores(local, mu)

        assignments = self._transport(logits, token_mass)  # [B, K, N]
        cluster_mass = assignments.sum(dim=-1, keepdim=True)  # [B, K, 1]
        gamma = assignments / cluster_mass.clamp_min(self.eps)

        mu_hat = torch.einsum("bkn,bdn->bkd", gamma, local)
        diff = local.transpose(1, 2).unsqueeze(1) - mu_hat.unsqueeze(2)
        var_hat = torch.einsum("bkn,bknd->bkd", gamma, diff.square()).clamp_min(self.eps)
        sigma_hat = torch.sqrt(var_hat)

        mean_shift = mu_hat - mu.unsqueeze(0)
        sigma_shift = sigma_hat - torch.exp(log_sigma).clamp_min(self.eps).unsqueeze(0)

        if self.mass_preserving:
            mass_weight = cluster_mass * (self.num_clusters / max(1.0 - self.dustbin_mass, self.eps))
            mass_weight = mass_weight.clamp_min(self.eps).pow(self.mass_power)
            mean_shift = mean_shift * mass_weight
            sigma_shift = sigma_shift * mass_weight

        return torch.cat([mean_shift, sigma_shift], dim=-1)
