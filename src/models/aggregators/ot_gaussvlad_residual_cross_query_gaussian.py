"""
Residual Cross-query Gaussian Aggregation.

This layer keeps the self-attentive Gaussian OT branch from
OTGaussVLADGaussianQueryCodebookLayer, but changes the final descriptor
construction:

    Gaussian OT residual -> residual queries -> cross-query descriptor.

Unlike OT-GaussVLAD-CS, the CS path is not a side branch fused with the OT
descriptor. It directly re-encodes the Gaussian residual tensor and produces
the final global descriptor.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ot_gaussvlad_gaussian_query_codebook import (
    OTGaussVLADGaussianQueryCodebookLayer,
    QuerySelfAttention,
)

__all__ = ["OTGaussVLADResidualCrossQueryGaussianLayer"]


class OTGaussVLADResidualCrossQueryGaussianLayer(OTGaussVLADGaussianQueryCodebookLayer):
    """
    OT-GaussVLAD with residual cross-query descriptor construction.

    The Gaussian OT branch first builds:
        r_ot = [mu_hat - mu, sigma_hat - sigma]  # [B, K, 2D]

    Then learned residual queries read r_ot with cross-attention, and an
    independent reference residual codebook maps the query-level residual
    features into the final descriptor:
        S = F_ref^T P_r  # [B, Cr, Cf]
    """

    def __init__(
        self,
        *args,
        residual_num_queries=256,
        residual_feature_dim=None,
        residual_reference_dim=None,
        residual_query_nheads=8,
        residual_reference_nheads=8,
        residual_self_attn=True,
        residual_attn_dropout=0.0,
        residual_intra_norm=True,
        residual_scale_mode="sqrt",
        residual_input_norm=True,
        descriptor_dim=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if residual_scale_mode not in {"none", "sqrt"}:
            raise ValueError("residual_scale_mode must be either 'none' or 'sqrt'.")

        self.residual_num_queries = int(residual_num_queries)
        self.residual_feature_dim = self.num_clusters if residual_feature_dim is None else int(residual_feature_dim)
        self.residual_reference_dim = (
            2 * self.cluster_dim if residual_reference_dim is None else int(residual_reference_dim)
        )
        self.residual_intra_norm = bool(residual_intra_norm)
        self.residual_scale_mode = residual_scale_mode
        self.residual_input_norm = bool(residual_input_norm)
        self.raw_cluster_dim = 2 * self.cluster_dim

        if self.raw_cluster_dim % int(residual_query_nheads) != 0:
            raise ValueError("2 * cluster_dim must be divisible by residual_query_nheads.")
        if self.residual_reference_dim % int(residual_reference_nheads) != 0:
            raise ValueError("residual_reference_dim must be divisible by residual_reference_nheads.")

        self.residual_norm = nn.LayerNorm(self.raw_cluster_dim)
        self.residual_queries = QuerySelfAttention(
            embed_dim=self.raw_cluster_dim,
            num_queries=self.residual_num_queries,
            num_heads=int(residual_query_nheads),
            dropout=float(residual_attn_dropout),
            self_attn=bool(residual_self_attn),
            out_norm=True,
        )
        self.residual_cross_attn = nn.MultiheadAttention(
            embed_dim=self.raw_cluster_dim,
            num_heads=int(residual_query_nheads),
            dropout=float(residual_attn_dropout),
            batch_first=True,
        )
        self.residual_cross_norm = nn.LayerNorm(self.raw_cluster_dim)
        self.residual_feature_proj = nn.Linear(self.raw_cluster_dim, self.residual_feature_dim)
        self.residual_feature_norm = nn.LayerNorm(self.residual_feature_dim)
        self.residual_reference_queries = QuerySelfAttention(
            embed_dim=self.residual_reference_dim,
            num_queries=self.residual_num_queries,
            num_heads=int(residual_reference_nheads),
            dropout=float(residual_attn_dropout),
            self_attn=bool(residual_self_attn),
            out_norm=True,
        )

        expected_dim = self.residual_feature_dim * self.residual_reference_dim
        if descriptor_dim is not None and int(descriptor_dim) != expected_dim:
            raise ValueError(f"descriptor_dim must be omitted or equal to {expected_dim}, got {descriptor_dim}.")
        self.output_dim = expected_dim

    def _compute_gaussian_ot_residual(self, local: torch.Tensor, token_mass: torch.Tensor) -> torch.Tensor:
        mu, log_sigma = self._gaussian_codebook()
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
            logits = logits.transpose(1, 2).contiguous()
        else:
            centers = F.normalize(mu, p=2, dim=-1)
            tokens = F.normalize(local, p=2, dim=1)
            scale = F.softplus(self.attention_scale) + self.eps
            logits = scale * torch.einsum("kd,bdn->bkn", centers, tokens)

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

        return torch.cat([mean_shift, sigma_shift], dim=-1)

    def _compute_residual_cs_descriptor(self, residual: torch.Tensor) -> torch.Tensor:
        b = residual.size(0)
        residual_tokens = self.residual_norm(residual) if self.residual_input_norm else residual

        queries = self.residual_queries().expand(b, -1, -1)
        query_features = self.residual_cross_attn(
            queries,
            residual_tokens,
            residual_tokens,
            need_weights=False,
        )[0]
        query_features = self.residual_cross_norm(query_features)
        query_features = self.residual_feature_norm(self.residual_feature_proj(query_features))  # [B, Q, Cf]

        reference_codebook = self.residual_reference_queries().squeeze(0)  # [Q, Cr]
        descriptor = torch.einsum("qr,bqf->brf", reference_codebook, query_features)
        if self.residual_scale_mode == "sqrt":
            descriptor = descriptor / math.sqrt(max(self.residual_num_queries, 1))
        if self.residual_intra_norm:
            descriptor = F.normalize(descriptor, p=2, dim=1)

        return descriptor.reshape(b, -1)

    def forward(self, x):
        x = self._split_backbone_output(x)
        local = self._extract_local_tokens(x)  # [B, D, N]
        token_mass = self._compute_token_mass(local)  # [B, N]
        residual = self._compute_gaussian_ot_residual(local, token_mass)  # [B, K, 2D]
        descriptor = self._compute_residual_cs_descriptor(residual)
        return F.normalize(descriptor, p=2, dim=-1)
