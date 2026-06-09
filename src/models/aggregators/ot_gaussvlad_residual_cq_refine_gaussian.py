"""
OT-anchored Residual Cross-query Gaussian Refinement.

This variant keeps the self-attentive Gaussian OT descriptor as the anchor and
uses residual cross-query attention only as a conservative refinement:

    r_ot -> residual queries -> cluster-aligned refinement
    r_final = r_ot + alpha * r_refine

The cluster identity embedding preserves the fixed Gaussian prototype order
while residual queries read and re-code the OT residual tensor.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ot_gaussvlad_gaussian_query_codebook import (
    OTGaussVLADGaussianQueryCodebookLayer,
    QuerySelfAttention,
    inverse_sigmoid,
)

__all__ = ["OTGaussVLADResidualCQRefineGaussianLayer"]


class OTGaussVLADResidualCQRefineGaussianLayer(OTGaussVLADGaussianQueryCodebookLayer):
    """
    OT-GaussVLAD with residual cross-query refinement.

    The base OT/Gaussian branch builds:
        r_ot = [mu_hat - mu, sigma_hat - sigma]  # [B, K, 2D]

    Residual queries read r_ot with cluster identity embeddings and predict a
    cluster-aligned refinement r_refine. The final descriptor remains anchored
    to the original OT residual:
        r_final = r_ot + alpha * r_refine
    """

    def __init__(
        self,
        *args,
        residual_num_queries=256,
        residual_query_nheads=8,
        residual_reference_nheads=8,
        residual_self_attn=True,
        residual_attn_dropout=0.0,
        residual_input_norm=True,
        use_cluster_embedding=True,
        cluster_embedding_init_std=0.02,
        refine_scale_init=0.05,
        refine_max_scale=1.0,
        refine_tanh=True,
        refine_scale_mode="sqrt",
        descriptor_dim=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if refine_scale_mode not in {"none", "sqrt"}:
            raise ValueError("refine_scale_mode must be either 'none' or 'sqrt'.")
        if not 0.0 < float(refine_scale_init) < float(refine_max_scale):
            raise ValueError("refine_scale_init must be in (0, refine_max_scale).")
        if float(refine_max_scale) <= 0:
            raise ValueError("refine_max_scale must be positive.")

        self.raw_cluster_dim = 2 * self.cluster_dim
        self.raw_dim = self.num_clusters * self.raw_cluster_dim
        self.residual_num_queries = int(residual_num_queries)
        self.residual_input_norm = bool(residual_input_norm)
        self.use_cluster_embedding = bool(use_cluster_embedding)
        self.refine_max_scale = float(refine_max_scale)
        self.refine_tanh = bool(refine_tanh)
        self.refine_scale_mode = refine_scale_mode

        if self.raw_cluster_dim % int(residual_query_nheads) != 0:
            raise ValueError("2 * cluster_dim must be divisible by residual_query_nheads.")
        if self.num_clusters % int(residual_reference_nheads) != 0:
            raise ValueError("num_clusters must be divisible by residual_reference_nheads.")

        self.residual_norm = nn.LayerNorm(self.raw_cluster_dim)
        if self.use_cluster_embedding:
            self.cluster_embedding = nn.Parameter(torch.empty(1, self.num_clusters, self.raw_cluster_dim))
            nn.init.normal_(self.cluster_embedding, mean=0.0, std=float(cluster_embedding_init_std))
        else:
            self.register_parameter("cluster_embedding", None)

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
        self.residual_value_proj = nn.Linear(self.raw_cluster_dim, self.raw_cluster_dim)
        self.residual_value_norm = nn.LayerNorm(self.raw_cluster_dim)

        # Query-to-cluster reference codebook. It maps query-level residual
        # features back to the fixed Gaussian cluster coordinates.
        self.residual_cluster_reference = QuerySelfAttention(
            embed_dim=self.num_clusters,
            num_queries=self.residual_num_queries,
            num_heads=int(residual_reference_nheads),
            dropout=float(residual_attn_dropout),
            self_attn=bool(residual_self_attn),
            out_norm=True,
        )

        scale_ratio = float(refine_scale_init) / self.refine_max_scale
        self.refine_scale_logit = nn.Parameter(torch.tensor(inverse_sigmoid(scale_ratio)))

        if descriptor_dim is not None and int(descriptor_dim) != self.raw_dim:
            raise ValueError(f"descriptor_dim must be omitted or equal to {self.raw_dim}, got {descriptor_dim}.")
        self.output_dim = self.raw_dim

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

    def _compute_residual_refinement(self, residual: torch.Tensor) -> torch.Tensor:
        b = residual.size(0)
        tokens = self.residual_norm(residual) if self.residual_input_norm else residual
        if self.cluster_embedding is not None:
            tokens = tokens + self.cluster_embedding.to(dtype=tokens.dtype, device=tokens.device)

        queries = self.residual_queries().expand(b, -1, -1)
        query_features = self.residual_cross_attn(queries, tokens, tokens, need_weights=False)[0]
        query_features = self.residual_cross_norm(query_features)
        query_features = self.residual_value_norm(self.residual_value_proj(query_features))  # [B, Q, 2D]

        reference = self.residual_cluster_reference().squeeze(0)  # [Q, K]
        refinement = torch.einsum("qk,bqc->bkc", reference, query_features)
        if self.refine_scale_mode == "sqrt":
            refinement = refinement / math.sqrt(max(self.residual_num_queries, 1))
        if self.refine_tanh:
            refinement = torch.tanh(refinement)
        return refinement

    def _refine_scale(self) -> torch.Tensor:
        return self.refine_max_scale * torch.sigmoid(self.refine_scale_logit)

    def forward(self, x):
        x = self._split_backbone_output(x)
        local = self._extract_local_tokens(x)  # [B, D, N]
        token_mass = self._compute_token_mass(local)  # [B, N]
        residual = self._compute_gaussian_ot_residual(local, token_mass)  # [B, K, 2D]
        refinement = self._compute_residual_refinement(residual)
        fused = residual + self._refine_scale().to(dtype=residual.dtype, device=residual.device) * refinement
        descriptor = fused.reshape(fused.size(0), -1)
        return F.normalize(descriptor, p=2, dim=-1)
