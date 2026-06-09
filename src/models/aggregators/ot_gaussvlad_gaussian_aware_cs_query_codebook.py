"""
OT-GaussVLAD + self-attentive Gaussian codebook + Gaussian-aware CS.

This variant keeps the current best OT/Gaussian backbone from
OTGaussVLADCSQueryCodebookLayer and replaces the plain QAA-style dot-product
CS branch with a distribution-aware CS branch:

    feature queries attend local tokens -> query-level diagonal Gaussians
    reference queries -> reference diagonal Gaussian codebook
    CS = -W2^2(query Gaussian, reference Gaussian)

The output shape remains aligned with the OT residual [B, K, 2D], so the
existing cluster_fusion path is preserved for a clean ablation against the
plain CS version.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ot_gaussvlad_cs_query_codebook import OTGaussVLADCSQueryCodebookLayer

__all__ = ["OTGaussVLADGaussianAwareCSQueryCodebookLayer"]


def inverse_softplus(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError("Softplus target must be positive.")
    return math.log(math.expm1(value))


class OTGaussVLADGaussianAwareCSQueryCodebookLayer(OTGaussVLADCSQueryCodebookLayer):
    """
    Distribution-aware Cross-query Similarity for OT-GaussVLAD.

    The parent class provides:
        - DINO local token projection
        - learned token mass
        - Gaussian OT assignment
        - self-attentive Gaussian prototype codebook
        - OT residual [mu_hat - mu, sigma_hat - sigma]
        - cluster-aligned fusion

    This subclass only changes the CS branch. Instead of computing a plain
    reference_query^T feature_query matrix, each image query and reference query
    is interpreted as a diagonal Gaussian, and their relation is computed by a
    W2-style negative distance.
    """

    def __init__(
        self,
        *args,
        gcs_reduce="sum_sqrt",
        gcs_center=True,
        gcs_score_scale_init=0.1,
        gcs_sigma_weight_init=1.0,
        gcs_learn_sigma_weight=True,
        gcs_log_sigma_bias=0.0,
        gcs_log_sigma_scale=2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if gcs_reduce not in {"mean", "sum_sqrt"}:
            raise ValueError("gcs_reduce must be either 'mean' or 'sum_sqrt'.")
        if float(gcs_log_sigma_scale) <= 0.0:
            raise ValueError("gcs_log_sigma_scale must be positive.")

        self.gcs_reduce = gcs_reduce
        self.gcs_center = bool(gcs_center)
        self.gcs_log_sigma_bias = float(gcs_log_sigma_bias)
        self.gcs_log_sigma_scale = float(gcs_log_sigma_scale)

        # The plain CS projection/norm from the parent is not used by this
        # Gaussian-aware branch. Replacing them avoids carrying dead parameters.
        self.cs_feature_proj = nn.Identity()
        self.cs_feature_norm = nn.Identity()

        self.gcs_image_mean = nn.Linear(self.cluster_dim, self.cs_feature_dim)
        self.gcs_image_log_sigma = nn.Linear(self.cluster_dim, self.cs_feature_dim)
        self.gcs_image_mean_norm = nn.LayerNorm(self.cs_feature_dim)

        self.gcs_reference_mean = nn.Linear(self.cs_reference_dim, self.cs_reference_dim)
        self.gcs_reference_log_sigma = nn.Linear(self.cs_reference_dim, self.cs_reference_dim)
        self.gcs_reference_mean_norm = nn.LayerNorm(self.cs_reference_dim)

        # Small but non-zero log-sigma initialization keeps the branch stable
        # while still allowing sigma terms to receive gradients from step 1.
        nn.init.normal_(self.gcs_image_log_sigma.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.gcs_image_log_sigma.bias)
        nn.init.normal_(self.gcs_reference_log_sigma.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.gcs_reference_log_sigma.bias)

        self.gcs_log_score_scale = nn.Parameter(torch.tensor(inverse_softplus(float(gcs_score_scale_init))))
        if gcs_learn_sigma_weight:
            self.gcs_log_sigma_weight = nn.Parameter(torch.tensor(inverse_softplus(float(gcs_sigma_weight_init))))
        else:
            self.register_buffer(
                "gcs_log_sigma_weight",
                torch.tensor(inverse_softplus(float(gcs_sigma_weight_init))),
                persistent=True,
            )

    def _bounded_log_sigma(self, raw: torch.Tensor) -> torch.Tensor:
        return self.gcs_log_sigma_bias + self.gcs_log_sigma_scale * torch.tanh(raw)

    def _query_gaussians(self, local: torch.Tensor):
        b = local.size(0)
        local_tokens = local.transpose(1, 2).contiguous()  # [B, N, D]

        feature_queries = self.cs_feature_queries().expand(b, -1, -1)
        query_state = self.cs_cross_attn(feature_queries, local_tokens, local_tokens, need_weights=False)[0]
        query_state = self.cs_cross_norm(query_state)

        image_mean = self.gcs_image_mean_norm(self.gcs_image_mean(query_state))  # [B, Q, Cf]
        image_log_sigma = self._bounded_log_sigma(self.gcs_image_log_sigma(query_state))

        reference_state = self.cs_reference_queries().squeeze(0)  # [Q, Cr]
        reference_mean = self.gcs_reference_mean_norm(self.gcs_reference_mean(reference_state))
        reference_log_sigma = self._bounded_log_sigma(self.gcs_reference_log_sigma(reference_state))

        return image_mean, image_log_sigma, reference_mean, reference_log_sigma

    def _reduce_query_scores(self, scores: torch.Tensor) -> torch.Tensor:
        if self.gcs_reduce == "mean":
            return scores.mean(dim=1)
        return scores.sum(dim=1) / math.sqrt(max(self.cs_num_queries, 1))

    def _w2_gaussian_cs(
        self,
        image_mean: torch.Tensor,
        image_log_sigma: torch.Tensor,
        reference_mean: torch.Tensor,
        reference_log_sigma: torch.Tensor,
    ) -> torch.Tensor:
        image_sigma = torch.exp(image_log_sigma).clamp_min(self.eps)
        reference_sigma = torch.exp(reference_log_sigma).clamp_min(self.eps)

        mean_diff = reference_mean.unsqueeze(0).unsqueeze(-1) - image_mean.unsqueeze(2)
        sigma_diff = reference_sigma.unsqueeze(0).unsqueeze(-1) - image_sigma.unsqueeze(2)

        sigma_weight = F.softplus(self.gcs_log_sigma_weight) + self.eps
        score_scale = F.softplus(self.gcs_log_score_scale) + self.eps

        cost = mean_diff.square() + sigma_weight * sigma_diff.square()
        cs = -score_scale * self._reduce_query_scores(cost)  # [B, Cr, Cf]

        if self.gcs_center:
            cs = cs - cs.mean(dim=1, keepdim=True)
        return cs

    def _compute_cs_residual(self, local: torch.Tensor) -> torch.Tensor:
        image_mean, image_log_sigma, reference_mean, reference_log_sigma = self._query_gaussians(local)
        cs = self._w2_gaussian_cs(image_mean, image_log_sigma, reference_mean, reference_log_sigma)

        if self.cs_intra_norm:
            cs = F.normalize(cs, p=2, dim=1)

        if self.fusion_mode in {"cluster_fusion", "residual_gated"}:
            return cs.transpose(1, 2).contiguous()  # [B, K, 2D]
        return cs.reshape(local.size(0), -1)
