"""
OT-GaussVLAD with self-attentive Gaussian codebook, OT-anchored refinement,
uncertainty gating, and a lightweight calibration regularizer.

This file intentionally does not modify the existing
ot_gaussvlad_gaussian_query_codebook_ot_anchored_refine.py implementation.

Pipeline:
    local tokens -> OT assignment -> OT image Gaussian moments
    Gaussian-conditioned cluster queries -> small moment corrections
    uncertainty gate(cluster mass, assignment entropy, sigma disagreement)
    -> gated corrections -> Gaussian residual descriptor.

The calibration loss is exposed through a small loss wrapper in this same
module. OpenVPRLab's training loop only passes descriptors and labels to the
loss, so the aggregator stores the latest auxiliary loss in a module-level
slot that the wrapper reads immediately after forward.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ot_gaussvlad_gaussian_query_codebook import (
    OTGaussVLADGaussianQueryCodebookLayer,
    inverse_sigmoid,
)
from .ot_gaussvlad_gaussian_query_codebook_ot_anchored_refine import (
    GaussianEstimateRefiner,
)
from src.losses.vpr_losses import VPRLossFunction

__all__ = [
    "OTGaussVLADGaussianQueryCodebookOTAnchoredUncertaintyLayer",
    "VPRLossWithAggregatorCalibration",
]


_LAST_CALIBRATION_LOSS: Optional[torch.Tensor] = None


def _set_last_calibration_loss(loss: torch.Tensor) -> None:
    global _LAST_CALIBRATION_LOSS
    _LAST_CALIBRATION_LOSS = loss


def _get_last_calibration_loss() -> Optional[torch.Tensor]:
    return _LAST_CALIBRATION_LOSS


class ClusterUncertaintyGate(nn.Module):
    """Predict per-cluster refinement gates from OT reliability signals."""

    def __init__(
        self,
        hidden_dim: int = 32,
        gate_init: float = 0.5,
        gate_min: float = 0.0,
        gate_max: float = 1.0,
        detach_stats: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if not 0.0 <= gate_min < gate_max <= 1.0:
            raise ValueError("Expected 0 <= gate_min < gate_max <= 1.")
        if not gate_min < gate_init < gate_max:
            raise ValueError("gate_init must be inside (gate_min, gate_max).")

        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)
        self.detach_stats = bool(detach_stats)
        self.eps = float(eps)

        self.net = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

        nn.init.zeros_(self.net[-1].weight)
        ratio = (float(gate_init) - self.gate_min) / (self.gate_max - self.gate_min)
        nn.init.constant_(self.net[-1].bias, inverse_sigmoid(ratio))

    def forward(
        self,
        cluster_mass: torch.Tensor,
        gamma: torch.Tensor,
        sigma_hat_ot: torch.Tensor,
        log_sigma_ref: torch.Tensor,
        num_clusters: int,
        dustbin_mass: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, n = gamma.shape
        log_n = math.log(max(n, 2))

        mass_quality = cluster_mass * (float(num_clusters) / max(1.0 - float(dustbin_mass), self.eps))
        mass_quality = mass_quality.clamp(0.0, 1.0)

        gamma_safe = gamma.clamp_min(self.eps)
        entropy = -(gamma_safe * gamma_safe.log()).sum(dim=-1, keepdim=True) / log_n
        entropy = entropy.clamp(0.0, 1.0)
        entropy_quality = 1.0 - entropy

        log_sigma_hat = torch.log(sigma_hat_ot.clamp_min(self.eps))
        sigma_disagreement = (log_sigma_hat - log_sigma_ref.unsqueeze(0)).abs().mean(dim=-1, keepdim=True)
        sigma_quality = torch.exp(-sigma_disagreement).clamp(0.0, 1.0)

        target = (mass_quality * entropy_quality * sigma_quality).clamp(0.0, 1.0)

        stats = torch.cat(
            [
                mass_quality,
                entropy,
                sigma_disagreement.clamp_max(5.0) / 5.0,
                sigma_quality,
            ],
            dim=-1,
        )
        if self.detach_stats:
            stats = stats.detach()

        gate_logits = self.net(stats)
        gate_unit = torch.sigmoid(gate_logits)
        gate = self.gate_min + (self.gate_max - self.gate_min) * gate_unit
        target_unit = ((target - self.gate_min) / (self.gate_max - self.gate_min)).clamp(0.0, 1.0)
        return gate, target.detach(), entropy, gate_logits, target_unit.detach()


class OTGaussVLADGaussianQueryCodebookOTAnchoredUncertaintyLayer(
    OTGaussVLADGaussianQueryCodebookLayer
):
    """
    OT-anchored Gaussian refinement with uncertainty-gated corrections.

    Compared with OTAnchoredRefineLayer, this version does not blindly apply
    every predicted correction. Each cluster predicts a scalar gate from OT
    reliability signals:
        cluster mass, assignment entropy, and sigma disagreement.

    A small calibration term encourages high gates only for reliable clusters
    and suppresses correction energy for uncertain clusters.
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
        uncertainty_gate_hidden_dim=32,
        uncertainty_gate_init=0.5,
        uncertainty_gate_min=0.0,
        uncertainty_gate_max=1.0,
        uncertainty_gate_detach_stats=True,
        calibration_bce_weight=1.0,
        calibration_refine_penalty_weight=0.05,
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

        self.uncertainty_gate = ClusterUncertaintyGate(
            hidden_dim=int(uncertainty_gate_hidden_dim),
            gate_init=float(uncertainty_gate_init),
            gate_min=float(uncertainty_gate_min),
            gate_max=float(uncertainty_gate_max),
            detach_stats=bool(uncertainty_gate_detach_stats),
            eps=self.eps,
        )
        self.calibration_bce_weight = float(calibration_bce_weight)
        self.calibration_refine_penalty_weight = float(calibration_refine_penalty_weight)

        self.last_calibration_loss = None
        self.last_gate_mean = None
        self.last_gate_target_mean = None
        self.last_assignment_entropy = None

    def _scores_with_codebook(
        self,
        local: torch.Tensor,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
    ) -> torch.Tensor:
        if self.score_mode == "gmm":
            x_tokens = local.transpose(1, 2)
            inv_sigma_sq = torch.exp(-2.0 * log_sigma)
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

    def _calibration_loss(
        self,
        gate_logits: torch.Tensor,
        gate_target: torch.Tensor,
        reliability_target: torch.Tensor,
        delta_mu: torch.Tensor,
        delta_log_sigma: torch.Tensor,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            gate_logits,
            gate_target.to(dtype=gate_logits.dtype, device=gate_logits.device),
            reduction="mean",
        )

        refine_energy = 0.5 * (
            delta_mu.square().mean(dim=-1, keepdim=True)
            + delta_log_sigma.square().mean(dim=-1, keepdim=True)
        )
        uncertain_penalty = ((1.0 - reliability_target) * refine_energy).mean()
        return (
            self.calibration_bce_weight * bce
            + self.calibration_refine_penalty_weight * uncertain_penalty
        )

    def forward(self, x):
        x = self._split_backbone_output(x)
        local = self._extract_local_tokens(x)
        token_mass = self._compute_token_mass(local)
        mu, log_sigma = self._gaussian_codebook()

        logits = self._scores_with_codebook(local, mu, log_sigma)
        assignments = self._transport(logits, token_mass)
        cluster_mass = assignments.sum(dim=-1, keepdim=True)
        gamma = assignments / cluster_mass.clamp_min(self.eps)

        mu_hat_ot = torch.einsum("bkn,bdn->bkd", gamma, local)
        second_moment = torch.einsum("bkn,bdn->bkd", gamma, local.square())
        var_hat = (second_moment - mu_hat_ot.square()).clamp_min(self.eps)
        sigma_hat_ot = torch.sqrt(var_hat)

        delta_mu, delta_log_sigma = self.estimate_refiner(local, mu, log_sigma)
        gate, gate_target, entropy, gate_logits, gate_target_unit = self.uncertainty_gate(
            cluster_mass=cluster_mass,
            gamma=gamma,
            sigma_hat_ot=sigma_hat_ot,
            log_sigma_ref=log_sigma,
            num_clusters=self.num_clusters,
            dustbin_mass=self.dustbin_mass,
        )

        gated_delta_mu = gate * delta_mu
        gated_delta_log_sigma = gate * delta_log_sigma

        mu_hat = mu_hat_ot + gated_delta_mu
        log_sigma_hat = torch.log(sigma_hat_ot.clamp_min(self.eps)) + gated_delta_log_sigma
        sigma_hat = torch.exp(log_sigma_hat).clamp_min(self.eps)

        mean_shift = mu_hat - mu.unsqueeze(0)
        sigma_shift = sigma_hat - torch.exp(log_sigma).clamp_min(self.eps).unsqueeze(0)

        if self.mass_preserving:
            mass_weight = cluster_mass * (self.num_clusters / max(1.0 - self.dustbin_mass, self.eps))
            mass_weight = mass_weight.clamp_min(self.eps).pow(self.mass_power)
            mean_shift = mean_shift * mass_weight
            sigma_shift = sigma_shift * mass_weight

        calibration_loss = self._calibration_loss(
            gate_logits=gate_logits,
            gate_target=gate_target_unit,
            reliability_target=gate_target,
            delta_mu=delta_mu,
            delta_log_sigma=delta_log_sigma,
        )
        self.last_calibration_loss = calibration_loss
        self.last_gate_mean = gate.detach().mean()
        self.last_gate_target_mean = gate_target.detach().mean()
        self.last_assignment_entropy = entropy.detach().mean()
        _set_last_calibration_loss(calibration_loss)

        residual = torch.cat([mean_shift, sigma_shift], dim=-1)
        descriptor = residual.reshape(local.size(0), -1)
        return F.normalize(descriptor, p=2, dim=-1)


class VPRLossWithAggregatorCalibration(nn.Module):
    """
    Standard VPR metric loss plus the latest aggregator calibration loss.

    This wrapper is intentionally local to this new experiment. Existing
    configs that use src.losses.VPRLossFunction are untouched.
    """

    def __init__(
        self,
        loss_fn_name: str = "MultiSimilarityLoss",
        miner_name: str = "MultiSimilarityMiner",
        calibration_weight: float = 0.02,
    ):
        super().__init__()
        if calibration_weight < 0:
            raise ValueError("calibration_weight must be non-negative.")
        self.base_loss = VPRLossFunction(
            loss_fn_name=loss_fn_name,
            miner_name=miner_name,
        )
        self.calibration_weight = float(calibration_weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        base_loss, batch_accuracy = self.base_loss(embeddings, labels)
        if self.calibration_weight == 0.0:
            return base_loss, batch_accuracy

        calibration_loss = _get_last_calibration_loss()
        if calibration_loss is None:
            calibration_loss = embeddings.new_zeros(())
        else:
            calibration_loss = calibration_loss.to(device=embeddings.device, dtype=base_loss.dtype)

        return base_loss + self.calibration_weight * calibration_loss, batch_accuracy
