"""
Pure Laplace OT-VLAD aggregation.

This layer replaces the Gaussian modeling used by OT-GaussVLAD with a diagonal
Laplace distribution. The transport score is the negative Laplace NLL, and the
default descriptor is a Laplace score residual:

    g_mu = E_gamma[sign(x - mu) / b]
    g_b  = E_gamma[|x - mu| / b - 1]

where mu is the Laplace location codebook, b is the positive Laplace scale
codebook, and gamma is the cluster-normalized OT assignment.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_softplus(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError("Softplus target must be positive.")
    return math.log(math.expm1(value))


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


class LaplaceOTVLADLayer(nn.Module):
    """
    OT-VLAD based on diagonal Laplace distributions.

    score:
        negative diagonal Laplace NLL
    assignment:
        Sinkhorn + dustbin
    aggregation:
        score:              [E sign(x-mu)/b, E |x-mu|/b - 1]
        moment:             [mu_hat - mu, b_hat - b]
        standardized_moment: [(mu_hat - mu)/b, log(b_hat/b)]
    """

    def __init__(
        self,
        in_channels=768,
        num_clusters=64,
        cluster_dim=64,
        hidden_channels=768,
        descriptor_dim=None,
        include_pi_in_scores=False,
        include_laplace_normalizer=True,
        normalize_cost_by_dim=True,
        sinkhorn_iters=3,
        dustbin_mass=0.05,
        dropout=0.0,
        mass_preserving=True,
        mass_power=1.0,
        token_mass_mode="learned",
        token_mass_hidden_channels=None,
        residual_mode="score",
        score_scale_init=1.0,
        smooth_abs_eps=1e-3,
        eps=1e-6,
    ):
        super().__init__()

        if token_mass_mode not in {"uniform", "learned"}:
            raise ValueError("token_mass_mode must be either 'uniform' or 'learned'.")
        if residual_mode not in {"score", "moment", "standardized_moment"}:
            raise ValueError("residual_mode must be one of: score, moment, standardized_moment.")
        if not 0.0 < dustbin_mass < 1.0:
            raise ValueError("dustbin_mass must be in (0, 1).")

        self.in_channels = int(in_channels)
        self.num_clusters = int(num_clusters)
        self.cluster_dim = int(cluster_dim)
        self.hidden_channels = int(hidden_channels)
        self.include_pi_in_scores = bool(include_pi_in_scores)
        self.include_laplace_normalizer = bool(include_laplace_normalizer)
        self.normalize_cost_by_dim = bool(normalize_cost_by_dim)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.dustbin_mass = float(dustbin_mass)
        self.mass_preserving = bool(mass_preserving)
        self.mass_power = float(mass_power)
        self.token_mass_mode = token_mass_mode
        self.residual_mode = residual_mode
        self.smooth_abs_eps = float(smooth_abs_eps)
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
        self.log_b = nn.Parameter(torch.zeros(self.num_clusters, self.cluster_dim))
        self.log_alpha = nn.Parameter(torch.zeros(self.num_clusters))

        self.log_score_scale = nn.Parameter(torch.tensor(inverse_softplus(score_scale_init)))
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

        self.raw_cluster_dim = 2 * self.cluster_dim
        self.raw_dim = self.num_clusters * self.raw_cluster_dim
        if descriptor_dim is not None and int(descriptor_dim) != self.raw_dim:
            raise ValueError(
                "Projection has been removed for OT alignment. "
                f"descriptor_dim must be omitted or equal to raw_dim={self.raw_dim}, got {descriptor_dim}."
            )
        self.output_dim = self.raw_dim

    @staticmethod
    def _split_backbone_output(x):
        if isinstance(x, tuple):
            return x[0]
        return x

    def _extract_local_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local_features(x).flatten(2).transpose(1, 2)
        x = self.token_norm(x)
        return x.transpose(1, 2).contiguous()

    def _cluster_prior(self) -> torch.Tensor:
        return F.softmax(self.log_alpha, dim=0)

    def _scale(self) -> torch.Tensor:
        return torch.exp(self.log_b).clamp_min(self.eps)

    def _smooth_abs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x.square() + self.smooth_abs_eps * self.smooth_abs_eps)

    def _compute_token_mass(self, local: torch.Tensor) -> torch.Tensor:
        b, _, n = local.shape
        if self.token_mass_mode == "uniform":
            return local.new_full((b, n), 1.0 / max(n, 1))

        token_logits = self.token_mass_head(local).squeeze(1)
        scale = F.softplus(self.token_mass_scale) + self.eps
        return F.softmax(scale * token_logits, dim=-1).clamp_min(self.eps)

    def _laplace_scores(self, local: torch.Tensor) -> torch.Tensor:
        tokens = local.transpose(1, 2).unsqueeze(2)  # [B, N, 1, D]
        mu = self.mu.view(1, 1, self.num_clusters, self.cluster_dim)
        scale = self._scale().view(1, 1, self.num_clusters, self.cluster_dim)

        abs_residual = self._smooth_abs(tokens - mu)
        cost = abs_residual / scale
        if self.include_laplace_normalizer:
            cost = cost + math.log(2.0) + self.log_b.view(1, 1, self.num_clusters, self.cluster_dim)
        cost = cost.sum(dim=-1)
        if self.normalize_cost_by_dim:
            cost = cost / max(self.cluster_dim, 1)

        score_scale = F.softplus(self.log_score_scale) + self.eps
        logits = -score_scale * cost
        if self.include_pi_in_scores:
            logits = logits + F.log_softmax(self.log_alpha, dim=0).view(1, 1, -1)
        return logits.transpose(1, 2).contiguous()

    def _transport(self, logits: torch.Tensor, token_mass: torch.Tensor) -> torch.Tensor:
        b, _, n = logits.shape
        dustbin = self.dust_bin.to(dtype=logits.dtype).expand(b, 1, n)
        scores = torch.cat([logits, dustbin], dim=1)

        cluster_mass = ((1.0 - self.dustbin_mass) * self._cluster_prior()).to(dtype=logits.dtype)
        cluster_mass = cluster_mass.unsqueeze(0).expand(b, -1)
        dustbin_mass = logits.new_full((b, 1), self.dustbin_mass)
        source_mass = torch.cat([cluster_mass, dustbin_mass], dim=1)

        target_mass = token_mass / token_mass.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        transport = log_sinkhorn_iterations(
            scores,
            torch.log(source_mass.clamp_min(self.eps)),
            torch.log(target_mass.clamp_min(self.eps)),
            self.sinkhorn_iters,
        )
        return torch.exp(transport[:, :-1, :])

    def _laplace_residual(self, local: torch.Tensor, gamma: torch.Tensor):
        tokens = local.transpose(1, 2).unsqueeze(1)  # [B, 1, N, D]
        mu = self.mu.view(1, self.num_clusters, 1, self.cluster_dim)
        scale = self._scale().view(1, self.num_clusters, 1, self.cluster_dim)
        residual = tokens - mu
        abs_residual = self._smooth_abs(residual)

        if self.residual_mode == "score":
            smooth_sign = residual / abs_residual.clamp_min(self.eps)
            location_feat = torch.einsum("bkn,bknd->bkd", gamma, smooth_sign / scale)
            scale_feat = torch.einsum("bkn,bknd->bkd", gamma, abs_residual / scale - 1.0)
            return location_feat, scale_feat

        mu_hat = torch.einsum("bkn,bdn->bkd", gamma, local)
        centered = local.transpose(1, 2).unsqueeze(1) - mu_hat.unsqueeze(2)
        b_hat = torch.einsum("bkn,bknd->bkd", gamma, self._smooth_abs(centered)).clamp_min(self.eps)
        cluster_scale = self._scale().unsqueeze(0)

        if self.residual_mode == "moment":
            location_feat = mu_hat - self.mu.unsqueeze(0)
            scale_feat = b_hat - cluster_scale
        else:
            location_feat = (mu_hat - self.mu.unsqueeze(0)) / cluster_scale.clamp_min(self.eps)
            scale_feat = torch.log(b_hat) - self.log_b.unsqueeze(0)
        return location_feat, scale_feat

    def forward(self, x):
        x = self._split_backbone_output(x)
        local = self._extract_local_tokens(x)
        token_mass = self._compute_token_mass(local)
        logits = self._laplace_scores(local)

        assignments = self._transport(logits, token_mass)
        cluster_mass = assignments.sum(dim=-1, keepdim=True)
        gamma = assignments / cluster_mass.clamp_min(self.eps)

        location_feat, scale_feat = self._laplace_residual(local, gamma)

        if self.mass_preserving:
            mass_weight = cluster_mass * (self.num_clusters / max(1.0 - self.dustbin_mass, self.eps))
            mass_weight = mass_weight.clamp_min(self.eps).pow(self.mass_power)
            location_feat = location_feat * mass_weight
            scale_feat = scale_feat * mass_weight

        descriptor = torch.cat([location_feat, scale_feat], dim=-1).reshape(local.size(0), -1)
        return F.normalize(descriptor, p=2, dim=-1)
