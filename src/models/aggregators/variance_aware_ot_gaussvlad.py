"""
Variance-aware OT-GaussVLAD aggregation.

This layer promotes local tokens from point descriptors to diagonal Gaussian
tokens. The transport score is a diagonal Gaussian W2-style cost between token
Gaussians and cluster Gaussians, followed by Sinkhorn normalization with a
dustbin. Aggregation uses the law of total variance:

    Var[X] = E[Var[X | token]] + Var[E[X | token]]

so the learned token uncertainty affects both assignment and the final
second-order residual.
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


class VarianceAwareOTGaussVLADDiagCovLayer(nn.Module):
    """
    Variance-aware OT-GaussVLAD with diagonal Gaussian tokens.

    score:
        negative diagonal Gaussian W2-style cost
    assignment:
        Sinkhorn + dustbin
    mass:
        cluster prior and optional uncertainty-aware token mass
    aggregation:
        raw:        [mu_hat - mu, sigma_hat - sigma]
        normalized: [(mu_hat - mu) / sigma, log(sigma_hat / sigma)]
    """

    def __init__(
        self,
        in_channels=768,
        num_clusters=64,
        cluster_dim=64,
        hidden_channels=768,
        descriptor_dim=None,
        include_pi_in_scores=False,
        sinkhorn_iters=3,
        dustbin_mass=0.05,
        dropout=0.0,
        mass_preserving=True,
        mass_power=1.0,
        token_mass_mode="learned_uncertainty",
        token_mass_hidden_channels=None,
        sigma_cost_weight=1.0,
        learn_sigma_cost_weight=True,
        score_scale_init=10.0,
        token_log_sigma_bias=0.0,
        token_log_sigma_scale=2.0,
        residual_mode="normalized",
        eps=1e-6,
    ):
        super().__init__()

        if token_mass_mode not in {"uniform", "learned", "uncertainty", "learned_uncertainty"}:
            raise ValueError(
                "token_mass_mode must be one of: uniform, learned, uncertainty, learned_uncertainty."
            )
        if residual_mode not in {"raw", "normalized"}:
            raise ValueError("residual_mode must be either 'raw' or 'normalized'.")
        if not 0.0 < dustbin_mass < 1.0:
            raise ValueError("dustbin_mass must be in (0, 1).")

        self.in_channels = int(in_channels)
        self.num_clusters = int(num_clusters)
        self.cluster_dim = int(cluster_dim)
        self.hidden_channels = int(hidden_channels)
        self.include_pi_in_scores = bool(include_pi_in_scores)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.dustbin_mass = float(dustbin_mass)
        self.mass_preserving = bool(mass_preserving)
        self.mass_power = float(mass_power)
        self.token_mass_mode = token_mass_mode
        self.token_log_sigma_bias = float(token_log_sigma_bias)
        self.token_log_sigma_scale = float(token_log_sigma_scale)
        self.residual_mode = residual_mode
        self.eps = float(eps)

        local_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.token_stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=1),
            local_dropout,
            nn.ReLU(),
        )
        self.token_mean_head = nn.Conv2d(self.hidden_channels, self.cluster_dim, kernel_size=1)
        self.token_log_sigma_head = nn.Conv2d(self.hidden_channels, self.cluster_dim, kernel_size=1)
        self.token_norm = nn.LayerNorm(self.cluster_dim)

        token_mass_hidden_channels = (
            max(32, self.cluster_dim) if token_mass_hidden_channels is None else int(token_mass_hidden_channels)
        )
        if self.token_mass_mode in {"learned", "learned_uncertainty"}:
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

        if self.token_mass_mode in {"uncertainty", "learned_uncertainty"}:
            self.uncertainty_mass_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_parameter("uncertainty_mass_scale", None)

        self.mu = nn.Parameter(torch.empty(self.num_clusters, self.cluster_dim))
        nn.init.xavier_uniform_(self.mu)
        self.log_sigma = nn.Parameter(torch.zeros(self.num_clusters, self.cluster_dim))
        self.log_alpha = nn.Parameter(torch.zeros(self.num_clusters))

        if learn_sigma_cost_weight:
            self.log_sigma_cost_weight = nn.Parameter(torch.tensor(inverse_softplus(sigma_cost_weight)))
        else:
            self.register_buffer(
                "log_sigma_cost_weight",
                torch.tensor(inverse_softplus(sigma_cost_weight)),
                persistent=True,
            )
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

    def _cluster_prior(self) -> torch.Tensor:
        return F.softmax(self.log_alpha, dim=0)

    def _cluster_sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma).clamp_min(self.eps)

    def _extract_token_gaussians(self, x: torch.Tensor):
        features = self.token_stem(x)

        token_mean = self.token_mean_head(features).flatten(2).transpose(1, 2)
        token_mean = self.token_norm(token_mean).transpose(1, 2).contiguous()

        raw_log_sigma = self.token_log_sigma_head(features).flatten(2)
        token_log_sigma = self.token_log_sigma_bias + self.token_log_sigma_scale * torch.tanh(raw_log_sigma)
        token_sigma = torch.exp(token_log_sigma).clamp_min(self.eps)
        return token_mean, token_sigma, token_log_sigma

    def _compute_token_mass(self, token_mean: torch.Tensor, token_log_sigma: torch.Tensor) -> torch.Tensor:
        b, _, n = token_mean.shape
        if self.token_mass_mode == "uniform":
            return token_mean.new_full((b, n), 1.0 / max(n, 1))

        mass_logits = token_mean.new_zeros((b, n))
        if self.token_mass_mode in {"learned", "learned_uncertainty"}:
            learned_logits = self.token_mass_head(token_mean).squeeze(1)
            learned_scale = F.softplus(self.token_mass_scale) + self.eps
            mass_logits = mass_logits + learned_scale * learned_logits

        if self.token_mass_mode in {"uncertainty", "learned_uncertainty"}:
            uncertainty = token_log_sigma.mean(dim=1)
            uncertainty_scale = F.softplus(self.uncertainty_mass_scale) + self.eps
            mass_logits = mass_logits - uncertainty_scale * uncertainty

        return F.softmax(mass_logits, dim=-1).clamp_min(self.eps)

    def _squared_l2_cost(self, x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        x_tokens = x.transpose(1, 2)  # [B, N, D]
        x_sq = x_tokens.square().sum(dim=-1, keepdim=True)
        c_sq = centers.square().sum(dim=-1).view(1, 1, -1)
        dot = torch.einsum("bnd,kd->bnk", x_tokens, centers)
        return (x_sq - 2.0 * dot + c_sq).clamp_min(0.0) / max(self.cluster_dim, 1)

    def _variance_aware_scores(self, token_mean: torch.Tensor, token_sigma: torch.Tensor) -> torch.Tensor:
        cluster_sigma = self._cluster_sigma()
        mean_cost = self._squared_l2_cost(token_mean, self.mu)
        sigma_cost = self._squared_l2_cost(token_sigma, cluster_sigma)
        sigma_weight = F.softplus(self.log_sigma_cost_weight) + self.eps
        score_scale = F.softplus(self.log_score_scale) + self.eps

        cost = mean_cost + sigma_weight * sigma_cost
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

    def _residual(self, mu_hat: torch.Tensor, sigma_hat: torch.Tensor):
        cluster_mu = self.mu.unsqueeze(0)
        cluster_sigma = self._cluster_sigma().unsqueeze(0)

        if self.residual_mode == "raw":
            mean_shift = mu_hat - cluster_mu
            sigma_shift = sigma_hat - cluster_sigma
        else:
            mean_shift = (mu_hat - cluster_mu) / cluster_sigma.clamp_min(self.eps)
            sigma_shift = torch.log(sigma_hat.clamp_min(self.eps)) - self.log_sigma.unsqueeze(0)
        return mean_shift, sigma_shift

    def forward(self, x):
        x = self._split_backbone_output(x)
        token_mean, token_sigma, token_log_sigma = self._extract_token_gaussians(x)
        token_mass = self._compute_token_mass(token_mean, token_log_sigma)

        logits = self._variance_aware_scores(token_mean, token_sigma)
        assignments = self._transport(logits, token_mass)
        cluster_mass = assignments.sum(dim=-1, keepdim=True)
        gamma = assignments / cluster_mass.clamp_min(self.eps)

        mu_hat = torch.einsum("bkn,bdn->bkd", gamma, token_mean)
        token_second_moment = token_mean.square() + token_sigma.square()
        second_moment = torch.einsum("bkn,bdn->bkd", gamma, token_second_moment)
        var_hat = (second_moment - mu_hat.square()).clamp_min(self.eps)
        sigma_hat = torch.sqrt(var_hat)

        mean_shift, sigma_shift = self._residual(mu_hat, sigma_hat)

        if self.mass_preserving:
            mass_weight = cluster_mass * (self.num_clusters / max(1.0 - self.dustbin_mass, self.eps))
            mass_weight = mass_weight.clamp_min(self.eps).pow(self.mass_power)
            mean_shift = mean_shift * mass_weight
            sigma_shift = sigma_shift * mass_weight

        residual = torch.cat([mean_shift, sigma_shift], dim=-1)
        descriptor = residual.reshape(token_mean.size(0), -1)
        return F.normalize(descriptor, p=2, dim=-1)
