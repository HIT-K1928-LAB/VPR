"""
Image-conditioned OT-GaussVLAD.

This layer keeps the OT-GaussVLAD descriptor but makes the source-side OT mass
image-adaptive:

    cluster prior  pi(x)      = softmax(global_context(x))
    dustbin mass   rho(x)     = rho_min + (rho_max - rho_min) * sigmoid(global_context(x))

The remaining mass, 1 - rho(x), is distributed over clusters according to
pi(x). This lets each image decide which prototypes should receive transport
mass and how much content should be rejected by the dustbin.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_sigmoid(value: float) -> float:
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError("Sigmoid target must be in (0, 1).")
    return math.log(value / (1.0 - value))


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


class ImageConditionedOTGaussVLADDiagCovLayer(nn.Module):
    """
    OT-GaussVLAD with image-conditioned cluster prior and dustbin mass.

    score:
        Gaussian logits or cosine attention logits
    assignment:
        Sinkhorn with per-image source mass [pi(x), rho(x)]
    aggregation:
        [mu_hat - mu, sigma_hat - sigma]
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
        dustbin_mass_init=0.05,
        dustbin_mass_min=0.01,
        dustbin_mass_max=0.30,
        dropout=0.0,
        mass_preserving=True,
        mass_power=1.0,
        token_mass_mode="learned",
        token_mass_hidden_channels=None,
        context_hidden_channels=None,
        prior_temperature_init=1.0,
        prior_residual_weight_init=0.0,
        eps=1e-6,
    ):
        super().__init__()

        if score_mode not in {"gmm", "attention"}:
            raise ValueError("score_mode must be either 'gmm' or 'attention'.")
        if token_mass_mode not in {"uniform", "learned"}:
            raise ValueError("token_mass_mode must be either 'uniform' or 'learned'.")
        if not 0.0 < dustbin_mass_min < dustbin_mass_max < 1.0:
            raise ValueError("dustbin_mass_min and dustbin_mass_max must satisfy 0 < min < max < 1.")
        if not dustbin_mass_min < dustbin_mass_init < dustbin_mass_max:
            raise ValueError("dustbin_mass_init must lie between dustbin_mass_min and dustbin_mass_max.")

        self.in_channels = int(in_channels)
        self.num_clusters = int(num_clusters)
        self.cluster_dim = int(cluster_dim)
        self.hidden_channels = int(hidden_channels)
        self.score_mode = score_mode
        self.include_pi_in_scores = bool(include_pi_in_scores)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.dustbin_mass_min = float(dustbin_mass_min)
        self.dustbin_mass_max = float(dustbin_mass_max)
        self.mass_preserving = bool(mass_preserving)
        self.mass_power = float(mass_power)
        self.token_mass_mode = token_mass_mode
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

        context_hidden_channels = (
            max(64, self.cluster_dim) if context_hidden_channels is None else int(context_hidden_channels)
        )
        context_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.context_head = nn.Sequential(
            nn.Linear(2 * self.cluster_dim, context_hidden_channels),
            nn.ReLU(),
            context_dropout,
        )
        self.prior_head = nn.Linear(context_hidden_channels, self.num_clusters)
        self.dustbin_head = nn.Linear(context_hidden_channels, 1)

        self.mu = nn.Parameter(torch.empty(self.num_clusters, self.cluster_dim))
        nn.init.xavier_uniform_(self.mu)
        self.log_sigma = nn.Parameter(torch.zeros(self.num_clusters, self.cluster_dim))
        self.log_alpha = nn.Parameter(torch.zeros(self.num_clusters))

        self.prior_temperature = nn.Parameter(torch.tensor(float(prior_temperature_init)))
        self.prior_residual_weight = nn.Parameter(torch.tensor(float(prior_residual_weight_init)))
        normalized_dustbin = (dustbin_mass_init - dustbin_mass_min) / (dustbin_mass_max - dustbin_mass_min)
        self.dustbin_bias = nn.Parameter(torch.tensor(inverse_sigmoid(normalized_dustbin)))

        self.attention_scale = nn.Parameter(torch.tensor(10.0))
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

    def _sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma).clamp_min(self.eps)

    def _compute_token_mass(self, local: torch.Tensor) -> torch.Tensor:
        b, _, n = local.shape
        if self.token_mass_mode == "uniform":
            return local.new_full((b, n), 1.0 / max(n, 1))

        token_logits = self.token_mass_head(local).squeeze(1)
        scale = F.softplus(self.token_mass_scale) + self.eps
        return F.softmax(scale * token_logits, dim=-1).clamp_min(self.eps)

    def _context(self, local: torch.Tensor) -> torch.Tensor:
        mean_context = local.mean(dim=-1)
        max_context = local.amax(dim=-1)
        return torch.cat([mean_context, max_context], dim=1)

    def _image_conditioned_source_mass(self, local: torch.Tensor):
        context = self.context_head(self._context(local))

        static_logits = self.log_alpha.unsqueeze(0)
        dynamic_logits = self.prior_head(context)
        residual_weight = torch.tanh(self.prior_residual_weight)
        temperature = F.softplus(self.prior_temperature) + self.eps
        cluster_prior = F.softmax((static_logits + residual_weight * dynamic_logits) / temperature, dim=-1)

        dustbin_unit = torch.sigmoid(self.dustbin_head(context).squeeze(-1) + self.dustbin_bias)
        dustbin_mass = self.dustbin_mass_min + (self.dustbin_mass_max - self.dustbin_mass_min) * dustbin_unit
        cluster_mass = (1.0 - dustbin_mass).unsqueeze(-1) * cluster_prior
        source_mass = torch.cat([cluster_mass, dustbin_mass.unsqueeze(-1)], dim=1)
        return source_mass, cluster_mass, dustbin_mass

    def _gmm_scores(self, x: torch.Tensor) -> torch.Tensor:
        x_tokens = x.transpose(1, 2)
        inv_sigma_sq = torch.exp(-2.0 * self.log_sigma)
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

    def _transport(
        self,
        logits: torch.Tensor,
        token_mass: torch.Tensor,
        source_mass: torch.Tensor,
    ) -> torch.Tensor:
        b, _, n = logits.shape
        dustbin = self.dust_bin.to(dtype=logits.dtype).expand(b, 1, n)
        scores = torch.cat([logits, dustbin], dim=1)

        target_mass = token_mass / token_mass.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        transport = log_sinkhorn_iterations(
            scores,
            torch.log(source_mass.to(dtype=logits.dtype).clamp_min(self.eps)),
            torch.log(target_mass.clamp_min(self.eps)),
            self.sinkhorn_iters,
        )
        return torch.exp(transport[:, :-1, :])

    def forward(self, x):
        x = self._split_backbone_output(x)
        local = self._extract_local_tokens(x)
        token_mass = self._compute_token_mass(local)
        source_mass, _, dustbin_mass = self._image_conditioned_source_mass(local)

        if self.score_mode == "gmm":
            logits = self._gmm_scores(local)
        else:
            logits = self._attention_scores(local)

        assignments = self._transport(logits, token_mass, source_mass)
        cluster_mass = assignments.sum(dim=-1, keepdim=True)
        gamma = assignments / cluster_mass.clamp_min(self.eps)

        mu_hat = torch.einsum("bkn,bdn->bkd", gamma, local)
        diff = local.transpose(1, 2).unsqueeze(1) - mu_hat.unsqueeze(2)
        var_hat = torch.einsum("bkn,bknd->bkd", gamma, diff.square()).clamp_min(self.eps)
        sigma_hat = torch.sqrt(var_hat)

        mean_shift = mu_hat - self.mu.unsqueeze(0)
        sigma_shift = sigma_hat - self._sigma().unsqueeze(0)

        if self.mass_preserving:
            normalizer = self.num_clusters / (1.0 - dustbin_mass).clamp_min(self.eps)
            mass_weight = cluster_mass * normalizer.view(-1, 1, 1)
            mass_weight = mass_weight.clamp_min(self.eps).pow(self.mass_power)
            mean_shift = mean_shift * mass_weight
            sigma_shift = sigma_shift * mass_weight

        residual = torch.cat([mean_shift, sigma_shift], dim=-1)
        descriptor = residual.reshape(local.size(0), -1)
        return F.normalize(descriptor, p=2, dim=-1)
