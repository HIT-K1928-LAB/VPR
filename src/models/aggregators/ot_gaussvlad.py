"""
    OT-GaussVLAD: Sinkhorn-style transport for Gaussian residual aggregation.

    This layer keeps the GaussVLAD / GaussVLAD++ interpretation of
    prototype-wise Gaussian residuals, but replaces the token-to-cluster
    posterior with OT assignment and a dustbin.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class OTGaussVLADDiagCovLayer(nn.Module):
    """
    OT-GaussVLAD with diagonal covariance residuals.

    score:
        Gaussian logits or cosine attention logits
    assignment:
        Sinkhorn + dustbin
    mass:
        cluster prior
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
        dustbin_mass=0.05,
        dropout=0.0,
        mass_preserving=True,
        mass_power=1.0,
        token_mass_mode="learned",
        token_mass_hidden_channels=None,
        eps=1e-6,
    ):
        super().__init__()

        if score_mode not in {"gmm", "attention"}:
            raise ValueError("score_mode must be either 'gmm' or 'attention'.")
        if token_mass_mode not in {"uniform", "learned"}:
            raise ValueError("token_mass_mode must be either 'uniform' or 'learned'.")
        if not 0.0 < dustbin_mass < 1.0:
            raise ValueError("dustbin_mass must be in (0, 1).")

        self.in_channels = int(in_channels)
        self.num_clusters = int(num_clusters)
        self.cluster_dim = int(cluster_dim)
        self.hidden_channels = int(hidden_channels)
        self.score_mode = score_mode
        self.include_pi_in_scores = bool(include_pi_in_scores)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.dustbin_mass = float(dustbin_mass)
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

        self.mu = nn.Parameter(torch.empty(self.num_clusters, self.cluster_dim))
        nn.init.xavier_uniform_(self.mu)
        self.log_sigma = nn.Parameter(torch.zeros(self.num_clusters, self.cluster_dim))
        self.log_alpha = nn.Parameter(torch.zeros(self.num_clusters))

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
        x = self.local_features(x).flatten(2).transpose(1, 2)  # [B, N, D]
        x = self.token_norm(x)
        return x.transpose(1, 2).contiguous()  # [B, D, N]

    def _cluster_prior(self) -> torch.Tensor:
        return F.softmax(self.log_alpha, dim=0)

    def _sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma).clamp_min(self.eps)

    def _compute_token_mass(self, local: torch.Tensor) -> torch.Tensor:
        b, _, n = local.shape
        if self.token_mass_mode == "uniform":
            return local.new_full((b, n), 1.0 / max(n, 1))

        token_logits = self.token_mass_head(local).squeeze(1)  # [B, N]
        scale = F.softplus(self.token_mass_scale) + self.eps
        token_mass = F.softmax(scale * token_logits, dim=-1)
        return token_mass.clamp_min(self.eps)

    def _gmm_scores(self, x: torch.Tensor) -> torch.Tensor:
        x_tokens = x.transpose(1, 2)  # [B, N, D]
        inv_sigma_sq = torch.exp(-2.0 * self.log_sigma)  # [K, D]
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

    def _transport(self, logits: torch.Tensor, token_mass: torch.Tensor) -> torch.Tensor:
        b, _, n = logits.shape
        dustbin = self.dust_bin.to(dtype=logits.dtype).expand(b, 1, n)
        scores = torch.cat([logits, dustbin], dim=1)  # [B, K+1, N]

        cluster_mass = ((1.0 - self.dustbin_mass) * self._cluster_prior()).to(dtype=logits.dtype)
        cluster_mass = cluster_mass.unsqueeze(0).expand(b, -1)  # [B, K]
        dustbin_mass = logits.new_full((b, 1), self.dustbin_mass)
        source_mass = torch.cat([cluster_mass, dustbin_mass], dim=1)  # [B, K+1]

        target_mass = token_mass / token_mass.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        transport = log_sinkhorn_iterations(
            scores,
            torch.log(source_mass.clamp_min(self.eps)),
            torch.log(target_mass.clamp_min(self.eps)),
            self.sinkhorn_iters,
        )
        return torch.exp(transport[:, :-1, :])

    def forward(self, x):
        x = self._split_backbone_output(x)
        local = self._extract_local_tokens(x)  # [B, D, N]
        token_mass = self._compute_token_mass(local)  # [B, N]

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

        mean_shift = mu_hat - self.mu.unsqueeze(0)
        sigma_shift = sigma_hat - self._sigma().unsqueeze(0)

        if self.mass_preserving:
            mass_weight = cluster_mass * (self.num_clusters / max(1.0 - self.dustbin_mass, self.eps))
            mass_weight = mass_weight.clamp_min(self.eps).pow(self.mass_power)
            mean_shift = mean_shift * mass_weight
            sigma_shift = sigma_shift * mass_weight

        residual = torch.cat([mean_shift, sigma_shift], dim=-1)
        descriptor = residual.reshape(local.size(0), -1)
        descriptor = F.normalize(descriptor, p=2, dim=-1)
        return descriptor
