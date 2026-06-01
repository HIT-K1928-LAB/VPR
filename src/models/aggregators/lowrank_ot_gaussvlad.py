"""
Low-rank plus diagonal OT-GaussVLAD.

This layer replaces the purely diagonal covariance assumption with

    Sigma_k = U_k U_k^T + diag(s_k^2),

where U_k is low-rank. The assignment score uses a Gaussian NLL computed with
Woodbury identities, so the full D x D covariance matrix is never explicitly
inverted. The descriptor keeps diagonal Wasserstein-style residuals and adds a
low-rank square-root covariance residual in the learned subspace.
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


def spd_matrix_sqrt(matrix: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute a stable square root for small SPD matrices."""
    matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    eigvals, eigvecs = torch.linalg.eigh(matrix.float())
    sqrt_eigvals = eigvals.clamp_min(eps).sqrt()
    return (eigvecs * sqrt_eigvals.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)


class LowRankOTGaussVLADLayer(nn.Module):
    """
    OT-GaussVLAD with low-rank plus diagonal covariance.

    score:
        lowrank_gmm, diag_gmm, or attention
    assignment:
        Sinkhorn + dustbin
    covariance:
        Sigma_k = U_k U_k^T + diag(s_k^2)
    aggregation:
        [mu_hat - mu, marginal_sigma_hat - marginal_sigma, lowrank_cov_sqrt_residual]
    """

    def __init__(
        self,
        in_channels=768,
        num_clusters=64,
        cluster_dim=64,
        hidden_channels=768,
        covariance_rank=4,
        descriptor_dim=None,
        score_mode="lowrank_gmm",
        include_pi_in_scores=False,
        sinkhorn_iters=3,
        dustbin_mass=0.05,
        dropout=0.0,
        mass_preserving=True,
        mass_power=1.0,
        token_mass_mode="learned",
        token_mass_hidden_channels=None,
        factor_scale_init=0.05,
        lowrank_residual_mode="sqrt",
        lowrank_residual_weight_init=0.25,
        eps=1e-6,
    ):
        super().__init__()

        if score_mode not in {"lowrank_gmm", "diag_gmm", "attention"}:
            raise ValueError("score_mode must be one of: lowrank_gmm, diag_gmm, attention.")
        if token_mass_mode not in {"uniform", "learned"}:
            raise ValueError("token_mass_mode must be either 'uniform' or 'learned'.")
        if lowrank_residual_mode not in {"sqrt", "raw"}:
            raise ValueError("lowrank_residual_mode must be either 'sqrt' or 'raw'.")
        if not 0.0 < dustbin_mass < 1.0:
            raise ValueError("dustbin_mass must be in (0, 1).")
        if int(covariance_rank) <= 0:
            raise ValueError("covariance_rank must be positive.")

        self.in_channels = int(in_channels)
        self.num_clusters = int(num_clusters)
        self.cluster_dim = int(cluster_dim)
        self.hidden_channels = int(hidden_channels)
        self.covariance_rank = int(covariance_rank)
        self.score_mode = score_mode
        self.include_pi_in_scores = bool(include_pi_in_scores)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.dustbin_mass = float(dustbin_mass)
        self.mass_preserving = bool(mass_preserving)
        self.mass_power = float(mass_power)
        self.token_mass_mode = token_mass_mode
        self.lowrank_residual_mode = lowrank_residual_mode
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
        self.raw_factor = nn.Parameter(
            torch.empty(self.num_clusters, self.cluster_dim, self.covariance_rank)
        )
        nn.init.xavier_uniform_(self.raw_factor)
        self.log_factor_scale = nn.Parameter(torch.tensor(inverse_softplus(factor_scale_init)))
        self.log_lowrank_residual_weight = nn.Parameter(
            torch.tensor(inverse_softplus(lowrank_residual_weight_init))
        )
        self.log_alpha = nn.Parameter(torch.zeros(self.num_clusters))

        self.attention_scale = nn.Parameter(torch.tensor(10.0))
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

        self.raw_cluster_dim = 2 * self.cluster_dim + self.covariance_rank * self.covariance_rank
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

    def _sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma).clamp_min(self.eps)

    def _factor(self) -> torch.Tensor:
        scale = F.softplus(self.log_factor_scale) + self.eps
        return scale * self.raw_factor

    def _factor_basis(self, factor: torch.Tensor) -> torch.Tensor:
        return F.normalize(factor, p=2, dim=1, eps=self.eps)

    def _marginal_sigma(self, factor: torch.Tensor) -> torch.Tensor:
        marginal_var = self._sigma().square() + factor.square().sum(dim=-1)
        return torch.sqrt(marginal_var.clamp_min(self.eps))

    def _compute_token_mass(self, local: torch.Tensor) -> torch.Tensor:
        b, _, n = local.shape
        if self.token_mass_mode == "uniform":
            return local.new_full((b, n), 1.0 / max(n, 1))

        token_logits = self.token_mass_head(local).squeeze(1)
        scale = F.softplus(self.token_mass_scale) + self.eps
        return F.softmax(scale * token_logits, dim=-1).clamp_min(self.eps)

    def _diag_gmm_scores(self, x: torch.Tensor) -> torch.Tensor:
        x_tokens = x.transpose(1, 2).float()
        mu = self.mu.float()
        log_sigma = self.log_sigma.float()
        inv_sigma_sq = torch.exp(-2.0 * log_sigma)
        quadratic = torch.einsum("bnd,kd->bnk", x_tokens.square(), inv_sigma_sq)
        quadratic = quadratic - 2.0 * torch.einsum("bnd,kd->bnk", x_tokens, mu * inv_sigma_sq)
        quadratic = quadratic + (mu.square() * inv_sigma_sq).sum(dim=-1).view(1, 1, -1)
        log_sigma_sum = log_sigma.sum(dim=-1).view(1, 1, -1)
        logits = -0.5 * quadratic - log_sigma_sum
        if self.include_pi_in_scores:
            logits = logits + F.log_softmax(self.log_alpha.float(), dim=0).view(1, 1, -1)
        return logits.transpose(1, 2).contiguous()

    def _lowrank_gmm_scores(self, x: torch.Tensor) -> torch.Tensor:
        x_tokens = x.transpose(1, 2).float()  # [B, N, D]
        mu = self.mu.float()
        log_sigma = self.log_sigma.float()
        factor = self._factor().float()  # [K, D, R]

        inv_diag = torch.exp(-2.0 * log_sigma)  # [K, D]
        weighted_factor = factor * inv_diag.unsqueeze(-1)  # D^-1 U

        base = torch.einsum("bnd,kd->bnk", x_tokens.square(), inv_diag)
        base = base - 2.0 * torch.einsum("bnd,kd->bnk", x_tokens, mu * inv_diag)
        base = base + (mu.square() * inv_diag).sum(dim=-1).view(1, 1, -1)

        q = torch.einsum("bnd,kdr->bnkr", x_tokens, weighted_factor)
        q = q - torch.einsum("kd,kdr->kr", mu, weighted_factor).view(1, 1, self.num_clusters, self.covariance_rank)

        eye = torch.eye(self.covariance_rank, device=x.device, dtype=torch.float32)
        small_system = (eye.unsqueeze(0) + torch.einsum("kdr,kds->krs", factor, weighted_factor)).float()
        q = q.float()
        solved = torch.linalg.solve(
            small_system.view(1, 1, self.num_clusters, self.covariance_rank, self.covariance_rank),
            q.unsqueeze(-1),
        ).squeeze(-1)
        correction = (q * solved).sum(dim=-1)
        quadratic = (base - correction).clamp_min(0.0)

        sign, logabsdet = torch.linalg.slogdet(small_system)
        logdet = (2.0 * log_sigma).sum(dim=-1) + logabsdet
        logits = -0.5 * (quadratic + logdet.view(1, 1, -1))
        if self.include_pi_in_scores:
            logits = logits + F.log_softmax(self.log_alpha.float(), dim=0).view(1, 1, -1)
        return logits.transpose(1, 2).contiguous()

    def _attention_scores(self, x: torch.Tensor) -> torch.Tensor:
        centers = F.normalize(self.mu, p=2, dim=-1)
        tokens = F.normalize(x, p=2, dim=1)
        scale = F.softplus(self.attention_scale) + self.eps
        return scale * torch.einsum("kd,bdn->bkn", centers, tokens)

    def _transport(self, logits: torch.Tensor, token_mass: torch.Tensor) -> torch.Tensor:
        b, _, n = logits.shape
        dustbin = self.dust_bin.to(dtype=logits.dtype).expand(b, 1, n)
        scores = torch.cat([logits, dustbin], dim=1)

        cluster_mass = ((1.0 - self.dustbin_mass) * self._cluster_prior()).to(dtype=logits.dtype)
        cluster_mass = cluster_mass.unsqueeze(0).expand(b, -1)
        dustbin_mass = logits.new_full((b, 1), self.dustbin_mass)
        source_mass = torch.cat([cluster_mass, dustbin_mass], dim=1)

        target_mass = token_mass.to(dtype=logits.dtype)
        target_mass = target_mass / target_mass.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        transport = log_sinkhorn_iterations(
            scores,
            torch.log(source_mass.clamp_min(self.eps)),
            torch.log(target_mass.clamp_min(self.eps)),
            self.sinkhorn_iters,
        )
        return torch.exp(transport[:, :-1, :])

    def _projected_model_covariance(self, factor: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        sigma_sq = self._sigma().float().square()
        diag_proj = torch.einsum("kdr,kd,kds->krs", basis, sigma_sq, basis)
        basis_factor = torch.einsum("kdr,kds->krs", basis, factor.float())
        factor_proj = basis_factor @ basis_factor.transpose(-1, -2)
        eye = torch.eye(self.covariance_rank, device=factor.device, dtype=torch.float32)
        return (diag_proj + factor_proj + self.eps * eye.unsqueeze(0)).float()

    def _lowrank_covariance_residual(
        self,
        local: torch.Tensor,
        gamma: torch.Tensor,
        mu_hat: torch.Tensor,
        factor: torch.Tensor,
    ) -> torch.Tensor:
        basis = self._factor_basis(factor.float())
        tokens = local.transpose(1, 2).float()
        x_proj = torch.einsum("bnd,kdr->bnkr", tokens, basis).permute(0, 2, 1, 3)
        mu_proj = torch.einsum("bkd,kdr->bkr", mu_hat.float(), basis)
        centered_proj = x_proj - mu_proj.unsqueeze(2)
        cov_hat = torch.einsum("bkn,bknr,bkns->bkrs", gamma.float(), centered_proj, centered_proj).float()

        cov_model = self._projected_model_covariance(factor, basis).float()
        if self.lowrank_residual_mode == "sqrt":
            cov_hat = cov_hat + self.eps * torch.eye(
                self.covariance_rank, device=local.device, dtype=torch.float32
            ).view(1, 1, self.covariance_rank, self.covariance_rank)
            cov_feat = spd_matrix_sqrt(cov_hat, self.eps) - spd_matrix_sqrt(cov_model, self.eps).unsqueeze(0)
        else:
            cov_feat = cov_hat - cov_model.unsqueeze(0)

        weight = F.softplus(self.log_lowrank_residual_weight) + self.eps
        return weight * cov_feat.reshape(local.size(0), self.num_clusters, -1)

    def forward(self, x):
        x = self._split_backbone_output(x)
        local = self._extract_local_tokens(x)
        token_mass = self._compute_token_mass(local)

        if self.score_mode == "lowrank_gmm":
            logits = self._lowrank_gmm_scores(local)
        elif self.score_mode == "diag_gmm":
            logits = self._diag_gmm_scores(local)
        else:
            logits = self._attention_scores(local)

        assignments = self._transport(logits, token_mass)
        cluster_mass = assignments.sum(dim=-1, keepdim=True)
        gamma = assignments / cluster_mass.clamp_min(self.eps)

        local_stats = local.to(dtype=assignments.dtype)
        mu_hat = torch.einsum("bkn,bdn->bkd", gamma, local)
        diff = local.transpose(1, 2).unsqueeze(1) - mu_hat.unsqueeze(2)
        var_hat = torch.einsum("bkn,bknd->bkd", gamma, diff.square()).clamp_min(self.eps)
        sigma_hat = torch.sqrt(var_hat)
        # mu_hat = torch.einsum("bkn,bdn->bkd", gamma, local_stats)
        # second_moment = torch.einsum("bkn,bdn->bkd", gamma, local_stats.square())
        # var_hat = (second_moment - mu_hat.square()).clamp_min(self.eps)
        # sigma_hat = torch.sqrt(var_hat)

        factor = self._factor()
        mean_shift = mu_hat - self.mu.unsqueeze(0).to(dtype=mu_hat.dtype)
        sigma_shift = sigma_hat - self._marginal_sigma(factor).unsqueeze(0).to(dtype=sigma_hat.dtype)
        lowrank_shift = self._lowrank_covariance_residual(local, gamma, mu_hat, factor)

        if self.mass_preserving:
            mass_weight = cluster_mass * (self.num_clusters / max(1.0 - self.dustbin_mass, self.eps))
            mass_weight = mass_weight.clamp_min(self.eps).pow(self.mass_power)
            mean_shift = mean_shift * mass_weight
            sigma_shift = sigma_shift * mass_weight
            lowrank_shift = lowrank_shift * mass_weight

        residual = torch.cat([mean_shift, sigma_shift, lowrank_shift.to(dtype=mean_shift.dtype)], dim=-1)
        descriptor = residual.reshape(local.size(0), -1)
        return F.normalize(descriptor, p=2, dim=-1)
