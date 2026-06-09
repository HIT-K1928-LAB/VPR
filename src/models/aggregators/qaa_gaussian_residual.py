"""
QAA-style Gaussian residual aggregation.

This is a standalone aggregator that builds a global descriptor without the
OT-GaussVLAD branch:

    image queries read local tokens -> query-level diagonal Gaussians
    reference queries -> reference Gaussian codebook
    W2 scores -> soft/OT assignment
    residual = [mu_hat - mu_ref, sigma_hat - sigma_ref]

The default output dimension is K * 2D = 64 * 128 = 8192.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["QAAGaussianResidualLayer"]


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
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(scores + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(scores + u.unsqueeze(2), dim=1)
    return scores + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor | None, iters: int) -> torch.Tensor:
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms = (m * one).to(scores)
    ns = (n * one).to(scores)
    bs = ((n - m) * one).to(scores)

    if alpha is not None:
        bins = alpha.expand(b, 1, n)
        scores = torch.cat([scores, bins], dim=1)

    norm = -torch.log(ms + ns)
    if alpha is not None:
        log_mu = torch.cat([norm.expand(m), bs.clamp_min(1).log()[None] + norm])
    else:
        log_mu = norm.expand(m)
    log_nu = norm.expand(n)
    log_mu = log_mu[None].expand(b, -1)
    log_nu = log_nu[None].expand(b, -1)

    transport = log_sinkhorn_iterations(scores, log_mu, log_nu, iters)
    return transport - norm


class QuerySelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_queries: int,
        num_heads: int,
        dropout: float = 0.0,
        self_attn: bool = True,
        out_norm: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.queries = nn.Parameter(torch.empty(1, num_queries, embed_dim))
        nn.init.xavier_uniform_(self.queries)
        self.self_attn_flag = bool(self_attn)
        self.out_norm = bool(out_norm)

        if self.self_attn_flag:
            self.self_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self) -> torch.Tensor:
        q = self.queries
        if self.self_attn_flag:
            q = q + self.self_attn(q, q, q, need_weights=False)[0]
        if self.out_norm:
            q = self.norm(q)
        return q


class QAAGaussianResidualLayer(nn.Module):
    """
    Query-conditioned Gaussian residual descriptor.

    Shapes with default hyperparameters:
        local tokens:      [B, N, 768]
        image Gaussians:  [B, Q=256, D=64]
        reference codebook [K=64, D=64]
        residual:         [B, K=64, 2D=128]
        descriptor:       [B, 8192]
    """

    def __init__(
        self,
        in_channels: int = 768,
        num_clusters: int = 64,
        gaussian_dim: int = 64,
        num_queries: int = 256,
        image_query_dim: int | None = None,
        reference_query_dim: int | None = None,
        image_nheads: int = 8,
        reference_nheads: int = 8,
        image_self_attn: bool = True,
        reference_self_attn: bool = True,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        assignment_norm: str = "ot",
        sinkhorn_iters: int = 3,
        dust_bin: bool = True,
        score_scale_init: float = 5.0,
        sigma_weight_init: float = 1.0,
        learn_sigma_weight: bool = True,
        log_sigma_bias: float = 0.0,
        log_sigma_scale: float = 2.0,
        aggregate_sigma: str = "moment",
        residual_intra_norm: bool = True,
        mass_preserving: bool = True,
        mass_power: float = 0.05,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if assignment_norm not in {"ot", "softmax", "query_softmax"}:
            raise ValueError("assignment_norm must be one of: ot, softmax, query_softmax.")
        if aggregate_sigma not in {"moment", "mean"}:
            raise ValueError("aggregate_sigma must be either 'moment' or 'mean'.")
        if assignment_norm == "ot" and num_queries < num_clusters:
            raise ValueError("OT assignment requires num_queries >= num_clusters.")
        if float(log_sigma_scale) <= 0.0:
            raise ValueError("log_sigma_scale must be positive.")

        self.in_channels = int(in_channels)
        self.num_clusters = int(num_clusters)
        self.gaussian_dim = int(gaussian_dim)
        self.num_queries = int(num_queries)
        self.assignment_norm = assignment_norm
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.aggregate_sigma = aggregate_sigma
        self.residual_intra_norm = bool(residual_intra_norm)
        self.mass_preserving = bool(mass_preserving)
        self.mass_power = float(mass_power)
        self.log_sigma_bias = float(log_sigma_bias)
        self.log_sigma_scale = float(log_sigma_scale)
        self.eps = float(eps)
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")

        self.image_query_dim = self.in_channels if image_query_dim is None else int(image_query_dim)
        self.reference_query_dim = self.gaussian_dim if reference_query_dim is None else int(reference_query_dim)

        if self.image_query_dim != self.in_channels:
            raise ValueError("v1 requires image_query_dim to match in_channels for cross-attention.")
        if self.image_query_dim % int(image_nheads) != 0:
            raise ValueError("image_query_dim must be divisible by image_nheads.")
        if self.reference_query_dim % int(reference_nheads) != 0:
            raise ValueError("reference_query_dim must be divisible by reference_nheads.")

        self.token_norm = nn.LayerNorm(self.in_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.image_queries = QuerySelfAttention(
            embed_dim=self.image_query_dim,
            num_queries=self.num_queries,
            num_heads=int(image_nheads),
            dropout=float(attn_dropout),
            self_attn=bool(image_self_attn),
            out_norm=True,
        )
        self.image_cross_attn = nn.MultiheadAttention(
            embed_dim=self.in_channels,
            num_heads=int(image_nheads),
            dropout=float(attn_dropout),
            batch_first=True,
        )
        self.image_cross_norm = nn.LayerNorm(self.in_channels)

        self.image_mu = nn.Linear(self.in_channels, self.gaussian_dim)
        self.image_log_sigma = nn.Linear(self.in_channels, self.gaussian_dim)
        self.image_mu_norm = nn.LayerNorm(self.gaussian_dim)

        self.reference_queries = QuerySelfAttention(
            embed_dim=self.reference_query_dim,
            num_queries=self.num_clusters,
            num_heads=int(reference_nheads),
            dropout=float(attn_dropout),
            self_attn=bool(reference_self_attn),
            out_norm=True,
        )
        self.reference_mu = nn.Linear(self.reference_query_dim, self.gaussian_dim)
        self.reference_log_sigma = nn.Linear(self.reference_query_dim, self.gaussian_dim)
        self.reference_mu_norm = nn.LayerNorm(self.gaussian_dim)

        nn.init.normal_(self.image_log_sigma.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.image_log_sigma.bias)
        nn.init.normal_(self.reference_log_sigma.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.reference_log_sigma.bias)

        self.log_score_scale = nn.Parameter(torch.tensor(inverse_softplus(float(score_scale_init))))
        if learn_sigma_weight:
            self.log_sigma_weight = nn.Parameter(torch.tensor(inverse_softplus(float(sigma_weight_init))))
        else:
            self.register_buffer(
                "log_sigma_weight",
                torch.tensor(inverse_softplus(float(sigma_weight_init))),
                persistent=True,
            )

        if dust_bin:
            self.dust_bin = nn.Parameter(torch.tensor(1.0))
        else:
            self.dust_bin = None

        self.output_dim = self.num_clusters * 2 * self.gaussian_dim

    @staticmethod
    def _split_input(x):
        if isinstance(x, (tuple, list)):
            if len(x) == 0:
                raise ValueError("Unexpected empty input tuple/list.")
            return x[0]
        return x

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        tokens = self.token_norm(tokens)
        return self.dropout(tokens)

    def _bounded_log_sigma(self, raw: torch.Tensor) -> torch.Tensor:
        return self.log_sigma_bias + self.log_sigma_scale * torch.tanh(raw)

    def _image_gaussians(self, tokens: torch.Tensor):
        b = tokens.size(0)
        queries = self.image_queries().expand(b, -1, -1)
        query_state = self.image_cross_attn(queries, tokens, tokens, need_weights=False)[0]
        query_state = self.image_cross_norm(query_state)

        mu_q = self.image_mu_norm(self.image_mu(query_state))
        log_sigma_q = self._bounded_log_sigma(self.image_log_sigma(query_state))
        sigma_q = torch.exp(log_sigma_q).clamp_min(self.eps)
        return mu_q, sigma_q

    def _reference_gaussians(self):
        reference_state = self.reference_queries().squeeze(0)
        mu_ref = self.reference_mu_norm(self.reference_mu(reference_state))
        log_sigma_ref = self._bounded_log_sigma(self.reference_log_sigma(reference_state))
        sigma_ref = torch.exp(log_sigma_ref).clamp_min(self.eps)
        return mu_ref, sigma_ref

    def _w2_scores(self, mu_q: torch.Tensor, sigma_q: torch.Tensor, mu_ref: torch.Tensor, sigma_ref: torch.Tensor):
        mean_diff = mu_q.unsqueeze(1) - mu_ref.unsqueeze(0).unsqueeze(2)
        sigma_diff = sigma_q.unsqueeze(1) - sigma_ref.unsqueeze(0).unsqueeze(2)
        sigma_weight = F.softplus(self.log_sigma_weight) + self.eps
        score_scale = F.softplus(self.log_score_scale) + self.eps
        cost = mean_diff.square().mean(dim=-1) + sigma_weight * sigma_diff.square().mean(dim=-1)
        return -score_scale * cost

    def _assign(self, scores: torch.Tensor) -> torch.Tensor:
        if self.assignment_norm == "softmax":
            return F.softmax(scores, dim=1)
        if self.assignment_norm == "query_softmax":
            return F.softmax(scores, dim=-1)

        transport = log_optimal_transport(scores, self.dust_bin, self.sinkhorn_iters)
        assignment = torch.exp(transport)
        if self.dust_bin is not None:
            assignment = assignment[:, :-1, :]
        return assignment

    def _aggregate(self, assignments: torch.Tensor, mu_q: torch.Tensor, sigma_q: torch.Tensor):
        cluster_mass = assignments.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        gamma = assignments / cluster_mass

        mu_hat = torch.einsum("bkq,bqd->bkd", gamma, mu_q)
        if self.aggregate_sigma == "mean":
            sigma_hat = torch.einsum("bkq,bqd->bkd", gamma, sigma_q)
        else:
            second = torch.einsum("bkq,bqd->bkd", gamma, sigma_q.square() + mu_q.square())
            var_hat = (second - mu_hat.square()).clamp_min(self.eps)
            sigma_hat = torch.sqrt(var_hat)
        return mu_hat, sigma_hat, cluster_mass

    def forward(self, x):
        x = self._split_input(x)
        tokens = self._tokens(x)

        mu_q, sigma_q = self._image_gaussians(tokens)
        mu_ref, sigma_ref = self._reference_gaussians()
        scores = self._w2_scores(mu_q, sigma_q, mu_ref, sigma_ref)
        assignments = self._assign(scores)
        mu_hat, sigma_hat, cluster_mass = self._aggregate(assignments, mu_q, sigma_q)

        mean_residual = mu_hat - mu_ref.unsqueeze(0)
        sigma_residual = sigma_hat - sigma_ref.unsqueeze(0)
        residual = torch.cat([mean_residual, sigma_residual], dim=-1)

        if self.residual_intra_norm:
            residual = F.normalize(residual, p=2, dim=-1)

        if self.mass_preserving:
            expected_mass = self.num_queries / max(self.num_clusters, 1)
            mass_weight = (cluster_mass / max(expected_mass, self.eps)).clamp_min(self.eps).pow(self.mass_power)
            residual = residual * mass_weight

        descriptor = residual.reshape(x.size(0), -1)
        return F.normalize(descriptor, p=2, dim=-1)
