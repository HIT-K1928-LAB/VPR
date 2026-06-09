"""
Pure QAA aggregator for OpenVPRLab.

This is a conservative, standalone implementation of the original QAA idea:

    learned feature queries + learned score queries + QAA-style CS

No OT/Gaussian branch is used here. The final descriptor is built directly
from the query-reference similarity matrix, so this file can serve as a clean
baseline for your "single QAA" ablation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6

__all__ = ["QAAOnlyLayer"]


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


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """Differentiable optimal transport in log-space."""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns, bs = (m * one).to(scores), (n * one).to(scores), ((n - m) * one).to(scores)

    if alpha is not None:
        bins = alpha.expand(b, 1, n)
        couplings = torch.cat([scores, bins], dim=1)
    else:
        couplings = scores

    norm = -torch.log(ms + ns)
    if alpha is not None:
        log_mu = torch.cat([norm.expand(m), bs.log()[None] + norm])
    else:
        log_mu = norm.expand(m)
    log_nu = norm.expand(n)
    log_mu = log_mu[None].expand(b, -1)
    log_nu = log_nu[None].expand(b, -1)

    z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    return z - norm


class QuerySelfAttn(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_queries: int,
        nheads: int = 8,
        self_attn_flag: bool = True,
        self_attn_out_norm: bool = True,
    ) -> None:
        super().__init__()
        if in_dim % nheads != 0:
            raise ValueError(f"in_dim ({in_dim}) must be divisible by nheads ({nheads}).")

        self.queries = nn.Parameter(torch.randn(1, num_queries, in_dim))
        self.self_attn_flag = bool(self_attn_flag)
        self.self_attn_out_norm = bool(self_attn_out_norm)

        if self.self_attn_flag:
            self.self_attn = nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = nn.LayerNorm(in_dim)

    def adjust_queries(self, num_queries: int) -> None:
        if num_queries > self.queries.shape[1]:
            repeat_factor = num_queries // self.queries.shape[1] + 1
            new_queries = self.queries.repeat(1, repeat_factor, 1)[:, :num_queries, :]
        else:
            new_queries = self.queries[:, :num_queries, :]
        self.queries = nn.Parameter(new_queries)

    def forward(self) -> torch.Tensor:
        q = self.queries
        if self.self_attn_flag:
            q = q + self.self_attn(q, q, q, need_weights=False)[0]
        if self.self_attn_out_norm:
            q = self.norm_q(q)
        return q


class QueryCrossAttn(nn.Module):
    def __init__(
        self,
        in_dim: int,
        output_dim: int,
        nheads: int = 8,
        arch: str = "conv",
        skip: str = "none",
        out_norm: bool = True,
    ) -> None:
        super().__init__()
        if in_dim % nheads != 0:
            raise ValueError(f"in_dim ({in_dim}) must be divisible by nheads ({nheads}).")
        if skip not in {"none", "cross", "full"}:
            raise ValueError("skip must be one of: none, cross, full.")

        self.cross_attn = nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = nn.LayerNorm(in_dim)
        self.arch = arch
        self.skip = skip
        self.out_norm = bool(out_norm)

        if arch == "conv":
            self.conv = nn.Conv1d(in_dim, output_dim, kernel_size=1)
            self.norm2_out = nn.LayerNorm(output_dim)
        elif arch == "linear":
            self.linear1 = nn.Linear(in_dim, 4 * in_dim)
            self.activation = nn.ReLU(inplace=True)
            self.linear2 = nn.Linear(4 * in_dim, output_dim)
            self.norm2_out = nn.LayerNorm(output_dim)
        elif arch == "linearproj":
            self.linear1 = nn.Linear(in_dim, 4 * in_dim)
            self.activation = nn.ReLU(inplace=True)
            self.linear2 = nn.Linear(4 * in_dim, in_dim)
            self.norm2_out = nn.LayerNorm(in_dim)
            self.linear3 = nn.Linear(in_dim, output_dim)
            self.norm3_out = nn.LayerNorm(output_dim)
        elif arch == "none":
            pass
        else:
            raise ValueError("arch must be one of: conv, linear, linearproj, none.")

    def forward(self, x: torch.Tensor, q: torch.Tensor):
        x_flatten = x.flatten(2).permute(0, 2, 1)

        out, attn = self.cross_attn(q, x_flatten, x_flatten, need_weights=True)
        if self.skip in {"full", "cross"}:
            out = q + out
        out = self.norm_out(out)

        if self.arch == "conv":
            cache = out
            out = self.conv(out.permute(0, 2, 1)).permute(0, 2, 1)
            if self.skip == "full":
                out = cache + out
            if self.out_norm:
                out = self.norm2_out(out)
        elif self.arch == "linear":
            cache = out
            out = self.linear2(self.activation(self.linear1(out)))
            if self.skip == "full":
                out = cache + out
            if self.out_norm:
                out = self.norm2_out(out)
        elif self.arch == "linearproj":
            cache = out
            out = self.linear2(self.activation(self.linear1(out)))
            if self.skip == "full":
                out = cache + out
            out = self.norm2_out(out)
            out = self.linear3(out)
            if self.out_norm:
                out = self.norm3_out(out)
        elif self.arch == "none":
            pass

        return out.permute(0, 2, 1), attn


class QAAOnlyLayer(nn.Module):
    """
    Pure QAA aggregation layer.

    Output dimension:
        token_dim + num_clusters * cluster_dim
    """

    def __init__(
        self,
        in_channels: int = 768,
        num_clusters: int = 64,
        cluster_dim: int = 128,
        token_dim: int = 0,
        dropout: float = 0.0,
        num_queries: int = 256,
        self_attn: str = "both",
        dust_bin: bool = True,
        feature_nheads: int = 8,
        score_nheads: int = 8,
        attn_arch: str = "conv",
        skip_connection: str = "none",
        out_norm: bool = True,
        self_attn_out_norm: bool = True,
        score_norm: str = "ot",
        intra_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if self_attn not in {"none", "feature", "score", "both"}:
            raise ValueError("self_attn must be one of: none, feature, score, both.")
        if score_norm not in {"ot", "softmax", "none"}:
            raise ValueError("score_norm must be one of: ot, softmax, none.")
        if score_norm == "ot" and num_queries < num_clusters:
            raise ValueError("For OT normalization, num_queries must be >= num_clusters.")

        self.in_channels = int(in_channels)
        self.num_clusters = int(num_clusters)
        self.cluster_dim = int(cluster_dim)
        self.token_dim = int(token_dim)
        self.num_queries = int(num_queries)
        self.score_norm = score_norm
        self.intra_norm = bool(intra_norm)
        self.eps = float(eps)
        if self.eps <= 0:
            raise ValueError("eps must be positive.")

        if self.cluster_dim % int(feature_nheads) != 0:
            raise ValueError("cluster_dim must be divisible by feature_nheads.")
        if self.in_channels % int(score_nheads) != 0:
            raise ValueError("in_channels must be divisible by score_nheads.")

        self.feature_attn_on = self_attn in {"both", "feature"}
        self.score_attn_on = self_attn in {"both", "score"}

        if dropout > 0:
            dropout_layer = nn.Dropout(dropout)
        else:
            dropout_layer = nn.Identity()

        if self.token_dim != 0:
            self.token_features = nn.Sequential(
                nn.Linear(self.in_channels, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, self.token_dim),
            )

        self.queries_feature = QuerySelfAttn(
            self.cluster_dim,
            self.num_queries,
            nheads=int(feature_nheads),
            self_attn_flag=self.feature_attn_on,
            self_attn_out_norm=self_attn_out_norm,
        )
        self.queries_score = QuerySelfAttn(
            self.in_channels,
            self.num_queries,
            nheads=int(score_nheads),
            self_attn_flag=self.score_attn_on,
            self_attn_out_norm=self_attn_out_norm,
        )
        self.score = QueryCrossAttn(
            self.in_channels,
            self.num_clusters,
            nheads=int(score_nheads),
            arch=attn_arch,
            skip=skip_connection,
            out_norm=out_norm,
        )
        self.dropout = dropout_layer

        if dust_bin:
            self.dust_bin = nn.Parameter(torch.tensor(1.0))
        else:
            self.dust_bin = None

        self.output_dim = self.token_dim + self.num_clusters * self.cluster_dim

    @torch.no_grad()
    def cache_query(self) -> None:
        self.cached_query_feature = self.queries_feature()
        self.cached_query_score = self.queries_score()

    def clean_cache(self) -> None:
        if hasattr(self, "cached_query_feature"):
            del self.cached_query_feature
        if hasattr(self, "cached_query_score"):
            del self.cached_query_score

    def adjust_queries(self, num_queries: int) -> None:
        self.queries_feature.adjust_queries(num_queries)
        self.queries_score.adjust_queries(num_queries)

    def _split_input(self, x):
        if isinstance(x, (tuple, list)):
            if len(x) == 2:
                return x[0], x[1]
            if len(x) == 1:
                return x[0], None
            raise ValueError("Unexpected input tuple/list length.")
        return x, None

    def forward(self, x):
        x, t = self._split_input(x)

        f_raw = self.cached_query_feature if hasattr(self, "cached_query_feature") else self.queries_feature()
        q_raw = self.cached_query_score if hasattr(self, "cached_query_score") else self.queries_score()

        f = f_raw.permute(0, 2, 1).repeat(x.shape[0], 1, 1)
        q = q_raw.repeat(x.shape[0], 1, 1)

        p, _ = self.score(self.dropout(x), q)

        if self.score_norm == "ot":
            p = log_optimal_transport(p, self.dust_bin, 3)
            p = torch.exp(p)
            if self.dust_bin is not None:
                p = p[:, :-1, :]
        elif self.score_norm == "softmax":
            p = torch.softmax(p, dim=1)
        elif self.score_norm == "none":
            pass

        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)

        desc = (f * p).sum(dim=-1)
        if self.intra_norm:
            desc = F.normalize(desc, p=2, dim=1)
        desc = desc.flatten(1)

        if self.token_dim != 0:
            if t is None:
                t = x.mean(dim=(-2, -1))
            t = self.token_features(t)
            desc = torch.cat([F.normalize(t, p=2, dim=-1), desc], dim=-1)

        return F.normalize(desc, p=2, dim=1)
