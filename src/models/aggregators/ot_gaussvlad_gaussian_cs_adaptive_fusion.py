"""
OT-GaussVLAD with Distribution-aware Cross-query Similarity and adaptive cluster fusion.

This layer keeps the Gaussian OT residual branch from OT-GaussVLAD v2 and
adds a Gaussian-aware CS branch. Instead of a plain dot-product CS matrix, the
side branch predicts query-level diagonal Gaussian features and compares them
with an independent Gaussian reference codebook through W2-style or
natural-parameter similarity.
"""

import math

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


def inverse_sigmoid(value: float) -> float:
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError("Sigmoid target must be in (0, 1).")
    return math.log(value / (1.0 - value))


def inverse_softplus(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError("Softplus target must be positive.")
    return math.log(math.expm1(value))


class QuerySelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_queries: int,
        num_heads: int,
        dropout: float = 0.0,
        self_attn: bool = True,
        out_norm: bool = True,
    ):
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


class OTGaussVLADGaussianCSLayer(nn.Module):
    """
    OT-GaussVLAD with a Gaussian-aware Cross-query Similarity branch.

    OT branch:
        Sinkhorn assignment over learned Gaussian prototypes followed by
        [mu_hat - mu, sigma_hat - sigma] residual aggregation.

    Gaussian-CS branch:
        feature queries attend local tokens and predict query-level Gaussian
        features. A learned reference Gaussian codebook provides the fixed
        frame. The branch computes W2-style or natural-parameter similarity
        along the query axis, yielding a [Cr, Cf] relation matrix.

    Fusion:
        default cluster_fusion keeps the final descriptor shape identical to
        OT-GaussVLAD v2 and initializes as OT + small Gaussian-CS residual.
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
        gcs_num_queries=256,
        gcs_feature_dim=None,
        gcs_reference_dim=None,
        gcs_feature_nheads=8,
        gcs_reference_nheads=8,
        gcs_self_attn=True,
        gcs_attn_dropout=0.0,
        gcs_metric="w2",
        gcs_reduce="mean",
        gcs_intra_norm=True,
        gcs_score_scale_init=1.0,
        gcs_sigma_weight_init=1.0,
        gcs_learn_sigma_weight=True,
        gcs_natural_normalize=True,
        gcs_hybrid_natural_init=0.5,
        gcs_log_sigma_bias=0.0,
        gcs_log_sigma_scale=2.0,
        fusion_mode="cluster_fusion",
        fusion_dropout=0.0,
        gcs_fusion_init=0.1,
        gcs_gate_init=0.1,
        eps=1e-6,
    ):
        super().__init__()

        if score_mode not in {"gmm", "attention"}:
            raise ValueError("score_mode must be either 'gmm' or 'attention'.")
        if token_mass_mode not in {"uniform", "learned"}:
            raise ValueError("token_mass_mode must be either 'uniform' or 'learned'.")
        if gcs_metric not in {"w2", "natural", "hybrid"}:
            raise ValueError("gcs_metric must be one of: w2, natural, hybrid.")
        if gcs_reduce not in {"mean", "sum_sqrt"}:
            raise ValueError("gcs_reduce must be either 'mean' or 'sum_sqrt'.")
        if fusion_mode not in {"cluster_fusion", "residual_gated", "concat"}:
            raise ValueError("fusion_mode must be one of: cluster_fusion, residual_gated, concat.")
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
        self.gcs_num_queries = int(gcs_num_queries)
        self.gcs_metric = gcs_metric
        self.gcs_reduce = gcs_reduce
        self.gcs_intra_norm = bool(gcs_intra_norm)
        self.gcs_natural_normalize = bool(gcs_natural_normalize)
        self.gcs_log_sigma_bias = float(gcs_log_sigma_bias)
        self.gcs_log_sigma_scale = float(gcs_log_sigma_scale)
        self.fusion_mode = fusion_mode
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

        self.gcs_feature_dim = self.num_clusters if gcs_feature_dim is None else int(gcs_feature_dim)
        self.gcs_reference_dim = self.raw_cluster_dim if gcs_reference_dim is None else int(gcs_reference_dim)
        self.gcs_dim = self.gcs_feature_dim * self.gcs_reference_dim

        if self.cluster_dim % int(gcs_feature_nheads) != 0:
            raise ValueError("cluster_dim must be divisible by gcs_feature_nheads.")
        if self.gcs_reference_dim % int(gcs_reference_nheads) != 0:
            raise ValueError("gcs_reference_dim must be divisible by gcs_reference_nheads.")

        self.gcs_feature_queries = QuerySelfAttention(
            embed_dim=self.cluster_dim,
            num_queries=self.gcs_num_queries,
            num_heads=int(gcs_feature_nheads),
            dropout=gcs_attn_dropout,
            self_attn=gcs_self_attn,
            out_norm=True,
        )
        self.gcs_reference_queries = QuerySelfAttention(
            embed_dim=self.gcs_reference_dim,
            num_queries=self.gcs_num_queries,
            num_heads=int(gcs_reference_nheads),
            dropout=gcs_attn_dropout,
            self_attn=gcs_self_attn,
            out_norm=True,
        )
        self.gcs_cross_attn = nn.MultiheadAttention(
            embed_dim=self.cluster_dim,
            num_heads=int(gcs_feature_nheads),
            dropout=gcs_attn_dropout,
            batch_first=True,
        )
        self.gcs_cross_norm = nn.LayerNorm(self.cluster_dim)

        self.gcs_image_mean = nn.Linear(self.cluster_dim, self.gcs_feature_dim)
        self.gcs_image_log_sigma = nn.Linear(self.cluster_dim, self.gcs_feature_dim)
        self.gcs_image_mean_norm = nn.LayerNorm(self.gcs_feature_dim)

        self.gcs_reference_mean = nn.Linear(self.gcs_reference_dim, self.gcs_reference_dim)
        self.gcs_reference_log_sigma = nn.Linear(self.gcs_reference_dim, self.gcs_reference_dim)
        self.gcs_reference_mean_norm = nn.LayerNorm(self.gcs_reference_dim)

        self.gcs_log_score_scale = nn.Parameter(torch.tensor(inverse_softplus(float(gcs_score_scale_init))))
        if gcs_learn_sigma_weight:
            self.gcs_log_sigma_weight = nn.Parameter(torch.tensor(inverse_softplus(float(gcs_sigma_weight_init))))
        else:
            self.register_buffer(
                "gcs_log_sigma_weight",
                torch.tensor(inverse_softplus(float(gcs_sigma_weight_init))),
                persistent=True,
            )
        self.gcs_hybrid_natural_logit = nn.Parameter(torch.tensor(inverse_sigmoid(float(gcs_hybrid_natural_init))))

        if self.fusion_mode in {"cluster_fusion", "residual_gated"}:
            if self.gcs_feature_dim != self.num_clusters or self.gcs_reference_dim != self.raw_cluster_dim:
                raise ValueError(
                    "cluster_fusion and residual_gated require gcs_feature_dim=num_clusters "
                    "and gcs_reference_dim=2*cluster_dim."
                )
            expected_dim = self.raw_dim
        else:
            expected_dim = self.raw_dim + self.gcs_dim

        if descriptor_dim is not None and int(descriptor_dim) != expected_dim:
            raise ValueError(f"descriptor_dim must be omitted or equal to {expected_dim}, got {descriptor_dim}.")
        self.output_dim = expected_dim

        fusion_dropout_layer = nn.Dropout(fusion_dropout) if fusion_dropout > 0 else nn.Identity()
        if self.fusion_mode == "cluster_fusion":
            self.cluster_fusion_dropout = fusion_dropout_layer
            self.cluster_fusion = nn.Linear(2 * self.raw_cluster_dim, self.raw_cluster_dim)
            self._init_cluster_fusion(float(gcs_fusion_init))
        elif self.fusion_mode == "residual_gated":
            self.gcs_gate_logit = nn.Parameter(torch.tensor(inverse_sigmoid(float(gcs_gate_init))))
        else:
            self.cluster_fusion_dropout = fusion_dropout_layer

    def _init_cluster_fusion(self, gcs_fusion_init: float) -> None:
        with torch.no_grad():
            self.cluster_fusion.weight.zero_()
            self.cluster_fusion.bias.zero_()
            eye = torch.eye(self.raw_cluster_dim)
            self.cluster_fusion.weight[:, : self.raw_cluster_dim].copy_(eye)
            self.cluster_fusion.weight[:, self.raw_cluster_dim :].copy_(gcs_fusion_init * eye)

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

    def _compute_token_mass(self, local: torch.Tensor) -> torch.Tensor:
        b, _, n = local.shape
        if self.token_mass_mode == "uniform":
            return local.new_full((b, n), 1.0 / max(n, 1))

        token_logits = self.token_mass_head(local).squeeze(1)
        scale = F.softplus(self.token_mass_scale) + self.eps
        token_mass = F.softmax(scale * token_logits, dim=-1)
        return token_mass.clamp_min(self.eps)

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

    def _compute_ot_residual(self, local: torch.Tensor, token_mass: torch.Tensor) -> torch.Tensor:
        if self.score_mode == "gmm":
            logits = self._gmm_scores(local)
        else:
            logits = self._attention_scores(local)

        assignments = self._transport(logits, token_mass)
        cluster_mass = assignments.sum(dim=-1, keepdim=True)
        gamma = assignments / cluster_mass.clamp_min(self.eps)

        mu_hat = torch.einsum("bkn,bdn->bkd", gamma, local)
        diff = local.transpose(1, 2).unsqueeze(1) - mu_hat.unsqueeze(2)
        var_hat = torch.einsum("bkn,bknd->bkd", gamma, diff.square()).clamp_min(self.eps)
        sigma_hat = torch.sqrt(var_hat)

        mean_shift = mu_hat - self.mu.unsqueeze(0)
        sigma_shift = sigma_hat - self._sigma().unsqueeze(0)

        if self.mass_preserving:
            mass_weight = cluster_mass * (self.num_clusters / max(1.0 - self.dustbin_mass, self.eps))
            mass_weight = mass_weight.clamp_min(self.eps).pow(self.mass_power)
            mean_shift = mean_shift * mass_weight
            sigma_shift = sigma_shift * mass_weight

        return torch.cat([mean_shift, sigma_shift], dim=-1)

    def _bounded_log_sigma(self, raw: torch.Tensor) -> torch.Tensor:
        return self.gcs_log_sigma_bias + self.gcs_log_sigma_scale * torch.tanh(raw)

    def _query_gaussians(self, local: torch.Tensor):
        b = local.size(0)
        local_tokens = local.transpose(1, 2).contiguous()

        feature_queries = self.gcs_feature_queries().expand(b, -1, -1)
        query_state = self.gcs_cross_attn(feature_queries, local_tokens, local_tokens, need_weights=False)[0]
        query_state = self.gcs_cross_norm(query_state)

        image_mean = self.gcs_image_mean_norm(self.gcs_image_mean(query_state))
        image_log_sigma = self._bounded_log_sigma(self.gcs_image_log_sigma(query_state))

        reference_state = self.gcs_reference_queries().squeeze(0)
        reference_mean = self.gcs_reference_mean_norm(self.gcs_reference_mean(reference_state))
        reference_log_sigma = self._bounded_log_sigma(self.gcs_reference_log_sigma(reference_state))

        return image_mean, image_log_sigma, reference_mean, reference_log_sigma

    def _reduce_query_scores(self, scores: torch.Tensor) -> torch.Tensor:
        if self.gcs_reduce == "mean":
            return scores.mean(dim=1)
        return scores.sum(dim=1) / math.sqrt(max(self.gcs_num_queries, 1))

    def _w2_similarity(
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
        return -score_scale * self._reduce_query_scores(cost)

    def _natural_similarity(
        self,
        image_mean: torch.Tensor,
        image_log_sigma: torch.Tensor,
        reference_mean: torch.Tensor,
        reference_log_sigma: torch.Tensor,
    ) -> torch.Tensor:
        image_inv_var = torch.exp(-2.0 * image_log_sigma).clamp_max(1.0 / self.eps)
        reference_inv_var = torch.exp(-2.0 * reference_log_sigma).clamp_max(1.0 / self.eps)

        image_eta1 = image_mean * image_inv_var
        image_eta2 = -0.5 * image_inv_var
        reference_eta1 = reference_mean * reference_inv_var
        reference_eta2 = -0.5 * reference_inv_var

        eta1_score = reference_eta1.unsqueeze(0).unsqueeze(-1) * image_eta1.unsqueeze(2)
        eta2_score = reference_eta2.unsqueeze(0).unsqueeze(-1) * image_eta2.unsqueeze(2)
        score = eta1_score + eta2_score

        if self.gcs_natural_normalize:
            ref_norm = torch.sqrt(reference_eta1.square() + reference_eta2.square() + self.eps)
            img_norm = torch.sqrt(image_eta1.square() + image_eta2.square() + self.eps)
            score = score / (ref_norm.unsqueeze(0).unsqueeze(-1) * img_norm.unsqueeze(2)).clamp_min(self.eps)

        return self._reduce_query_scores(score)

    def _compute_gaussian_cs_residual(self, local: torch.Tensor) -> torch.Tensor:
        image_mean, image_log_sigma, reference_mean, reference_log_sigma = self._query_gaussians(local)

        if self.gcs_metric == "w2":
            gcs = self._w2_similarity(image_mean, image_log_sigma, reference_mean, reference_log_sigma)
        elif self.gcs_metric == "natural":
            gcs = self._natural_similarity(image_mean, image_log_sigma, reference_mean, reference_log_sigma)
        else:
            w2_score = self._w2_similarity(image_mean, image_log_sigma, reference_mean, reference_log_sigma)
            natural_score = self._natural_similarity(image_mean, image_log_sigma, reference_mean, reference_log_sigma)
            natural_weight = torch.sigmoid(self.gcs_hybrid_natural_logit)
            gcs = w2_score + natural_weight * natural_score

        if self.gcs_intra_norm:
            gcs = F.normalize(gcs, p=2, dim=1)

        if self.fusion_mode in {"cluster_fusion", "residual_gated"}:
            return gcs.transpose(1, 2).contiguous()
        return gcs.reshape(local.size(0), -1)

    def _fuse(self, ot_residual: torch.Tensor, gcs_residual: torch.Tensor) -> torch.Tensor:
        if self.fusion_mode == "cluster_fusion":
            fused = torch.cat([ot_residual, gcs_residual], dim=-1)
            fused = self.cluster_fusion_dropout(fused)
            return self.cluster_fusion(fused).reshape(ot_residual.size(0), -1)
        if self.fusion_mode == "residual_gated":
            gate = torch.sigmoid(self.gcs_gate_logit)
            return (ot_residual + gate * gcs_residual).reshape(ot_residual.size(0), -1)

        ot_descriptor = ot_residual.reshape(ot_residual.size(0), -1)
        gcs_descriptor = self.cluster_fusion_dropout(gcs_residual)
        return torch.cat([ot_descriptor, gcs_descriptor], dim=-1)

    def forward(self, x):
        x = self._split_backbone_output(x)
        local = self._extract_local_tokens(x)
        token_mass = self._compute_token_mass(local)

        ot_residual = self._compute_ot_residual(local, token_mass)
        gcs_residual = self._compute_gaussian_cs_residual(local)
        descriptor = self._fuse(ot_residual, gcs_residual)
        return F.normalize(descriptor, p=2, dim=-1)


class OTGaussVLADGaussianCSAdaptiveFusionLayer(OTGaussVLADGaussianCSLayer):
    """
    OT-GaussVLAD with Gaussian-aware CS and adaptive cluster-wise fusion.

    For each cluster k:
        r_k = r_ot_k + g_k * r_gcs_k

    The gate g_k is predicted from cluster mass, OT confidence,
    Gaussian-CS confidence, and uncertainty cues from both branches.
    """

    def __init__(
        self,
        *args,
        adaptive_gate_hidden_dim=64,
        adaptive_gate_dropout=0.0,
        adaptive_gate_init=0.1,
        adaptive_gate_temperature=1.0,
        **kwargs,
    ):
        kwargs = dict(kwargs)
        kwargs['fusion_mode'] = 'cluster_fusion'
        kwargs['gcs_fusion_init'] = 0.0
        super().__init__(*args, **kwargs)

        if int(adaptive_gate_hidden_dim) < 1:
            raise ValueError('adaptive_gate_hidden_dim must be at least 1.')
        if float(adaptive_gate_temperature) <= 0.0:
            raise ValueError('adaptive_gate_temperature must be positive.')
        if not 0.0 < float(adaptive_gate_init) < 1.0:
            raise ValueError('adaptive_gate_init must be in (0, 1).')

        self.adaptive_gate_hidden_dim = int(adaptive_gate_hidden_dim)
        self.adaptive_gate_temperature = float(adaptive_gate_temperature)
        self.adaptive_gate_bias = nn.Parameter(torch.tensor(inverse_sigmoid(float(adaptive_gate_init))))

        gate_dropout = nn.Dropout(adaptive_gate_dropout) if adaptive_gate_dropout > 0 else nn.Identity()
        self.adaptive_gate = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, self.adaptive_gate_hidden_dim),
            nn.ReLU(),
            gate_dropout,
            nn.Linear(self.adaptive_gate_hidden_dim, 1),
        )
        with torch.no_grad():
            self.adaptive_gate[-1].weight.zero_()
            self.adaptive_gate[-1].bias.zero_()

    def _normalized_cluster_mass(self, cluster_mass: torch.Tensor) -> torch.Tensor:
        scale = self.num_clusters / max(1.0 - self.dustbin_mass, self.eps)
        return cluster_mass * scale

    def _compute_ot_terms(self, local: torch.Tensor, token_mass: torch.Tensor):
        if self.score_mode == 'gmm':
            logits = self._gmm_scores(local)
        else:
            logits = self._attention_scores(local)

        assignments = self._transport(logits, token_mass)
        cluster_mass = assignments.sum(dim=-1, keepdim=True)
        gamma = assignments / cluster_mass.clamp_min(self.eps)

        mu_hat = torch.einsum('bkn,bdn->bkd', gamma, local)
        diff = local.transpose(1, 2).unsqueeze(1) - mu_hat.unsqueeze(2)
        var_hat = torch.einsum('bkn,bknd->bkd', gamma, diff.square()).clamp_min(self.eps)
        sigma_hat = torch.sqrt(var_hat)

        mean_shift = mu_hat - self.mu.unsqueeze(0)
        sigma_shift = sigma_hat - self._sigma().unsqueeze(0)
        raw_residual = torch.cat([mean_shift, sigma_shift], dim=-1)

        if self.mass_preserving:
            mass_weight = self._normalized_cluster_mass(cluster_mass).clamp_min(self.eps).pow(self.mass_power)
            mean_shift = mean_shift * mass_weight
            sigma_shift = sigma_shift * mass_weight

        residual = torch.cat([mean_shift, sigma_shift], dim=-1)
        cluster_mass_weight = self._normalized_cluster_mass(cluster_mass)
        ot_confidence = torch.exp(-torch.sqrt(raw_residual.square().mean(dim=-1, keepdim=True) + self.eps))
        ot_uncertainty = torch.log1p(sigma_hat.mean(dim=-1, keepdim=True))
        return residual, cluster_mass_weight, ot_confidence, ot_uncertainty

    def _compute_gaussian_cs_terms(self, local: torch.Tensor):
        image_mean, image_log_sigma, reference_mean, reference_log_sigma = self._query_gaussians(local)

        if self.gcs_metric == 'w2':
            gcs_raw = self._w2_similarity(image_mean, image_log_sigma, reference_mean, reference_log_sigma)
        elif self.gcs_metric == 'natural':
            gcs_raw = self._natural_similarity(image_mean, image_log_sigma, reference_mean, reference_log_sigma)
        else:
            w2_score = self._w2_similarity(image_mean, image_log_sigma, reference_mean, reference_log_sigma)
            natural_score = self._natural_similarity(image_mean, image_log_sigma, reference_mean, reference_log_sigma)
            natural_weight = torch.sigmoid(self.gcs_hybrid_natural_logit)
            gcs_raw = w2_score + natural_weight * natural_score

        if self.gcs_intra_norm:
            gcs_residual = F.normalize(gcs_raw, p=2, dim=1)
        else:
            gcs_residual = gcs_raw
        gcs_residual = gcs_residual.transpose(1, 2).contiguous()

        gcs_confidence = torch.exp(-torch.sqrt(gcs_residual.square().mean(dim=-1, keepdim=True) + self.eps))
        gcs_uncertainty = torch.log1p(torch.exp(image_log_sigma).mean(dim=(1, 2), keepdim=True))
        gcs_uncertainty = gcs_uncertainty.expand(-1, self.num_clusters, -1)
        return gcs_residual, gcs_confidence, gcs_uncertainty

    def _adaptive_gate_features(
        self,
        cluster_mass: torch.Tensor,
        ot_confidence: torch.Tensor,
        ot_uncertainty: torch.Tensor,
        gcs_confidence: torch.Tensor,
        gcs_uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        if ot_confidence.shape[1] != cluster_mass.shape[1]:
            ot_confidence = ot_confidence.mean(dim=1, keepdim=True).expand(-1, cluster_mass.shape[1], -1)
        if ot_uncertainty.shape[1] != cluster_mass.shape[1]:
            ot_uncertainty = ot_uncertainty.mean(dim=1, keepdim=True).expand(-1, cluster_mass.shape[1], -1)
        if gcs_confidence.shape[1] != cluster_mass.shape[1]:
            gcs_confidence = gcs_confidence.mean(dim=1, keepdim=True).expand(-1, cluster_mass.shape[1], -1)
        if gcs_uncertainty.shape[1] != cluster_mass.shape[1]:
            gcs_uncertainty = gcs_uncertainty.mean(dim=1, keepdim=True).expand(-1, cluster_mass.shape[1], -1)
        return torch.cat([cluster_mass, ot_confidence, gcs_confidence, ot_uncertainty, gcs_uncertainty], dim=-1)

    def _predict_gate(self, gate_features: torch.Tensor) -> torch.Tensor:
        gate_logits = self.adaptive_gate(gate_features) + self.adaptive_gate_bias
        gate_logits = gate_logits / self.adaptive_gate_temperature
        return torch.sigmoid(gate_logits)

    def forward(self, x):
        x = self._split_backbone_output(x)
        local = self._extract_local_tokens(x)
        token_mass = self._compute_token_mass(local)

        ot_residual, cluster_mass, ot_confidence, ot_uncertainty = self._compute_ot_terms(local, token_mass)
        gcs_residual, gcs_confidence, gcs_uncertainty = self._compute_gaussian_cs_terms(local)

        gate_features = self._adaptive_gate_features(
            cluster_mass,
            ot_confidence,
            ot_uncertainty,
            gcs_confidence,
            gcs_uncertainty,
        )
        gate = self._predict_gate(gate_features)
        fused = ot_residual + gate * gcs_residual
        descriptor = fused.reshape(local.size(0), -1)
        return F.normalize(descriptor, p=2, dim=-1)
