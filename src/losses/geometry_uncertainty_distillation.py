"""
Uncertainty-aware geometric ranking distillation for VPR.

This loss combines a normal metric-learning objective with a listwise
distillation term. The teacher is a geometric re-ranking cache produced offline
from local feature matching. The cache should provide pair-level geometric
scores and inlier counts. Inlier counts are converted into reliability weights:
pairs with few inliers are treated as uncertain and contribute little to the
distillation term. Setting base_loss_weight to 0.0 yields a distillation-only
fine-tuning objective.

Supported teacher cache formats:

1. CSV with columns such as:
   query_label,candidate_label,geo_score,inliers
   q_label,db_label,score,inlier_count

2. Torch .pt/.pth dictionary in sparse form:
   {
       "label_pairs": LongTensor[M, 2],
       "geo_scores": FloatTensor[M],
       "inlier_counts": FloatTensor[M],
   }

3. Torch .pt/.pth dictionary in dense form:
   {
       "labels": LongTensor[L],
       "geo_scores": FloatTensor[L, L],
       "inlier_counts": FloatTensor[L, L],
   }

The current OpenVPRLab training loop passes only descriptors and labels to the
loss. Therefore this implementation uses label-pair teacher scores. For
image-pair teachers, use a dataset/framework that also returns stable image ids.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vpr_losses import VPRLossFunction


PairKey = Tuple[int, int]


def _first_present(row: Dict[str, str], names: Iterable[str], default=None):
    for name in names:
        if name in row and row[name] not in {"", None}:
            return row[name]
    return default


class GeometryUncertaintyDistillationLoss(nn.Module):
    """Metric loss plus uncertainty-aware geometric listwise distillation.

    Args:
        base_loss_fn_name: Base VPR loss name used by VPRLossFunction.
        miner_name: Miner name used by VPRLossFunction.
        teacher_path: Optional path to a CSV/PT cache with geometric teacher
            pair scores.
        base_loss_weight: Weight of the base VPR metric-learning term. Set to
            0.0 for distribution-only distillation fine-tuning.
        distill_weight: Weight of the geometric distillation term.
        teacher_temperature: Softmax temperature for teacher logits.
        student_temperature: Softmax temperature for descriptor similarities.
        inlier_threshold: Inlier count at which the reliability is about 0.5.
        inlier_temperature: Smoothness of the reliability sigmoid.
        min_reliable_inliers: Pairs below this inlier count are ignored.
        positive_fallback: Add same-label positives even when absent from cache.
        positive_fallback_score: Teacher score for same-label pairs.
        positive_fallback_inliers: Inlier count assigned to fallback positives.
        symmetric_teacher: If true, cache entries are inserted both ways.
        max_teacher_pairs: Optional cap for loading very large sparse caches.
    """

    def __init__(
        self,
        base_loss_fn_name: str = "MultiSimilarityLoss",
        miner_name: str = "MultiSimilarityMiner",
        teacher_path: Optional[str] = None,
        base_loss_weight: float = 1.0,
        distill_weight: float = 0.05,
        teacher_temperature: float = 0.07,
        student_temperature: float = 0.07,
        inlier_threshold: float = 20.0,
        inlier_temperature: float = 6.0,
        min_reliable_inliers: float = 8.0,
        positive_fallback: bool = True,
        positive_fallback_score: float = 1.0,
        positive_fallback_inliers: float = 50.0,
        symmetric_teacher: bool = True,
        max_teacher_pairs: Optional[int] = None,
    ):
        super().__init__()
        if base_loss_weight < 0:
            raise ValueError("base_loss_weight must be non-negative.")
        if distill_weight < 0:
            raise ValueError("distill_weight must be non-negative.")
        if teacher_temperature <= 0 or student_temperature <= 0:
            raise ValueError("teacher_temperature and student_temperature must be positive.")
        if inlier_temperature <= 0:
            raise ValueError("inlier_temperature must be positive.")

        self.base_loss = VPRLossFunction(
            loss_fn_name=base_loss_fn_name,
            miner_name=miner_name,
        )
        self.teacher_path = teacher_path
        self.base_loss_weight = float(base_loss_weight)
        self.distill_weight = float(distill_weight)
        self.teacher_temperature = float(teacher_temperature)
        self.student_temperature = float(student_temperature)
        self.inlier_threshold = float(inlier_threshold)
        self.inlier_temperature = float(inlier_temperature)
        self.min_reliable_inliers = float(min_reliable_inliers)
        self.positive_fallback = bool(positive_fallback)
        self.positive_fallback_score = float(positive_fallback_score)
        self.positive_fallback_inliers = float(positive_fallback_inliers)
        self.symmetric_teacher = bool(symmetric_teacher)
        self.max_teacher_pairs = max_teacher_pairs

        self._teacher_pairs: Dict[PairKey, Tuple[float, float]] = {}
        self._dense_labels = None
        self._dense_scores = None
        self._dense_inliers = None

        if teacher_path:
            self._load_teacher_cache(Path(teacher_path))

    def _load_teacher_cache(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(
                f"Geometric teacher cache not found: {path}. "
                "Generate it first or set teacher_path to null."
            )

        suffix = path.suffix.lower()
        if suffix == ".csv":
            self._load_csv_cache(path)
        elif suffix in {".pt", ".pth"}:
            self._load_torch_cache(path)
        else:
            raise ValueError(f"Unsupported teacher cache format: {path.suffix}")

    def _add_pair(self, q_label: int, d_label: int, score: float, inliers: float) -> None:
        self._teacher_pairs[(int(q_label), int(d_label))] = (float(score), float(inliers))
        if self.symmetric_teacher:
            self._teacher_pairs[(int(d_label), int(q_label))] = (float(score), float(inliers))

    def _load_csv_cache(self, path: Path) -> None:
        loaded = 0
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q_label = _first_present(row, ("query_label", "q_label", "query", "src_label"))
                d_label = _first_present(row, ("candidate_label", "db_label", "database_label", "dst_label", "target_label"))
                if q_label is None or d_label is None:
                    raise ValueError(
                        "CSV teacher cache must contain query/candidate label columns."
                    )

                inliers = _first_present(row, ("inliers", "inlier_count", "num_inliers"), default=None)
                score = _first_present(row, ("geo_score", "score", "teacher_score"), default=None)
                if inliers is None and score is None:
                    raise ValueError("CSV teacher cache needs either score or inlier count.")

                inliers_f = float(inliers) if inliers is not None else float(score)
                score_f = float(score) if score is not None else math.log1p(max(inliers_f, 0.0))
                self._add_pair(int(q_label), int(d_label), score_f, inliers_f)

                loaded += 1
                if self.max_teacher_pairs is not None and loaded >= int(self.max_teacher_pairs):
                    break

    def _load_torch_cache(self, path: Path) -> None:
        cache = torch.load(path, map_location="cpu")
        if not isinstance(cache, dict):
            raise ValueError("Torch teacher cache must be a dictionary.")

        if {"labels", "geo_scores", "inlier_counts"}.issubset(cache.keys()):
            self._dense_labels = torch.as_tensor(cache["labels"], dtype=torch.long).cpu()
            self._dense_scores = torch.as_tensor(cache["geo_scores"], dtype=torch.float32).cpu()
            self._dense_inliers = torch.as_tensor(cache["inlier_counts"], dtype=torch.float32).cpu()
            if self._dense_scores.ndim != 2 or self._dense_inliers.ndim != 2:
                raise ValueError("Dense geo_scores and inlier_counts must be matrices.")
            return

        if "label_pairs" not in cache:
            raise ValueError("Sparse torch cache must contain label_pairs.")

        pairs = torch.as_tensor(cache["label_pairs"], dtype=torch.long).cpu()
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("label_pairs must have shape [M, 2].")

        if "inlier_counts" in cache:
            inliers = torch.as_tensor(cache["inlier_counts"], dtype=torch.float32).cpu()
        elif "inliers" in cache:
            inliers = torch.as_tensor(cache["inliers"], dtype=torch.float32).cpu()
        else:
            inliers = None

        if "geo_scores" in cache:
            scores = torch.as_tensor(cache["geo_scores"], dtype=torch.float32).cpu()
        elif "scores" in cache:
            scores = torch.as_tensor(cache["scores"], dtype=torch.float32).cpu()
        else:
            scores = torch.log1p(inliers.clamp_min(0.0)) if inliers is not None else None

        if scores is None and inliers is None:
            raise ValueError("Sparse torch cache needs geo_scores or inlier_counts.")
        if inliers is None:
            inliers = scores
        if scores is None:
            scores = torch.log1p(inliers.clamp_min(0.0))

        total = pairs.shape[0]
        if self.max_teacher_pairs is not None:
            total = min(total, int(self.max_teacher_pairs))
        for idx in range(total):
            self._add_pair(
                int(pairs[idx, 0].item()),
                int(pairs[idx, 1].item()),
                float(scores[idx].item()),
                float(inliers[idx].item()),
            )

    def _lookup_dense(self, labels_cpu: torch.Tensor, device: torch.device):
        dense_labels = self._dense_labels
        if dense_labels is None:
            return None

        label_to_index = {int(v.item()): i for i, v in enumerate(dense_labels)}
        n = labels_cpu.numel()
        scores = torch.zeros(n, n, dtype=torch.float32, device=device)
        inliers = torch.zeros(n, n, dtype=torch.float32, device=device)
        valid = torch.zeros(n, n, dtype=torch.bool, device=device)

        for i, q in enumerate(labels_cpu.tolist()):
            qi = label_to_index.get(int(q))
            if qi is None:
                continue
            for j, d in enumerate(labels_cpu.tolist()):
                dj = label_to_index.get(int(d))
                if dj is None:
                    continue
                scores[i, j] = self._dense_scores[qi, dj].to(device)
                inliers[i, j] = self._dense_inliers[qi, dj].to(device)
                valid[i, j] = True
        return scores, inliers, valid

    def _teacher_matrices(self, labels: torch.Tensor):
        device = labels.device
        labels_cpu = labels.detach().cpu().long()
        n = labels_cpu.numel()

        dense = self._lookup_dense(labels_cpu, device)
        if dense is not None:
            scores, inliers, valid = dense
        else:
            scores = torch.zeros(n, n, dtype=torch.float32, device=device)
            inliers = torch.zeros(n, n, dtype=torch.float32, device=device)
            valid = torch.zeros(n, n, dtype=torch.bool, device=device)

            labels_list = [int(x) for x in labels_cpu.tolist()]
            for i, q_label in enumerate(labels_list):
                for j, d_label in enumerate(labels_list):
                    pair = self._teacher_pairs.get((q_label, d_label))
                    if pair is None:
                        continue
                    score, inlier_count = pair
                    scores[i, j] = float(score)
                    inliers[i, j] = float(inlier_count)
                    valid[i, j] = True

        if self.positive_fallback:
            same_label = labels.view(-1, 1).eq(labels.view(1, -1))
            fallback = same_label & ~torch.eye(n, dtype=torch.bool, device=device)
            missing = fallback & ~valid
            scores = torch.where(
                missing,
                torch.full_like(scores, self.positive_fallback_score),
                scores,
            )
            inliers = torch.where(
                missing,
                torch.full_like(inliers, self.positive_fallback_inliers),
                inliers,
            )
            valid = valid | missing

        valid = valid & ~torch.eye(n, dtype=torch.bool, device=device)
        return scores, inliers, valid

    def _distillation_loss(self, descriptors: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        scores, inliers, valid = self._teacher_matrices(labels)
        if not valid.any():
            return descriptors.new_zeros(())

        reliability = torch.sigmoid((inliers - self.inlier_threshold) / self.inlier_temperature)
        reliability = reliability * (inliers >= self.min_reliable_inliers).float()
        valid = valid & (reliability > 0)
        if not valid.any():
            return descriptors.new_zeros(())

        descriptors = F.normalize(descriptors, p=2, dim=1)
        student_logits = descriptors @ descriptors.t()
        student_logits = student_logits / self.student_temperature

        teacher_logits = scores / self.teacher_temperature
        neg_inf = torch.finfo(student_logits.dtype).min
        teacher_logits = teacher_logits.masked_fill(~valid, neg_inf)
        student_logits = student_logits.masked_fill(~valid, neg_inf)

        row_valid = valid.sum(dim=1) >= 2
        if not row_valid.any():
            return descriptors.new_zeros(())

        teacher_prob = F.softmax(teacher_logits[row_valid], dim=1)
        student_log_prob = F.log_softmax(student_logits[row_valid], dim=1)

        row_reliability = (teacher_prob * reliability[row_valid]).sum(dim=1).detach()
        kl = F.kl_div(student_log_prob, teacher_prob.detach(), reduction="none").sum(dim=1)
        denom = row_reliability.sum().clamp_min(1e-6)
        return (kl * row_reliability).sum() / denom

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        base_loss, batch_accuracy = self.base_loss(embeddings, labels)
        if self.distill_weight == 0:
            return self.base_loss_weight * base_loss, batch_accuracy

        distill_loss = self._distillation_loss(embeddings, labels)
        total_loss = self.base_loss_weight * base_loss + self.distill_weight * distill_loss
        return total_loss, batch_accuracy
