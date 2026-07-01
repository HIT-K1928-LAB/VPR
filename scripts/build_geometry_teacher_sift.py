"""
Build a geometry-teacher CSV for uncertainty-aware ranking distillation.

Pipeline:
    GSV-Cities images
    -> OpenVPRLab global descriptors from a checkpoint
    -> top-K retrieval
    -> SIFT matching + fundamental-matrix RANSAC
    -> label-pair teacher CSV

The output CSV is compatible with
src.losses.geometry_uncertainty_distillation.GeometryUncertaintyDistillationLoss:

    query_label,candidate_label,geo_score,inliers

Extra columns are written for analysis but ignored by the loss.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from rich.progress import track
from torchvision import transforms as T

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run import IMAGENET_MEAN_STD, get_instance
from src.core.vpr_framework import VPRFramework
from src.dataloaders.train.gsv_cities import GSVCitiesDataset
from src.losses.vpr_losses import VPRLossFunction
from src.utils import config_manager


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    rel_name: str
    label: int


@dataclass
class MatchScore:
    geo_score: float
    inliers: int
    inlier_ratio: float
    num_matches: int


def load_config(config_path: Path) -> dict:
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build label-pair geometric teacher CSV.")
    parser.add_argument("--config", type=Path, required=True, help="OpenVPRLab model config.")
    parser.add_argument("--ckpt", type=Path, default=None, help="Checkpoint used for global retrieval.")
    parser.add_argument("--output", type=Path, required=True, help="Output teacher CSV path.")
    parser.add_argument("--train_set", type=str, default=None, help="Override train dataset name.")
    parser.add_argument("--cities", nargs="+", default=None, help="Override city list. Use one or more city names.")
    parser.add_argument("--max_places", type=int, default=2000, help="Limit number of places for a first run.")
    parser.add_argument("--max_images_per_place", type=int, default=1, help="Images sampled per place.")
    parser.add_argument("--topk", type=int, default=20, help="Global retrieval candidates per image.")
    parser.add_argument("--max_pairs", type=int, default=40000, help="Maximum image pairs to verify.")
    parser.add_argument("--batch_size", type=int, default=64, help="Descriptor extraction batch size.")
    parser.add_argument("--image_size", type=int, nargs=2, default=None, help="Descriptor image size H W.")
    parser.add_argument("--local_resize_max", type=int, default=1024, help="Max side for local matching images.")
    parser.add_argument("--sift_max_keypoints", type=int, default=4096, help="SIFT features per image.")
    parser.add_argument("--ratio_thresh", type=float, default=0.75, help="Lowe ratio threshold.")
    parser.add_argument("--ransac_thresh", type=float, default=1.5, help="RANSAC reprojection threshold.")
    parser.add_argument("--ransac_conf", type=float, default=0.999, help="RANSAC confidence.")
    parser.add_argument("--min_matches", type=int, default=8, help="Minimum tentative matches before RANSAC.")
    parser.add_argument("--exclude_same_label", action="store_true", help="Do not verify same-place image pairs.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    return parser.parse_args()


def build_records(config: dict, args: argparse.Namespace) -> List[ImageRecord]:
    dataset_name = args.train_set or config["datamodule"]["train_set_name"]
    dataset_path = config_manager.get_dataset_path(dataset_name=dataset_name, dataset_type="train")
    cities = args.cities if args.cities is not None else config["datamodule"]["cities"]
    if isinstance(cities, list) and len(cities) == 1 and cities[0].lower() == "all":
        cities = "all"

    dataset = GSVCitiesDataset(
        dataset_path=dataset_path,
        cities=cities,
        img_per_place=max(1, args.max_images_per_place),
        random_sample_from_each_place=False,
        transform=None,
    )

    dataframe = dataset.dataframe.sort_values(
        by=["year", "month", "lat"],
        ascending=[False, False, False],
    )

    records: List[ImageRecord] = []
    num_places = 0
    for label, place in dataframe.groupby(level=0, sort=False):
        if args.max_places is not None and num_places >= args.max_places:
            break
        place = place.head(args.max_images_per_place)
        for _, row in place.iterrows():
            img_name = dataset.get_img_name(row)
            path = dataset.base_path / "Images" / row["city_id"] / img_name
            rel_name = f"{row['city_id']}/{img_name}"
            if path.is_file():
                records.append(ImageRecord(path=path, rel_name=rel_name, label=int(label)))
        num_places += 1

    if not records:
        raise RuntimeError(f"No images found under {dataset.base_path}")
    return records


def build_model(config: dict, ckpt_path: Path, device: torch.device) -> VPRFramework:
    backbone = get_instance(config["backbone"]["module"], config["backbone"]["class"], config["backbone"]["params"])
    out_channels = backbone.out_channels
    if "in_channels" in config["aggregator"]["params"]:
        if config["aggregator"]["params"]["in_channels"] is None:
            config["aggregator"]["params"]["in_channels"] = out_channels
    aggregator = get_instance(
        config["aggregator"]["module"],
        config["aggregator"]["class"],
        config["aggregator"]["params"],
    )

    model = VPRFramework(
        backbone=backbone,
        aggregator=aggregator,
        loss_function=VPRLossFunction(),
        optimizer=config["trainer"]["optimizer"],
        lr=config["trainer"]["lr"],
        weight_decay=config["trainer"]["wd"],
        warmup_steps=config["trainer"]["warmup"],
        milestones=config["trainer"]["milestones"],
        lr_mult=config["trainer"]["lr_mult"],
        verbose=False,
        config_dict=config,
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def descriptor_transform(image_size: Tuple[int, int]):
    return T.Compose(
        [
            T.Resize(size=image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"]),
        ]
    )


@torch.inference_mode()
def extract_descriptors(
    model: VPRFramework,
    records: List[ImageRecord],
    image_size: Tuple[int, int],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    transform = descriptor_transform(image_size)
    descriptors = []
    batch = []

    for record in track(records, description="Extracting global descriptors"):
        image = Image.open(record.path).convert("RGB")
        batch.append(transform(image))
        if len(batch) == batch_size:
            images = torch.stack(batch, dim=0).to(device, non_blocking=True)
            desc = model._extract_descriptors(model(images))
            descriptors.append(F.normalize(desc.float(), p=2, dim=1).cpu())
            batch.clear()

    if batch:
        images = torch.stack(batch, dim=0).to(device, non_blocking=True)
        desc = model._extract_descriptors(model(images))
        descriptors.append(F.normalize(desc.float(), p=2, dim=1).cpu())

    return torch.cat(descriptors, dim=0).numpy().astype("float32")


def retrieve_topk(descriptors: np.ndarray, topk: int) -> np.ndarray:
    try:
        import faiss

        index = faiss.IndexFlatIP(descriptors.shape[1])
        index.add(descriptors)
        _, indices = index.search(descriptors, topk + 1)
    except Exception:
        sims = descriptors @ descriptors.T
        indices = np.argsort(-sims, axis=1)[:, : topk + 1]

    clean = []
    for i, row in enumerate(indices):
        row = [int(j) for j in row if int(j) >= 0 and int(j) != i]
        clean.append(row[:topk])
    return np.asarray(clean, dtype=np.int64)


def resize_for_local(image: np.ndarray, resize_max: int) -> np.ndarray:
    if resize_max <= 0:
        return image
    h, w = image.shape[:2]
    scale = resize_max / max(h, w)
    if scale >= 1.0:
        return image
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


class SIFTFeatureCache:
    def __init__(self, max_keypoints: int, resize_max: int):
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("OpenCV SIFT is unavailable. Install opencv-contrib-python or use an environment with SIFT.")
        self.extractor = cv2.SIFT_create(nfeatures=int(max_keypoints))
        self.resize_max = int(resize_max)
        self.cache: Dict[Path, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}

    def get(self, path: Path):
        if path in self.cache:
            return self.cache[path]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            keypoints, descriptors = None, None
        else:
            image = resize_for_local(image, self.resize_max)
            keypoints, descriptors = self.extractor.detectAndCompute(image, None)
            if keypoints is not None:
                keypoints = np.asarray([kp.pt for kp in keypoints], dtype=np.float32)
        self.cache[path] = (keypoints, descriptors)
        return self.cache[path]


def verify_pair(
    cache: SIFTFeatureCache,
    path_q: Path,
    path_d: Path,
    ratio_thresh: float,
    min_matches: int,
    ransac_thresh: float,
    ransac_conf: float,
) -> MatchScore:
    kps_q, desc_q = cache.get(path_q)
    kps_d, desc_d = cache.get(path_d)
    if kps_q is None or kps_d is None or desc_q is None or desc_d is None:
        return MatchScore(0.0, 0, 0.0, 0)
    if len(desc_q) < 2 or len(desc_d) < 2:
        return MatchScore(0.0, 0, 0.0, 0)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(desc_q, desc_d, k=2)
    good = [m for m, n in knn if m.distance < ratio_thresh * n.distance]
    if len(good) < min_matches:
        return MatchScore(0.0, 0, 0.0, len(good))

    pts_q = np.float32([kps_q[m.queryIdx] for m in good])
    pts_d = np.float32([kps_d[m.trainIdx] for m in good])
    _, mask = cv2.findFundamentalMat(
        pts_q,
        pts_d,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=float(ransac_thresh),
        confidence=float(ransac_conf),
    )
    if mask is None:
        return MatchScore(0.0, 0, 0.0, len(good))

    inliers = int(mask.ravel().astype(bool).sum())
    inlier_ratio = float(inliers / max(len(good), 1))
    geo_score = float(math.log1p(max(inliers, 0)) * inlier_ratio)
    return MatchScore(geo_score, inliers, inlier_ratio, len(good))


def aggregate_teacher(
    records: List[ImageRecord],
    topk_indices: np.ndarray,
    args: argparse.Namespace,
) -> Dict[Tuple[int, int], Tuple[MatchScore, ImageRecord, ImageRecord]]:
    cache = SIFTFeatureCache(args.sift_max_keypoints, args.local_resize_max)
    aggregated: Dict[Tuple[int, int], Tuple[MatchScore, ImageRecord, ImageRecord]] = {}
    verified = 0

    total = min(len(records) * topk_indices.shape[1], int(args.max_pairs))
    progress = track(range(total), description="Verifying SIFT+RANSAC pairs")

    flat_pairs = []
    for q_idx, row in enumerate(topk_indices):
        for d_idx in row:
            if args.exclude_same_label and records[q_idx].label == records[int(d_idx)].label:
                continue
            flat_pairs.append((q_idx, int(d_idx)))
            if len(flat_pairs) >= int(args.max_pairs):
                break
        if len(flat_pairs) >= int(args.max_pairs):
            break

    for pair_idx in progress:
        if pair_idx >= len(flat_pairs):
            break
        q_idx, d_idx = flat_pairs[pair_idx]
        q_record = records[q_idx]
        d_record = records[d_idx]
        score = verify_pair(
            cache,
            q_record.path,
            d_record.path,
            args.ratio_thresh,
            args.min_matches,
            args.ransac_thresh,
            args.ransac_conf,
        )
        key = (q_record.label, d_record.label)
        old = aggregated.get(key)
        if old is None or (score.geo_score, score.inliers) > (old[0].geo_score, old[0].inliers):
            aggregated[key] = (score, q_record, d_record)
        verified += 1

    print(f"Verified image pairs: {verified}")
    print(f"Aggregated label pairs: {len(aggregated)}")
    return aggregated


def write_teacher_csv(
    output: Path,
    aggregated: Dict[Tuple[int, int], Tuple[MatchScore, ImageRecord, ImageRecord]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_label",
        "candidate_label",
        "geo_score",
        "inliers",
        "inlier_ratio",
        "num_matches",
        "query_image",
        "candidate_image",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (q_label, d_label), (score, q_record, d_record) in sorted(aggregated.items()):
            writer.writerow(
                {
                    "query_label": q_label,
                    "candidate_label": d_label,
                    "geo_score": f"{score.geo_score:.8f}",
                    "inliers": score.inliers,
                    "inlier_ratio": f"{score.inlier_ratio:.8f}",
                    "num_matches": score.num_matches,
                    "query_image": q_record.rel_name,
                    "candidate_image": d_record.rel_name,
                }
            )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ckpt_path = args.ckpt or Path(config.get("init_ckpt_path") or config.get("ckpt_path") or "")
    if not ckpt_path.is_file():
        raise FileNotFoundError("Provide --ckpt or set init_ckpt_path/ckpt_path in the config.")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    image_size = args.image_size
    if image_size is None:
        image_size = config["datamodule"].get("val_image_size") or config["datamodule"]["train_image_size"]
    image_size = (int(image_size[0]), int(image_size[1]))

    records = build_records(config, args)
    print(f"Selected images: {len(records)}")

    model = build_model(config, ckpt_path, device)
    descriptors = extract_descriptors(model, records, image_size, args.batch_size, device)
    topk_indices = retrieve_topk(descriptors, args.topk)
    aggregated = aggregate_teacher(records, topk_indices, args)
    write_teacher_csv(args.output, aggregated)
    print(f"Wrote teacher CSV: {args.output}")


if __name__ == "__main__":
    main()
