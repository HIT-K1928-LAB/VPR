from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COORD_RE = re.compile(r"@(-?\d+(?:\.\d+)?)@(-?\d+(?:\.\d+)?)@")
MSLS_GROUP_RE = re.compile(r"@\d{8}@([^@]+)@\.jpg$", re.IGNORECASE)


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def _parse_coords(path: Path) -> Optional[Tuple[float, float]]:
    match = COORD_RE.search(path.name)
    if match is None:
        return None
    try:
        return float(match.group(1)), float(match.group(2))
    except ValueError:
        return None


def _group_token(dataset_name: str, path: Path) -> str:
    lower = dataset_name.lower()
    if lower in {"msls", "mapillary_sls"}:
        match = MSLS_GROUP_RE.search(path.name)
        if match is not None:
            return match.group(1)
        return path.parent.name
    if lower in {"sf_xl", "sf-xl"}:
        return path.parent.name
    return path.parent.name


def _list_images(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    return sorted([p for p in root.rglob("*") if _is_image(p)])


class MultiDatasetPlaceDataset(Dataset):
    """Coordinate-aware place dataset with QAA-like grouping.

    We first bucket images spatially, then optionally split each bucket into
    tighter connected components and sample a clique-like subset inside each
    place. This keeps the sampling cheap while making it more structured than
    plain random buckets.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path,
        img_per_place: int = 4,
        transform=None,
        bucket_size_m: float = 25.0,
        cluster_radius_m: Optional[float] = None,
        min_images_per_place: Optional[int] = None,
        sample_mode: str = "random",
        include_queries: bool = True,
        database_subdir: str = "database",
        query_subdir: str = "queries",
        hard_mining: bool = False,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")

        self.img_per_place = int(img_per_place)
        self.transform = transform
        self.bucket_size_m = float(bucket_size_m)
        self.cluster_radius_m = float(
            cluster_radius_m if cluster_radius_m is not None else min(self.bucket_size_m, 20.0)
        )
        self.min_images_per_place = int(min_images_per_place or img_per_place)
        self.sample_mode = sample_mode.lower().strip()
        if self.sample_mode not in {"random", "recent", "cluster", "clique"}:
            raise ValueError(
                f"Unsupported sample_mode {sample_mode!r}. "
                "Choose from 'random', 'recent', 'cluster', or 'clique'."
            )
        self.include_queries = bool(include_queries)
        self.database_subdir = database_subdir
        self.query_subdir = query_subdir
        self.hard_mining = hard_mining

        image_paths: List[Path] = []
        if self.database_subdir == "":
            image_paths.extend(_list_images(self.dataset_path))
        else:
            image_paths.extend(_list_images(self.dataset_path / self.database_subdir))
        if self.include_queries and self.query_subdir:
            image_paths.extend(_list_images(self.dataset_path / self.query_subdir))

        unique_paths: List[Path] = []
        seen = set()
        for path in image_paths:
            key = path.as_posix()
            if key not in seen:
                seen.add(key)
                unique_paths.append(path)
        self.image_paths = unique_paths
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No training images found under {self.dataset_path}")

        coords: List[Tuple[float, float]] = []
        tokens: List[str] = []
        for path in self.image_paths:
            coord = _parse_coords(path)
            if coord is None:
                raise ValueError(f"Could not parse coordinates from image name: {path}")
            coords.append(coord)
            tokens.append(_group_token(self.dataset_name, path))

        self.coords = np.asarray(coords, dtype=np.float32)
        self.tokens = tokens
        bucket_coords = np.floor(self.coords / self.bucket_size_m).astype(np.int64)

        grouped: DefaultDict[Tuple[str, int, int], List[int]] = defaultdict(list)
        for idx, (token, bucket) in enumerate(zip(tokens, bucket_coords)):
            grouped[(token, int(bucket[0]), int(bucket[1]))].append(idx)

        ordered_groups = sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]))
        self.place_keys: List[Tuple] = []
        self.places: List[np.ndarray] = []
        for group_key, indices in ordered_groups:
            indices_np = np.asarray(indices, dtype=np.int64)
            if self.sample_mode in {"cluster", "clique"}:
                components = self._split_into_components(indices_np, self.cluster_radius_m)
                for component_idx, component in enumerate(components):
                    if len(component) >= self.min_images_per_place:
                        self.place_keys.append((*group_key, component_idx))
                        self.places.append(component)
            else:
                if len(indices_np) >= self.min_images_per_place:
                    self.place_keys.append(group_key)
                    self.places.append(indices_np)

        if len(self.places) == 0:
            raise RuntimeError(
                f"No valid places found in {self.dataset_name}. "
                f"Try lowering bucket_size_m ({self.bucket_size_m}) or "
                f"min_images_per_place ({self.min_images_per_place})."
            )

        self.total_nb_images = len(self.image_paths)
        self.total_nb_places = len(self.places)
        self.cities = [self.dataset_name]
        self._bad_image_paths = set()

    def _split_into_components(self, place_indices: np.ndarray, radius: float) -> List[np.ndarray]:
        if len(place_indices) == 0:
            return []
        if len(place_indices) == 1:
            return [place_indices]

        coords = self.coords[place_indices].astype(np.float64)
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        adjacency = dist <= radius
        np.fill_diagonal(adjacency, True)

        visited = np.zeros(len(place_indices), dtype=bool)
        components: List[np.ndarray] = []
        for start in range(len(place_indices)):
            if visited[start]:
                continue
            stack = [start]
            visited[start] = True
            component: List[int] = []
            while stack:
                node = stack.pop()
                component.append(node)
                neighbors = np.where(adjacency[node] & ~visited)[0]
                if len(neighbors) > 0:
                    visited[neighbors] = True
                    stack.extend(neighbors.tolist())
            component_indices = place_indices[np.asarray(component, dtype=np.int64)]
            if len(component_indices) >= self.min_images_per_place:
                components.append(component_indices)
        components.sort(key=len, reverse=True)
        return components

    def reload(self, model=None, recompute=False):
        order = list(range(len(self.places)))
        random.shuffle(order)
        self.places = [self.places[i] for i in order]
        self.place_keys = [self.place_keys[i] for i in order]

    def __len__(self):
        return len(self.places)

    def _is_bad_image(self, path: Path) -> bool:
        return path.as_posix() in self._bad_image_paths

    def _mark_bad_image(self, path: Path) -> None:
        self._bad_image_paths.add(path.as_posix())

    def _valid_place_indices(self, place_indices: np.ndarray) -> np.ndarray:
        if len(place_indices) == 0:
            return place_indices
        keep = [not self._is_bad_image(self.image_paths[int(idx)]) for idx in place_indices]
        return place_indices[np.asarray(keep, dtype=bool)]

    @staticmethod
    def image_loader(path: Path):
        try:
            with Image.open(path) as img:
                return img.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError):
            return None

    def _sample_random_indices(self, place_indices: np.ndarray) -> np.ndarray:
        if len(place_indices) <= self.img_per_place:
            return np.random.choice(place_indices, self.img_per_place, replace=True)
        return np.random.choice(place_indices, self.img_per_place, replace=False)

    def _sample_recent_indices(self, place_indices: np.ndarray) -> np.ndarray:
        if len(place_indices) <= self.img_per_place:
            return np.random.choice(place_indices, self.img_per_place, replace=True)
        return place_indices[-self.img_per_place :]

    def _sample_clique_indices(self, place_indices: np.ndarray) -> np.ndarray:
        if len(place_indices) <= self.img_per_place:
            return np.random.choice(place_indices, self.img_per_place, replace=True)

        coords = self.coords[place_indices].astype(np.float64)
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)

        center = int(np.argmin(dist.mean(axis=1)))
        selected = [center]
        available = [i for i in range(len(place_indices)) if i != center]

        while len(selected) < self.img_per_place:
            feasible = [
                idx for idx in available if np.all(dist[idx, selected] <= self.cluster_radius_m)
            ]
            if not feasible:
                break
            scores = [float(dist[idx, selected].mean()) for idx in feasible]
            best = feasible[int(np.argmin(scores))]
            selected.append(best)
            available.remove(best)

        if len(selected) < self.img_per_place:
            ordering = np.argsort(dist[center])
            for idx in ordering:
                idx = int(idx)
                if idx not in selected:
                    selected.append(idx)
                if len(selected) >= self.img_per_place:
                    break

        if len(selected) < self.img_per_place:
            selected.extend(
                np.random.choice(
                    np.arange(len(place_indices)),
                    self.img_per_place - len(selected),
                    replace=True,
                ).tolist()
            )

        return place_indices[np.asarray(selected[: self.img_per_place], dtype=np.int64)]

    def _sample_indices(self, place_indices: np.ndarray) -> np.ndarray:
        if self.sample_mode == "recent":
            return self._sample_recent_indices(place_indices)
        if self.sample_mode == "clique":
            return self._sample_clique_indices(place_indices)
        if self.sample_mode == "cluster":
            return self._sample_random_indices(place_indices)
        return self._sample_random_indices(place_indices)

    def __getitem__(self, index):
        place_indices = self._valid_place_indices(self.places[index])
        if len(place_indices) == 0:
            raise RuntimeError(f"No readable images remain in place {index} of {self.dataset_name}.")

        imgs = []
        labels = []
        attempts = 0
        max_attempts = max(self.img_per_place * 10, len(place_indices) * 4)

        while len(imgs) < self.img_per_place and attempts < max_attempts:
            sampled = np.atleast_1d(self._sample_indices(place_indices))
            progress_made = False

            for idx in sampled:
                if len(imgs) >= self.img_per_place:
                    break
                img_path = self.image_paths[int(idx)]
                img = self.image_loader(img_path)
                attempts += 1
                if img is None:
                    self._mark_bad_image(img_path)
                    place_indices = self._valid_place_indices(place_indices)
                    if len(place_indices) == 0:
                        break
                    continue
                if self.transform is not None:
                    img = self.transform(img)
                imgs.append(img)
                labels.append(index)
                progress_made = True

            if len(place_indices) == 0:
                break
            if not progress_made and len(imgs) < self.img_per_place:
                attempts += 1

        if len(imgs) < self.img_per_place:
            raise RuntimeError(
                f"Could not load {self.img_per_place} valid images for place {index} in {self.dataset_name}. "
                f"Found only {len(imgs)} after filtering unreadable files."
            )

        return torch.stack(imgs), torch.tensor(labels, dtype=torch.long)
