from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional, Tuple, Any, List, Sequence

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from src.utils import config_manager


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COORD_RE = re.compile(r"@(-?\d+(?:\.\d+)?)@(-?\d+(?:\.\d+)?)@")


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def _list_images(directory: Path) -> List[Path]:
    return sorted([p for p in directory.iterdir() if _is_image(p)])

def _read_path_list(list_file: Path) -> List[str]:
    entries: List[str] = []
    with list_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(line)
    return entries


def _resolve_entries(
    dataset_path: Path,
    entries: Sequence[str],
    candidate_roots: Sequence[Path],
    allow_missing: bool = False,
) -> List[Path]:
    resolved: List[Path] = []
    for raw_entry in entries:
        entry = Path(raw_entry)
        candidates: List[Path] = []
        if entry.is_absolute():
            candidates.append(entry)
        else:
            candidates.append(dataset_path / entry)
            for root in candidate_roots:
                candidates.append(root / entry)

        found: Optional[Path] = None
        for candidate in candidates:
            if candidate.is_file():
                found = candidate
                break
        if found is None:
            if allow_missing:
                continue
            raise FileNotFoundError(
                f"Could not resolve image path '{raw_entry}' relative to {dataset_path}."
            )
        resolved.append(found)
    return resolved


def _to_dataset_relative_strings(dataset_path: Path, paths: Sequence[Path]) -> np.ndarray:
    rel_paths: List[str] = []
    for path in paths:
        try:
            rel_paths.append(path.relative_to(dataset_path).as_posix())
        except ValueError:
            rel_paths.append(path.as_posix())
    return np.asarray(rel_paths, dtype=np.str_)


def _split_manifest_entries(entries: Sequence[str]) -> Tuple[List[str], List[str]]:
    db_entries: List[str] = []
    q_entries: List[str] = []
    for raw_entry in entries:
        parts = Path(raw_entry).parts
        if "database" in parts:
            db_entries.append(raw_entry)
        elif "queries" in parts:
            q_entries.append(raw_entry)
    return db_entries, q_entries


def _parse_coords(path: Path) -> Optional[Tuple[float, float]]:
    match = COORD_RE.search(path.name)
    if match is None:
        return None
    try:
        return float(match.group(1)), float(match.group(2))
    except ValueError:
        return None


def _build_ground_truth(db_paths: List[Path], q_paths: List[Path], radius_m: float) -> List[np.ndarray]:
    if len(q_paths) == 0 or len(db_paths) == 0:
        return []

    db_coords = []
    q_coords = []

    for path in db_paths:
        coords = _parse_coords(path)
        if coords is None:
            raise ValueError(f"Could not parse coordinates from database image: {path}")
        db_coords.append(coords)

    for path in q_paths:
        coords = _parse_coords(path)
        if coords is None:
            raise ValueError(f"Could not parse coordinates from query image: {path}")
        q_coords.append(coords)

    db_coords_np = np.asarray(db_coords, dtype=np.float32)
    q_coords_np = np.asarray(q_coords, dtype=np.float32)

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(db_coords_np)
        positives = tree.query_ball_point(q_coords_np, r=float(radius_m))
        return [np.asarray(idx, dtype=np.int64) for idx in positives]
    except Exception:
        diff = q_coords_np[:, None, :] - db_coords_np[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        radius2 = float(radius_m) ** 2
        return [np.flatnonzero(row <= radius2).astype(np.int64) for row in dist2]


class CoordinateVPRDataset(Dataset):
    """
    Standard coordinate-encoded test split:
    - images/test/database
    - images/test/queries
    - filenames contain @UTM_east@UTM_north@
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str] = None,
        input_transform: Optional[Callable] = None,
        positive_radius_m: float = 25.0,
    ):
        self.dataset_name = dataset_name
        self.input_transform = input_transform
        self.positive_radius_m = float(positive_radius_m)

        if dataset_path is None:
            dataset_path = config_manager.get_dataset_path(dataset_name=dataset_name, dataset_type="val")
        else:
            dataset_path = Path(dataset_path)
            if not dataset_path.is_dir():
                raise FileNotFoundError(f"The directory {dataset_path} does not exist. Please check the path.")

        self.dataset_path = Path(dataset_path)

        self.database_dir = self.dataset_path / "images" / "test" / "database"
        self.queries_dir = self.dataset_path / "images" / "test" / "queries"
        database_list = self.dataset_path / "images" / "test" / "database_images_paths.txt"
        queries_list = self.dataset_path / "images" / "test" / "queries_images_paths.txt"
        all_images_list = self.dataset_path / "all_images_paths.txt"

        if database_list.is_file() and queries_list.is_file():
            if not self.database_dir.is_dir():
                raise FileNotFoundError(f"Missing database directory: {self.database_dir}")
            db_entries = _read_path_list(database_list)
            q_entries = _read_path_list(queries_list)
            db_paths = _resolve_entries(
                self.dataset_path,
                db_entries,
                candidate_roots=[self.database_dir],
                allow_missing=True,
            )
            q_roots = [self.queries_dir, self.database_dir] if self.queries_dir.is_dir() else [self.database_dir]
            q_paths = _resolve_entries(
                self.dataset_path,
                q_entries,
                candidate_roots=q_roots,
                allow_missing=True,
            )
        elif all_images_list.is_file():
            all_entries = _read_path_list(all_images_list)
            db_entries, q_entries = _split_manifest_entries(all_entries)
            if len(db_entries) == 0 or len(q_entries) == 0:
                raise RuntimeError(
                    f"Could not split manifest {all_images_list} into database and query entries."
                )
            db_paths = _resolve_entries(
                self.dataset_path,
                db_entries,
                candidate_roots=[self.dataset_path],
                allow_missing=True,
            )
            q_paths = _resolve_entries(
                self.dataset_path,
                q_entries,
                candidate_roots=[self.dataset_path],
                allow_missing=True,
            )
        else:
            if not self.database_dir.is_dir():
                raise FileNotFoundError(f"Missing database directory: {self.database_dir}")
            if not self.queries_dir.is_dir():
                raise FileNotFoundError(f"Missing queries directory: {self.queries_dir}")
            db_paths = _list_images(self.database_dir)
            q_paths = _list_images(self.queries_dir)

        if len(db_paths) == 0:
            raise RuntimeError(f"No database images found in {self.database_dir}")

        self.dbImages = _to_dataset_relative_strings(self.dataset_path, db_paths)
        self.qImages = _to_dataset_relative_strings(self.dataset_path, q_paths)

        self.ground_truth = _build_ground_truth(
            db_paths,
            q_paths,
            self.positive_radius_m,
        )

        self.image_paths = np.concatenate((self.dbImages, self.qImages))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        img_path = self.image_paths[index]
        with Image.open(self.dataset_path / img_path) as img:
            img = img.convert("RGB")
        if self.input_transform:
            img = self.input_transform(img)
        return img, index

    def __len__(self) -> int:
        return len(self.image_paths)

