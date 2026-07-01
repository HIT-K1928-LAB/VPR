# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/OpenVPRLab
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch
import lightning as L
from collections import OrderedDict
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
from torchvision.transforms import v2 as T2

from src.dataloaders.train.gsv_cities import GSVCitiesDataset
from src.dataloaders.train.multi_dataset_places import MultiDatasetPlaceDataset
from src.utils import config_manager
from src.dataloaders.valid.mapillary_sls import MapillarySLSDataset
from src.dataloaders.valid.pittsburgh import PittsburghDataset
from src.dataloaders.valid.coordinate_vpr import CoordinateVPRDataset


class VPRDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_set_name="gsv-cities",
        train_set_names=None,
        cities=["Osaka"],
        train_image_size=(224, 224),
        batch_size=60,
        img_per_place=4,
        shuffle_all=False,
        random_sample_from_each_place=True,
        val_set_names=["pitts30k_val", "msls_val"],
        val_image_size=None,
        val_positive_radius_m=25.0,
        train_loader_mode="max_size_cycle",
        train_dataset_weights=None,
        msls_sampling=None,
        sf_xl_sampling=None,
        generic_sampling=None,
        num_workers=4,
        batch_sampler=None,
        mean_std={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    ):
        super().__init__()
        self.train_set_name = train_set_name
        self.train_set_names = list(train_set_names) if train_set_names is not None else [train_set_name]
        self.batch_size = batch_size
        self.img_per_place = img_per_place
        self.shuffle_all = shuffle_all
        self.train_image_size = train_image_size
        self.val_image_size = val_image_size if val_image_size is not None else train_image_size
        self.val_positive_radius_m = float(val_positive_radius_m)
        self.train_loader_mode = str(train_loader_mode or "max_size_cycle").lower()
        self.train_dataset_weights = train_dataset_weights
        self.msls_sampling = dict(msls_sampling or {})
        self.sf_xl_sampling = dict(sf_xl_sampling or {})
        self.generic_sampling = dict(generic_sampling or {})
        self.num_workers = num_workers
        self.batch_sampler = batch_sampler
        self.cities = cities
        self.mean_std = mean_std
        self.random_sample_from_each_place = random_sample_from_each_place
        self.val_set_names = val_set_names

        self.train_set_paths = {}
        for ds_name in self.train_set_names:
            self.train_set_paths[ds_name] = config_manager.get_dataset_path(
                dataset_name=ds_name,
                dataset_type="train",
            )

        self.val_set_paths = {}
        for ds_name in self.val_set_names:
            ds_path = config_manager.get_dataset_path(dataset_name=ds_name, dataset_type="val")
            self.val_set_paths[ds_name] = ds_path

        self.train_transform = T2.Compose([
            T2.ToImage(),
            T2.Resize(size=self.train_image_size, interpolation=T2.InterpolationMode.BICUBIC, antialias=True),
            T2.RandAugment(num_ops=3, magnitude=15, interpolation=T2.InterpolationMode.BILINEAR),
            T2.ToDtype(torch.float32, scale=True),
            T2.Normalize(mean=self.mean_std["mean"], std=self.mean_std["std"]),
        ])

        self.val_transform = T2.Compose([
            T2.ToImage(),
            T2.Resize(size=self.val_image_size, interpolation=T2.InterpolationMode.BICUBIC, antialias=True),
            T2.ToDtype(torch.float32, scale=True),
            T2.Normalize(mean=self.mean_std["mean"], std=self.mean_std["std"]),
        ])

        self.train_dataset = None
        self.train_datasets = None
        self.val_datasets = None
        self.train_dataset_weight_map = self._resolve_train_dataset_weights(self.train_set_names)

    def setup(self, stage=None):
        if stage == "fit":
            self.train_datasets = [self._get_train_dataset(ds_name) for ds_name in self.train_set_names]
            self.train_dataset = self._summarize_train_datasets()
            self.val_datasets = [self._get_val_dataset(ds_name) for ds_name in self.val_set_names]
        if stage == "test":
            self.val_datasets = [self._get_val_dataset(ds_name) for ds_name in self.val_set_names]
        if stage == "predict":
            self.val_datasets = [self._get_val_dataset(ds_name) for ds_name in self.val_set_names]

    def on_train_epoch_start(self):
        self._refresh_train_datasets()

    def train_dataloader(self):
        if len(self.train_set_names) == 1 and self.train_set_names[0].lower() in {"gsv-cities", "gsv-cities-light"}:
            if self.train_datasets is not None and len(self.train_datasets) > 0:
                self.train_dataset = self.train_datasets[0]
            else:
                self.train_dataset = self._get_train_dataset(self.train_set_names[0])
                self.train_datasets = [self.train_dataset]
            self._refresh_train_datasets()
            return self._single_train_loader(self.train_dataset)

        if self.train_datasets is None:
            self.train_datasets = [self._get_train_dataset(ds_name) for ds_name in self.train_set_names]
            self.train_dataset = self._summarize_train_datasets()

        self._refresh_train_datasets()

        train_dataloaders = OrderedDict()
        for ds_name, dataset in zip(self.train_set_names, self.train_datasets):
            train_dataloaders[ds_name] = self._single_train_loader(dataset)
        if self.train_loader_mode not in {"min_size", "max_size_cycle", "max_size"}:
            raise ValueError(
                f"Unsupported train_loader_mode {self.train_loader_mode!r}. "
                "Choose from 'min_size', 'max_size_cycle', or 'max_size'."
            )
        if len(train_dataloaders) == 1:
            return next(iter(train_dataloaders.values()))
        return CombinedLoader(train_dataloaders, mode=self.train_loader_mode)

    def val_dataloader(self):
        val_dataloaders = []
        for dataset in self.val_datasets:
            dl = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=False,
                pin_memory=True,
                shuffle=False,
            )
            val_dataloaders.append(dl)
        return val_dataloaders

    def test_dataloader(self):
        return self.val_dataloader()

    def _single_train_loader(self, dataset):
        if self.batch_sampler is not None:
            return DataLoader(
                dataset=dataset,
                num_workers=self.num_workers,
                batch_sampler=self.batch_sampler,
                pin_memory=True,
            )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            shuffle=self.shuffle_all,
        )

    def _refresh_train_datasets(self):
        if self.train_datasets is None:
            return
        for dataset in self.train_datasets:
            reload_fn = getattr(dataset, "reload", None)
            if callable(reload_fn):
                reload_fn()

    def _resolve_train_dataset_weights(self, dataset_names):
        if self.train_dataset_weights is None:
            return {name: 1.0 for name in dataset_names}
        if isinstance(self.train_dataset_weights, dict):
            return {name: float(self.train_dataset_weights.get(name, 1.0)) for name in dataset_names}
        weights = list(self.train_dataset_weights)
        if len(weights) != len(dataset_names):
            raise ValueError(
                "train_dataset_weights must have the same length as train_set_names "
                f"({len(weights)} vs {len(dataset_names)})."
            )
        return {name: float(weight) for name, weight in zip(dataset_names, weights)}

    def _dataset_sampling_config(self, ds_name):
        ds_name_lower = ds_name.lower()
        sampling = {
            "sample_mode": "clique",
            "bucket_size_m": 25.0,
            "cluster_radius_m": 20.0,
            "min_images_per_place": self.img_per_place,
            "include_queries": True,
            "database_subdir": "database",
            "query_subdir": "queries",
        }
        if ds_name_lower in {"msls", "msls-train", "mapillary_sls"}:
            sampling.update(self.generic_sampling)
            sampling.update(self.msls_sampling)
        elif ds_name_lower in {"sf_xl", "sf-xl"}:
            sampling.update({
                "include_queries": False,
                "database_subdir": "",
                "query_subdir": "queries",
            })
            sampling.update(self.generic_sampling)
            sampling.update(self.sf_xl_sampling)
        else:
            sampling.update(self.generic_sampling)
        sampling["min_images_per_place"] = int(sampling.get("min_images_per_place", self.img_per_place))
        sampling["bucket_size_m"] = float(sampling.get("bucket_size_m", 25.0))
        sampling["cluster_radius_m"] = float(sampling.get("cluster_radius_m", 20.0))
        sampling["include_queries"] = bool(sampling.get("include_queries", True))
        sampling["database_subdir"] = str(sampling.get("database_subdir", "database"))
        sampling["query_subdir"] = str(sampling.get("query_subdir", "queries"))
        sampling["sample_mode"] = str(sampling.get("sample_mode", "clique"))
        return sampling

    def _get_train_dataset(self, ds_name):
        hard_mining = self.batch_sampler is not None

        if ds_name.lower() in {"gsv-cities", "gsv-cities-light"}:
            return GSVCitiesDataset(
                dataset_path=self.train_set_paths[ds_name],
                cities=self.cities,
                img_per_place=self.img_per_place,
                random_sample_from_each_place=self.random_sample_from_each_place,
                transform=self.train_transform,
                hard_mining=hard_mining,
            )

        ds_name_lower = ds_name.lower()
        sampling = self._dataset_sampling_config(ds_name)
        if ds_name_lower in {"msls", "msls-train", "mapillary_sls"}:
            return MultiDatasetPlaceDataset(
                dataset_name=ds_name,
                dataset_path=self.train_set_paths[ds_name],
                img_per_place=self.img_per_place,
                transform=self.train_transform,
                bucket_size_m=sampling["bucket_size_m"],
                cluster_radius_m=sampling["cluster_radius_m"],
                min_images_per_place=sampling["min_images_per_place"],
                sample_mode=sampling["sample_mode"],
                include_queries=sampling["include_queries"],
                database_subdir=sampling["database_subdir"],
                query_subdir=sampling["query_subdir"],
                hard_mining=hard_mining,
            )

        if ds_name_lower in {"sf_xl", "sf-xl"}:
            return MultiDatasetPlaceDataset(
                dataset_name=ds_name,
                dataset_path=self.train_set_paths[ds_name],
                img_per_place=self.img_per_place,
                transform=self.train_transform,
                bucket_size_m=sampling["bucket_size_m"],
                cluster_radius_m=sampling["cluster_radius_m"],
                min_images_per_place=sampling["min_images_per_place"],
                sample_mode=sampling["sample_mode"],
                include_queries=sampling["include_queries"],
                database_subdir=sampling["database_subdir"],
                query_subdir=sampling["query_subdir"],
                hard_mining=hard_mining,
            )

        return MultiDatasetPlaceDataset(
            dataset_name=ds_name,
            dataset_path=self.train_set_paths[ds_name],
            img_per_place=self.img_per_place,
            transform=self.train_transform,
            bucket_size_m=sampling["bucket_size_m"],
            cluster_radius_m=sampling["cluster_radius_m"],
            min_images_per_place=sampling["min_images_per_place"],
            sample_mode=sampling["sample_mode"],
            include_queries=sampling["include_queries"],
            database_subdir=sampling["database_subdir"],
            query_subdir=sampling["query_subdir"],
            hard_mining=hard_mining,
        )

    def _summarize_train_datasets(self):
        if self.train_datasets is None:
            return None
        return {
            "names": self.train_set_names,
            "num_datasets": len(self.train_datasets),
            "total_places": sum(len(ds) for ds in self.train_datasets),
            "total_images": sum(getattr(ds, "total_nb_images", 0) for ds in self.train_datasets),
            "loader_mode": self.train_loader_mode,
            "dataset_weights": self.train_dataset_weight_map,
            "datasets": self.train_datasets,
        }

    def _get_val_dataset(self, ds_name):
        ds_name_lower = ds_name.lower()
        coordinate_datasets = {
            "tokyo247",
            "st_lucia",
            "nordland",
            "amstertime",
            "baidu",
            "sped",
            "eynsham",
        }
        if ds_name_lower in coordinate_datasets:
            return CoordinateVPRDataset(
                dataset_name=ds_name,
                dataset_path=self.val_set_paths[ds_name],
                input_transform=self.val_transform,
                positive_radius_m=self.val_positive_radius_m,
            )
        if "msls" in ds_name_lower:
            return MapillarySLSDataset(
                dataset_path=self.val_set_paths[ds_name],
                input_transform=self.val_transform,
            )
        if "pitts30k" in ds_name_lower or "pitts250k" in ds_name_lower:
            return PittsburghDataset(
                dataset_path=self.val_set_paths[ds_name],
                input_transform=self.val_transform,
            )
        raise ValueError(f"Unknown dataset name: {ds_name}")
