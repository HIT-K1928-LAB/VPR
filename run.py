# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/OpenVPRLab
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import json
import torch
import yaml
import numpy as np
import importlib
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path
from src.core.vpr_datamodule import VPRDataModule
from src.core.vpr_framework import VPRFramework
from src.losses.vpr_losses import VPRLossFunction

from rich.traceback import install
install()

IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

ALL_CITIES = [
    'Bangkok', 'BuenosAires', 'LosAngeles', 'MexicoCity', 'OSL', 'Rome',
    'Barcelona', 'Chicago', 'Madrid', 'Miami', 'Phoenix', 'TRT', 'Boston',
    'Lisbon', 'Medellin', 'Minneapolis', 'PRG', 'WashingtonDC', 'Brussels',
    'London', 'Melbourne', 'Osaka', 'PRS',
]


def load_config(config_path='model_config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def get_instance(module_name, class_name, params):
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_(**params)


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _normalize_trainer_devices(devices):
    if devices is None:
        return 1
    if isinstance(devices, (int, np.integer)):
        return int(devices)
    if isinstance(devices, str):
        stripped = devices.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped
    if isinstance(devices, (list, tuple)):
        normalized = []
        for item in devices:
            if isinstance(item, (int, np.integer)):
                normalized.append(int(item))
            elif isinstance(item, str):
                stripped = item.strip()
                normalized.append(int(stripped) if stripped.isdigit() else stripped)
            else:
                normalized.append(item)
        if len(normalized) == 1:
            only = normalized[0]
            if isinstance(only, int) and only == 0:
                return [0]
            return only
        return normalized
    return devices


def _resolve_trainer_strategy(devices, strategy):
    if strategy is not None:
        if isinstance(strategy, str) and strategy.lower() == 'ddp':
            return DDPStrategy(find_unused_parameters=True)
        return strategy
    if isinstance(devices, int) and devices > 1:
        return DDPStrategy(find_unused_parameters=True)
    if isinstance(devices, (list, tuple)) and len(devices) > 1:
        return DDPStrategy(find_unused_parameters=True)
    return None


def _flatten_dataset_weights(train_set_names, weights_cfg):
    if weights_cfg is None:
        return None
    if isinstance(weights_cfg, dict):
        return {str(k): float(v) for k, v in weights_cfg.items()}
    weights = list(weights_cfg)
    if len(weights) != len(train_set_names):
        raise ValueError(
            f"train_dataset_weights must match train_set_names length ({len(weights)} vs {len(train_set_names)})."
        )
    return {str(name): float(weight) for name, weight in zip(train_set_names, weights)}


def train(config):
    seed_everything(config["seed"], workers=True)
    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True)
    torch.backends.cuda.enable_flash_sdp(True)

    datamodule_cfg = config['datamodule']
    train_set_names = datamodule_cfg.get('train_set_names') or [datamodule_cfg['train_set_name']]
    train_dataset_weights = _flatten_dataset_weights(train_set_names, datamodule_cfg.get('train_dataset_weights'))
    datamodule = VPRDataModule(
        train_set_name=datamodule_cfg['train_set_name'],
        train_set_names=train_set_names,
        cities=datamodule_cfg['cities'],
        train_image_size=datamodule_cfg['train_image_size'],
        batch_size=datamodule_cfg['batch_size'],
        img_per_place=datamodule_cfg['img_per_place'],
        random_sample_from_each_place=True,
        shuffle_all=False,
        num_workers=datamodule_cfg['num_workers'],
        batch_sampler=None,
        mean_std=IMAGENET_MEAN_STD,
        val_set_names=datamodule_cfg['val_set_names'],
        val_image_size=datamodule_cfg['val_image_size'],
        val_positive_radius_m=datamodule_cfg.get('val_positive_radius_m', 25.0),
        train_loader_mode=datamodule_cfg.get('train_loader_mode', 'max_size_cycle'),
        train_dataset_weights=train_dataset_weights,
        msls_sampling=datamodule_cfg.get('msls_sampling'),
        sf_xl_sampling=datamodule_cfg.get('sf_xl_sampling'),
        generic_sampling=datamodule_cfg.get('generic_sampling'),
    )

    backbone = get_instance(config['backbone']['module'], config['backbone']['class'], config['backbone']['params'])
    out_channels = backbone.out_channels
    if 'in_channels' in config['aggregator']['params']:
        if config['aggregator']['params']['in_channels'] is None:
            config['aggregator']['params']['in_channels'] = out_channels

    aggregator = get_instance(config['aggregator']['module'], config['aggregator']['class'], config['aggregator']['params'])
    loss_function = get_instance(config['loss_function']['module'], config['loss_function']['class'], config['loss_function']['params'])

    config['datamodule']['train_dataset_weights'] = train_dataset_weights
    vpr_model = VPRFramework(
        backbone=backbone,
        aggregator=aggregator,
        loss_function=loss_function,
        optimizer=config['trainer']['optimizer'],
        lr=config['trainer']['lr'],
        weight_decay=config['trainer']['wd'],
        warmup_steps=config['trainer']['warmup'],
        milestones=config['trainer']['milestones'],
        lr_mult=config['trainer']['lr_mult'],
        verbose= not config["silent"],
        config_dict=config,
    )

    if config["compile"]:
        vpr_model = torch.compile(vpr_model)

    tensorboard_logger = TensorBoardLogger(
        save_dir=f"./logs/{backbone.backbone_name}",
        name=f"{aggregator.__class__.__name__}",
        default_hp_metric=False
    )

    val_monitor_dataset = datamodule_cfg.get("val_set_names") or ["msls-val"]
    if isinstance(val_monitor_dataset, str):
        val_monitor_dataset = [val_monitor_dataset]
    val_monitor_dataset = val_monitor_dataset[0]
    val_monitor_r1 = f"{val_monitor_dataset}/R1"
    val_monitor_r5 = f"{val_monitor_dataset}/R5"
    checkpoint_filename = (
        "epoch({epoch:02d})_step({step:04d})_"
        "R1[{METRIC_R1:.4f}]_R5[{METRIC_R5:.4f}]"
    ).replace("METRIC_R1", val_monitor_r1).replace("METRIC_R5", val_monitor_r5)
    checkpoint_cb = ModelCheckpoint(
        monitor=val_monitor_r1,
        filename=checkpoint_filename,
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_top_k=3,
        mode="max",
        save_on_train_epoch_end=False,
    )

    from src.utils.callbacks import CustomRichProgressBar, CustomRRichModelSummary, DatamoduleSummary
    progress_bar_cb = CustomRichProgressBar(config["display_theme"])
    model_summary_cb = CustomRRichModelSummary(config["display_theme"])
    data_summary_cb = DatamoduleSummary(config["display_theme"])

    trainer_cfg = config['trainer']
    trainer_devices = _normalize_trainer_devices(trainer_cfg.get('devices', 1))
    trainer_strategy = _resolve_trainer_strategy(trainer_devices, trainer_cfg.get('strategy'))

    trainer_kwargs = dict(
        accelerator=trainer_cfg.get('accelerator', 'gpu'),
        devices=trainer_devices,
        logger=tensorboard_logger,
        num_sanity_val_steps=0,
        precision='16-mixed',
        max_epochs=trainer_cfg['max_epochs'],
        check_val_every_n_epoch=1,
        callbacks=[
            checkpoint_cb,
            data_summary_cb,
            model_summary_cb,
            progress_bar_cb,
        ],
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=10,
        fast_dev_run=config["dev"],
        enable_model_summary=False,
        num_nodes=int(trainer_cfg.get('num_nodes', 1) or 1),
    )
    if trainer_strategy is not None:
        trainer_kwargs['strategy'] = trainer_strategy
    trainer = Trainer(**trainer_kwargs)

    trainer.fit(model=vpr_model, datamodule=datamodule)


def evaluate(config):
    seed_everything(config["seed"], workers=True)
    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True)
    torch.backends.cuda.enable_flash_sdp(True)

    ckpt_path = config.get("ckpt_path")
    if not ckpt_path:
        raise ValueError("Please provide --ckpt_path when running in test mode.")
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    datamodule_cfg = config['datamodule']
    train_set_names = datamodule_cfg.get('train_set_names') or [datamodule_cfg['train_set_name']]
    train_dataset_weights = _flatten_dataset_weights(train_set_names, datamodule_cfg.get('train_dataset_weights'))
    datamodule = VPRDataModule(
        train_set_name=datamodule_cfg['train_set_name'],
        train_set_names=train_set_names,
        cities=datamodule_cfg['cities'],
        train_image_size=datamodule_cfg['train_image_size'],
        batch_size=datamodule_cfg['batch_size'],
        img_per_place=datamodule_cfg['img_per_place'],
        random_sample_from_each_place=True,
        shuffle_all=False,
        num_workers=datamodule_cfg['num_workers'],
        batch_sampler=None,
        mean_std=IMAGENET_MEAN_STD,
        val_set_names=datamodule_cfg['val_set_names'],
        val_image_size=datamodule_cfg['val_image_size'],
        val_positive_radius_m=datamodule_cfg.get('val_positive_radius_m', 25.0),
        train_loader_mode=datamodule_cfg.get('train_loader_mode', 'max_size_cycle'),
        train_dataset_weights=train_dataset_weights,
        msls_sampling=datamodule_cfg.get('msls_sampling'),
        sf_xl_sampling=datamodule_cfg.get('sf_xl_sampling'),
        generic_sampling=datamodule_cfg.get('generic_sampling'),
    )

    backbone = get_instance(config['backbone']['module'], config['backbone']['class'], config['backbone']['params'])
    out_channels = backbone.out_channels
    if 'in_channels' in config['aggregator']['params'] and config['aggregator']['params']['in_channels'] is None:
        config['aggregator']['params']['in_channels'] = out_channels

    aggregator = get_instance(config['aggregator']['module'], config['aggregator']['class'], config['aggregator']['params'])
    loss_function = get_instance(config['loss_function']['module'], config['loss_function']['class'], config['loss_function']['params'])

    config['datamodule']['train_dataset_weights'] = train_dataset_weights
    vpr_model = VPRFramework(
        backbone=backbone,
        aggregator=aggregator,
        loss_function=loss_function,
        optimizer=config['trainer']['optimizer'],
        lr=config['trainer']['lr'],
        weight_decay=config['trainer']['wd'],
        warmup_steps=config['trainer']['warmup'],
        milestones=config['trainer']['milestones'],
        lr_mult=config['trainer']['lr_mult'],
        verbose=not config["silent"],
        config_dict=config,
    )

    if config["compile"]:
        vpr_model = torch.compile(vpr_model)

    tensorboard_logger = TensorBoardLogger(
        save_dir=f"./logs/{backbone.backbone_name}",
        name=f"{aggregator.__class__.__name__}_test",
        default_hp_metric=False,
    )

    trainer_cfg = config['trainer']
    trainer_devices = _normalize_trainer_devices(trainer_cfg.get('devices', 1))
    trainer_strategy = _resolve_trainer_strategy(trainer_devices, trainer_cfg.get('strategy'))

    trainer_kwargs = dict(
        accelerator=trainer_cfg.get('accelerator', 'gpu'),
        devices=trainer_devices,
        logger=tensorboard_logger,
        num_sanity_val_steps=0,
        precision='16-mixed',
        callbacks=[],
        enable_model_summary=False,
        num_nodes=int(trainer_cfg.get('num_nodes', 1) or 1),
    )
    if trainer_strategy is not None:
        trainer_kwargs['strategy'] = trainer_strategy
    trainer = Trainer(**trainer_kwargs)

    results = trainer.test(model=vpr_model, datamodule=datamodule, ckpt_path=str(ckpt_path))

    metrics_dir = Path(tensorboard_logger.log_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "test_metrics.json"
    payload = {
        "checkpoint": str(ckpt_path),
        "log_dir": str(tensorboard_logger.log_dir),
        "results": results,
        "summary": getattr(vpr_model, "last_eval_summary", None),
        "config": config,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2, ensure_ascii=False)
    if not config["silent"]:
        print(results)
        print(f"Saved test metrics to {metrics_path}")
    return results


def main():
    from argparser import parse_args
    config = parse_args()
    if config["train"]:
        train(config)
    else:
        evaluate(config)


if __name__ == "__main__":
    main()
