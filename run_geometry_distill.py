"""
Train OpenVPRLab with uncertainty-aware geometric ranking distillation.

This script intentionally lives next to run.py instead of modifying it. It uses
the same config format and model construction, but can initialize the backbone
and aggregator from an existing checkpoint while using a new distillation loss.
"""

from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from rich.traceback import install

from argparser import parse_args
from run import ALL_CITIES, IMAGENET_MEAN_STD, get_instance
from src.core.vpr_datamodule import VPRDataModule
from src.core.vpr_framework import VPRFramework

install()


def _load_initial_weights(model: VPRFramework, ckpt_path: str | None) -> None:
    if not ckpt_path:
        return

    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Initial checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Initialized from: {path}")
    if missing:
        print(f"Missing keys while loading init checkpoint: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys while loading init checkpoint: {len(unexpected)}")


def train(config):
    seed_everything(config["seed"], workers=True)
    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True)
    torch.backends.cuda.enable_flash_sdp(True)

    datamodule = VPRDataModule(
        train_set_name=config["datamodule"]["train_set_name"],
        cities=config["datamodule"]["cities"],
        train_image_size=config["datamodule"]["train_image_size"],
        batch_size=config["datamodule"]["batch_size"],
        img_per_place=config["datamodule"]["img_per_place"],
        random_sample_from_each_place=True,
        shuffle_all=False,
        num_workers=config["datamodule"]["num_workers"],
        batch_sampler=None,
        mean_std=IMAGENET_MEAN_STD,
        val_set_names=config["datamodule"]["val_set_names"],
        val_image_size=config["datamodule"]["val_image_size"],
    )

    backbone = get_instance(
        config["backbone"]["module"],
        config["backbone"]["class"],
        config["backbone"]["params"],
    )
    out_channels = backbone.out_channels
    if "in_channels" in config["aggregator"]["params"]:
        if config["aggregator"]["params"]["in_channels"] is None:
            config["aggregator"]["params"]["in_channels"] = out_channels

    aggregator = get_instance(
        config["aggregator"]["module"],
        config["aggregator"]["class"],
        config["aggregator"]["params"],
    )
    loss_function = get_instance(
        config["loss_function"]["module"],
        config["loss_function"]["class"],
        config["loss_function"]["params"],
    )

    vpr_model = VPRFramework(
        backbone=backbone,
        aggregator=aggregator,
        loss_function=loss_function,
        optimizer=config["trainer"]["optimizer"],
        lr=config["trainer"]["lr"],
        weight_decay=config["trainer"]["wd"],
        warmup_steps=config["trainer"]["warmup"],
        milestones=config["trainer"]["milestones"],
        lr_mult=config["trainer"]["lr_mult"],
        verbose=not config["silent"],
        config_dict=config,
    )

    init_ckpt_path = config.get("init_ckpt_path") or config.get("ckpt_path")
    _load_initial_weights(vpr_model, init_ckpt_path)

    if config["compile"]:
        vpr_model = torch.compile(vpr_model)

    logger_name = f"{aggregator.__class__.__name__}_GeoDistill"
    tensorboard_logger = TensorBoardLogger(
        save_dir=f"./logs/{backbone.backbone_name}",
        name=logger_name,
        default_hp_metric=False,
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="msls-val/R1",
        filename="epoch({epoch:02d})_step({step:04d})_R1[{msls-val/R1:.4f}]_R5[{msls-val/R5:.4f}]",
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_top_k=3,
        mode="max",
    )

    from src.utils.callbacks import CustomRichProgressBar, CustomRRichModelSummary, DatamoduleSummary

    callbacks = [
        checkpoint_cb,
        DatamoduleSummary(config["display_theme"]),
        CustomRRichModelSummary(config["display_theme"]),
        CustomRichProgressBar(config["display_theme"]),
    ]

    trainer = Trainer(
        accelerator="gpu",
        devices=[0],
        logger=tensorboard_logger,
        num_sanity_val_steps=0,
        precision="16-mixed",
        max_epochs=config["trainer"]["max_epochs"],
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=10,
        fast_dev_run=config["dev"],
        enable_model_summary=False,
    )

    trainer.fit(model=vpr_model, datamodule=datamodule)


def main():
    config = parse_args()
    if not config.get("train", True):
        raise ValueError("run_geometry_distill.py is for training. Use run.py for testing.")
    train(config)


if __name__ == "__main__":
    main()
