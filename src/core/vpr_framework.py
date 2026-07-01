# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/OpenVPRLab
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import numpy as np
import torch
import torch.distributed as dist
import lightning as L
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms import v2 as T2
import src.utils as utils
import yaml


class VPRFramework(L.LightningModule):
    def __init__(
        self,
        backbone,
        aggregator,
        loss_function,
        lr=1e-4,
        optimizer="adamw",
        weight_decay=1e-3,
        warmup_steps=1500,
        milestones=[5, 10, 15],
        lr_mult=0.25,
        verbose=True,
        config_dict=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator
        self.loss_function = loss_function
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult
        self.verbose = verbose
        self.last_eval_summary = None
        self.last_eval_metrics = None
        self.config_dict = config_dict or {}

        self.save_hyperparameters(self.config_dict)

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x

    def configure_optimizers(self):
        optimizer_params = [
            {"params": self.backbone.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
            {"params": self.aggregator.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
        ]

        if self.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(optimizer_params)
        else:
            raise ValueError(f"Optimizer {self.optimizer} not supported")

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones, gamma=self.lr_mult
        )
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * pg["initial_lr"]

        optimizer.step(closure=optimizer_closure)
        self.log('_LR', optimizer.param_groups[-1]['lr'], prog_bar=False, logger=True)

    @torch.compiler.disable()
    def compute_loss(self, descriptors, labels):
        loss, batch_accuracy = self.loss_function(descriptors, labels)
        return loss, batch_accuracy

    def on_train_start(self):
        pass

    def on_train_epoch_start(self):
        pass

    def _flatten_place_batch(self, images, labels):
        P, K, c, h, w = images.shape
        images = images.view(P * K, c, h, w)
        labels = labels.view(-1)
        return images, labels

    def _forward_loss(self, images, labels):
        model_output = self(images)
        if isinstance(model_output, tuple) or isinstance(model_output, list):
            descriptors = model_output[0]
        else:
            descriptors = model_output
        return self.compute_loss(descriptors, labels)

    def _dataset_weights(self):
        dm_cfg = self.config_dict.get("datamodule", {})
        weights = dm_cfg.get("train_dataset_weights", None)
        names = dm_cfg.get("train_set_names", None)
        if weights is None or not names:
            return {}
        if isinstance(weights, dict):
            return {str(k): float(v) for k, v in weights.items()}
        weights = list(weights)
        if len(weights) != len(names):
            return {}
        return {str(name): float(weight) for name, weight in zip(names, weights)}

    def training_step(self, batch, batch_idx):
        if isinstance(batch, tuple) and len(batch) == 3 and isinstance(batch[0], dict):
            batch, _, _ = batch

        if isinstance(batch, dict):
            total_loss = None
            total_weight = 0.0
            batch_logs = {}
            weight_map = self._dataset_weights()
            for dataset_name, dataset_batch in batch.items():
                if dataset_batch is None:
                    continue
                images, labels = dataset_batch
                images, labels = self._flatten_place_batch(images, labels)
                loss, batch_accuracy = self._forward_loss(images, labels)
                weight = float(weight_map.get(dataset_name, 1.0))
                weighted_loss = loss * weight
                total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
                total_weight += weight
                batch_logs[f"loss/{dataset_name}"] = loss
                batch_logs[f"weighted_loss/{dataset_name}"] = weighted_loss
                batch_logs[f"batch_acc/{dataset_name}"] = batch_accuracy
            if total_loss is None:
                total_loss = torch.tensor(0.0, device=self.device)
            elif total_weight > 0:
                total_loss = total_loss / total_weight
            self.log("loss", total_loss, prog_bar=True, logger=True)
            for key, value in batch_logs.items():
                self.log(key, value, prog_bar=False, logger=True)
            return total_loss

        images, labels = batch
        images, labels = self._flatten_place_batch(images, labels)
        loss, batch_accuracy = self._forward_loss(images, labels)

        self.log("loss", loss, prog_bar=True, logger=True)
        self.log("batch_acc", batch_accuracy, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        pass

    def _extract_descriptors(self, model_output):
        if isinstance(model_output, tuple) or isinstance(model_output, list):
            return model_output[0]
        return model_output

    def _append_eval_output(self, storage, dataloader_idx, indices, descriptors):
        if dataloader_idx not in storage:
            storage[dataloader_idx] = []
        storage[dataloader_idx].append(
            {
                "indices": indices.detach().cpu().numpy(),
                "descriptors": descriptors.detach().cpu().numpy(),
            }
        )

    def on_validation_epoch_start(self):
        self.validation_step_outputs = {}

    def on_test_epoch_start(self):
        self.test_step_outputs = {}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images, labels = batch
        descriptors = self._extract_descriptors(self(images))
        self._append_eval_output(self.validation_step_outputs, dataloader_idx, labels, descriptors)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        images, labels = batch
        descriptors = self._extract_descriptors(self(images))
        self._append_eval_output(self.test_step_outputs, dataloader_idx, labels, descriptors)

    def _gather_eval_outputs(self, outputs):
        if not dist.is_available() or not dist.is_initialized():
            return outputs

        world_size = dist.get_world_size()
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, outputs)

        merged = {}
        for rank_outputs in gathered:
            if not rank_outputs:
                continue
            for dataloader_idx, batches in rank_outputs.items():
                merged.setdefault(dataloader_idx, []).extend(batches)
        return merged

    def _finalize_eval(self, outputs, k_values, title):
        dm = self.trainer.datamodule
        outputs = self._gather_eval_outputs(outputs)
        list_of_recalls = []
        evaluated_names = []
        summary = {
            "title": title,
            "k_values": [int(k) for k in k_values],
            "datasets": [],
            "evaluated": [],
            "skipped": [],
        }

        for dataloader_idx, dataset_name in enumerate(dm.val_set_names):
            descriptors_list = outputs.get(dataloader_idx, [])
            dataset = dm.val_datasets[dataloader_idx]
            num_references = int(getattr(dataset, "num_references", 0))
            num_queries = int(getattr(dataset, "num_queries", 0))

            if self.trainer.fast_dev_run:
                if dataloader_idx == 0:
                    print("\nFast dev run: skipping recall@k computation\n")
                summary["datasets"].append(
                    {
                        "name": dataset_name,
                        "status": "skipped",
                        "reason": "fast_dev_run",
                        "num_references": num_references,
                        "num_queries": num_queries,
                    }
                )
                summary["skipped"].append(dataset_name)
                continue

            if len(descriptors_list) == 0:
                summary["datasets"].append(
                    {
                        "name": dataset_name,
                        "status": "skipped",
                        "reason": "empty_outputs",
                        "num_references": num_references,
                        "num_queries": num_queries,
                    }
                )
                summary["skipped"].append(dataset_name)
                continue

            if num_queries == 0:
                summary["datasets"].append(
                    {
                        "name": dataset_name,
                        "status": "skipped",
                        "reason": "no_queries",
                        "num_references": num_references,
                        "num_queries": num_queries,
                    }
                )
                summary["skipped"].append(dataset_name)
                continue

            indices = np.concatenate([item["indices"] for item in descriptors_list], axis=0)
            descriptors = np.concatenate([item["descriptors"] for item in descriptors_list], axis=0)
            order = np.argsort(indices, kind="stable")
            indices = indices[order]
            descriptors = descriptors[order]
            _, unique_positions = np.unique(indices, return_index=True)
            if len(unique_positions) != len(indices):
                unique_positions = np.sort(unique_positions)
                indices = indices[unique_positions]
                descriptors = descriptors[unique_positions]

            expected = num_references + num_queries
            if descriptors.shape[0] != expected:
                summary["datasets"].append(
                    {
                        "name": dataset_name,
                        "status": "skipped",
                        "reason": "descriptor_count_mismatch",
                        "num_references": num_references,
                        "num_queries": num_queries,
                        "num_descriptors": int(descriptors.shape[0]),
                        "expected_descriptors": int(expected),
                    }
                )
                summary["skipped"].append(dataset_name)
                continue

            recalls_dict = utils.compute_recall_performance(
                descriptors,
                num_references,
                num_queries,
                dataset.ground_truth,
                k_values=k_values,
            )
            recalls_dict = {k: float(recalls_dict[k]) for k in k_values}
            recalls_log = {f"{dataset_name}/R{k}": float(recalls_dict[k]) for k in k_values}
            self.log_dict(recalls_log, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            if hasattr(self.trainer, "callback_metrics"):
                for metric_name, metric_value in recalls_log.items():
                    self.trainer.callback_metrics[metric_name] = torch.tensor(metric_value, device=self.device)
            list_of_recalls.append(recalls_dict)
            evaluated_names.append(dataset_name)
            summary["datasets"].append(
                {
                    "name": dataset_name,
                    "status": "evaluated",
                    "num_references": num_references,
                    "num_queries": num_queries,
                    "recall": {f"R@{k}": float(recalls_dict[k]) for k in k_values},
                }
            )
            summary["evaluated"].append(dataset_name)

        if self.verbose:
            utils.display_recall_performance(list_of_recalls, evaluated_names, title=title)
            if summary["skipped"]:
                print("Skipped datasets:", ", ".join(summary["skipped"]))
        self.last_eval_summary = summary
        self.last_eval_metrics = list_of_recalls
        outputs.clear()

    def on_validation_epoch_end(self):
        self._finalize_eval(
            self.validation_step_outputs,
            k_values=[1, 5, 10, 15],
            title="Validation Recall@k Performance",
        )

    def on_test_epoch_end(self):
        self._finalize_eval(
            self.test_step_outputs,
            k_values=[1, 5, 10, 15, 20],
            title="Test Recall@k Performance",
        )
