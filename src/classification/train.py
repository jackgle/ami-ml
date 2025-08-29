#!/usr/bin/env python
# coding: utf-8

""" main script for training classification models (supports e2e or head→all) """

import pathlib
import time
import typing as tp
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from timm.utils import AverageMeter

import wandb
from src.classification.dataloader import build_webdataset_pipeline
from src.classification.utils import (
    build_model,
    get_learning_rate_scheduler,
    get_loss_function,
    get_optimizer,  # still used for plain e2e path
    get_webdataset_length,
    set_random_seeds,
)


# -----------------------------
# helpers for two-stage training
# -----------------------------

def _set_bn_eval(module: torch.nn.Module) -> None:
    # keep bn running stats frozen; leave affine params trainable
    if isinstance(
        module,
        (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.SyncBatchNorm,
        ),
    ):
        module.eval()


def _maybe_freeze_bn_stats(model: torch.nn.Module, freeze_bn_stats: bool) -> None:
    # apply bn eval if requested
    if freeze_bn_stats:
        model.apply(_set_bn_eval)


def _split_head_backbone(
    model: torch.nn.Module, head_param_patterns: list[str]
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    # split parameters by name patterns
    head, backbone = [], []
    for n, p in model.named_parameters():
        if any(h in n for h in head_param_patterns):
            head.append(p)
        else:
            backbone.append(p)
    return head, backbone


def _freeze_to_head_only(
    model: torch.nn.Module, head_param_patterns: list[str]
) -> None:
    # freeze everything except head
    for n, p in model.named_parameters():
        p.requires_grad = any(h in n for h in head_param_patterns)


def _unfreeze_all(model: torch.nn.Module) -> None:
    # make everything trainable
    for p in model.parameters():
        p.requires_grad = True


def _build_optimizer_param_groups(
    optimizer_type: str,
    params: tp.Union[tp.Iterable[torch.nn.Parameter], list[dict]],
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    # minimal local factory to support param groups; falls back to utils.get_optimizer for simple case
    opt_type = optimizer_type.lower()
    if isinstance(params, list) and len(params) > 0 and isinstance(params[0], dict):
        if opt_type == "adamw":
            return torch.optim.AdamW(params, weight_decay=weight_decay)
        elif opt_type == "sgd":
            return torch.optim.SGD(params, momentum=0.9, nesterov=True, weight_decay=weight_decay)
        else:
            # default to adamw when param groups are used
            return torch.optim.AdamW(params, weight_decay=weight_decay)
    else:
        # simple path if caller passes a flat iterable
        if opt_type == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif opt_type == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
        else:
            # unknown type -> default adamw
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


# -----------------------------
# core training / eval loops
# -----------------------------

def _save_model_checkpoint(
    model: torch.nn.Module,
    model_save_path: pathlib.Path,
    optimizer: torch.optim.Optimizer,
    learning_rate_scheduler: tp.Any,
    epoch: int,
    train_loss: float,
    val_loss: float,
) -> None:
    """save model to disk"""

    if torch.cuda.device_count() > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    model_checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler": learning_rate_scheduler.state_dict()
        if learning_rate_scheduler is not None
        else None,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    torch.save(model_checkpoint, f"{model_save_path}_checkpoint.pt")


def _train_model_for_one_epoch(
    model: torch.nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    learning_rate_scheduler: Optional[tp.Any],
    total_train_steps: int,
    grad_clip: Optional[float] = None,
) -> tuple[dict, int]:  # first element is a metrics dict
    """training model for one epoch"""

    total_train_steps_current = total_train_steps
    running_loss = AverageMeter()
    running_accuracy = AverageMeter()

    model.train()
    # Check if model supports sex prediction
    sex_prediction = hasattr(model, 'predict_sex') and getattr(model, 'predict_sex', False)
    running_sex_loss = AverageMeter() if sex_prediction else None
    running_sex_accuracy = AverageMeter() if sex_prediction else None

    for batch_data in train_dataloader:
        if sex_prediction:
            images, species_labels, sex_labels, *rest = batch_data
            sex_labels = sex_labels.to(device, non_blocking=True).float()
        else:
            images, species_labels, *rest = batch_data
        images = images.to(device, non_blocking=True)
        species_labels = species_labels.to(device, non_blocking=True)
        static_feats = rest[0].to(device, non_blocking=True) if rest else None

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images, static_feats) if static_feats is not None else model(images)
        if sex_prediction:
            species_logits, sex_logits = outputs
            species_loss = loss_function(species_logits, species_labels)
            sex_loss_fn = torch.nn.BCEWithLogitsLoss()
            sex_loss = sex_loss_fn(sex_logits, sex_labels)
            loss = species_loss + sex_loss
        else:
            species_logits = outputs
            loss = loss_function(species_logits, species_labels)

        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss.update(loss.item())
        _, predicted = torch.max(species_logits, 1)
        running_accuracy.update((predicted == species_labels).sum().item() / species_labels.size(0))
        if sex_prediction:
            running_sex_loss.update(sex_loss.item())
            sex_pred = (torch.sigmoid(sex_logits) > 0.5).float()
            running_sex_accuracy.update((sex_pred == sex_labels).sum().item() / sex_labels.size(0))

        if learning_rate_scheduler is not None:
            total_train_steps_current += 1
            step_update = getattr(learning_rate_scheduler, "step_update", None)
            if callable(step_update):
                learning_rate_scheduler.step_update(num_updates=total_train_steps_current)

    metrics = {"train_loss": running_loss.avg, "train_accuracy": running_accuracy.avg}
    if sex_prediction:
        metrics["sex_loss"] = running_sex_loss.avg
        metrics["sex_accuracy"] = running_sex_accuracy.avg
    return metrics, total_train_steps_current


def _evaluate_model(
    model: torch.nn.Module,
    device: str,
    loss_function: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    set_type: str,
) -> dict:
    """evaluate model either for validation or test set"""

    sex_prediction = hasattr(model, 'predict_sex') and getattr(model, 'predict_sex', False)
    running_sex_loss = AverageMeter() if sex_prediction else None
    running_sex_accuracy = AverageMeter() if sex_prediction else None
    running_loss = AverageMeter()
    running_accuracy = AverageMeter()

    model.eval()
    for batch_data in dataloader:
        if sex_prediction:
            images, species_labels, sex_labels, *rest = batch_data
            sex_labels = sex_labels.to(device, non_blocking=True).float()
        else:
            images, species_labels, *rest = batch_data
        images = images.to(device, non_blocking=True)
        species_labels = species_labels.to(device, non_blocking=True)
        static_feats = rest[0].to(device, non_blocking=True) if rest else None

        with torch.no_grad():
            outputs = model(images, static_feats) if static_feats is not None else model(images)
            if sex_prediction:
                species_logits, sex_logits = outputs
                species_loss = loss_function(species_logits, species_labels)
                sex_loss_fn = torch.nn.BCEWithLogitsLoss()
                sex_loss = sex_loss_fn(sex_logits, sex_labels)
                loss = species_loss + sex_loss
            else:
                species_logits = outputs
                loss = loss_function(species_logits, species_labels)

        running_loss.update(loss.item())
        _, predicted = torch.max(species_logits, 1)
        running_accuracy.update((predicted == species_labels).sum().item() / species_labels.size(0))
        if sex_prediction:
            running_sex_loss.update(sex_loss.item())
            sex_pred = (torch.sigmoid(sex_logits) > 0.5).float()
            running_sex_accuracy.update((sex_pred == sex_labels).sum().item() / sex_labels.size(0))

    metrics = {f"{set_type}_loss": running_loss.avg, f"{set_type}_accuracy": running_accuracy.avg}
    if sex_prediction:
        metrics[f"{set_type}_sex_loss"] = running_sex_loss.avg
        metrics[f"{set_type}_sex_accuracy"] = running_sex_accuracy.avg
    return metrics


# -----------------------------
# main entry
# -----------------------------

def train_model(
    random_seed: int,
    model_type: str,
    num_classes: int,
    existing_weights: Optional[str],
    static_features: bool,
    static_feature_keys: Optional[list[str]],
    static_feat_num_categories: tp.Optional[list[int]],
    predict_sex: bool,
    total_epochs: int,
    warmup_epochs: int,
    early_stopping: int,
    train_webdataset: str,
    val_webdataset: str,
    test_webdataset: str,
    image_input_size: int,
    batch_size: int,
    preprocess_mode: str,
    optimizer_type: str,
    learning_rate: float,
    learning_rate_scheduler: Optional[str],
    weight_decay: float,
    loss_function_type: str,
    weight_on_order_loss: float,
    label_smoothing: float,
    mixed_resolution_data_aug: bool,
    model_save_directory: str,
    wandb_entity: Optional[str],
    wandb_project: Optional[str],
    wandb_run_name: Optional[str],
    save_model_artifact: bool,
    # new knobs for two-stage
    train_strategy: str,  # "e2e" or "head_all"
    stage1_epochs: int,
    backbone_lr_scale: float,
    freeze_bn_stats: bool,
    reset_opt_on_unfreeze: bool,
    head_param_patterns: list[str],
) -> None:
    """main training function"""

    # set random seeds
    set_random_seeds(random_seed)

    if static_features and "embeddings" not in head_param_patterns:
        head_param_patterns.append("embeddings")  # train static feature embeddings in stage 1

    # model initialization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"the available device is {device}.")
    # Add predict_sex argument if you want to enable sex prediction
    model = build_model(
        device,
        model_type,
        num_classes,
        existing_weights,
        static_features=static_features,
        static_feat_num_categories=static_feat_num_categories,
        predict_sex=predict_sex,  # Set to True to enable sex prediction
    )

    # setup dataloaders
    train_dataloader = build_webdataset_pipeline(
        train_webdataset,
        image_input_size,
        batch_size,
        preprocess_mode,
        mixed_resolution_data_aug=mixed_resolution_data_aug,
        is_training=True,
        use_static_features=static_features,
        static_feature_keys=static_feature_keys,
        use_sex_label=predict_sex
    )
    val_dataloader = build_webdataset_pipeline(
        val_webdataset,
        image_input_size,
        batch_size,
        preprocess_mode,
        use_static_features=static_features,
        static_feature_keys=static_feature_keys,
        use_sex_label=predict_sex
    )
    test_dataloader = build_webdataset_pipeline(
        test_webdataset,
        image_input_size,
        batch_size,
        preprocess_mode,
        use_static_features=static_features,
        static_feature_keys=static_feature_keys,
        use_sex_label=predict_sex
    )

    # compute steps per epoch once for schedulers
    steps_per_epoch = None
    if learning_rate_scheduler:
        train_data_length = get_webdataset_length(train_webdataset)
        steps_per_epoch = int((train_data_length - 1) / batch_size) + 1

    # loss
    loss_function = get_loss_function(
        loss_function_type,
        label_smoothing=label_smoothing,
        weight_on_order=weight_on_order_loss,
    )

    # save path
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = Path(model_save_directory) / f"{model_type}_{current_date}"

    # start W&B logging
    if wandb_entity or wandb_project:
        training_configuration = {
            "random_seed": random_seed,
            "model_type": model_type,
            "num_classes": num_classes,
            "existing_weights": existing_weights,
            "total_epochs": total_epochs,
            "warmup_epochs": warmup_epochs,
            "early_stopping": early_stopping,
            "train_webdataset": train_webdataset,
            "val_webdataset": val_webdataset,
            "test_webdataset": test_webdataset,
            "image_input_size": image_input_size,
            "batch_size": batch_size,
            "preprocess_mode": preprocess_mode,
            "optimizer_type": optimizer_type,
            "learning_rate": learning_rate,
            "learning_rate_scheduler": learning_rate_scheduler,
            "weight_decay": weight_decay,
            "loss_function_type": loss_function_type,
            "label_smoothing": label_smoothing,
            "model_save_directory": model_save_directory,
            # new config
            "train_strategy": train_strategy,
            "stage1_epochs": stage1_epochs,
            "backbone_lr_scale": backbone_lr_scale,
            "freeze_bn_stats": freeze_bn_stats,
            "reset_opt_on_unfreeze": reset_opt_on_unfreeze,
            "head_param_patterns": head_param_patterns,
        }
        wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            name=wandb_run_name,
            config=training_configuration,
        )

    # bookkeeping
    total_train_steps = 0
    early_stopping_count = 0
    lowest_val_loss = float("inf")

    # build optimizer + scheduler for a given stage
    def _make_optimizer_and_scheduler(
        params_or_model: tp.Union[torch.nn.Module, tp.Iterable[torch.nn.Parameter], list[dict]],
        stage_epochs: int,
        head_backbone_groups: bool = False,
    ) -> tuple[torch.optim.Optimizer, Optional[tp.Any]]:
        # create optimizer
        if head_backbone_groups:
            optimizer = _build_optimizer_param_groups(
                optimizer_type,
                params_or_model,  # expects param groups list[dict]
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            if isinstance(params_or_model, torch.nn.Module):
                optimizer = get_optimizer(optimizer_type, params_or_model, learning_rate, weight_decay)
            else:
                optimizer = _build_optimizer_param_groups(optimizer_type, params_or_model, learning_rate, weight_decay)

        # create scheduler for this stage if requested
        if learning_rate_scheduler and steps_per_epoch is not None:
            # use stage-specific total epochs for per-stage cosine/warmup
            lr_sched = get_learning_rate_scheduler(
                optimizer,
                learning_rate_scheduler,
                stage_epochs,
                steps_per_epoch,
                min(warmup_epochs, max(stage_epochs - 1, 0)),
            )
        else:
            lr_sched = None
        return optimizer, lr_sched

    # apply bn freezing choice
    _maybe_freeze_bn_stats(model, freeze_bn_stats)

    # choose strategy
    if train_strategy == "e2e":
        # plain end-to-end for total_epochs
        optimizer, lr_sched = _make_optimizer_and_scheduler(model, total_epochs)

        for epoch in range(1, total_epochs + 1):
            epoch_start_time = time.time()
            train_metrics, total_train_steps = _train_model_for_one_epoch(
                model,
                device,
                optimizer,
                loss_function,
                train_dataloader,
                lr_sched,
                total_train_steps,
                grad_clip=1.0,
            )
            early_stopping_count += 1
            val_metrics = _evaluate_model(model, device, loss_function, val_dataloader, "val")

            if val_metrics["val_loss"] < lowest_val_loss:
                _save_model_checkpoint(
                    model,
                    model_save_path,
                    optimizer,
                    lr_sched,
                    epoch,
                    train_metrics["train_loss"],
                    val_metrics["val_loss"],
                )
                lowest_val_loss = val_metrics["val_loss"]
                early_stopping_count = 0

            sex_prediction = hasattr(model, 'predict_sex') and getattr(model, 'predict_sex', False)
            print_str = (
                f"Epoch [{epoch:02d}/{total_epochs}]: "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Train Acc: {train_metrics['train_accuracy']*100:.2f}%, "
                f"Val Acc: {val_metrics['val_accuracy']*100:.2f}%, "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
            if sex_prediction:
                print_str += (
                    f", Train Sex Acc: {train_metrics['sex_accuracy']*100:.2f}%, "
                    f"Val Sex Acc: {val_metrics['val_sex_accuracy']*100:.2f}%"
                )
            print(print_str, flush=True)

            if wandb_entity or wandb_project:
                log_dict = {
                    "epoch": epoch,
                    "time_per_epoch_mins": (time.time() - epoch_start_time) / 60,
                    "train_loss": train_metrics["train_loss"],
                    "val_loss": val_metrics["val_loss"],
                    "train_accuracy": train_metrics["train_accuracy"],
                    "val_accuracy": val_metrics["val_accuracy"],
                }
                if sex_prediction:
                    log_dict["train_sex_accuracy"] = train_metrics["sex_accuracy"]
                    log_dict["val_sex_accuracy"] = val_metrics["val_sex_accuracy"]
                wandb.log(log_dict)

            if early_stopping_count >= early_stopping:
                print(
                    f"early stopping at epoch {epoch} with lowest validation loss: {lowest_val_loss:.4f}.",
                    flush=True,
                )
                break

    elif train_strategy == "head_all":
        # stage 1: head-only
        _freeze_to_head_only(model, head_param_patterns)
        _maybe_freeze_bn_stats(model, freeze_bn_stats)
        head_params, _ = _split_head_backbone(model, head_param_patterns)
        optimizer, lr_sched = _make_optimizer_and_scheduler(head_params, stage1_epochs)

        for epoch in range(1, stage1_epochs + 1):
            epoch_start_time = time.time()
            train_metrics, total_train_steps = _train_model_for_one_epoch(
                model,
                device,
                optimizer,
                loss_function,
                train_dataloader,
                lr_sched,
                total_train_steps,
                grad_clip=1.0,
            )
            early_stopping_count += 1
            val_metrics = _evaluate_model(model, device, loss_function, val_dataloader, "val")

            if val_metrics["val_loss"] < lowest_val_loss:
                _save_model_checkpoint(
                    model,
                    model_save_path,
                    optimizer,
                    lr_sched,
                    epoch,
                    train_metrics["train_loss"],
                    val_metrics["val_loss"],
                )
                lowest_val_loss = val_metrics["val_loss"]
                early_stopping_count = 0


                sex_prediction = hasattr(model, 'predict_sex') and getattr(model, 'predict_sex', False)
                print_str = (
                    f"[Stage1 head-only] Epoch [{epoch:02d}/{stage1_epochs}]: "
                    f"Train Loss: {train_metrics['train_loss']:.4f}, "
                    f"Val Loss: {val_metrics['val_loss']:.4f}, "
                    f"Train Acc: {train_metrics['train_accuracy']*100:.2f}%, "
                    f"Val Acc: {val_metrics['val_accuracy']*100:.2f}%, "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )
                if sex_prediction:
                    print_str += (
                        f", Train Sex Acc: {train_metrics['sex_accuracy']*100:.2f}%, "
                        f"Val Sex Acc: {val_metrics['val_sex_accuracy']*100:.2f}%"
                    )
                print(print_str, flush=True)

                if wandb_entity or wandb_project:
                    log_dict = {
                        "stage": 1,
                        "epoch": epoch,
                        "time_per_epoch_mins": (time.time() - epoch_start_time) / 60,
                        "train_loss": train_metrics["train_loss"],
                        "val_loss": val_metrics["val_loss"],
                        "train_accuracy": train_metrics["train_accuracy"],
                        "val_accuracy": val_metrics["val_accuracy"],
                    }
                    if sex_prediction:
                        log_dict["train_sex_accuracy"] = train_metrics["sex_accuracy"]
                        log_dict["val_sex_accuracy"] = val_metrics["val_sex_accuracy"]
                    wandb.log(log_dict)

            if early_stopping_count >= early_stopping:
                print(
                    f"early stopping during stage 1 at epoch {epoch} with lowest validation loss: {lowest_val_loss:.4f}.",
                    flush=True,
                )
                break

        # stage 2: unfreeze all + small backbone lr
        _unfreeze_all(model)
        _maybe_freeze_bn_stats(model, freeze_bn_stats)
        head_params, backbone_params = _split_head_backbone(model, head_param_patterns)
        param_groups = [
            {"params": head_params, "lr": learning_rate},
            {"params": backbone_params, "lr": learning_rate * backbone_lr_scale},
        ]

        if reset_opt_on_unfreeze:
            optimizer, lr_sched = _make_optimizer_and_scheduler(
                param_groups,
                stage_epochs=max(total_epochs - stage1_epochs, 1),
                head_backbone_groups=True,
            )
        else:
            # reuse existing optimizer: clear and set param groups (kept simple by rebuilding)
            optimizer, lr_sched = _make_optimizer_and_scheduler(
                param_groups,
                stage_epochs=max(total_epochs - stage1_epochs, 1),
                head_backbone_groups=True,
            )

        stage2_epochs = max(total_epochs - stage1_epochs, 1)
        for s2_epoch in range(1, stage2_epochs + 1):
            global_epoch = stage1_epochs + s2_epoch
            epoch_start_time = time.time()
            train_metrics, total_train_steps = _train_model_for_one_epoch(
                model,
                device,
                optimizer,
                loss_function,
                train_dataloader,
                lr_sched,
                total_train_steps,
                grad_clip=1.0,
            )
            early_stopping_count += 1
            val_metrics = _evaluate_model(model, device, loss_function, val_dataloader, "val")

            if val_metrics["val_loss"] < lowest_val_loss:
                _save_model_checkpoint(
                    model,
                    model_save_path,
                    optimizer,
                    lr_sched,
                    global_epoch,
                    train_metrics["train_loss"],
                    val_metrics["val_loss"],
                )
                lowest_val_loss = val_metrics["val_loss"]
                early_stopping_count = 0


                sex_prediction = hasattr(model, 'predict_sex') and getattr(model, 'predict_sex', False)
                print_str = (
                    f"[Stage2 unfreeze-all] Epoch [{global_epoch:02d}/{total_epochs}]: "
                    f"Train Loss: {train_metrics['train_loss']:.4f}, "
                    f"Val Loss: {val_metrics['val_loss']:.4f}, "
                    f"Train Acc: {train_metrics['train_accuracy']*100:.2f}%, "
                    f"Val Acc: {val_metrics['val_accuracy']*100:.2f}%, "
                    f"Head LR: {optimizer.param_groups[0]['lr']:.6f}, "
                    f"Backbone LR: {optimizer.param_groups[1]['lr']:.6f}"
                )
                if sex_prediction:
                    print_str += (
                        f", Train Sex Acc: {train_metrics['sex_accuracy']*100:.2f}%, "
                        f"Val Sex Acc: {val_metrics['val_sex_accuracy']*100:.2f}%"
                    )
                print(print_str, flush=True)

                if wandb_entity or wandb_project:
                    log_dict = {
                        "stage": 2,
                        "epoch": global_epoch,
                        "time_per_epoch_mins": (time.time() - epoch_start_time) / 60,
                        "train_loss": train_metrics["train_loss"],
                        "val_loss": val_metrics["val_loss"],
                        "train_accuracy": train_metrics["train_accuracy"],
                        "val_accuracy": val_metrics["val_accuracy"],
                        "head_lr": optimizer.param_groups[0]["lr"],
                        "backbone_lr": optimizer.param_groups[1]["lr"],
                    }
                    if sex_prediction:
                        log_dict["train_sex_accuracy"] = train_metrics["sex_accuracy"]
                        log_dict["val_sex_accuracy"] = val_metrics["val_sex_accuracy"]
                    wandb.log(log_dict)

            if early_stopping_count >= early_stopping:
                print(
                    f"early stopping during stage 2 at epoch {global_epoch} with lowest validation loss: {lowest_val_loss:.4f}.",
                    flush=True,
                )
                break

    else:
        raise ValueError(f"unknown train_strategy: {train_strategy}")

    # final evaluation
    test_metrics = _evaluate_model(model, device, loss_function, test_dataloader, "test")
    print_str = f"the test accuracy is {test_metrics['test_accuracy']*100:.2f}%."
    if 'test_sex_accuracy' in test_metrics:
        print_str += f" Test sex accuracy: {test_metrics['test_sex_accuracy']*100:.2f}%"
    print(print_str, flush=True)

    if wandb_entity or wandb_project:
        log_dict = {"test_accuracy": test_metrics["test_accuracy"]}
        if 'test_sex_accuracy' in test_metrics:
            log_dict["test_sex_accuracy"] = test_metrics["test_sex_accuracy"]
        wandb.log(log_dict)
        if save_model_artifact:
            wandb.log_artifact(f"{model_save_path}_checkpoint.pt", type="model", name=wandb_run_name)
        wandb.finish()
