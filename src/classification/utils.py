#!/usr/bin/env python
# coding: utf-8

""" Utility functions
"""

import random
import tarfile
import typing as tp

import braceexpand
import numpy as np
import timm
import torch
import torch.nn as nn
from timm.scheduler import CosineLRScheduler

from src.classification.constants import (
    AVAILABLE_MODELS,
    COSINE_LR_SCHEDULER,
    CROSS_ENTROPY_LOSS,
    VIT_B16_128,
    WEIGHTED_ORDER_AND_BINARY_LOSS,
)
from src.classification.custom_loss_functions import (
    WeightedOrderAndBinaryCrossEntropyLoss,
)


def set_random_seeds(random_seed: int) -> None:
    """Set random seeds for reproducibility"""

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True


def get_optimizer(
    optimizer_type: str,
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """Optimizer definitions"""

    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        raise RuntimeError(f"{optimizer_type} optimizer is not implemented.")


def get_learning_rate_scheduler(
    optimizer: torch.optim.Optimizer,
    lr_scheduler_type: str,
    total_epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int,
) -> tp.Any:
    """Learning rate scheduler definitions"""

    total_steps = int(total_epochs * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    if lr_scheduler_type == COSINE_LR_SCHEDULER:
        return CosineLRScheduler(
            optimizer,
            t_initial=(total_steps - warmup_steps),
            warmup_t=warmup_steps,
            warmup_prefix=True,
            cycle_limit=1,
            t_in_epochs=False,
        )
    else:
        raise RuntimeError(
            f"{lr_scheduler_type} learning rate scheduler is not implemented."
        )


def get_loss_function(
    loss_function_name: str, label_smoothing: float = 0.0, weight_on_order: float = 0.5
) -> torch.nn.Module:
    """Loss function definitions"""

    if loss_function_name == CROSS_ENTROPY_LOSS:
        return torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif loss_function_name == WEIGHTED_ORDER_AND_BINARY_LOSS:
        return WeightedOrderAndBinaryCrossEntropyLoss(weight_on_order=weight_on_order)
    else:
        raise RuntimeError(f"{loss_function_name} loss is not implemented.")


def _count_files_from_tar(tar_filename: str, ext="jpg") -> int:
    """Count the number of images in a single tar archive"""

    tar = tarfile.open(tar_filename)
    files = [f for f in tar.getmembers() if f.name.endswith(ext)]
    count_files = len(files)
    tar.close()
    return count_files


def get_webdataset_length(sharedurl: str) -> int:
    """Get the total number of images in all webdataset files for a given dataset"""

    tar_filenames = list(braceexpand.braceexpand(sharedurl))
    counts = [_count_files_from_tar(tar_f) for tar_f in tar_filenames]
    return int(sum(counts))


class ImageModelWithStaticFeatures(nn.Module):
    def __init__(
        self,
        model_type: str,
        num_classes: int,
        static_feat_dim: int,
        static_embed_dim: int = 32,
        pretrained: bool = True,
        img_size: int = None,
    ):
        super().__init__()
        model_args = {"pretrained": pretrained, "num_classes": 0}
        if img_size is not None:
            model_args["img_size"] = img_size
        self.backbone = timm.create_model(model_type, **model_args)
        backbone_out_dim = self.backbone.num_features

        self.static_embed = nn.Sequential(
            nn.Linear(static_feat_dim, static_embed_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(backbone_out_dim + static_embed_dim, num_classes)

    def forward(self, x_img, x_static):
        img_feat = self.backbone(x_img)
        static_feat = self.static_embed(x_static)
        x = torch.cat([img_feat, static_feat], dim=1)
        return self.classifier(x)


def build_model(
    device: str,
    model_type: str,
    num_classes: int,
    existing_weights: tp.Optional[str],
    pretrained: bool = True,
    checkpoint: bool = False,
    static_features: bool = False,
    static_feat_dim: int = None,
    static_embed_dim: int = 4,
) -> torch.nn.Module:
    """Model builder"""

    if model_type not in AVAILABLE_MODELS:
        raise RuntimeError(f"Model {model_type} not implemented")

    if static_features:
        # For ViT special case
        img_size = 128 if model_type == VIT_B16_128 else None
        if model_type == VIT_B16_128:
            model_type = "vit_base_patch16_224_in21k"
        model = ImageWithStaticFeaturesModel(
            model_type=model_type,
            num_classes=num_classes,
            static_feat_dim=static_feat_dim,
            static_embed_dim=static_embed_dim,
            pretrained=pretrained,
            img_size=img_size,
        )
    else:
        model_arguments = {"pretrained": pretrained, "num_classes": num_classes}
        if model_type == VIT_B16_128:
            model_type = "vit_base_patch16_224_in21k"
            model_arguments["img_size"] = 128
        model = timm.create_model(model_type, **model_arguments)

    # If available, load existing weights
    if existing_weights:
        print("Loading existing model weights.")
        state_dict = torch.load(existing_weights, map_location=torch.device(device))
        if checkpoint:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict, strict=False)

    # Make use of multiple GPUs, if available
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    return model
