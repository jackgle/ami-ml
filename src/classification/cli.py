"""
Command Line Interface for the training module

To add a new command, create a new function below following these instructions:

- Create a new Command key constant
   <NEW_COMMAND_NAME>_CMD = "<NEW_COMMAND_NAME>_cmd"

- Add the new Command key constant to the COMMAND_KEYS frozenset

- Add the command name and help text in the COMMANDS and COMMANDS_HELP dictionaries

- Create a new function named <COMMAND_NAME>_command(), alongside the appropriate
  options
    - Make sure to reuse appropriate options before creating duplicate ones.
    - The contents of the CLI command should be minimal : execution of an imported
      function

- If unsure, take some time to look at how other commands have been implemented

- Make sure to use lazy loading when importing modules that are only used by 1 command
"""

import typing as tp
from typing import Optional

import click

from src.classification.constants import (
    ADAMW,
    AVAILABLE_LOSS_FUNCIONS,
    AVAILABLE_LR_SCHEDULERS,
    AVAILABLE_MODELS,
    AVAILABLE_OPTIMIZERS,
    CROSS_ENTROPY_LOSS,
    RESNET50,
)

SupportedModels = tp.Literal[*AVAILABLE_MODELS]
SupportedLossFunctions = tp.Literal[*AVAILABLE_LOSS_FUNCIONS]
SupportedOptimizers = tp.Literal[*AVAILABLE_OPTIMIZERS]
SupportedLearningRateSchedulers = tp.Literal[*AVAILABLE_LR_SCHEDULERS]

# Command key constants
# make sure to add them to COMMAND_KEYS frozenset
TRAIN_CMD = "train_cmd"

# this is most useful to automatically test the CLI
COMMAND_KEYS = frozenset([TRAIN_CMD])

# command dictionary
COMMANDS = {
    TRAIN_CMD: "train-model",
}

# command help text dictionary
COMMANDS_HELP = {TRAIN_CMD: "Train a classification model"}


# # # # # # #
# Commands  #
# # # # # # #

# the order of declaration of the commands affect the order
# in which they appear in the CLI


#
# Train Model Command
#
@click.command(
    name=COMMANDS[TRAIN_CMD],
    help=COMMANDS_HELP[TRAIN_CMD],
    context_settings={"show_default": True},
)
@click.option(
    "--random_seed",
    type=int,
    default=42,
    help="Random seed for reproducibility",
)
@click.option(
    "--model_type",
    type=click.Choice(tp.get_args(SupportedModels)),
    default=RESNET50,
    help="Model architecture",
)
@click.option(
    "--num_classes",
    type=int,
    required=True,
    help="Number of model's output classes",
)
@click.option(
    "--existing_weights",
    type=str,
    default=None,
    help="Existing weights to be loaded, if available",
)
@click.option(
    "--static_features",
    is_flag=True,
    default=False,
    help="Whether to use static features in the model",
)
@click.option(
    "--static_feature_keys",
    type=str,
    multiple=True,
    default=None,
    help="Keys of static features to be used in the model",
)
@click.option(
    "--static_feat_num_categories",
    type=int,
    multiple=True,
    default=None,
    help="Number of categories for each static feature (e.g., [num_regions, num_countries])",
)
@click.option(
    "--predict_sex/--no-predict_sex",
    default=False,
    help="Enable binary sex prediction head in the model",
)
@click.option(
    "--total_epochs",
    type=int,
    default=30,
    help="Total number of training epochs",
)
@click.option(
    "--warmup_epochs",
    type=int,
    default=0,
    help="Number of warmup epochs, if using a learning rate scehduler",
)
@click.option(
    "--early_stopping",
    type=int,
    default=3,
    help="Stop training if validation loss does not decrease after this many epochs",
)
@click.option(
    "--train_webdataset",
    type=str,
    required=True,
    help="Webdataset files for the training set",
)
@click.option(
    "--val_webdataset",
    type=str,
    required=True,
    help="Webdataset files for the validation set",
)
@click.option(
    "--test_webdataset",
    type=str,
    required=True,
    help="Webdataset files for the test set",
)
@click.option(
    "--image_input_size",
    type=int,
    default=128,
    help="Image input size for training and inference",
)
@click.option(
    "--batch_size",
    type=int,
    default=64,
    help="Batch size for training",
)
@click.option(
    "--preprocess_mode",
    type=click.Choice(["torch", "tf", "other"]),
    default="torch",
    help="Preprocessing mode for normalization",
)
@click.option(
    "--optimizer_type",
    type=click.Choice(tp.get_args(SupportedOptimizers)),
    default=ADAMW,
    help="Optimizer type",
)
@click.option(
    "--learning_rate",
    type=float,
    default=0.001,
    help="Initial learning rate",
)
@click.option(
    "--learning_rate_scheduler",
    type=click.Choice(tp.get_args(SupportedLearningRateSchedulers)),
    default=None,
    help="Learning rate scheduler",
)
@click.option(
    "--weight_decay",
    type=float,
    default=1e-5,
    help="Weight decay for regularization",
)
@click.option(
    "--loss_function_type",
    type=click.Choice(tp.get_args(SupportedLossFunctions)),
    default=CROSS_ENTROPY_LOSS,
    help="Loss function",
)
@click.option(
    "--weight_on_order_loss",
    type=float,
    default=0.5,
    help=(
        "Weight on order-level classification for the custom loss function. "
        "The weight on binary is 1-order."
    ),
)
@click.option(
    "--label_smoothing",
    type=float,
    default=0.1,
    help="Label smoothing for model regularization. No smoothing if 0.0",
)
@click.option(
    "--mixed_resolution_data_aug",
    type=bool,
    default=True,
    help=(
        "A custom mixed resolution data augmentation technique. "
        "See The AMI Dataset ECCV 2024 paper for more details."
    ),
)
@click.option(
    "--model_save_directory",
    type=str,
    required=True,
    help="Directory to save the trained model",
)
@click.option(
    "--wandb_entity",
    type=str,
    default=None,
    help="Weights and Biases entity",
)
@click.option(
    "--wandb_project",
    type=str,
    default=None,
    help="Weights and Biases project",
)
@click.option(
    "--wandb_run_name",
    type=str,
    default=None,
    help="User-defined training run name",
)
# new: two-stage training knobs (restricted to e2e or head_all)
@click.option(
    "--train_strategy",
    type=click.Choice(["e2e", "head_all"]),
    default="e2e",
    help="Training flow: end-to-end, or head-only then unfreeze all",
)
@click.option(
    "--stage1_epochs",
    type=int,
    default=8,
    help="Epochs to train head-only before unfreezing (if head_all)",
)
@click.option(
    "--backbone_lr_scale",
    type=float,
    default=0.1,
    help="Multiplier for backbone LR after unfreezing: lr_backbone = learning_rate * scale",
)
@click.option(
    "--freeze_bn_stats/--no-freeze_bn_stats",
    default=True,
    help="Keep BatchNorm running stats frozen (eval mode) on small datasets",
)
@click.option(
    "--reset_opt_on_unfreeze/--no-reset_opt_on_unfreeze",
    default=True,
    help="Reinitialize optimizer when unfreezing to avoid stale moments",
)
@click.option(
    "--head_param_patterns",
    type=str,
    multiple=True,
    default=("head", "fc", "classifier"),
    help="Parameter-name substrings that identify the classification head",
)

def train_model_command(
    random_seed: int,
    model_type: str,
    num_classes: int,
    existing_weights: Optional[str],
    static_features: bool,
    static_feature_keys: Optional[list[str]],
    static_feat_num_categories: tp.Optional[list[int]],
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
    # new args
    train_strategy: str,
    stage1_epochs: int,
    backbone_lr_scale: float,
    freeze_bn_stats: bool,
    reset_opt_on_unfreeze: bool,
    head_param_patterns: tuple[str, ...],
    predict_sex: bool,
):
    from src.classification.train import train_model

    train_model(
        random_seed=random_seed,
        model_type=model_type,
        num_classes=num_classes,
        existing_weights=existing_weights,
        static_features=static_features,
        static_feature_keys=static_feature_keys,
        static_feat_num_categories=static_feat_num_categories,
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        early_stopping=early_stopping,
        train_webdataset=train_webdataset,
        val_webdataset=val_webdataset,
        test_webdataset=test_webdataset,
        image_input_size=image_input_size,
        batch_size=batch_size,
        preprocess_mode=preprocess_mode,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        learning_rate_scheduler=learning_rate_scheduler,
        weight_decay=weight_decay,
        loss_function_type=loss_function_type,
        weight_on_order_loss=weight_on_order_loss,
        label_smoothing=label_smoothing,
        mixed_resolution_data_aug=mixed_resolution_data_aug,
        model_save_directory=model_save_directory,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        # new args forwarded to trainer
        train_strategy=train_strategy,
        stage1_epochs=stage1_epochs,
        backbone_lr_scale=backbone_lr_scale,
        freeze_bn_stats=freeze_bn_stats,
        reset_opt_on_unfreeze=reset_opt_on_unfreeze,
        head_param_patterns=list(head_param_patterns),
        predict_sex=predict_sex,
    )


# # # # # # # # # # # # # #
# Main CLI configuration  #
# # # # # # # # # # # # # #
class OrderCommands(click.Group):
    """this class is necessary to order the commands the way we want to."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        return list(self.commands)


@click.group(cls=OrderCommands)
def cli():
    """this is the main command line interface for the classification tools."""


# following is an automated way to add all functions containing the word `command`
# in their name instead of manually having to add them.
all_objects = globals()
functions = [
    obj for name, obj in all_objects.items() if callable(obj) and "command" in name
]

for command_function in functions:
    cli.add_command(command_function)

if __name__ == "__main__":
    cli()
