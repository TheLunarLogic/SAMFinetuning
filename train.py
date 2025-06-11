import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

# Your classes are assumed to be defined in the notebook or imported directly:
# SAMPointPromptDataset, SAMForSemanticSegmentation, SemanticSegmentationLitModule, StructureLoss

# This code is referenced from: 
# https://github.com/autogluon/autogluon/blob/master/multimodal/src/autogluon/multimodal/models/sam.py
from sam_model import SAMForSemanticSegmentation

# https://github.com/autogluon/autogluon/blob/master/multimodal/src/autogluon/multimodal/optim/losses/structure_loss.py
from loss import StructureLoss

# https://github.com/autogluon/autogluon/blob/master/multimodal/src/autogluon/multimodal/optim/lit_module.py
from lit_module import SemanticSegmentationLitModule

from dataset import SAMPointPromptDataset



def get_trainable_params_efficient_finetune(norm_param_names, efficient_finetune, extra_params=None):
    """Get trainable parameter names for efficient fine-tuning, following AutoGluon's implementation"""
    if efficient_finetune == "lora":
        trainable_param_names = [
            "lora_",  # LoRA parameters
            "bias",   # Bias terms
        ]
        # Add normalization layers
        trainable_param_names.extend(norm_param_names)
        # Add any extra parameters specified
        if extra_params:
            trainable_param_names.extend(extra_params)
        return trainable_param_names
    else:
        return None

def get_norm_layer_param_names(model):
    """Get normalization layer parameter names, following AutoGluon's implementation"""
    norm_param_names = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
                              torch.nn.LayerNorm, torch.nn.GroupNorm)):
            norm_param_names.extend([f"{name}.weight", f"{name}.bias"])
    return norm_param_names

def run_training(
    train_csv,
    val_csv,
    checkpoint="facebook/sam-vit-base",
    save_dir="./checkpoints",
    batch_size=2,
    num_epochs=5,
    learning_rate=0.0001,
    weight_decay=0.0001,
    patience=10,
    image_size=1024,
    num_workers=4,
    num_gpus=1
):
    os.makedirs(save_dir, exist_ok=True)

    train_dataset = SAMPointPromptDataset(train_csv, image_size=image_size)
    val_dataset = SAMPointPromptDataset(val_csv, image_size=image_size) if val_csv else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    ) if val_dataset else None

    model = SAMForSemanticSegmentation(
        checkpoint_name=checkpoint,
        num_classes=1,
        pretrained=True,
        frozen_layers=["mask_decoder.iou_prediction_head", "prompt_encoder"]
    )

    norm_param_names = get_norm_layer_param_names(model)
    trainable_param_names = get_trainable_params_efficient_finetune(
        norm_param_names=norm_param_names,
        efficient_finetune="lora",
        extra_params=[".*mask_decoder"]
    )

    loss_func = StructureLoss()

    lit_module = SemanticSegmentationLitModule(
        model=model,
        loss_func=loss_func,
        trainable_param_names=trainable_param_names,
        optim_type="adamw",
        lr=learning_rate,
        weight_decay=weight_decay,
        validation_metric="iou",
        validation_metric_name="iou"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename='sam-lora-{epoch:02d}-{val_iou:.4f}',
        save_top_k=3,
        verbose=True,
        monitor='val_iou',
        mode='max'
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_iou',
        patience=patience,
        mode='max',
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=min(num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else None,
        strategy='ddp_find_unused_parameters_true' if num_gpus > 1 else 'auto',
        callbacks=[checkpoint_callback, early_stopping_callback],
        default_root_dir=save_dir,
        precision='16-mixed' if torch.cuda.is_available() else 32,
    )

    trainer.fit(lit_module, train_loader, val_loader)
    trainer.save_checkpoint(os.path.join(save_dir, 'sam_lora_final.ckpt'))

    print(f"✅ Training completed. Best model saved at: {checkpoint_callback.best_model_path}")
