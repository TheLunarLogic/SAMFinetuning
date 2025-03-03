import os
import argparse
import math
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from dataset import SAMPointPromptDataset
from sam_model import SAMForSemanticSegmentation
from lit_module import SemanticSegmentationLitModule
from loss import StructureLoss

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

def main():
    parser = argparse.ArgumentParser(description='Train SAM with LoRA using point prompts')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, default=None, help='Path to validation CSV file')
    parser.add_argument('--checkpoint', type=str, default='facebook/sam-vit-base', help='SAM checkpoint name or path')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size per GPU')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--image_size', type=int, default=1024, help='Size to resize images to')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create datasets
    train_dataset = SAMPointPromptDataset(args.train_csv, image_size=args.image_size)
    val_dataset = SAMPointPromptDataset(args.val_csv, image_size=args.image_size) if args.val_csv else None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    ) if val_dataset else None
    
    # Create model
    model = SAMForSemanticSegmentation(
        checkpoint_name=args.checkpoint,
        num_classes=1,
        pretrained=True,
        # Use the frozen layers from your config
        frozen_layers=["mask_decoder.iou_prediction_head", "prompt_encoder"]
    )
    
    # Get trainable parameter names for LoRA
    norm_param_names = get_norm_layer_param_names(model)
    trainable_param_names = get_trainable_params_efficient_finetune(
        norm_param_names=norm_param_names,
        efficient_finetune="lora",
        extra_params=[".*mask_decoder"]  # From your config
    )
    
    # Create loss function
    loss_func = StructureLoss()
    
    # Create Lightning module
    lit_module = SemanticSegmentationLitModule(
        model=model,
        loss_func=loss_func,
        trainable_param_names=trainable_param_names,
        optim_type="adamw",
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_metric="iou",
        validation_metric_name="iou"
    )
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename='sam-lora-{epoch:02d}-{val_iou:.4f}',
        save_top_k=3,  # From your config
        verbose=True,
        monitor='val_iou',
        mode='max'
    )
    
    early_stopping_callback = EarlyStopping(
        monitor='val_iou',
        patience=args.patience,  # From your config
        mode='max',
        verbose=True
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=min(args.num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else None,
        strategy='ddp_find_unused_parameters_true' if args.num_gpus > 1 else 'auto',  # Match your config
        callbacks=[checkpoint_callback, early_stopping_callback],
        default_root_dir=args.save_dir,
        precision='16-mixed' if torch.cuda.is_available() else 32,  # Match your config
    )
    
    # Train model
    trainer.fit(lit_module, train_loader, val_loader)
    
    # Save final model
    trainer.save_checkpoint(os.path.join(args.save_dir, 'sam_lora_final.ckpt'))
    
    print(f"Training completed. Best model saved at: {checkpoint_callback.best_model_path}")

if __name__ == '__main__':
    main()