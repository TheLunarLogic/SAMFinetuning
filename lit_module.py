import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class SemanticSegmentationLitModule(pl.LightningModule):
    """
    Lightning module for semantic segmentation, following AutoGluon's implementation
    """
    def __init__(
        self,
        model,
        loss_func=None,
        model_postprocess_fn=None,
        trainable_param_names=None,
        optim_type="adamw",
        lr=1e-4,
        weight_decay=1e-4,
        lr_decay=0,  # From your config
        lr_mult=1.0,
        validation_metric="iou",
        validation_metric_name="iou",
        custom_metric_func=None,
    ):
        super().__init__()
        self.model = model
        self.model_postprocess_fn = model_postprocess_fn
        self.trainable_param_names = trainable_param_names
        
        self.optim_type = optim_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.lr_mult = lr_mult
        
        self.validation_metric = validation_metric
        self.validation_metric_name = validation_metric_name
        self.custom_metric_func = custom_metric_func
        
        # Loss function - use provided or default to BCEWithLogitsLoss
        self.loss_func = loss_func if loss_func is not None else nn.BCEWithLogitsLoss()
        
        # Set which parameters to train based on trainable_param_names
        if self.trainable_param_names:
            self.set_trainable_params()
    
    def set_trainable_params(self):
        """Set which parameters require gradients based on trainable_param_names"""
        for name, param in self.model.named_parameters():
            if not any(trainable_name in name for trainable_name in self.trainable_param_names):
                param.requires_grad = False
    
    def forward(self, batch):
        return self.model(batch)
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        
        # Extract predictions and ground truth
        prefix = list(outputs.keys())[0]
        logits = outputs[prefix]["logits"]  # Shape: [B, 1, H, W]
        masks = batch[f"{prefix}_label"]    # Shape: [B, H, W]
        
        # Compute loss
        loss = self.loss_func(logits, masks)
        
        # Compute IoU
        pred_masks = (torch.sigmoid(logits) > 0.5).float()
        pred_masks = pred_masks.squeeze(1)  # Shape: [B, H, W]
        
        intersection = (pred_masks * masks).sum((1, 2))
        union = pred_masks.sum((1, 2)) + masks.sum((1, 2)) - intersection
        iou = (intersection / (union + 1e-6)).mean()
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log(f"val_{self.validation_metric_name}", iou, prog_bar=True)
        
        return {"val_loss": loss, f"val_{self.validation_metric_name}": iou}

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        
        # Extract predictions and ground truth
        prefix = list(outputs.keys())[0]
        logits = outputs[prefix]["logits"]  # Shape: [B, 1, H, W]
        masks = batch[f"{prefix}_label"]    # Shape: [B, H, W]
        
        # Compute loss
        loss = self.loss_func(logits, masks)
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        
        # Add MOE loss if present
        if "moe_loss" in outputs[prefix]:
            moe_loss = outputs[prefix]["moe_loss"]
            loss = loss + moe_loss
            self.log("train_moe_loss", moe_loss, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers following AutoGluon's pattern"""
        # Get trainable parameters
        if self.trainable_param_names:
            trainable_parameters = [
                p for n, p in self.model.named_parameters() 
                if any(trainable_name in n for trainable_name in self.trainable_param_names) and p.requires_grad
            ]
        else:
            trainable_parameters = self.model.parameters()
        
        # Create optimizer
        if self.optim_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                trainable_parameters,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.SGD(
                trainable_parameters,
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        
        # Create scheduler if lr_decay is set
        if self.lr_decay > 0:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=1,
                gamma=self.lr_decay
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        else:
            return optimizer