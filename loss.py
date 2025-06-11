import torch
import torch.nn as nn
import torch.nn.functional as F

class StructureLoss(nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init__()
    
    def forward(self, pred, mask):
        """
        Args:
            pred: Tensor of shape [B, 1, H, W]
            mask: Tensor of shape [B, H, W]
        """
        # Ensure mask has shape [B, 1, H, W] for operations
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        
        # Binary cross entropy
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        
        # Calculate IoU loss
        pred = torch.sigmoid(pred)
        inter = (pred * mask).sum(dim=(2, 3))
        union = (pred + mask).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        
        # Combined loss
        return (wbce.mean(dim=(2, 3)) + wiou).mean()