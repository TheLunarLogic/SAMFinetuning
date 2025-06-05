import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SamModel, SamConfig
from peft import LoraConfig, get_peft_model

class SAMForSemanticSegmentation(nn.Module):
    """
    SAM model for semantic segmentation with LoRA, following AutoGluon's implementation
    """
    def __init__(
        self,
        prefix="sam",
        checkpoint_name="facebook/sam-vit-base",  # Using base model as in your config
        num_classes=1,
        pretrained=True,
        frozen_layers=None,
    ):
        super().__init__()
        self.prefix = prefix
        self.pretrained = pretrained
        self.checkpoint_name = checkpoint_name
        self.num_classes = num_classes
        
        # Default frozen layers from your config
        if frozen_layers is None:
            self.frozen_layers = ["mask_decoder.iou_prediction_head", "prompt_encoder"]
        else:
            self.frozen_layers = frozen_layers
        
        self.device = None
        self.name_to_id = {}
        
        self._load_checkpoint(checkpoint_name)
        
        # Apply LoRA to the model
        self.apply_lora()
        
        # Freeze model layers based on frozen_layers list
        self.freeze_model_layers()
        
        self.image_size = self.model.vision_encoder.image_size
        self.config = self.model.config
        
        # Modify mask decoder for single mask output
        self.model.mask_decoder.num_mask_tokens = 1
        mask_token_data = self.model.mask_decoder.mask_tokens.weight.data[0]
        self.model.mask_decoder.mask_tokens = nn.Embedding(1, self.model.mask_decoder.hidden_size)
        self.model.mask_decoder.mask_tokens.weight.data[0] = mask_token_data
        
        # For Conv-LoRA
        self.output_moe_loss = False
    
    def _load_checkpoint(self, checkpoint_name):
        if self.pretrained:
            self.model = SamModel.from_pretrained(checkpoint_name)
        else:
            configuration = SamConfig(name_or_path=checkpoint_name)
            self.model = SamModel(configuration)
    
    def apply_lora(self):
        """Apply LoRA to the model following AutoGluon's configuration"""
        # Configure LoRA as per your settings
        lora_config = LoraConfig(
            r=128,  # LoRA rank from your config
            lora_alpha=32,  # LoRA alpha from your config
            target_modules=["q_proj", "v_proj"],  # Target q and v projections as per your config
            lora_dropout=0.0,
            bias="none",
            modules_to_save=[],
        )
        
        # First freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Apply LoRA only to vision encoder attention layers
        for name, module in self.model.vision_encoder.named_modules():
            if "attn" in name:
                # Find q_proj and v_proj in attention modules
                if hasattr(module, "q_proj"):
                    module.q_proj = get_peft_layer(module.q_proj, r=128, lora_alpha=32)
                    module.q_proj.weight.requires_grad = True
                if hasattr(module, "v_proj"):
                    module.v_proj = get_peft_layer(module.v_proj, r=128, lora_alpha=32)
                    module.v_proj.weight.requires_grad = True
    
    def freeze_model_layers(self):
        """Freeze layers based on frozen_layers list"""
        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in self.frozen_layers):
                param.requires_grad = False
            # Make mask_decoder trainable as per your config's extra_trainable_params
            elif "mask_decoder" in name and "iou_prediction_head" not in name:
                param.requires_grad = True
    
    @property
    def image_key(self):
        return f"{self.prefix}_image"
    
    @property
    def points_key(self):
        return f"{self.prefix}_points"
    
    @property
    def point_labels_key(self):
        return f"{self.prefix}_point_labels"
    
    @property
    def label_key(self):
        return f"{self.prefix}_label"
    
    def forward(self, batch):
        """
        Forward pass following AutoGluon's implementation
        """
        # Check if points are provided in the batch
        has_points = self.points_key in batch and batch[self.points_key] is not None
        
        # Prepare forward arguments
        forward_args = {
            "pixel_values": batch[self.image_key],
            "multimask_output": False,
            "output_moe_loss": self.output_moe_loss
        }
        
        # Add points if available
        if has_points:
            forward_args["input_points"] = batch[self.points_key]
            forward_args["input_labels"] = batch[self.point_labels_key]
        
        # Call the model with appropriate arguments
        rets = self.model(**forward_args)
        
        # Process results
        if self.num_classes == 1:
            pred_masks = rets.pred_masks[:, 0, :, :, :]
            pred_masks = F.interpolate(
                pred_masks, (self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
            
            # In sam_model.py, modify the forward method:
            if self.training:
                rets_dict = {self.prefix: {"logits": pred_masks}}
            else:
                rets_dict = {self.prefix: {"logits": pred_masks}}
                if self.label_key in batch:
                    rets_dict[self.prefix]["label"] = batch[self.label_key]
        else:
            # Multi-class case would be handled here
            pass
        
        if self.output_moe_loss:
            rets_dict[self.prefix].update({"moe_loss": rets.vision_moe_loss})
        
        return rets_dict
    
    def get_trainable_parameters(self):
        """Return only the parameters that require gradients"""
        return [p for p in self.parameters() if p.requires_grad]

# Helper function to create LoRA layers
def get_peft_layer(layer, r=4, lora_alpha=1):
    """Create a LoRA layer from a regular layer"""
    if isinstance(layer, nn.Linear):
        in_features, out_features = layer.in_features, layer.out_features
        bias = layer.bias is not None
        
        # Create LoRA layer
        lora_layer = LoRALinear(
            in_features=in_features,
            out_features=out_features,
            r=r,
            lora_alpha=lora_alpha,
            bias=bias
        )
        
        # Copy weights from original layer
        lora_layer.weight.data = layer.weight.data.clone()
        if bias:
            lora_layer.bias.data = layer.bias.data.clone()
        
        return lora_layer
    return layer

# LoRA implementation
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, lora_alpha=1, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        
        # Original weight
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # LoRA weights
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Mark as trainable
        self.weight.requires_grad = False  # Freeze original weights
        self.lora_A.requires_grad = True
        self.lora_B.requires_grad = True
        if bias:
            self.bias.requires_grad = True
    
    def forward(self, x):
        # Original forward
        original_output = F.linear(x, self.weight, self.bias)
        
        # LoRA forward
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * (self.lora_alpha / self.r)
        
        return original_output + lora_output