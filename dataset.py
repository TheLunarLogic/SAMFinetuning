import json
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SAMPointPromptDataset(Dataset):
    def __init__(self, csv_file, image_size=1024):
        """
        Dataset for SAM with point prompts, following AutoGluon's pattern
        
        Args:
            csv_file: Path to CSV file with columns: image, label, points
            image_size: Size to resize images to (SAM typically uses 1024x1024)
        """
        import pandas as pd
        self.data = pd.read_csv(csv_file)
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.data.iloc[idx]['image']
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        # Resize image for SAM
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # Load mask
        mask_path = self.data.iloc[idx]['label']
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        mask = torch.from_numpy(np.array(mask))
        if mask.max() > 1:
            mask = (mask > 0).float()
        else:
            mask = mask.float()
            
        # Load points
        points_data = self.data.iloc[idx]['points']
        
        # Check if points_data is a file path or direct JSON string
        if os.path.exists(points_data):
            with open(points_data, 'r') as f:
                points_info = json.load(f)
        else:
            # Try to parse as JSON string
            try:
                points_info = json.loads(points_data)
            except:
                # Fallback: assume it's a string representation of a list of points
                points_info = eval(points_data)
        
        # Extract points and labels
        if isinstance(points_info, dict):
            points = points_info.get('points', [])
            point_labels = points_info.get('labels', [1] * len(points))  # Default to foreground
        else:
            # Assume it's a list of points
            points = points_info
            point_labels = [1] * len(points)  # Default all to foreground
        
        # Scale points to match resized image
        scale_x = self.image_size / original_size[0]
        scale_y = self.image_size / original_size[1]
        
        scaled_points = []
        for point in points:
            scaled_points.append([point[0] * scale_x, point[1] * scale_y])
        
        # Format points for SAM
        # SAM expects: [batch_size, point_batch_size, num_points, 2]
        # For single image: [1, 1, num_points, 2]
        if len(scaled_points) > 0:
            formatted_points = torch.tensor([scaled_points], dtype=torch.float)  # [1, num_points, 2]
            formatted_labels = torch.tensor([point_labels], dtype=torch.long)    # [1, num_points]
        else:
            # Handle case with no points
            formatted_points = torch.zeros((1, 0, 2), dtype=torch.float)
            formatted_labels = torch.zeros((1, 0), dtype=torch.long)
        
        # Apply transforms to image
        image = self.transform(image)
        
        # Following AutoGluon's batch format
        return {
            'sam_image': image,
            'sam_points': formatted_points,
            'sam_point_labels': formatted_labels,
            'sam_label': mask
        }