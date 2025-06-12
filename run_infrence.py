# our fine-tuned model :

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import pandas as pd
from collections import Counter, defaultdict
import ast
import random
from tqdm import tqdm

# This code is referenced from: 
# https://github.com/autogluon/autogluon/blob/master/multimodal/src/autogluon/multimodal/models/sam.py
from sam_model import SAMForSemanticSegmentation

# https://github.com/autogluon/autogluon/blob/master/multimodal/src/autogluon/multimodal/optim/losses/structure_loss.py
from loss import StructureLoss

# https://github.com/autogluon/autogluon/blob/master/multimodal/src/autogluon/multimodal/optim/lit_module.py
from lit_module import SemanticSegmentationLitModule



# Function to load the trained model
def load_model_from_checkpoint(checkpoint_path):
    """
    Load a trained SAM model with LoRA weights from a checkpoint
    """
    # Create the model
    model = SAMForSemanticSegmentation(
        checkpoint_name="facebook/sam-vit-base",  # Same as used in training
        num_classes=1,
        pretrained=True
    )

    # Create the LitModule
    lit_module = SemanticSegmentationLitModule(
        model=model,
        loss_func=StructureLoss()
    )

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    lit_module.load_state_dict(checkpoint['state_dict'])

    # Set to evaluation mode
    lit_module.eval()

    return lit_module

# Function to predict mask with point prompts
def predict_mask_with_points(model, image_path, points, point_labels=None, device='cuda'):
    """
    Predict a mask for an image with point prompts

    Args:
        model: The SAM model
        image_path: Path to the input image
        points: List of [x, y] coordinates on the original image scale
        point_labels: List of point labels (1 for foreground, 0 for background)
                     If None, all points are assumed to be foreground (1)
        device: Device to run inference on

    Returns:
        pred_mask: Predicted mask as numpy array
        image: Original image
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size

    # If point_labels not provided, assume all points are foreground
    if point_labels is None:
        point_labels = [1] * len(points)

    # Resize to model's expected size
    image_size = model.model.image_size
    image_resized = image.resize((image_size, image_size), Image.BILINEAR)

    # Convert to tensor and normalize
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image_resized).unsqueeze(0).to(device)

    # Scale points to match resized image
    scale_x = image_size / original_size[0]
    scale_y = image_size / original_size[1]

    scaled_points = []
    for point in points:
        scaled_points.append([point[0] * scale_x, point[1] * scale_y])

    # Format points for SAM - correct shape: [batch_size, point_batch_size, nb_points_per_image, 2]
    points_tensor = torch.tensor([[scaled_points]], dtype=torch.float).to(device)  # Shape: [1, 1, num_points, 2]
    point_labels_tensor = torch.tensor([[point_labels]], dtype=torch.long).to(device)  # Shape: [1, 1, num_points]

    # Create batch with points
    batch = {
        'sam_image': image_tensor,
        'sam_points': points_tensor,
        'sam_point_labels': point_labels_tensor
    }

    # Run inference
    with torch.no_grad():
        outputs = model(batch)

    # Extract mask prediction
    prefix = list(outputs.keys())[0]
    logits = outputs[prefix]["logits"]

    # Convert to binary mask
    pred_mask = (torch.sigmoid(logits) > 0.5).float()

    # Resize back to original image size
    pred_mask = F.interpolate(
        pred_mask,
        size=original_size[::-1],  # (width, height) -> (height, width)
        mode='bilinear',
        align_corners=False
    )

    # Convert to numpy array
    pred_mask = pred_mask.squeeze().cpu().numpy()

    return pred_mask, image

# Function to calculate IoU between predicted and ground truth masks
def calculate_iou(pred_mask, gt_mask):
    """
    Calculate IoU between predicted and ground truth masks
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    if union == 0:
        return 0.0

    return intersection / union

# Function to visualize results with points and ground truth
def visualize_results(image, pred_mask, gt_mask, points, iou, class_name):
    """
    Visualize the image, predicted mask, and ground truth mask with points
    """
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot image with points
    axes[0].imshow(image)
    for point in points:
        axes[0].plot(point[0], point[1], marker='o', markersize=8, color='r')
    axes[0].set_title(f'Input Image with Points\n{class_name}')
    axes[0].axis('off')

    # Plot predicted mask
    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title(f'Fine-tuned SAM Predicted Mask (IoU: {iou:.4f})')
    axes[1].axis('off')

    # Plot ground truth mask
    axes[2].imshow(gt_mask, cmap='gray')
    axes[2].set_title('Ground Truth Mask')
    axes[2].axis('off')

    plt.tight_layout()
    return fig

def select_samples(test_df, num_samples=30):
    """
    Select samples from test_df with class distribution proportional to the dataset
    Returns a deterministic selection (same samples each time)
    """
    # Extract class names from image paths
    def extract_class_name(path):
        # Check if the path starts with 'dataset_yolo'
        if path.startswith('dataset_yolo'):
            parts = path.split('/')
            if len(parts) > 1:
                # parts[1] should be something like "53_Grains"
                class_folder = parts[1]
                # Return the digits before the underscore, e.g. "53"
                return class_folder.split('_')[0]
        else:
            # Fallback for other paths: look for parts starting with "Class_"
            parts = path.split('/')
            for part in parts:
                if part.startswith('Class_'):
                    return part
        return 'Unknown'


    test_df['class'] = test_df['image'].apply(extract_class_name)

    # Count instances per class
    class_counts = Counter(test_df['class'])
    total_instances = len(test_df)

    # Calculate how many samples to take from each class (proportional to their frequency)
    class_samples = {}

    for class_name, count in class_counts.items():
        # Calculate proportion and round to nearest integer
        proportion = count / total_instances
        samples_to_take = max(1, round(proportion * num_samples))
        class_samples[class_name] = samples_to_take

    # Adjust to ensure we get exactly the desired number of samples
    total_samples = sum(class_samples.values())

    if total_samples > num_samples:
        # Remove samples from classes with the most samples
        classes_sorted = sorted(class_samples.keys(), key=lambda x: class_samples[x], reverse=True)
        for class_name in classes_sorted:
            if total_samples <= num_samples:
                break
            if class_samples[class_name] > 1:
                class_samples[class_name] -= 1
                total_samples -= 1

    elif total_samples < num_samples:
        # Add samples to classes with the most instances
        classes_sorted = sorted(class_samples.keys(), key=lambda x: class_counts[x], reverse=True)
        for class_name in classes_sorted:
            if total_samples >= num_samples:
                break
            class_samples[class_name] += 1
            total_samples += 1

    # Group test samples by class
    class_to_samples = defaultdict(list)
    for _, row in test_df.iterrows():
        class_to_samples[row['class']].append(row)

    # Select samples from each class - use a fixed seed for reproducibility
    random.seed(42)  # Fixed seed ensures same selection each time
    selected_samples = []

    for class_name, samples_to_take in class_samples.items():
        if class_name in class_to_samples:
            # Take random samples from this class
            available_samples = class_to_samples[class_name]
            if len(available_samples) <= samples_to_take:
                selected_samples.extend(available_samples)
            else:
                selected_samples.extend(random.sample(available_samples, samples_to_take))

    # Return the selected samples
    return selected_samples

def evaluate_finetuned_sam_model(checkpoint_path , test_df , output_dir):
    # Set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the model
    checkpoint_path = checkpoint_path
    model = load_model_from_checkpoint(checkpoint_path)
    model = model.to(device)
    print("Model loaded successfully!")


    # Load test.csv
    test_df = pd.read_csv(test_df)
    print(f"Loaded {len(test_df)} test samples")

    # Select samples - using fixed seed for reproducibility
    selected_samples = select_samples(test_df, num_samples=30)
    print(f"Selected {len(selected_samples)} samples for evaluation")

    # Create output directory for visualizations
    output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Run inference on selected samples
    results = []

    for i, sample in enumerate(tqdm(selected_samples)):
        image_path = sample['image']
        mask_path = sample['label']

        # Parse points from string representation
        points = ast.literal_eval(sample['points'])

        # Predict mask with points
        pred_mask, image = predict_mask_with_points(model, image_path, points, device=device)

        # Load ground truth mask
        gt_mask = np.array(Image.open(mask_path).convert('L')) > 0

        # Calculate IoU
        iou = calculate_iou(pred_mask > 0.5, gt_mask)

        # Extract class name for display
        class_name = sample['class'].split('_', 1)[1] if '_' in sample['class'] else sample['class']

        # Visualize results
        fig = visualize_results(image, pred_mask, gt_mask, points, iou, class_name)

        # Save visualization
        output_path = os.path.join(output_dir, f'sample_{i+1}_{class_name}.png')
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

        # Store results
        results.append({
            'sample_id': i+1,
            'class': class_name,
            'image_path': image_path,
            'iou': iou,
            'visualization_path': output_path
        })

        print(f"Sample {i+1}/{len(selected_samples)}: Class={class_name}, IoU={iou:.4f}")

    # Create summary DataFrame
    results_df = pd.DataFrame(results)

    # Calculate average IoU per class
    class_iou = results_df.groupby('class')['iou'].mean().reset_index()
    class_iou = class_iou.sort_values('iou', ascending=False)

    # Calculate overall average IoU
    avg_iou = results_df['iou'].mean()

    # Print summary
    print("\nFine-tuned SAM Inference Results Summary:")
    print(f"Overall Average IoU: {avg_iou:.4f}")
    print("\nAverage IoU per Class:")
    for _, row in class_iou.iterrows():
        print(f"{row['class']}: {row['iou']:.4f}")

    # Save results to CSV
    results_df.to_csv(os.path.join(output_dir, 'finetuned_sam_results.csv'), index=False)
    class_iou.to_csv(os.path.join(output_dir, 'finetuned_sam_class_iou_summary.csv'), index=False)

    print(f"\nResults saved to {output_dir}")
    print(f"Visualizations for {len(results_df)} samples saved to {output_dir}")
