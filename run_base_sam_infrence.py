# using base model

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
from transformers import SamModel

def calculate_iou(pred_mask, gt_mask):
    """
    Calculate IoU between predicted and ground truth masks
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    if union == 0:
        return 0.0

    return intersection / union

def predict_mask_with_base_sam(model, image_path, points, device='cuda'):
    """
    Predict a mask using the base SAM model with point prompts

    Args:
        model: The base SAM model from HuggingFace
        image_path: Path to the input image
        points: List of [x, y] coordinates on the original image scale
        device: Device to run inference on

    Returns:
        pred_mask: Predicted mask as numpy array
        image: Original image
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)

    # Resize image for SAM (1024x1024)
    image_size = 1024
    image_resized = image.resize((image_size, image_size), Image.BILINEAR)

    # Preprocess image
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

    # Format points for SAM
    input_points = torch.tensor([[scaled_points]], dtype=torch.float).to(device)
    input_labels = torch.ones((1, 1, len(points)), dtype=torch.long).to(device)  # All points are foreground

    # Generate mask prediction
    with torch.no_grad():
        outputs = model(
            pixel_values=image_tensor,
            input_points=input_points,
            input_labels=input_labels,
            multimask_output=False
        )

    # Get the predicted mask
    pred_mask = torch.sigmoid(outputs.pred_masks[:, 0, :, :, :])
    pred_mask = (pred_mask > 0.5).float()

    # Resize back to original image size
    pred_mask = F.interpolate(
        pred_mask,
        size=original_size[::-1],  # (width, height) -> (height, width)
        mode='bilinear',
        align_corners=False
    )

    # Convert to numpy
    pred_mask = pred_mask.squeeze().cpu().numpy()

    return pred_mask, image

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
    axes[1].set_title(f'Base SAM Predicted Mask (IoU: {iou:.4f})')
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

def base_main_function(test , output):
    # Set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the base SAM model
    print("Loading base SAM model...")
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    model.to(device)
    model.eval()
    print("Base SAM model loaded successfully!")

    # Load test.csv
       # Load test.csv
    test_df = test
    print(f"Loaded {len(test_df)} test samples")

    # Select samples - using fixed seed for reproducibility
    selected_samples = select_samples(test_df, num_samples=30)
    print(f"Selected {len(selected_samples)} samples for evaluation")

    # Create output directory for visualizations
    output_dir = output
    os.makedirs(output_dir, exist_ok=True)

    # Run inference on selected samples
    results = []

    for i, sample in enumerate(tqdm(selected_samples)):
        image_path = sample['image']
        mask_path = sample['label']

        # Parse points from string representation
        points = ast.literal_eval(sample['points'])

        # Predict mask with points using base SAM
        pred_mask, image = predict_mask_with_base_sam(model, image_path, points, device=device)

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
    print("\nBase SAM Inference Results Summary:")
    print(f"Overall Average IoU: {avg_iou:.4f}")
    print("\nAverage IoU per Class:")
    for _, row in class_iou.iterrows():
        print(f"{row['class']}: {row['iou']:.4f}")

    # Save results to CSV
    results_df.to_csv(os.path.join(output_dir, 'base_sam_results.csv'), index=False)
    class_iou.to_csv(os.path.join(output_dir, 'base_sam_class_iou_summary.csv'), index=False)

    print(f"\nResults saved to {output_dir}")
    print(f"Visualizations for {len(results_df)} samples saved to {output_dir}")
