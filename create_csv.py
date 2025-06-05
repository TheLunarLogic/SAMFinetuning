import os
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd
import random
import glob

def select_points_for_mask(mask):
    """
    Select three points from a mask:
    - One in the middle of the mask
    - Two others equidistant from the middle point
    """
    # Find all points in the mask
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0:
        return None
    
    # Calculate the center of mass of the mask
    center_y = int(np.mean(y_indices))
    center_x = int(np.mean(x_indices))
    
    # Make sure the center point is actually in the mask
    if mask[center_y, center_x] == 0:
        # Find the closest point in the mask to the calculated center
        distances = (y_indices - center_y)**2 + (x_indices - center_x)**2
        closest_idx = np.argmin(distances)
        center_y, center_x = y_indices[closest_idx], x_indices[closest_idx]
    
    # Get all points in the mask
    mask_points = list(zip(y_indices, x_indices))
    
    # Calculate distances from center to all points
    distances = [(y - center_y)**2 + (x - center_x)**2 for y, x in mask_points]
    
    # Sort points by distance from center
    sorted_points = [p for _, p in sorted(zip(distances, mask_points))]
    
    # Choose two points that are roughly equidistant from center
    # Try to get points that are about 2/3 of the way to the edge
    target_idx = len(sorted_points) * 2 // 3
    if target_idx >= len(sorted_points):
        target_idx = len(sorted_points) // 2
    
    # Get the point at target_idx
    p1 = sorted_points[target_idx]
    
    # Find a second point that's similar in distance but in a different direction
    # by calculating the angle from center to p1
    angle1 = np.arctan2(p1[0] - center_y, p1[1] - center_x)
    
    # Look for a point with similar distance but roughly opposite angle
    target_angle = angle1 + np.pi  # opposite direction
    
    # Filter points with similar distance to target_idx
    similar_distance_points = sorted_points[target_idx-10:target_idx+10]
    if len(similar_distance_points) < 3:
        similar_distance_points = sorted_points[target_idx//2:]
    
    # Find point with angle closest to target_angle
    angles = [np.arctan2(y - center_y, x - center_x) for y, x in similar_distance_points]
    angle_diffs = [min(abs(a - target_angle), abs(a - target_angle + 2*np.pi), abs(a - target_angle - 2*np.pi)) for a in angles]
    p2 = similar_distance_points[np.argmin(angle_diffs)]
    
    return [(center_y, center_x), p1, p2]

def main():
    # Paths
    dataset_dir = 'dataset2'  # Directory containing class folders
    output_dir = 'processed_dataset'  # Directory to save CSV files
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find all class directories
    class_dirs = [d for d in os.listdir(dataset_dir) 
                 if os.path.isdir(os.path.join(dataset_dir, d)) and d.startswith('Class_')]
    
    print(f"Found {len(class_dirs)} class directories")
    
    # Count instances per class and collect data
    class_data = {}
    class_counts = {}
    
    for class_dir in class_dirs:
        # Extract class ID and name from directory name
        # Format: Class_ID_Name
        parts = class_dir.split('_', 2)
        if len(parts) < 3:
            print(f"Warning: Invalid class directory name format: {class_dir}. Skipping...")
            continue
        
        class_id = int(parts[1])
        class_name = parts[2]
        
        # Find all image files for this class
        image_files = glob.glob(os.path.join(dataset_dir, class_dir, 'image_*.png'))
        
        # Count instances
        instance_count = len(image_files)
        class_counts[class_id] = instance_count
        
        print(f"Class {class_id} ({class_name}): {instance_count} instances")
        
        # Skip classes with fewer than 100 instances
        if instance_count < 100:
            print(f"  Skipping class {class_id} ({class_name}) - fewer than 100 instances")
            continue
        
        # Process each image and its corresponding mask
        class_instances = []
        
        for image_path in tqdm(image_files, desc=f"Processing {class_name}"):
            # Get corresponding mask path
            mask_path = image_path.replace('image_', 'mask_')
            if not os.path.exists(mask_path):
                print(f"Warning: Mask not found for {image_path}. Skipping...")
                continue
            
            # Read mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Could not read mask {mask_path}. Skipping...")
                continue
            
            # Binarize mask if needed (in case it's not already binary)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask = mask // 255  # Convert to 0/1
            
            # Select three points from the mask
            points = select_points_for_mask(mask)
            if not points:
                print(f"Warning: Could not select points for {image_path}. Skipping...")
                continue
            
            # Store data
            class_instances.append({
                'image_path': image_path,
                'mask_path': mask_path,
                'class_id': class_id,
                'class_name': class_name,
                'point1_y': points[0][0],
                'point1_x': points[0][1],
                'point2_y': points[1][0],
                'point2_x': points[1][1],
                'point3_y': points[2][0],
                'point3_x': points[2][1]
            })
        
        class_data[class_id] = class_instances
    
    # Filter valid classes (with at least 100 instances)
    valid_classes = {class_id: instances for class_id, instances in class_data.items() 
                    if len(instances) >= 100}
    
    if not valid_classes:
        print("No classes with at least 100 instances found. Exiting...")
        return
    
    print(f"Found {len(valid_classes)} classes with at least 100 instances each.")
    
    # Create CSV files with proper distribution
    create_csv_files(valid_classes, output_dir)
    
    print(f"Processing complete. CSV files saved to {output_dir}")

def create_csv_files(class_data, output_dir):
    """
    Create train.csv, val.csv, and test.csv files with proper distribution.
    
    Args:
        class_data: Dictionary mapping class_id to list of instance data
        output_dir: Directory to save CSV files
    """
    # Calculate total instances and class proportions
    class_counts = {class_id: len(instances) for class_id, instances in class_data.items()}
    total_instances = sum(class_counts.values())
    class_proportions = {class_id: count / total_instances for class_id, count in class_counts.items()}
    
    # Calculate how many instances to take from each class
    train_size = 10000
    val_size = 1000
    test_size = 1000
    
    train_instances = {class_id: int(train_size * proportion) 
                       for class_id, proportion in class_proportions.items()}
    val_instances = {class_id: int(val_size * proportion) 
                     for class_id, proportion in class_proportions.items()}
    test_instances = {class_id: int(test_size * proportion) 
                      for class_id, proportion in class_proportions.items()}
    
    # Ensure we have exactly the right number of instances
    train_deficit = train_size - sum(train_instances.values())
    val_deficit = val_size - sum(val_instances.values())
    test_deficit = test_size - sum(test_instances.values())
    
    # Distribute any deficit to the largest classes
    for deficit, instances_dict in [(train_deficit, train_instances), 
                                   (val_deficit, val_instances), 
                                   (test_deficit, test_instances)]:
        if deficit > 0:
            sorted_classes = sorted(instances_dict.keys(), 
                                   key=lambda c: instances_dict[c], 
                                   reverse=True)
            for i in range(deficit):
                instances_dict[sorted_classes[i % len(sorted_classes)]] += 1
    
    # Create datasets
    train_data = []
    val_data = []
    test_data = []
    
    for class_id, instances in class_data.items():
        # Shuffle instances for this class
        random.shuffle(instances)
        
        # Take required number for each split
        train_count = train_instances[class_id]
        val_count = val_instances[class_id]
        test_count = test_instances[class_id]
        
        # Ensure we don't request more than available
        available = len(instances)
        if train_count + val_count + test_count > available:
            # Scale down proportionally
            total_needed = train_count + val_count + test_count
            scale = available / total_needed
            train_count = int(train_count * scale)
            val_count = int(val_count * scale)
            test_count = available - train_count - val_count
        
        train_data.extend(instances[:train_count])
        val_data.extend(instances[train_count:train_count+val_count])
        test_data.extend(instances[train_count+val_count:train_count+val_count+test_count])
    
    # Convert to DataFrames
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)
    
    # Save to CSV
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Created CSV files:")
    print(f"  - train.csv: {len(train_df)} instances")
    print(f"  - val.csv: {len(val_df)} instances")
    print(f"  - test.csv: {len(test_df)} instances")

if __name__ == "__main__":
    main()

