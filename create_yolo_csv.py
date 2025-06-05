import os
import cv2
import numpy as np
import pandas as pd
import random
from pathlib import Path

def get_three_distant_points(mask_path):
    print(f"Processing mask: {mask_path}")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print("Error: Unable to read mask file.")
        return []
    
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) < 3:
        print("Skipping: Not enough non-zero pixels.")
        return []
    
    points = list(zip(x_indices, y_indices))
    center_idx = len(points) // 2
    center_point = points[center_idx]
    
    farthest_point_1 = max(points, key=lambda p: np.linalg.norm(np.array(p) - np.array(center_point)))
    farthest_point_2 = max(points, key=lambda p: np.linalg.norm(np.array(p) - np.array(farthest_point_1)))
    
    print(f"Selected points: {farthest_point_1}, {center_point}, {farthest_point_2}")
    return [list(farthest_point_1), list(center_point), list(farthest_point_2)]

def generate_csv(dataset_path, output_csv):
    print(f"Starting CSV generation from dataset: {dataset_path}")
    data = []
    class_folders = os.listdir(dataset_path)
    print(f"Found {len(class_folders)} classes.")
    
    for class_folder in class_folders:
        class_path = os.path.join(dataset_path, class_folder)
        images_path = os.path.join(class_path, "images")
        masks_path = os.path.join(class_path, "masks")
        
        if not os.path.exists(images_path) or not os.path.exists(masks_path):
            print(f"Skipping {class_folder}: Missing images or masks folder.")
            continue
        
        image_files = [f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Processing class {class_folder}: Found {len(image_files)} images.")
        
        for image_file in image_files:
            base_name = Path(image_file).stem
            mask_file = base_name + ".png"
            mask_path = os.path.join(masks_path, mask_file)
            
            if not os.path.exists(mask_path):
                print(f"Skipping {image_file}: Corresponding mask not found.")
                continue
            
            points = get_three_distant_points(mask_path)
            if not points:
                print(f"Skipping {image_file}: No valid points found.")
                continue
            
            image_rel_path = os.path.join(dataset_path, class_folder, "images", image_file)
            mask_rel_path = os.path.join(dataset_path, class_folder, "masks", mask_file)
            
            data.append([len(data), image_rel_path, mask_rel_path, str(points)])
    
    if not data:
        print("No valid data to save in CSV.")
        return
    
    df = pd.DataFrame(data, columns=["Unnamed: 0", "image", "label", "points"])
    df.to_csv(output_csv, index=False)
    print(f"CSV saved successfully: {output_csv}")

# Example usage
dataset_yolo = "dataset_yolo"
output_csv = "output.csv"
generate_csv(dataset_yolo, output_csv)