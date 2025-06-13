import os
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd
import random
import glob

def select_points_for_mask(mask):
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0:
        return None

    center_y = int(np.mean(y_indices))
    center_x = int(np.mean(x_indices))

    if mask[center_y, center_x] == 0:
        distances = (y_indices - center_y)**2 + (x_indices - center_x)**2
        closest_idx = np.argmin(distances)
        center_y, center_x = y_indices[closest_idx], x_indices[closest_idx]

    mask_points = list(zip(y_indices, x_indices))
    distances = [(y - center_y)**2 + (x - center_x)**2 for y, x in mask_points]
    sorted_points = [p for _, p in sorted(zip(distances, mask_points))]

    target_idx = len(sorted_points) * 2 // 3
    if target_idx >= len(sorted_points):
        target_idx = len(sorted_points) // 2

    p1 = sorted_points[target_idx]
    angle1 = np.arctan2(p1[0] - center_y, p1[1] - center_x)
    target_angle = angle1 + np.pi

    similar_distance_points = sorted_points[target_idx-10:target_idx+10]
    if len(similar_distance_points) < 3:
        similar_distance_points = sorted_points[target_idx//2:]

    angles = [np.arctan2(y - center_y, x - center_x) for y, x in similar_distance_points]
    angle_diffs = [min(abs(a - target_angle), abs(a - target_angle + 2*np.pi), abs(a - target_angle - 2*np.pi)) for a in angles]
    p2 = similar_distance_points[np.argmin(angle_diffs)]

    return [(center_y, center_x), p1, p2]

def generate_coco_csv_splits(dataset_dir,output_dir):
    dataset_dir = dataset_dir
    output_dir = output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    class_dirs = [d for d in os.listdir(dataset_dir)
                 if os.path.isdir(os.path.join(dataset_dir, d)) and d.startswith('Class_')]

    print(f"Found {len(class_dirs)} class directories")

    class_data = {}

    for class_dir in class_dirs:
        parts = class_dir.split('_', 2)
        if len(parts) < 3:
            print(f"Warning: Invalid class directory name format: {class_dir}. Skipping...")
            continue

        class_id = int(parts[1])
        class_name = parts[2]
        image_files = glob.glob(os.path.join(dataset_dir, class_dir, 'image_*.png'))
        print(f"Class {class_id} ({class_name}): {len(image_files)} instances")

        class_instances = []

        for image_path in tqdm(image_files, desc=f"Processing {class_name}"):
            mask_path = image_path.replace('image_', 'mask_')
            if not os.path.exists(mask_path):
                continue

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask = mask // 255

            points = select_points_for_mask(mask)
            if not points:
                continue

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

        if class_instances:
            class_data[class_id] = class_instances

    if not class_data:
        print("No valid instances found. Exiting...")
        return

    create_csv_files(class_data, output_dir)
    print(f"Processing complete. CSV files saved to {output_dir}")

def create_csv_files(class_data, output_dir):
    train_data = []
    val_data = []
    test_data = []

    for class_id, instances in class_data.items():
        random.shuffle(instances)

        total = len(instances)
        train_count = int(total * 0.6)
        val_count = int(total * 0.2)
        test_count = total - train_count - val_count

        train_data.extend(instances[:train_count])
        val_data.extend(instances[train_count:train_count+val_count])
        test_data.extend(instances[train_count+val_count:])

    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print(f"Created CSV files:")
    print(f"  - train.csv: {len(train_df)} instances")
    print(f"  - val.csv: {len(val_df)} instances")
    print(f"  - test.csv: {len(test_df)} instances")
