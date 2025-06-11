import os
import cv2
import numpy as np
import pandas as pd
import random
from pathlib import Path
from tqdm import tqdm

def get_three_distant_points(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) < 3:
        return []
    points = list(zip(x_indices, y_indices))
    center_idx = len(points) // 2
    center_point = points[center_idx]
    farthest_point_1 = max(points, key=lambda p: np.linalg.norm(np.array(p) - np.array(center_point)))
    farthest_point_2 = max(points, key=lambda p: np.linalg.norm(np.array(p) - np.array(farthest_point_1)))
    return [list(farthest_point_1), list(center_point), list(farthest_point_2)]

def create_csv_splits(class_data, output_dir):
    train_data, val_data, test_data = [], [], []

    for cls, instances in class_data.items():
        random.shuffle(instances)
        total = len(instances)
        split_1 = total // 3
        split_2 = 2 * total // 3

        train_data.extend(instances[:split_1])
        val_data.extend(instances[split_1:split_2])
        test_data.extend(instances[split_2:])

    # Save CSV files
    pd.DataFrame(train_data).to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    pd.DataFrame(val_data).to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    pd.DataFrame(test_data).to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print(f"CSV files created with sizes: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

def generate_yolo_csv_splits(dataset_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    class_folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"Found {len(class_folders)} class folders.")

    class_data = {}

    for class_folder in class_folders:
        class_path = os.path.join(dataset_path, class_folder)
        images_path = os.path.join(class_path, "images")
        masks_path = os.path.join(class_path, "masks")

        if not os.path.exists(images_path) or not os.path.exists(masks_path):
            print(f"Skipping {class_folder} due to missing images or masks folder.")
            continue

        image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_files) == 0:
            print(f"Skipping {class_folder}: no images found.")
            continue

        print(f"Processing class {class_folder} with {len(image_files)} images.")
        instances = []
        for image_file in tqdm(image_files, desc=f"Processing {class_folder}"):
            base_name = Path(image_file).stem
            mask_file = base_name + ".png"
            mask_path = os.path.join(masks_path, mask_file)
            if not os.path.exists(mask_path):
                #print(f"Mask missing for {image_file}, skipping.")
                continue

            points = get_three_distant_points(mask_path)
            if not points:
                #print(f"No valid points in mask for {image_file}, skipping.")
                continue

            instances.append({
                "image_path": os.path.join(class_path, "images", image_file),
                "mask_path": mask_path,
                "class_name": class_folder,
                "point1_x": points[0][0],
                "point1_y": points[0][1],
                "point2_x": points[1][0],
                "point2_y": points[1][1],
                "point3_x": points[2][0],
                "point3_y": points[2][1]
            })

        if len(instances) > 0:
            class_data[class_folder] = instances
        else:
            print(f"Class {class_folder} has no valid instances after filtering, skipping.")

    if not class_data:
        print("No classes with valid instances found. Exiting.")
        return

    create_csv_splits(class_data, output_dir)
