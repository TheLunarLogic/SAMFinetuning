import os
import yaml
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

def find_yolo_dataset_dir():
    cwd = Path.cwd()
    for dir in cwd.iterdir():
        if dir.is_dir():
            images = dir / 'images'
            labels = dir / 'labels'
            data_yaml = dir / 'data.yaml'
            if images.exists() and labels.exists() and data_yaml.exists():
                return str(dir)
    raise FileNotFoundError("YOLO dataset folder with images/, labels/, and data.yaml not found.")

def load_class_mapping(data_yaml_path):
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    class_names = data['names']
    return {i: class_names[i] for i in range(len(class_names))}

def create_class_dirs(class_mapping):
    for class_id, class_name in class_mapping.items():
        safe_name = class_name.replace(' ', '_').replace('/', '_').replace('&', 'and')
        class_dir = os.path.join(OUTPUT_DIR, f"{class_id}_{safe_name}")
        os.makedirs(os.path.join(class_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(class_dir, "masks"), exist_ok=True)

def parse_yolo_polygon(line):
    parts = line.strip().split()
    class_id = int(parts[0])
    points = [(float(parts[i]), float(parts[i+1])) for i in range(1, len(parts), 2) if i+1 < len(parts)]
    return class_id, points

def process_image_and_label(image_path, label_path, class_mapping, counter_dict):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return

    height, width = image.shape[:2]
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        class_id, points = parse_yolo_polygon(line)
        if class_id not in class_mapping:
            print(f"Warning: Class ID {class_id} not found in mapping")
            continue
        pixel_points = [(int(x * width), int(y * height)) for x, y in points]
        mask = np.zeros((height, width), dtype=np.uint8)

        if len(pixel_points) >= 3:
            pts = np.array(pixel_points, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)

        if mask.any():
            y_indices, x_indices = np.where(mask > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            padding_x, padding_y = int((x_max - x_min) * 0.1), int((y_max - y_min) * 0.1)
            x_min, y_min = max(0, x_min - padding_x), max(0, y_min - padding_y)
            x_max, y_max = min(width, x_max + padding_x), min(height, y_max + padding_y)
            cropped_image = image[y_min:y_max, x_min:x_max]
            cropped_mask = mask[y_min:y_max, x_min:x_max]
            if cropped_image.shape[0] < 10 or cropped_image.shape[1] < 10:
                continue
            class_name = class_mapping[class_id]
            safe_name = class_name.replace(' ', '_').replace('/', '_').replace('&', 'and')
            class_dir = os.path.join(OUTPUT_DIR, f"{class_id}_{safe_name}")
            counter_dict[class_id] = counter_dict.get(class_id, 0) + 1
            base_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{counter_dict[class_id]}"
            cv2.imwrite(os.path.join(class_dir, "images", f"{base_filename}.png"), cropped_image)
            cv2.imwrite(os.path.join(class_dir, "masks", f"{base_filename}.png"), cropped_mask)

def process_yolo_to_classwise():
    global OUTPUT_DIR
    YOLO_VAL_DIR = find_yolo_dataset_dir()
    print(f"Detected YOLO dataset directory: {YOLO_VAL_DIR}")

    IMAGES_DIR = os.path.join(YOLO_VAL_DIR, "images")
    LABELS_DIR = os.path.join(YOLO_VAL_DIR, "labels")
    data_yaml_path = os.path.join(YOLO_VAL_DIR, "data.yaml")
    OUTPUT_DIR = "dataset_yolo"

    class_mapping = load_class_mapping(data_yaml_path)
    create_class_dirs(class_mapping)
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    counter_dict = {}

    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(IMAGES_DIR, image_file)
        label_path = os.path.join(LABELS_DIR, os.path.splitext(image_file)[0] + '.txt')
        if not os.path.exists(label_path):
            print(f"Warning: Label file not found for {image_file}")
            continue

        with open(label_path, 'r') as f:
            if any(line.startswith(('0 ', '102 ')) for line in f):
                print(f"\033[91mHighlight: {label_path} contains class 0 or 102\033[0m")

        process_image_and_label(image_path, label_path, class_mapping, counter_dict)

    print("✅ Dataset processing complete!")


