import matplotlib.pyplot as plt
import cv2
import os
import random
import yaml

# Define paths
DATA_YAML_PATH = "/content/yolo_dataset/valid/data.yaml"
OUTPUT_DIR = "dataset_yolo"

# Load class names from data.yaml
with open(DATA_YAML_PATH, "r") as f:
    data_yaml = yaml.safe_load(f)
CLASS_NAMES = data_yaml["names"]

def visualize_images_and_masks_from_classes(class_ids, num_images=2):
    plt.figure(figsize=(15, 6))

    valid_classes = []     # (class_id, class_name)
    selected_data = []     # [(image_path, mask_path), ...]

    for class_id in class_ids:
        class_name = CLASS_NAMES[class_id].replace(' ', '_').replace('/', '_').replace('&', 'and')
        class_folder = os.path.join(OUTPUT_DIR, f"{class_id}_{class_name}")
        image_dir = os.path.join(class_folder, "images")
        mask_dir = os.path.join(class_folder, "masks")

        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            print(f"Missing directory: {image_dir} or {mask_dir}")
            continue

        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_files) == 0:
            print(f"No images in: {image_dir}")
            continue

        selected = image_files[:num_images] if len(image_files) < num_images else random.sample(image_files, num_images)

        image_mask_pairs = []
        for img_file in selected:
            img_path = os.path.join(image_dir, img_file)
            mask_filename = os.path.splitext(img_file)[0] + ".png"
            mask_path = os.path.join(mask_dir, mask_filename)

            if not os.path.exists(mask_path):
                print(f"Mask not found for {img_file}")
                continue

            image_mask_pairs.append((img_path, mask_path))

        if len(image_mask_pairs) > 0:
            valid_classes.append((class_id, class_name))
            selected_data.append(image_mask_pairs)

    # Plot images and masks
    for i, (image_mask_pairs, (class_id, class_name)) in enumerate(zip(selected_data, valid_classes)):
        for j, (img_path, mask_path) in enumerate(image_mask_pairs):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

            idx = j * len(valid_classes) * 2 + i * 2
            ax_img = plt.subplot(num_images, len(valid_classes)*2, idx + 1)
            ax_mask = plt.subplot(num_images, len(valid_classes)*2, idx + 2)

            ax_img.imshow(img)
            ax_img.axis('off')
            ax_img.set_title(f"{class_id}_{class_name} - Image")

            ax_mask.imshow(mask)
            ax_mask.axis('off')
            ax_mask.set_title(f"{class_id}_{class_name} - Mask")

    plt.tight_layout()
    plt.show()


