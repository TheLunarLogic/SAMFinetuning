import os
import json
import numpy as np
import cv2
from pycocotools import mask as mask_utils
from tqdm import tqdm
import glob

def create_mask_from_polygon(polygon, height, width):
    """Create a binary mask from polygon vertices."""
    # Convert polygon to the format expected by cv2.fillPoly
    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 2))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask

def merge_coco_jsons(json_paths):
    """Merge multiple COCO JSON files into a single structure."""
    print(f"Merging {len(json_paths)} COCO JSON files...")

    merged_data = {
        'images': [],
        'annotations': [],
        'categories': []
    }

    # Keep track of IDs to avoid conflicts
    next_image_id = 1
    next_annotation_id = 1
    category_id_mapping = {}  # Maps original category IDs to new ones
    image_id_mapping = {}  # Maps original image IDs to new ones

    # Process each JSON file
    for json_path in json_paths:
        print(f"Processing {json_path}...")
        with open(json_path, 'r') as f:
            coco_data = json.load(f)

        # Process categories first
        for category in coco_data.get('categories', []):
            original_id = category['id']
            # Check if this category already exists (by name)
            existing_category = next((c for c in merged_data['categories']
                                    if c['name'] == category['name']), None)

            if existing_category:
                # Map to existing category
                category_id_mapping[(json_path, original_id)] = existing_category['id']
            else:
                # Add new category with a unique ID
                new_id = len(merged_data['categories']) + 1
                category_id_mapping[(json_path, original_id)] = new_id

                category_copy = category.copy()
                category_copy['id'] = new_id
                merged_data['categories'].append(category_copy)

        # Process images
        for image in coco_data.get('images', []):
            original_id = image['id']
            image_copy = image.copy()
            image_copy['id'] = next_image_id

            # Store mapping from original to new ID
            image_id_mapping[(json_path, original_id)] = next_image_id

            merged_data['images'].append(image_copy)
            next_image_id += 1

        # Process annotations
        for annotation in coco_data.get('annotations', []):
            original_image_id = annotation['image_id']
            original_category_id = annotation['category_id']

            # Skip if we can't map the image or category
            if (json_path, original_image_id) not in image_id_mapping or \
               (json_path, original_category_id) not in category_id_mapping:
                continue

            annotation_copy = annotation.copy()
            annotation_copy['id'] = next_annotation_id
            annotation_copy['image_id'] = image_id_mapping[(json_path, original_image_id)]
            annotation_copy['category_id'] = category_id_mapping[(json_path, original_category_id)]

            merged_data['annotations'].append(annotation_copy)
            next_annotation_id += 1

    print(f"Merged data contains {len(merged_data['images'])} images, "
          f"{len(merged_data['annotations'])} annotations, and "
          f"{len(merged_data['categories'])} categories.")

    return merged_data

def generate_classwise_crops(coco_json_path,images_dir,output_dir):
    # Paths
    coco_json_path = coco_json_path
    images_dir = images_dir
    output_dir = output_dir

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load COCO JSON
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create image info mapping using file_name
    image_id_to_info = {}
    for image in coco_data['images']:
        image_id_to_info[image['id']] = {
            'file_name': image['file_name'],
            'width': image['width'],
            'height': image['height'],
        }

    # Create a mapping from category_id to category_name
    category_id_to_name = {
        category['id']: category['name'].replace(' ', '_')
        for category in coco_data['categories']
    }

    # Process annotations
    print("Processing annotations...")
    for annotation in tqdm(coco_data['annotations']):
        image_id = annotation['image_id']
        category_id = annotation['category_id']

        image_info = image_id_to_info.get(image_id)
        if not image_info:
            print(f"Warning: Image ID {image_id} not found. Skipping...")
            continue

        file_name = image_info['file_name']
        image_path = os.path.join(images_dir, file_name)

        if not os.path.isfile(image_path):
            print(f"Warning: Image file {file_name} not found. Skipping...")
            continue

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Unable to read {image_path}. Skipping...")
            continue

        expected_height, expected_width = image_info['height'], image_info['width']
        actual_height, actual_width = image.shape[:2]
        if (expected_height, expected_width) != (actual_height, actual_width):
            print(f"Resizing image: {file_name}")
            image = cv2.resize(image, (expected_width, expected_height))

        # Create mask
        segmentation = annotation.get('segmentation', [])
        if not segmentation:
            print(f"Warning: No segmentation for annotation {annotation['id']}. Skipping...")
            continue

        # Handle both RLE and Polygon formats explicitly
        if isinstance(segmentation, dict) and 'counts' in segmentation and 'size' in segmentation: # RLE
             # Ensure the RLE format is correct before decoding
            try:
                 # Pass the RLE dictionary directly to decode
                 # pycocotools expects {"size": [h, w], "counts": "..."}, counts as a string
                 mask = mask_utils.decode(segmentation)
            except Exception as e:
                 print(f"Error decoding RLE for annotation {annotation['id']}: {e}. Skipping...")
                 continue # Skip this annotation if RLE decoding fails

        elif isinstance(segmentation, list):
            if len(segmentation) > 0 and isinstance(segmentation[0], list):  # Polygon
                 # For polygons, pycocotools.mask.toBbox and .toArea work on the polygon list
                 # For creating a mask from polygon, use the create_mask_from_polygon helper
                 mask = create_mask_from_polygon(segmentation[0], expected_height, expected_width)
            else:
                print(f"Warning: Unknown list segmentation format for annotation {annotation['id']}. Skipping...")
                continue
        else:
            print(f"Warning: Invalid segmentation format for annotation {annotation['id']}. Skipping...")
            continue

        # Ensure mask is a numpy array and binary (0 or 1)
        if mask is None or mask.ndim != 2 or mask.shape != (expected_height, expected_width):
             print(f"Warning: Mask creation failed or resulted in incorrect shape for annotation {annotation['id']}. Skipping...")
             continue

        # Convert mask to binary if it's not already (e.g., from RLE which might be uint8 > 0)
        mask = (mask > 0).astype(np.uint8)


        # Get bounding box
        bbox = annotation.get('bbox')
        if bbox:
            x, y, w, h = map(int, bbox)
        else:
            # Calculate bbox from mask if not provided
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) == 0 or len(x_indices) == 0:
                print(f"Warning: Empty mask after decoding for annotation {annotation['id']}. Skipping...")
                continue
            x, y = int(np.min(x_indices)), int(np.min(y_indices))
            w = int(np.max(x_indices) - x + 1) # +1 to include the last pixel
            h = int(np.max(y_indices) - y + 1) # +1 to include the last pixel

        # Add padding
        # Ensure padding calculations don't go out of bounds
        pad_x = int(w * 0.1)
        pad_y = int(h * 0.1)

        x_p = max(0, x - pad_x)
        y_p = max(0, y - pad_y)

        # Calculate padded width and height, clamping to image dimensions
        x_p_end = min(expected_width, x + w + pad_x)
        y_p_end = min(expected_height, y + h + pad_y)

        w_p = x_p_end - x_p
        h_p = y_p_end - y_p


        cropped_image = image[y_p:y_p+h_p, x_p:x_p+w_p]
        cropped_mask = mask[y_p:y_p+h_p, x_p:x_p+w_p]

        if cropped_image.size == 0 or cropped_mask.size == 0:
            print(f"Warning: Empty crop for annotation {annotation['id']}. Skipping...")
            continue

        # Ensure mask_vis is 3 channels for saving as color image, even if binary
        mask_vis = np.stack([cropped_mask * 255] * 3, axis=-1).astype(np.uint8)

        class_name = category_id_to_name.get(category_id, f"unknown_{category_id}")
        class_dir = os.path.join(output_dir, f"Class_{category_id}_{class_name}")
        os.makedirs(class_dir, exist_ok=True)

        # Use a unique identifier for the saved files
        annotation_id = annotation.get('id', f"img{image_id}_ann{annotation['id']}") # Fallback if 'id' is missing
        image_save_path = os.path.join(class_dir, f"image_{annotation_id}.png")
        mask_save_path = os.path.join(class_dir, f"mask_{annotation_id}.png")

        try:
            cv2.imwrite(image_save_path, cropped_image)
            cv2.imwrite(mask_save_path, mask_vis)
        except Exception as e:
            print(f"Error saving images for annotation {annotation['id']}: {e}. Skipping...")
            continue


    print(f"Processing complete. Output saved to {output_dir}")
