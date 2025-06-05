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

def main():
    # Paths
    coco_json_paths = ['coco1.json', 'coco2.json', 'coco3.json']  # List of JSON files
    images_dir = 'Images'
    output_dir = 'dataset2'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Merge COCO JSON files
    coco_data = merge_coco_jsons(coco_json_paths)
    
    # Create a mapping from image_id to file_name and other info
    image_id_to_info = {}
    for image in coco_data['images']:
        image_id_to_info[image['id']] = {
            'width': image['width'],
            'height': image['height'],
            'labellerr_image_id': image.get('labellerr_image_id', '')
        }
    
    # Create a mapping from category_id to category_name
    category_id_to_name = {}
    for category in coco_data['categories']:
        category_id_to_name[category['id']] = category['name']
    
    # Process annotations
    print("Processing annotations...")
    for annotation in tqdm(coco_data['annotations']):
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        
        # Get image info
        image_info = image_id_to_info.get(image_id)
        if not image_info:
            print(f"Warning: Image ID {image_id} not found in images list. Skipping...")
            continue
        
        # Get category name
        category_name = category_id_to_name.get(category_id, f"unknown_{category_id}")
        category_name = category_name.replace(' ', '_')  # Replace spaces with underscores
        
        # Create class directory if it doesn't exist
        class_dir = os.path.join(output_dir, f"Class_{category_id}_{category_name}")
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        
        # Find the image file in the Images directory using labellerr_image_id
        labellerr_id = image_info['labellerr_image_id']
        if not labellerr_id:
            print(f"Warning: No labellerr_image_id for image ID {image_id}. Skipping...")
            continue
        
        # Look for image file with this ID
        image_path = None
        for file in os.listdir(images_dir):
            if file.startswith(labellerr_id) or labellerr_id in file:
                image_path = os.path.join(images_dir, file)
                break
        
        if not image_path:
            print(f"Warning: Could not find image for labellerr_image_id {labellerr_id}. Skipping...")
            continue
        
        # Read the image
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}. Skipping...")
                continue
            
            # Ensure image dimensions match what's in the COCO JSON
            actual_height, actual_width = image.shape[:2]
            expected_height = image_info['height']
            expected_width = image_info['width']
            
            if actual_height != expected_height or actual_width != expected_width:
                print(f"Warning: Image dimensions mismatch for {image_path}. "
                      f"Expected {expected_width}x{expected_height}, got {actual_width}x{actual_height}. "
                      f"Resizing...")
                image = cv2.resize(image, (expected_width, expected_height))
        except Exception as e:
            print(f"Error reading image {image_path}: {e}. Skipping...")
            continue
        
        # Create mask from segmentation
        segmentation = annotation.get('segmentation', [])
        if not segmentation:
            print(f"Warning: No segmentation data for annotation {annotation.get('id')}. Skipping...")
            continue
        
        # Handle different segmentation formats
        if isinstance(segmentation, dict):  # RLE format
            mask = mask_utils.decode(segmentation)
        elif isinstance(segmentation, list):
            if isinstance(segmentation[0], list):  # Polygon format
                mask = create_mask_from_polygon(segmentation[0], expected_height, expected_width)
            else:
                print(f"Warning: Unrecognized segmentation format. Skipping...")
                continue
        else:
            print(f"Warning: Unrecognized segmentation format. Skipping...")
            continue
        
        # Get bounding box (either from annotation or compute from mask)
        bbox = annotation.get('bbox')
        if bbox:
            x, y, w, h = [int(coord) for coord in bbox]
        else:
            # Compute bounding box from mask
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) == 0 or len(x_indices) == 0:
                print(f"Warning: Empty mask for annotation {annotation.get('id')}. Skipping...")
                continue
            x = int(np.min(x_indices))
            y = int(np.min(y_indices))
            w = int(np.max(x_indices) - x)
            h = int(np.max(y_indices) - y)
        
        # Add some padding to the bounding box (10% on each side)
        padding_x = int(w * 0.1)
        padding_y = int(h * 0.1)
        
        # Ensure the padded box is within image boundaries
        x_padded = max(0, x - padding_x)
        y_padded = max(0, y - padding_y)
        w_padded = min(expected_width - x_padded, w + 2 * padding_x)
        h_padded = min(expected_height - y_padded, h + 2 * padding_y)
        
        # Crop the image and mask
        cropped_image = image[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]
        cropped_mask = mask[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]
        
        # Skip if cropped image or mask is empty
        if cropped_image.size == 0 or cropped_mask.size == 0:
            print(f"Warning: Empty crop for annotation {annotation.get('id')}. Skipping...")
            continue
        
        # Convert mask to 3-channel for visualization (255 for white)
        mask_vis = np.stack([cropped_mask * 255] * 3, axis=-1).astype(np.uint8)
        
        # Save the cropped image and mask
        annotation_id = annotation.get('id', f"{image_id}_{category_id}_{len(os.listdir(class_dir)) // 2}")
        
        cropped_image_path = os.path.join(class_dir, f"image_{annotation_id}.png")
        mask_path = os.path.join(class_dir, f"mask_{annotation_id}.png")
        
        cv2.imwrite(cropped_image_path, cropped_image)
        cv2.imwrite(mask_path, mask_vis)
    
    print(f"Processing complete. Dataset saved to {output_dir}")

if __name__ == "__main__":
    main()