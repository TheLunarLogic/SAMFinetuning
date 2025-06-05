import yaml
import os
from collections import Counter
import json
from tqdm import tqdm
def _count_instances_in_file(annotation_path):
    class_counts = Counter()
    with open(annotation_path, 'r') as f:
        for line in f:
            class_id = int(line.split()[0])
            class_counts[class_id] += 1
    return class_counts

def _process_dataset_path(dataset_path):
    class_counts = Counter()
    if not os.path.isdir(dataset_path):
        print(f"Dataset path {dataset_path} is not a valid directory")
        return class_counts
        
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.txt'):
                annotation_path = os.path.join(root, file)
                class_counts.update(_count_instances_in_file(annotation_path))
    return class_counts

def calculate_class_instances(data_yaml_path):
    with open(data_yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    
    class_names = data['names']
    total_counts = Counter()
    
    for dataset_path in [data['train'], data['val']]:
        dataset_full_path = os.path.join(os.path.dirname(data_yaml_path), dataset_path)
        dataset_full_path = dataset_full_path.replace("images", "labels")
        total_counts.update(_process_dataset_path(dataset_full_path))

    return {class_names[class_id]: count 
            for class_id, count in total_counts.items()}

def filter_class_instances(class_instances, percentage_threshold=0.01):
    total_instances = sum(class_instances.values())
    return {class_name: count for class_name, count in class_instances.items() if count / total_instances >= percentage_threshold}

def _setup_new_directory(original_dir, new_dataset_name):
    new_dir = original_dir + '_' + new_dataset_name
    for subset in ['train', 'val']:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(new_dir, subset, subdir), exist_ok=True)
    return new_dir

def _create_class_mapping(data, filtered_class_names):
    return {data['names'].index(class_name): idx 
            for idx, class_name in enumerate(filtered_class_names)}

def _process_label_file(src_path, dst_path, old_to_new_class_map):
    keep_file = False
    new_annotations = []
    
    with open(src_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            if class_id in old_to_new_class_map:
                keep_file = True
                parts[0] = str(old_to_new_class_map[class_id])
                new_annotations.append(' '.join(parts))
    
    if keep_file:
        with open(dst_path, 'w') as f:
            f.write('\n'.join(new_annotations))
    
    return keep_file

def _copy_image_if_exists(image_name, src_dir, dst_dir):
    for ext in ['.jpg', '.jpeg', '.png']:
        src_path = os.path.join(src_dir, image_name + ext)
        if os.path.exists(src_path):
            dst_path = os.path.join(dst_dir, image_name + ext)
            os.system(f'cp "{src_path}" "{dst_path}"')
            break

def create_yolo_subdataset(data_yaml_path: str, filtered_class_names: list[str], new_dataset_name: str):
    """ Create a new YOLO dataset with a subset of classes from the original dataset.

    Args:
        data_yaml_path (str): Path to the original data.yaml file.
        filtered_class_names (list): List of class names to keep.
        new_dataset_name (str): Name of the new dataset.
    """
    # Load original data yaml
    with open(data_yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    
    # Setup directory structure
    original_dir = os.path.dirname(data_yaml_path)
    new_dir = _setup_new_directory(original_dir, new_dataset_name)
    
    # Create class mapping
    old_to_new_class_map = _create_class_mapping(data, filtered_class_names)
    
    # Process files
    for subset in ['train', 'val']:
        labels_src_dir = os.path.join(original_dir, subset, 'labels')
        labels_dst_dir = os.path.join(new_dir, subset, 'labels')
        images_src_dir = os.path.join(original_dir, subset, 'images')
        images_dst_dir = os.path.join(new_dir, subset, 'images')
        
        for label_file in tqdm(os.listdir(labels_src_dir), desc=f'Processing {subset}'):
            if not label_file.endswith('.txt'):
                continue
                
            src_label_path = os.path.join(labels_src_dir, label_file)
            dst_label_path = os.path.join(labels_dst_dir, label_file)
            
            if _process_label_file(src_label_path, dst_label_path, old_to_new_class_map):
                image_name = os.path.splitext(label_file)[0]
                _copy_image_if_exists(image_name, images_src_dir, images_dst_dir)
    
    # Update and save new data.yaml and classes.txt
    data.update({
        'path': new_dir,
        'names': filtered_class_names,
        'nc': len(filtered_class_names)
    })
    
    with open(os.path.join(new_dir, 'classes.txt'), 'w') as f:
        f.write('\n'.join(filtered_class_names))
        
    with open(os.path.join(new_dir, 'data.yaml'), 'w') as file:
        yaml.dump(data, file)
# Example usage:
data_yaml_path = '/home/Ximi-Hoque/MLTraining/datasets/yolo_dataset/data.yaml'
class_instances = calculate_class_instances(data_yaml_path)

filtered_class_instances = filter_class_instances(class_instances)
print("Filtered class-level instance counts:")
print(json.dumps(filtered_class_instances, indent=4))

filtered_class_names = ['Person', 'Wall', 'Sidewalk', 'Obstacle', 'Vegetation', 'Road', 'Fence', 'Sky', 'Car', 'Pole', 'Pedestrian crossing']
create_yolo_subdataset(data_yaml_path, filtered_class_names, '1_percent')