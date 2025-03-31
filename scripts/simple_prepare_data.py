#!/usr/bin/env python3
"""
Simplified preparation script for YOLO helmet detection dataset.
This script focuses on core functionality and avoids memory issues.
"""

import os
import sys
import json
import shutil
import random
from pathlib import Path
import logging
import argparse
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Simplified data preparation for YOLO training')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the directory containing the image data')
    parser.add_argument('--annotation_dir', type=str, required=True,
                        help='Path to the directory containing COCO annotation files')
    parser.add_argument('--output_dir', type=str, default='./data/yolo_dataset',
                        help='Path to the output directory to save prepared data')
    parser.add_argument('--split', type=str, default='0.7,0.15,0.15',
                        help='Split ratios for train, validation, and test sets')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Maximum number of samples to process (0 for all)')
    return parser.parse_args()

def create_directory_structure(output_dir):
    """Create the required directory structure for YOLO dataset."""
    directories = {
        'train_images': os.path.join(output_dir, 'train', 'images'),
        'train_labels': os.path.join(output_dir, 'train', 'labels'),
        'val_images': os.path.join(output_dir, 'val', 'images'),
        'val_labels': os.path.join(output_dir, 'val', 'labels'),
        'test_images': os.path.join(output_dir, 'test', 'images'),
        'test_labels': os.path.join(output_dir, 'test', 'labels')
    }
    
    # Create each directory in the structure
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    return directories

def load_annotations(annotation_dir):
    """Load annotation files in COCO format."""
    annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith('.json')]
    logger.info(f"Found {len(annotation_files)} annotation files in {annotation_dir}")
    
    all_annotations = {}
    total_images = 0
    total_annotations = 0
    
    for annotation_file in annotation_files:
        logger.info(f"Processing annotation file: {annotation_file}")
        
        try:
            with open(os.path.join(annotation_dir, annotation_file), 'r') as f:
                data = json.load(f)
            
            # Extract images and annotations
            images = {img['id']: img for img in data.get('images', [])}
            
            # Process annotations
            for ann in data.get('annotations', []):
                img_id = ann.get('image_id')
                if img_id not in images:
                    continue
                
                img_info = images[img_id]
                filename = img_info['file_name']
                
                # Extract just the filename (no path)
                if '/' in filename:
                    filename = filename.split('/')[-1]
                
                # Create annotation entry if it doesn't exist
                if filename not in all_annotations:
                    all_annotations[filename] = []
                
                # Convert COCO bbox format to YOLO format
                # COCO: [x, y, width, height]
                # YOLO: [class_id, center_x, center_y, width, height] (normalized)
                bbox = ann['bbox']
                x, y, w, h = bbox
                img_w, img_h = img_info['width'], img_info['height']
                
                # Normalize coordinates
                center_x = (x + w / 2) / img_w
                center_y = (y + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                
                # Category ID: 0 for helmet, 1 for no_helmet
                category_id = ann['category_id'] - 1  # Adjust to 0-indexed
                
                all_annotations[filename].append([category_id, center_x, center_y, norm_w, norm_h])
                total_annotations += 1
            
            total_images += len(images)
            
        except Exception as e:
            logger.error(f"Error processing {annotation_file}: {e}")
    
    logger.info(f"Loaded annotations for {len(all_annotations)} unique images with {total_annotations} total annotations")
    return all_annotations

def find_image_files(data_dir):
    """Find all image files recursively in data_dir."""
    image_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(image_files)} image files")
    return image_files

def match_images_with_annotations(image_files, annotations):
    """Match image files with their annotations using filenames."""
    matched_data = {}
    
    for image_path in image_files:
        filename = os.path.basename(image_path)
        if filename in annotations:
            matched_data[image_path] = annotations[filename]
    
    logger.info(f"Matched {len(matched_data)} images with annotations")
    return matched_data

def split_dataset(matched_data, split_ratios):
    """Split dataset into train, validation, and test sets."""
    image_paths = list(matched_data.keys())
    random.shuffle(image_paths)
    
    train_end = int(len(image_paths) * split_ratios[0])
    val_end = train_end + int(len(image_paths) * split_ratios[1])
    
    train_paths = image_paths[:train_end]
    val_paths = image_paths[train_end:val_end]
    test_paths = image_paths[val_end:]
    
    logger.info(f"Split into {len(train_paths)} training, {len(val_paths)} validation, and {len(test_paths)} test images")
    return train_paths, val_paths, test_paths

def process_dataset(image_paths, annotations, output_dirs, split_name):
    """Process images and annotations for a dataset split."""
    images_dir = output_dirs[f'{split_name}_images']
    labels_dir = output_dirs[f'{split_name}_labels']
    
    success_count = 0
    logger.info(f"Processing {split_name} dataset...")
    
    for image_path in tqdm(image_paths, desc=f"Processing {split_name}"):
        try:
            # Get filename
            filename = os.path.basename(image_path)
            basename = os.path.splitext(filename)[0]
            
            # Copy image
            dest_image_path = os.path.join(images_dir, filename)
            shutil.copy2(image_path, dest_image_path)
            
            # Write labels
            label_path = os.path.join(labels_dir, f"{basename}.txt")
            
            with open(label_path, 'w') as f:
                for ann in annotations[filename]:
                    class_id, cx, cy, w, h = ann
                    f.write(f"{int(class_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            
            success_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
    
    logger.info(f"Successfully processed {success_count} images for {split_name} dataset")
    return success_count

def create_dataset_yaml(output_dir, class_names):
    """Create a YAML configuration file for YOLOv11 training."""
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    
    yaml_content = f"""# YOLOv11 dataset configuration
path: {output_dir}  # dataset root dir
train: {os.path.join(output_dir, 'train')}  # train images relative to path
val: {os.path.join(output_dir, 'val')}  # val images relative to path
test: {os.path.join(output_dir, 'test')}  # test images relative to path

# Classes
nc: {len(class_names)}  # number of classes
names: {str(class_names)}  # class names
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    logger.info(f"Created dataset YAML configuration at {yaml_path}")
    return yaml_path

def main():
    args = parse_args()
    
    # Parse split ratios
    split_ratios = [float(x) for x in args.split.split(',')]
    if len(split_ratios) != 3 or abs(sum(split_ratios) - 1.0) > 0.01:
        logger.error("Split ratios must be three numbers that sum to 1.0")
        sys.exit(1)
    
    # Create output directories
    output_dir = args.output_dir
    directories = create_directory_structure(output_dir)
    
    # Load annotations
    logger.info(f"Loading annotations from {args.annotation_dir}")
    annotations = load_annotations(args.annotation_dir)
    
    # Find image files
    logger.info(f"Finding image files in {args.data_dir}")
    image_files = find_image_files(args.data_dir)
    
    # Limit the number of samples if specified
    if args.max_samples > 0 and args.max_samples < len(image_files):
        logger.info(f"Limiting to {args.max_samples} samples")
        image_files = image_files[:args.max_samples]
    
    # Match images with annotations
    matched_data = match_images_with_annotations(image_files, annotations)
    
    # Split the dataset
    train_paths, val_paths, test_paths = split_dataset(matched_data, split_ratios)
    
    # Process each dataset split
    process_dataset(train_paths, annotations, directories, 'train')
    process_dataset(val_paths, annotations, directories, 'val')
    process_dataset(test_paths, annotations, directories, 'test')
    
    # Create dataset.yaml
    class_names = ['helmet', 'no_helmet']
    yaml_path = create_dataset_yaml(output_dir, class_names)
    
    logger.info("Dataset preparation completed successfully")

if __name__ == "__main__":
    main() 