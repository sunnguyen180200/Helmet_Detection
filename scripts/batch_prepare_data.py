#!/usr/bin/env python3
"""
Batch processing script for preparing YOLO helmet detection dataset.
This script processes images in small batches to avoid memory issues.
"""

import os
import sys
import json
import shutil
import random
import argparse
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Batch data preparation for YOLO training')
    parser.add_argument('--data_dir', type=str, default='D:/2025/yolo/main_src/data/part_1/part_1',
                        help='Path to the directory containing the image data')
    parser.add_argument('--annotation_dir', type=str, default='D:/2025/yolo/main_src/data/annotation',
                        help='Path to the directory containing COCO annotation files')
    parser.add_argument('--output_dir', type=str, default='./data/prepared',
                        help='Path to the output directory to save prepared data')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of images to process in each batch')
    parser.add_argument('--split', type=str, default='0.7,0.15,0.15',
                        help='Split ratios for train, validation, and test sets')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Maximum number of samples to process (0 for all)')
    return parser.parse_args()

def create_directories(output_dir):
    """Create the required directory structure for YOLO dataset."""
    directories = {
        'train/images': os.path.join(output_dir, 'train', 'images'),
        'train/labels': os.path.join(output_dir, 'train', 'labels'),
        'val/images': os.path.join(output_dir, 'val', 'images'),
        'val/labels': os.path.join(output_dir, 'val', 'labels'),
        'test/images': os.path.join(output_dir, 'test', 'images'),
        'test/labels': os.path.join(output_dir, 'test', 'labels')
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    return output_dir

def find_images(data_dir):
    """Find all image files in the data directory."""
    image_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} image files")
    return image_files

def load_annotation_file(file_path):
    """Load a single annotation file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract images and annotations
        images = {img['id']: img for img in data.get('images', [])}
        
        # Create a mapping from filename to annotations
        filename_to_annotations = {}
        
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
            if filename not in filename_to_annotations:
                filename_to_annotations[filename] = []
            
            # Convert COCO bbox format to YOLO format
            bbox = ann['bbox']
            x, y, w, h = bbox
            img_w, img_h = img_info['width'], img_info['height']
            
            # Normalize coordinates
            center_x = (x + w / 2) / img_w
            center_y = (y + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h
            
            # Category ID: 0 for helmet, 1 for no_helmet
            category_id = ann['category_id'] # Adjust to 0-indexed
            
            filename_to_annotations[filename].append({
                'class_id': category_id,
                'bbox': [center_x, center_y, norm_w, norm_h]
            })
        
        return filename_to_annotations
    
    except Exception as e:
        print(f"Error loading annotation file {file_path}: {e}")
        return {}

def load_all_annotations(annotation_dir):
    """Load all annotation files and merge them."""
    annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith('.json')]
    print(f"Found {len(annotation_files)} annotation files")
    
    all_annotations = {}
    
    for file in annotation_files:
        file_path = os.path.join(annotation_dir, file)
        print(f"Processing annotation file: {file}")
        
        file_annotations = load_annotation_file(file_path)
        all_annotations.update(file_annotations)
    
    print(f"Loaded annotations for {len(all_annotations)} unique images")
    return all_annotations

def get_folder_name(image_path, data_dir):
    """Extract folder name from image path."""
    rel_path = os.path.relpath(image_path, data_dir)
    folder_name = os.path.basename(os.path.dirname(rel_path))
    return folder_name if folder_name != '.' else ""

def process_batch(batch, annotations, output_dir, split_map, data_dir):
    """Process a batch of images."""
    for image_path in batch:
        try:
            # Get the filename and folder name
            filename = os.path.basename(image_path)
            folder_name = get_folder_name(image_path, data_dir)
            basename = os.path.splitext(filename)[0]
            
            # Skip if no annotations
            if filename not in annotations:
                continue
            
            # Determine destination based on split map
            split = split_map.get(image_path)
            if not split:
                continue
            
            # Set output paths with folder name prefix
            dest_image_dir = os.path.join(output_dir, split, 'images')
            dest_label_dir = os.path.join(output_dir, split, 'labels')
            
            if folder_name:
                dest_image_filename = f"{folder_name}_{filename}"
                dest_label_filename = f"{folder_name}_{basename}.txt"
            else:
                dest_image_filename = filename
                dest_label_filename = f"{basename}.txt"
            
            dest_image_path = os.path.join(dest_image_dir, dest_image_filename)
            dest_label_path = os.path.join(dest_label_dir, dest_label_filename)
            
            # Copy the image
            shutil.copy2(image_path, dest_image_path)
            
            # Write the label file
            with open(dest_label_path, 'w') as f:
                for ann in annotations[filename]:
                    class_id = ann['class_id']
                    bbox = ann['bbox']
                    f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

def create_yaml_config(output_dir):
    """Create a YAML configuration file for YOLOv11 training."""
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    
    yaml_content = f"""# YOLOv11 dataset configuration
path: {output_dir}  # dataset root dir
train: {os.path.join(output_dir, 'train')}  # train images
val: {os.path.join(output_dir, 'val')}  # validation images
test: {os.path.join(output_dir, 'test')}  # test images

# Classes
nc: 2  # number of classes
names: ['helmet', 'no_helmet']  # class names
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created YAML configuration at {yaml_path}")

def main():
    args = parse_args()
    
    # Use hardcoded directories
    data_dir = "D:/2025/yolo/main_src/data/part_1/part_1"
    annotation_dir = "D:/2025/yolo/main_src/data/annotation"
    output_dir = args.output_dir
    
    print(f"Using data directory: {data_dir}")
    print(f"Using annotation directory: {annotation_dir}")
    print(f"Using output directory: {output_dir}")
    
    # Create output directories
    create_directories(output_dir)
    
    # Find all image files
    all_images = find_images(data_dir)
    if args.max_samples > 0 and args.max_samples < len(all_images):
        print(f"Limiting to {args.max_samples} samples")
        all_images = all_images[:args.max_samples]
    
    # Load all annotations
    annotations = load_all_annotations(annotation_dir)
    
    # Filter images that have annotations
    valid_images = [img for img in all_images if os.path.basename(img) in annotations]
    print(f"Found {len(valid_images)} images with valid annotations")
    
    # Create split ratios
    split_ratios = [float(x) for x in args.split.split(',')]
    
    # Shuffle and split images
    random.shuffle(valid_images)
    
    train_end = int(len(valid_images) * split_ratios[0])
    val_end = train_end + int(len(valid_images) * split_ratios[1])
    
    train_images = valid_images[:train_end]
    val_images = valid_images[train_end:val_end]
    test_images = valid_images[val_end:]
    
    print(f"Split into {len(train_images)} training, {len(val_images)} validation, and {len(test_images)} test images")
    
    # Create split mapping
    split_map = {}
    for img in train_images:
        split_map[img] = 'train'
    for img in val_images:
        split_map[img] = 'val'
    for img in test_images:
        split_map[img] = 'test'
    
    # Process in batches
    batch_size = args.batch_size
    batches = [valid_images[i:i + batch_size] for i in range(0, len(valid_images), batch_size)]
    
    print(f"Processing {len(batches)} batches of size {batch_size}")
    
    for i, batch in enumerate(tqdm(batches, desc="Processing batches")):
        process_batch(batch, annotations, output_dir, split_map, data_dir)
        if (i + 1) % 10 == 0:
            print(f"Completed {i+1}/{len(batches)} batches")
    
    # Create YAML configuration
    create_yaml_config(output_dir)
    
    print("Dataset preparation completed successfully")

if __name__ == "__main__":
    main() 