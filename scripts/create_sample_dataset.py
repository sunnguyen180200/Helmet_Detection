#!/usr/bin/env python3
"""
Create a small sample dataset for YOLOv11 training.
This script will copy a small set of images and create corresponding YOLO format annotations.
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser(description='Create a sample dataset for YOLO training')
    parser.add_argument('--source_dir', type=str, default='D:/2025/yolo/part_1/part_1',
                        help='Source directory containing images')
    parser.add_argument('--output_dir', type=str, default='./data/yolov11_sample',
                        help='Output directory for sample dataset')
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of images to include in sample')
    return parser.parse_args()

def create_directory_structure(output_dir):
    """Create YOLO directory structure."""
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            path = os.path.join(output_dir, split, subdir)
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")
    
    return output_dir

def find_random_images(source_dir, num_images):
    """Find random images from source directory."""
    all_images = []
    
    for ext in ['jpg', 'jpeg', 'png']:
        all_images.extend(glob(os.path.join(source_dir, '**', f'*.{ext}'), recursive=True))
    
    print(f"Found {len(all_images)} images in source directory")
    
    if len(all_images) <= num_images:
        return all_images
    
    return random.sample(all_images, num_images)

def create_sample_annotation(img_path, class_id=0):
    """Create a simple YOLO format annotation."""
    # This creates a random bounding box in the center region of the image
    x_center = random.uniform(0.3, 0.7)
    y_center = random.uniform(0.3, 0.7)
    width = random.uniform(0.1, 0.3)
    height = random.uniform(0.1, 0.3)
    
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def process_images(images, output_dir):
    """Process images and create sample dataset."""
    # Split images into train, val, test
    random.shuffle(images)
    
    num_images = len(images)
    train_idx = int(num_images * 0.7)
    val_idx = int(num_images * 0.85)
    
    train_images = images[:train_idx]
    val_images = images[train_idx:val_idx]
    test_images = images[val_idx:]
    
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    
    print(f"Split into {len(train_images)} train, {len(val_images)} val, {len(test_images)} test images")
    
    # Process each split
    for split_name, split_images in splits.items():
        print(f"Processing {split_name} split...")
        
        for img_path in split_images:
            try:
                # Get filename and destination paths
                filename = os.path.basename(img_path)
                basename = os.path.splitext(filename)[0]
                
                dest_img_path = os.path.join(output_dir, split_name, 'images', filename)
                dest_label_path = os.path.join(output_dir, split_name, 'labels', f"{basename}.txt")
                
                # Copy image
                shutil.copy2(img_path, dest_img_path)
                
                # Create annotation
                # Randomly choose between helmet (0) and no_helmet (1)
                class_id = random.choice([0, 1])
                annotation = create_sample_annotation(img_path, class_id)
                
                # Write annotation to file
                with open(dest_label_path, 'w') as f:
                    f.write(annotation + '\n')
                    
                    # Randomly add a second annotation for some images
                    if random.random() < 0.3:
                        second_class = random.choice([0, 1])
                        second_annotation = create_sample_annotation(img_path, second_class)
                        f.write(second_annotation + '\n')
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    print(f"Successfully created sample dataset in {output_dir}")

def create_yaml_config(output_dir):
    """Create YAML config file for YOLOv11."""
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    
    content = f"""# YOLOv11 dataset configuration
path: {output_dir}  # dataset root dir
train: {os.path.join(output_dir, 'train')}  # train images
val: {os.path.join(output_dir, 'val')}  # validation images
test: {os.path.join(output_dir, 'test')}  # test images

# Classes
nc: 2  # number of classes
names: ['helmet', 'no_helmet']  # class names
"""
    
    with open(yaml_path, 'w') as f:
        f.write(content)
    
    print(f"Created YAML configuration file at {yaml_path}")

def main():
    args = parse_args()
    
    print(f"Creating sample dataset with {args.num_images} images")
    
    # Create directory structure
    output_dir = args.output_dir
    create_directory_structure(output_dir)
    
    # Find random images
    images = find_random_images(args.source_dir, args.num_images)
    
    # Process images
    process_images(images, output_dir)
    
    # Create YAML config
    create_yaml_config(output_dir)
    
    print("Sample dataset creation completed successfully!")
    print("WARNING: This dataset uses randomly generated annotations and should only be used for testing the pipeline.")

if __name__ == "__main__":
    main() 