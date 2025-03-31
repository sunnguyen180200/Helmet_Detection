#!/usr/bin/env python3
"""
Final data preparation script for YOLOv11 training.
This version focuses on reliability and fault tolerance.
"""

import os
import sys
import json
import shutil
import random
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data_preparation.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare data for YOLOv11 training')
    parser.add_argument('--source_dir', type=str, default='D:/2025/yolo/part_1/part_1',
                        help='Path to the directory containing images')
    parser.add_argument('--annotation_dir', type=str, default='D:/2025/yolo/main_src/data/annotation',
                        help='Path to the directory containing COCO annotation files')
    parser.add_argument('--output_dir', type=str, default='./data/yolov11',
                        help='Path to the output directory for prepared dataset')
    parser.add_argument('--validate_only', action='store_true',
                        help='Only validate the dataset without copying files')
    parser.add_argument('--num_samples', type=int, default=0,
                        help='Number of samples to process (0 for all)')
    parser.add_argument('--safe_mode', action='store_true',
                        help='Use safe mode with more error checking and handling')
    return parser.parse_args()

def create_directory_structure(output_dir):
    """Create YOLO directory structure."""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    directories = {}
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            path = Path(output_dir) / split / subdir
            path.mkdir(exist_ok=True, parents=True)
            directories[f"{split}_{subdir}"] = path
            logger.info(f"Created directory: {path}")
    
    return directories

def find_images(source_dir):
    """Find all image files in the source directory."""
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    
    # Walk through the directory recursively
    for ext in image_extensions:
        pattern = f"**/*{ext}"
        image_files.extend([str(p) for p in Path(source_dir).glob(pattern)])
    
    logger.info(f"Found {len(image_files)} image files in {source_dir}")
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
            category_id = ann['category_id'] - 1  # Adjust to 0-indexed
            
            filename_to_annotations[filename].append({
                'class_id': category_id,
                'bbox': [center_x, center_y, norm_w, norm_h]
            })
        
        return filename_to_annotations
    
    except Exception as e:
        logger.error(f"Error loading annotation file {file_path}: {e}")
        return {}

def load_all_annotations(annotation_dir):
    """Load all annotation files and merge them."""
    annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith('.json')]
    logger.info(f"Found {len(annotation_files)} annotation files in {annotation_dir}")
    
    all_annotations = {}
    total_annotations = 0
    
    for file in annotation_files:
        file_path = os.path.join(annotation_dir, file)
        logger.info(f"Processing annotation file: {file}")
        
        file_annotations = load_annotation_file(file_path)
        
        # Count annotations
        for filename, annotations in file_annotations.items():
            total_annotations += len(annotations)
        
        # Merge with main dictionary
        all_annotations.update(file_annotations)
    
    logger.info(f"Loaded annotations for {len(all_annotations)} unique images with {total_annotations} total annotations")
    return all_annotations

def match_images_with_annotations(image_files, annotations):
    """Match image files with their annotations."""
    matched_data = {}
    
    for image_path in image_files:
        filename = os.path.basename(image_path)
        if filename in annotations and annotations[filename]:
            matched_data[image_path] = annotations[filename]
    
    logger.info(f"Matched {len(matched_data)} images with annotations")
    return matched_data

def split_dataset(matched_data):
    """Split dataset into train, validation, and test sets."""
    image_paths = list(matched_data.keys())
    random.shuffle(image_paths)
    
    # Calculate split indices (70% train, 15% val, 15% test)
    train_idx = int(len(image_paths) * 0.7)
    val_idx = int(len(image_paths) * 0.85)
    
    train_paths = image_paths[:train_idx]
    val_paths = image_paths[train_idx:val_idx]
    test_paths = image_paths[val_idx:]
    
    logger.info(f"Split dataset into {len(train_paths)} training, {len(val_paths)} validation, {len(test_paths)} test images")
    
    return {
        'train': train_paths,
        'val': val_paths,
        'test': test_paths
    }

def process_split(image_paths, annotations, directories, split_name, validate_only=False):
    """Process images for a specific split."""
    images_dir = directories[f"{split_name}_images"]
    labels_dir = directories[f"{split_name}_labels"]
    
    success_count = 0
    error_count = 0
    
    logger.info(f"Processing {len(image_paths)} images for {split_name} split")
    
    for image_path in tqdm(image_paths, desc=f"Processing {split_name}"):
        try:
            filename = os.path.basename(image_path)
            basename = os.path.splitext(filename)[0]
            
            # Destination paths
            dest_image_path = os.path.join(images_dir, filename)
            dest_label_path = os.path.join(labels_dir, f"{basename}.txt")
            
            if not validate_only:
                # Copy image
                shutil.copy2(image_path, dest_image_path)
                
                # Write label file
                with open(dest_label_path, 'w') as f:
                    for ann in annotations[filename]:
                        class_id = ann['class_id']
                        bbox = ann['bbox']
                        f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
            
            success_count += 1
            
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing {image_path}: {e}")
            
            # If more than 10% of files fail, abort
            if error_count > len(image_paths) * 0.1:
                logger.error(f"Too many errors ({error_count}), aborting {split_name} processing")
                break
    
    logger.info(f"Successfully processed {success_count}/{len(image_paths)} images for {split_name} split")
    return success_count

def create_yaml_config(output_dir):
    """Create YAML configuration file for YOLOv11."""
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    
    # Define dataset paths
    train_path = os.path.join(output_dir, 'train')
    val_path = os.path.join(output_dir, 'val')
    test_path = os.path.join(output_dir, 'test')
    
    # Create YAML content
    yaml_content = f"""# YOLOv11 dataset configuration
path: {output_dir}
train: {train_path}
val: {val_path}
test: {test_path}

# Classes
nc: 2
names: ['helmet', 'no_helmet']
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    logger.info(f"Created dataset YAML configuration at {yaml_path}")
    return yaml_path

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    logger.info(f"Starting YOLOv11 dataset preparation")
    logger.info(f"Source directory: {args.source_dir}")
    logger.info(f"Annotation directory: {args.annotation_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Validate directories
    if not os.path.exists(args.source_dir):
        logger.error(f"Source directory does not exist: {args.source_dir}")
        return 1
    
    if not os.path.exists(args.annotation_dir):
        logger.error(f"Annotation directory does not exist: {args.annotation_dir}")
        return 1
    
    # Create output directories
    directories = create_directory_structure(args.output_dir)
    
    # Find all image files
    image_files = find_images(args.source_dir)
    
    # Limit number of samples if specified
    if args.num_samples > 0 and args.num_samples < len(image_files):
        logger.info(f"Limiting to {args.num_samples} samples")
        image_files = image_files[:args.num_samples]
    
    # Load annotations
    annotations = load_all_annotations(args.annotation_dir)
    
    # Match images with annotations
    matched_data = match_images_with_annotations(image_files, annotations)
    
    if not matched_data:
        logger.error("No images with annotations found. Check source and annotation directories.")
        return 1
    
    # Split dataset
    splits = split_dataset(matched_data)
    
    # Process each split
    total_count = 0
    for split_name, image_paths in splits.items():
        success_count = process_split(
            image_paths,
            annotations,
            directories,
            split_name,
            validate_only=args.validate_only
        )
        total_count += success_count
    
    # Create YAML config
    if not args.validate_only:
        yaml_path = create_yaml_config(args.output_dir)
    
    logger.info(f"Dataset preparation completed successfully: {total_count} images processed")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 