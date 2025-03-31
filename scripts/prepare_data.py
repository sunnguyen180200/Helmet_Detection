#!/usr/bin/env python3
"""
Prepare and process data for helmet detection training.
"""

import argparse
import os
import sys
import yaml
import json
import shutil
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils import setup_logging

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare data for YOLO format training.')
    parser.add_argument('--data_dir', type=str, default='D:/2025/yolo/main_src/data/part_1/part_1',
                        help='Path to the directory containing the image data')
    parser.add_argument('--annotation_dir', type=str, default='D:/2025/yolo/main_src/data/annotation',
                        help='Path to the directory containing COCO annotation files')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Path to the output directory to save prepared data')
    parser.add_argument('--split', type=str, default='0.7,0.15,0.15',
                        help='Split ratios for train, validation, and test sets')
    parser.add_argument('--augment', action='store_true',
                        help='Apply data augmentation to training data')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Maximum number of samples to process (0 for all)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')
    parser.add_argument('--max_image_size', type=int, default=1280,
                        help='Maximum image size (larger images will be resized)')
    parser.add_argument('--resize_all', action='store_true',
                        help='Resize all images to the maximum image size')
    parser.add_argument('--skip_large_images', action='store_true',
                        help='Skip images larger than max_image_size instead of resizing')
    parser.add_argument('--memory_limit', action='store_true',
                        help='Process images with memory optimizations (slower but safer)')
    return parser.parse_args()

def create_directory_structure(output_dir: Path) -> Dict[str, Path]:
    """Create the required directory structure for YOLO dataset."""
    directories = {
        'train_images': output_dir / 'processed' / 'train' / 'images',
        'train_labels': output_dir / 'processed' / 'train' / 'labels',
        'val_images': output_dir / 'processed' / 'val' / 'images',
        'val_labels': output_dir / 'processed' / 'val' / 'labels',
        'test_images': output_dir / 'processed' / 'test' / 'images',
        'test_labels': output_dir / 'processed' / 'test' / 'labels'
    }
    
    # Ensure output_dir exists first
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create each directory in the structure
    for dir_path in directories.values():
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")
        except Exception as e:
            print(f"Error creating directory {dir_path}: {e}")
    
    return directories

def load_annotations(annotation_dir: Path, logger) -> Dict[str, Dict]:
    """Load annotation files in COCO format and organize them by image file."""
    annotation_files = list(annotation_dir.glob('*.json'))
    logger.info(f"Found {len(annotation_files)} annotation files in {annotation_dir}")
    
    image_annotations = {}
    total_annotations = 0
    total_images = 0
    
    for annotation_file in annotation_files:
        logger.info(f"Processing annotation file: {annotation_file.name}")
        
        try:
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            
            # Extract images and annotations
            images = {img['id']: img for img in data['images']}
            total_images += len(images)
            
            # Process annotations
            for ann in data['annotations']:
                img_id = ann['image_id']
                img_info = images.get(img_id)
                
                if not img_info:
                    continue
                
                filename = img_info['file_name']
                
                # Extract full path from relative path in annotation
                if '/' in filename:
                    # Extract the subdirectory and filename
                    parts = filename.split('/')
                    if len(parts) > 1:
                        subdir, img_name = parts[-2], parts[-1]
                        full_path = f"{subdir}/{img_name}"
                    else:
                        full_path = filename
                else:
                    full_path = filename
                
                if full_path not in image_annotations:
                    image_annotations[full_path] = {
                        'width': img_info['width'],
                        'height': img_info['height'],
                        'annotations': []
                    }
                
                # COCO bbox format: [x, y, width, height]
                bbox = ann['bbox']
                # Convert to YOLO format: [center_x, center_y, width, height] (normalized)
                x, y, w, h = bbox
                img_w, img_h = img_info['width'], img_info['height']
                
                # Normalize coordinates
                center_x = (x + w / 2) / img_w
                center_y = (y + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                
                # Category ID: 0 for helmet, 1 for no_helmet
                category_id = ann['category_id'] - 1  # Adjust to 0-indexed
                
                image_annotations[full_path]['annotations'].append({
                    'bbox': [center_x, center_y, norm_w, norm_h],
                    'category_id': category_id
                })
                
                total_annotations += 1
        
        except Exception as e:
            logger.error(f"Error processing annotation file {annotation_file.name}: {e}")
    
    logger.info(f"Loaded {total_images} images and {total_annotations} annotations from all annotation files")
    return image_annotations

def find_image_files(data_dir: Path, logger) -> List[Path]:
    """Find all image files recursively in data_dir."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(data_dir.glob(f'**/*{ext}')))
    
    logger.info(f"Found {len(image_files)} image files")
    if len(image_files) > 0:
        logger.info(f"Sample image paths:")
        for i in range(min(3, len(image_files))):
            logger.info(f"  - Sample image {i+1}: {image_files[i]}")
    
    return image_files

def match_images_with_annotations(image_files: List[Path], annotations: Dict[str, Dict], data_dir: Path, logger) -> Dict[Path, Dict]:
    """Match image files with their annotations."""
    matched_data = {}
    total_images = len(image_files)
    matched_count = 0
    
    for image_path in image_files:
        # Get relative path from data_dir
        rel_path = image_path.relative_to(data_dir) if image_path.is_relative_to(data_dir) else image_path.name
        rel_path_str = str(rel_path).replace('\\', '/')
        
        # Try different ways to match the image path with annotations
        annotation_key = None
        
        # Try 1: Full relative path
        if rel_path_str in annotations:
            annotation_key = rel_path_str
        
        # Try 2: Just filename with immediate parent directory
        if annotation_key is None:
            parts = rel_path_str.split('/')
            if len(parts) > 1:
                short_path = f"{parts[-2]}/{parts[-1]}"
                if short_path in annotations:
                    annotation_key = short_path
        
        # Try 3: Just filename
        if annotation_key is None:
            filename = image_path.name
            matching_keys = [k for k in annotations.keys() if k.endswith(f"/{filename}") or k == filename]
            if matching_keys:
                annotation_key = matching_keys[0]
        
        if annotation_key:
            matched_data[image_path] = annotations[annotation_key]
            matched_count += 1
    
    logger.info(f"Found {matched_count} valid images with annotations out of {total_images} images")
    return matched_data

def split_dataset(matched_data: Dict[Path, Dict], split_ratios: List[float], logger) -> Tuple[Dict[Path, Dict], Dict[Path, Dict], Dict[Path, Dict]]:
    """Split dataset into train, validation, and test sets."""
    # Get list of all image paths
    all_paths = list(matched_data.keys())
    
    # Shuffle the list
    random.shuffle(all_paths)
    
    # Calculate split indices
    train_end_idx = int(len(all_paths) * split_ratios[0])
    val_end_idx = train_end_idx + int(len(all_paths) * split_ratios[1])
    
    # Split dataset
    train_paths = all_paths[:train_end_idx]
    val_paths = all_paths[train_end_idx:val_end_idx]
    test_paths = all_paths[val_end_idx:]
    
    train_data = {path: matched_data[path] for path in train_paths}
    val_data = {path: matched_data[path] for path in val_paths}
    test_data = {path: matched_data[path] for path in test_paths}
    
    logger.info(f"Split into {len(train_data)} training, {len(val_data)} validation, and {len(test_data)} test images")
    
    return train_data, val_data, test_data

def get_augmentations():
    """Define data augmentations for training images."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomGamma(p=0.2),
        A.CLAHE(p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.1),
        A.GaussNoise(p=0.1),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def save_image_safe(image_path: Path, output_path: Path, annotations: Dict, 
                    max_size: int = 1280, resize_all: bool = False, 
                    skip_large: bool = False, memory_opt: bool = False, logger=None) -> bool:
    """Save image and its annotations with memory optimization options."""
    try:
        # Get image info
        if memory_opt:
            # Just get image dimensions without loading the full image
            # Use iminfo instead of IMREAD_HEADER_ONLY which doesn't exist in some OpenCV versions
            img_info = cv2.imread(str(image_path), cv2.IMREAD_REDUCED_COLOR_2)  # Load at reduced size
            if img_info is None:
                if logger:
                    logger.error(f"Could not read image {image_path}")
                return False
                
            # Get original dimensions from the reduced image
            h, w = img_info.shape[:2]
            h, w = h * 2, w * 2  # Multiply by 2 because we loaded at reduced resolution
        else:
            # Standard approach - load full image
            img = cv2.imread(str(image_path))
            if img is None:
                if logger:
                    logger.error(f"Could not read image {image_path}")
                return False
            h, w = img.shape[:2]
        
        # Check image size and decide if it needs resizing
        needs_resize = (max(h, w) > max_size) or resize_all
        
        if needs_resize and skip_large:
            if logger:
                logger.warning(f"Skipping large image {image_path} ({w}x{h})")
            return False
        
        # Process image with resize if needed
        if needs_resize or not memory_opt:
            # We need to load the full image for resize or if we're not in memory optimization mode
            if memory_opt and needs_resize:
                # Load the full image now that we know we need to resize it
                img = cv2.imread(str(image_path))
                if img is None:
                    if logger:
                        logger.error(f"Could not read full image {image_path}")
                    return False
                h, w = img.shape[:2]
            
            if needs_resize:
                # Calculate new dimensions maintaining aspect ratio
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                img = cv2.resize(img, (new_w, new_h))
            
            # Save processed image
            cv2.imwrite(str(output_path), img)
        else:
            # Memory optimization - just copy the file
            shutil.copy2(str(image_path), str(output_path))
        
        # Process annotations
        # For resized images, annotations need adjustment only if the image was actually resized
        # (the YOLO format is already normalized)
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Error processing {image_path}: {e}")
        return False

def process_dataset(data: Dict[Path, Dict], dirs: Dict[str, Path], data_dir: Path, 
                    split_name: str, args, logger) -> int:
    """Process and save a dataset split."""
    # Set up augmentations if needed
    augmentations = get_augmentations() if args.augment and split_name == 'train' else None
    
    # Get directory paths
    images_dir = dirs[f'{split_name}_images']
    labels_dir = dirs[f'{split_name}_labels']
    
    logger.info(f"Processing {split_name} data...")
    success_count = 0
    
    for img_path, annotations in tqdm(data.items(), desc=f"Processing {split_name} set"):
        try:
            # Get folder name and filename
            rel_path = img_path.relative_to(data_dir) if img_path.is_relative_to(data_dir) else img_path.name
            folder_name = rel_path.parent.name if rel_path.parent != Path('.') else ""
            
            # Generate output paths with folder name as prefix
            if folder_name:
                img_filename = f"{folder_name}_{img_path.name}"
                label_filename = f"{folder_name}_{img_path.stem}.txt"
            else:
                img_filename = img_path.name
                label_filename = f"{img_path.stem}.txt"
            
            output_img_path = images_dir / img_filename
            output_label_path = labels_dir / label_filename
            
            # Process and save image
            success = save_image_safe(
                img_path, output_img_path, 
                annotations, 
                max_size=args.max_image_size, 
                resize_all=args.resize_all,
                skip_large=args.skip_large_images,
                memory_opt=args.memory_limit,
                logger=logger
            )
            
            if not success:
                continue
            
            # Save annotations in YOLO format
            with open(output_label_path, 'w') as f:
                for ann in annotations['annotations']:
                    bbox = ann['bbox']
                    class_id = ann['category_id']
                    f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
            
            success_count += 1
            
            # Apply augmentations for training set if enabled
            if augmentations and args.augment and split_name == 'train' and not args.memory_limit:
                try:
                    # Load image for augmentation
                    img = cv2.imread(str(output_img_path))
                    if img is None:
                        continue
                    
                    # Extract bounding boxes and labels
                    bboxes = [ann['bbox'] for ann in annotations['annotations']]
                    class_labels = [ann['category_id'] for ann in annotations['annotations']]
                    
                    # Skip if no annotations
                    if not bboxes:
                        continue
                    
                    # Apply augmentations
                    augmented = augmentations(image=img, bboxes=bboxes, class_labels=class_labels)
                    aug_img = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_labels = augmented['class_labels']
                    
                    # Save augmented image and labels with folder prefix
                    if folder_name:
                        aug_img_filename = f"{folder_name}_{img_path.stem}_aug.jpg"
                        aug_label_filename = f"{folder_name}_{img_path.stem}_aug.txt"
                    else:
                        aug_img_filename = f"{img_path.stem}_aug.jpg"
                        aug_label_filename = f"{img_path.stem}_aug.txt"
                    
                    aug_img_path = images_dir / aug_img_filename
                    aug_label_path = labels_dir / aug_label_filename
                    
                    cv2.imwrite(str(aug_img_path), aug_img)
                    
                    with open(aug_label_path, 'w') as f:
                        for box, label in zip(aug_bboxes, aug_labels):
                            f.write(f"{label} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
                    
                    success_count += 1
                except Exception as e:
                    logger.error(f"Error during augmentation for {img_path}: {e}")
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
    
    return success_count

def create_yaml_config(output_dir: Path, class_names: List[str], logger) -> Path:
    """Create a YAML configuration file for YOLOv11 training."""
    config_path = output_dir / 'dataset.yaml'
    
    # Get absolute paths for train, val, and test directories
    train_path = str(output_dir / 'processed' / 'train')
    val_path = str(output_dir / 'processed' / 'val')
    test_path = str(output_dir / 'processed' / 'test')
    
    # Create config dictionary
    config = {
        'path': str(output_dir),
        'train': train_path,
        'val': val_path,
        'test': test_path,
        'nc': len(class_names),
        'names': class_names
    }
    
    # Write config to YAML file
    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    logger.info(f"Created YAML configuration file at {config_path}")
    return config_path

def main():
    """Main function to prepare data."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging(name='prepare_data')
    
    # Convert paths to Path objects - use the specified paths
    data_dir = Path('D:/2025/yolo/main_src/data/part_1/part_1')
    annotation_dir = Path('D:/2025/yolo/main_src/data/annotation')
    output_dir = Path(args.output_dir)
    
    logger.info(f"Using specified data directory: {data_dir}")
    logger.info(f"Using specified annotation directory: {annotation_dir}")
    
    # Parse split ratios
    split_ratios = [float(x) for x in args.split.split(',')]
    if len(split_ratios) != 3 or sum(split_ratios) != 1.0:
        logger.error("Split ratios must be three numbers that sum to 1.0")
        sys.exit(1)
    
    # Log configuration
    logger.info(f"Preparing data from {data_dir} with annotations from {annotation_dir}")
    logger.info(f"Using split ratios: train={split_ratios[0]}, val={split_ratios[1]}, test={split_ratios[2]}")
    
    # Check if directories exist
    logger.info("Using absolute paths:")
    logger.info(f"  - Current working directory: {Path.cwd()}")
    logger.info(f"  - Script directory: {Path(__file__).resolve().parent}")
    logger.info(f"  - Project directory: {Path(__file__).resolve().parent.parent}")
    logger.info(f"  - Data directory: {data_dir.resolve()}")
    logger.info(f"  - Annotation directory: {annotation_dir.resolve()}")
    logger.info(f"  - Output directory: {output_dir.resolve()}")
    
    logger.info("Directory existence checks:")
    logger.info(f"  - Data directory exists: {data_dir.exists()}")
    logger.info(f"  - Annotation directory exists: {annotation_dir.exists()}")
    logger.info(f"  - Output directory exists: {output_dir.exists()}")
    
    if not data_dir.exists() or not annotation_dir.exists():
        logger.error("Data directory or annotation directory does not exist")
        sys.exit(1)
    
    # Create directory structure
    directories = create_directory_structure(output_dir)
    logger.info(f"Created directory structure in {output_dir}")
    
    # Verify directories were created
    for name, path in directories.items():
        logger.info(f"Directory '{name}' exists: {path.exists()}")
    
    # Load annotations
    annotations = load_annotations(annotation_dir, logger)
    
    # Find image files
    logger.info(f"Processing {args.max_samples if args.max_samples > 0 else 'all'} images from {data_dir}")
    image_files = find_image_files(data_dir, logger)
    if args.max_samples > 0:
        image_files = image_files[:args.max_samples]
    
    # Match images with annotations
    matched_data = match_images_with_annotations(image_files, annotations, data_dir, logger)
    
    # Split dataset
    train_data, val_data, test_data = split_dataset(matched_data, split_ratios, logger)
    
    # Process datasets
    train_count = process_dataset(train_data, directories, data_dir, 'train', args, logger)
    val_count = process_dataset(val_data, directories, data_dir, 'val', args, logger)
    test_count = process_dataset(test_data, directories, data_dir, 'test', args, logger)
    
    # Create YAML config file
    class_names = ['helmet', 'no_helmet']
    yaml_path = create_yaml_config(output_dir, class_names, logger)
    
    # Log summary
    logger.info("=== Data Preparation Summary ===")
    logger.info(f"Successfully processed:")
    logger.info(f"  - Training images: {train_count}")
    logger.info(f"  - Validation images: {val_count}")
    logger.info(f"  - Test images: {test_count}")
    logger.info(f"Total: {train_count + val_count + test_count}")
    logger.info(f"Dataset configuration: {yaml_path}")
    logger.info("Dataset preparation completed successfully")

if __name__ == "__main__":
    main()
