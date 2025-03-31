#!/usr/bin/env python3
"""
Convert COCO format annotations to YOLO format for helmet detection.
"""

import argparse
import json
import os
import sys
import shutil
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils import setup_logging

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert COCO format annotations to YOLO format')
    parser.add_argument('--coco-json', type=str, required=True,
                        help='Path to COCO format annotation JSON file')
    parser.add_argument('--image-dir', type=str, required=True,
                        help='Path to directory containing images')
    parser.add_argument('--output-dir', type=str, default='../data/raw',
                        help='Path to output directory for YOLO format annotations')
    parser.add_argument('--class-map', type=str, default='0:motorcyclist_with_helmet,1:motorcyclist_without_helmet',
                        help='Mapping from COCO category IDs to YOLO class IDs (format: coco_id:yolo_id,coco_id:yolo_id,...)')
    return parser.parse_args()

def coco_to_yolo_bbox(bbox, img_width, img_height):
    """
    Convert COCO bbox (x, y, width, height) to YOLO format (x_center, y_center, width, height).
    
    Args:
        bbox (list): COCO format [x, y, width, height]
        img_width (int): Image width
        img_height (int): Image height
        
    Returns:
        list: YOLO format [x_center, y_center, width, height] normalized
    """
    x, y, width, height = bbox
    
    # Convert to center coordinates
    x_center = x + width / 2
    y_center = y + height / 2
    
    # Normalize
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return [x_center, y_center, width, height]

def main():
    """Main function to convert COCO to YOLO format annotations."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging('coco_to_yolo')
    logger.info(f"Converting COCO annotations from {args.coco_json} to YOLO format")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse class mapping
    class_map = {}
    for mapping in args.class_map.split(','):
        coco_id, yolo_id = mapping.split(':')
        class_map[int(coco_id)] = int(yolo_id) if yolo_id.isdigit() else yolo_id
    
    logger.info(f"Using class mapping: {class_map}")
    
    # Load COCO annotations
    try:
        with open(args.coco_json, 'r') as f:
            coco_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading COCO annotations: {e}")
        return
    
    # Check necessary COCO components
    if not all(k in coco_data for k in ['images', 'annotations', 'categories']):
        logger.error("COCO JSON is missing required fields (images, annotations, or categories)")
        return
    
    # Create image ID to filename mapping
    image_id_to_name = {}
    image_dimensions = {}
    for image in coco_data['images']:
        image_id_to_name[image['id']] = image['file_name']
        image_dimensions[image['id']] = (image['width'], image['height'])
    
    # Create category ID to name mapping
    category_id_to_name = {}
    for category in coco_data['categories']:
        category_id_to_name[category['id']] = category['name']
    
    # Group annotations by image ID
    annotations_by_image = {}
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(annotation)
    
    # Process annotations for each image
    images_processed = 0
    annotations_processed = 0
    
    for image_id, annotations in tqdm(annotations_by_image.items(), desc="Processing images"):
        if image_id not in image_id_to_name:
            logger.warning(f"Image ID {image_id} not found in images list")
            continue
        
        image_name = image_id_to_name[image_id]
        img_width, img_height = image_dimensions[image_id]
        
        # Create YOLO annotation file path
        yolo_file = output_dir / Path(image_name).with_suffix('.txt').name
        
        with open(yolo_file, 'w') as f:
            for annotation in annotations:
                category_id = annotation['category_id']
                
                # Skip categories not in mapping
                if category_id not in class_map:
                    continue
                
                # Get YOLO class ID
                yolo_class_id = class_map[category_id]
                
                # Get bbox in COCO format [x, y, width, height]
                bbox = annotation['bbox']
                
                # Convert to YOLO format [x_center, y_center, width, height]
                yolo_bbox = coco_to_yolo_bbox(bbox, img_width, img_height)
                
                # Write annotation line: class_id x_center y_center width height
                f.write(f"{yolo_class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
                annotations_processed += 1
        
        # Copy image if not already in output directory
        src_img_path = Path(args.image_dir) / image_name
        dst_img_path = output_dir / image_name
        
        if not dst_img_path.exists() and src_img_path.exists():
            try:
                shutil.copy2(src_img_path, dst_img_path)
            except Exception as e:
                logger.error(f"Error copying image {src_img_path} to {dst_img_path}: {e}")
        
        images_processed += 1
    
    logger.info(f"Conversion completed: {images_processed} images, {annotations_processed} annotations processed")
    logger.info(f"YOLO format annotations saved to {output_dir}")

if __name__ == '__main__':
    main() 