#!/usr/bin/env python3
"""
Script to prepare YOLOv11 dataset with folder names included in filenames.
This runs the optimized data preparation script with the correct parameters.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare YOLOv11 dataset')
    parser.add_argument('--output_dir', type=str, default='./data/yolov11_dataset',
                        help='Output directory for the prepared dataset')
    parser.add_argument('--method', type=str, choices=['standard', 'batch'], default='batch',
                        help='Method to use for preparation: standard or batch')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Maximum number of samples to process (0 for all)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for batch processing method')
    parser.add_argument('--memory_limit', action='store_true',
                        help='Use memory optimization (for standard method only)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Fixed paths
    data_dir = "D:/2025/yolo/main_src/data/part_1/part_1"
    annotation_dir = "D:/2025/yolo/main_src/data/annotation"
    
    # Verify directories exist
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return 1
    
    if not os.path.exists(annotation_dir):
        print(f"Error: Annotation directory not found: {annotation_dir}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the appropriate script
    if args.method == 'standard':
        # Run standard preparation script
        cmd = [
            "python", "scripts/prepare_data.py",
            "--output_dir", args.output_dir,
            "--max_samples", str(args.max_samples)
        ]
        
        if args.memory_limit:
            cmd.append("--memory_limit")
            
        print(f"Running standard preparation with command: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    else:  # batch method
        # Run batch preparation script
        cmd = [
            "python", "scripts/batch_prepare_data.py",
            "--output_dir", args.output_dir,
            "--batch_size", str(args.batch_size),
            "--max_samples", str(args.max_samples)
        ]
        
        print(f"Running batch preparation with command: {' '.join(cmd)}")
        subprocess.run(cmd)
    
    # Verify dataset creation
    train_images_dir = os.path.join(args.output_dir, 'train', 'images')
    if os.path.exists(train_images_dir):
        num_images = len([f for f in os.listdir(train_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"Successfully created dataset with {num_images} training images")
        print(f"Dataset is ready at: {args.output_dir}")
        return 0
    else:
        print("Error: Failed to create dataset")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 