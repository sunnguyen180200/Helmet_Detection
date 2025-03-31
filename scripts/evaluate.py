#!/usr/bin/env python3
"""
Evaluate a trained YOLOv11 model on helmet detection test data.
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

import torch
import numpy as np
from ultralytics import YOLO

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils import setup_logging

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate YOLOv11 model for helmet detection')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--data', type=str, default='../data/dataset.yaml',
                        help='Path to dataset YAML file')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for evaluation')
    parser.add_argument('--device', type=str, default='',
                        help='Device to use (cuda device, i.e. 0 or cpu)')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='Confidence threshold for detections')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--output', type=str, default='./results',
                        help='Directory to save evaluation results')
    return parser.parse_args()

def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging('evaluate')
    logger.info(f"Evaluating model: {args.model}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the model
    try:
        model = YOLO(args.model)
        logger.info(f"Model loaded successfully: {args.model}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Set evaluation device
    device = args.device if args.device else ('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset configuration
    try:
        with open(args.data, 'r') as f:
            data_cfg = yaml.safe_load(f)
        logger.info(f"Dataset configuration loaded from: {args.data}")
    except Exception as e:
        logger.error(f"Error loading dataset configuration: {e}")
        return
    
    # Run validation
    try:
        results = model.val(
            data=args.data,
            batch=args.batch_size,
            imgsz=args.imgsz,
            device=device,
            conf=args.conf_thres,
            iou=args.iou_thres,
            plots=True,
            save_json=True,
            save_dir=str(output_dir)
        )
        
        # Log results
        metrics = results.results_dict
        logger.info("Evaluation Results:")
        logger.info(f"mAP@0.5: {metrics.get('metrics/mAP50(B)', 0):.4f}")
        logger.info(f"mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        logger.info(f"Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
        logger.info(f"Recall: {metrics.get('metrics/recall(B)', 0):.4f}")
        
        # Save results to file
        with open(output_dir / 'results.txt', 'w') as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
        
        logger.info(f"Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

if __name__ == '__main__':
    main() 