#!/usr/bin/env python3
"""
Run inference with a trained YOLOv11 model for helmet detection.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils import setup_logging

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference with YOLOv11 model for helmet detection')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image, video, or directory of images')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for inference')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='Confidence threshold for detections')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='',
                        help='Device to use (cuda device, i.e. 0 or cpu)')
    parser.add_argument('--output', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save results to *.txt file')
    parser.add_argument('--save-conf', action='store_true',
                        help='Save confidences in --save-txt labels')
    parser.add_argument('--view-img', action='store_true',
                        help='Show results')
    return parser.parse_args()

def main():
    """Main inference function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging('predict')
    logger.info(f"Running inference with model: {args.model} on source: {args.source}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    try:
        model = YOLO(args.model)
        logger.info(f"Model loaded successfully: {args.model}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Set device
    device = args.device if args.device else ('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Run inference
    try:
        results = model.predict(
            source=args.source,
            conf=args.conf_thres,
            iou=args.iou_thres,
            imgsz=args.imgsz,
            device=device,
            save=True,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            save_crop=False,
            show=args.view_img,
            project=str(output_dir.parent),
            name=output_dir.name,
            exist_ok=True
        )
        
        # Log results
        logger.info(f"Inference completed, results saved to {output_dir}")
        
        # Process and display results if requested
        if args.view_img:
            for r in results:
                boxes = r.boxes  # Boxes object for bbox outputs
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                scores = boxes.conf.cpu().numpy()
                
                logger.info(f"Detected {len(boxes)} objects in {r.path}")
                for i, score in enumerate(scores):
                    class_id = cls_ids[i]
                    class_name = model.names[class_id]
                    logger.info(f"  {class_name}: {score:.4f}")
    
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise

if __name__ == '__main__':
    main() 