#!/usr/bin/env python3
"""
Train YOLOv11 model for helmet detection using Ultralytics.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

import torch
from ultralytics import YOLO

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils import setup_logging, seed_everything

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLOv11 model for helmet detection')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed for reproducibility
    seed_everything(42)
    
    
    # Setup logging
    logger = setup_logging('train')
    logger.info(f"Starting training with config: {args.config}")
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    try:
        # Try YOLOv11 when available
        model = YOLO(config['model'])
        logger.info(f"Using model: {config['model']}")
    except Exception as e:
        # Fallback to latest available YOLO model
        logger.warning(f"YOLOv11 not available yet, using YOLOv8 instead: {e}")
        model = YOLO('yolov8n.pt')
    
    # Set training device
    device = config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path('./models') / config.get('project', 'helmet_detection') / config.get('name', 'yolov11')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train the model
    try:
        model.train(
            data=config['data'],
            epochs=config.get('epochs', 100),
            patience=config.get('patience', 50),
            batch=config.get('batch_size', 16),
            imgsz=config.get('imgsz', 640),
            device=device,
            workers=config.get('workers', 8),
            project=str(output_dir.parent),
            name=output_dir.name,
            exist_ok=True,
            pretrained=True,
            amp=config.get('amp', True),
            lr0=config.get('lr0', 0.01),
            lrf=config.get('lrf', 0.01),
            momentum=config.get('momentum', 0.937),
            weight_decay=config.get('weight_decay', 0.0005),
            warmup_epochs=config.get('warmup_epochs', 3.0),
            warmup_momentum=config.get('warmup_momentum', 0.8),
            warmup_bias_lr=config.get('warmup_bias_lr', 0.1),
            save=config.get('save', True),
            save_period=config.get('save_period', 10),
            resume=args.resume,
        )
        logger.info("Training completed successfully.")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

if __name__ == '__main__':
    main() 