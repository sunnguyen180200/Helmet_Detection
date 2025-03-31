"""
Utility functions for YOLOv11 helmet detection project.
"""

import os
import sys
import logging
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

def seed_everything(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def setup_logging(name='helmet_detection', level=logging.INFO):
    """Set up logging configuration."""
    log_dir = Path('./logs')
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{name}_{timestamp}.log'
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def check_gpu_availability():
    """Check if GPU is available and return device name."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        return f"cuda:0 ({device_name}, {device_count} devices)"
    return "cpu"

def get_model_info(model):
    """Get model information."""
    try:
        from thop import profile
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
        flops, params = profile(model, inputs=(img,), verbose=False)
        return {
            'params': params / 1e6,
            'flops': flops / 1e9,
            'layers': len(list(model.modules()))
        }
    except ImportError:
        return {
            'params': sum(x.numel() for x in model.parameters()) / 1e6,
            'layers': len(list(model.modules()))
        }
        
def draw_detection(image, boxes, scores, classes, class_names, color_map=None):
    """
    Draw detection bounding boxes on an image.
    
    Args:
        image (numpy.ndarray): The image to draw on
        boxes (list): List of bounding boxes in [x1, y1, x2, y2] format
        scores (list): List of confidence scores
        classes (list): List of class indices
        class_names (dict): Dict mapping class indices to names
        color_map (dict, optional): Dict mapping class indices to colors
    
    Returns:
        numpy.ndarray: Image with drawn bounding boxes
    """
    import cv2
    
    if color_map is None:
        # Default colors for classes
        color_map = {
            0: (0, 255, 0),    # Green for class 0 (with helmet)
            1: (0, 0, 255),    # Red for class 1 (without helmet)
        }
    
    # Make a copy of the image to avoid modifying the original
    result_image = image.copy()
    
    for box, score, cls in zip(boxes, scores, classes):
        # Get coordinates
        x1, y1, x2, y2 = [int(coord) for coord in box]
        
        # Get color for class
        color = color_map.get(cls, (255, 255, 0))  # Default yellow if class not in color_map
        
        # Get class name
        class_name = class_names.get(cls, f"Class {cls}")
        
        # Draw bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        text = f"{class_name} {score:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(result_image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
        
        # Draw text
        cv2.putText(result_image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result_image

def xywh_to_xyxy(bbox, img_width, img_height):
    """
    Convert bbox from YOLO format (center_x, center_y, width, height) to (x1, y1, x2, y2).
    
    Args:
        bbox (list): [center_x, center_y, width, height] in normalized coordinates
        img_width (int): Image width
        img_height (int): Image height
        
    Returns:
        list: [x1, y1, x2, y2] in pixel coordinates
    """
    center_x, center_y, width, height = bbox
    
    # Convert normalized to pixel coordinates
    center_x *= img_width
    center_y *= img_height
    width *= img_width
    height *= img_height
    
    # Convert center format to corner format
    x1 = int(center_x - width / 2)
    y1 = int(center_y - height / 2)
    x2 = int(center_x + width / 2)
    y2 = int(center_y + height / 2)
    
    return [x1, y1, x2, y2]

def xyxy_to_xywh(bbox, img_width, img_height):
    """
    Convert bbox from (x1, y1, x2, y2) to YOLO format (center_x, center_y, width, height).
    
    Args:
        bbox (list): [x1, y1, x2, y2] in pixel coordinates
        img_width (int): Image width
        img_height (int): Image height
        
    Returns:
        list: [center_x, center_y, width, height] in normalized coordinates
    """
    x1, y1, x2, y2 = bbox
    
    # Convert corner format to center format
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    
    # Convert pixel to normalized coordinates
    center_x /= img_width
    center_y /= img_height
    width /= img_width
    height /= img_height
    
    return [center_x, center_y, width, height] 