"""
Data augmentation utilities for YOLOv11 helmet detection project.
"""

import cv2
import numpy as np
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

def create_train_transforms(img_size=640, augment=True):
    """
    Create training transformations pipeline using Albumentations.
    
    Args:
        img_size (int): Target image size
        augment (bool): Whether to use augmentations
        
    Returns:
        A.Compose: Albumentations composition of transforms
    """
    if not augment:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    return A.Compose([
        # Resize
        A.Resize(img_size, img_size),
        
        # Color augmentations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
        ], p=0.5),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.3),
        
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, 
                          border_mode=cv2.BORDER_CONSTANT, p=0.5),
        
        # Normalization
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def create_val_transforms(img_size=640):
    """
    Create validation transformations pipeline.
    
    Args:
        img_size (int): Target image size
        
    Returns:
        A.Compose: Albumentations composition of transforms
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def apply_augmentation(image, bboxes, class_labels, transforms=None):
    """
    Apply augmentations to image and bounding boxes.
    
    Args:
        image (numpy.ndarray): Input image
        bboxes (list): List of bounding boxes in YOLO format [x_center, y_center, width, height]
        class_labels (list): List of class labels for each bbox
        transforms (A.Compose, optional): Albumentations transforms to apply
        
    Returns:
        tuple: (augmented_image, augmented_bboxes, augmented_class_labels)
    """
    if transforms is None:
        transforms = create_train_transforms()
    
    # Apply transformations
    transformed = transforms(image=image, bboxes=bboxes, class_labels=class_labels)
    
    return transformed['image'], transformed['bboxes'], transformed['class_labels'] 