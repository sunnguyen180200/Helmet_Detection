"""
Dataset utilities for YOLOv11 helmet detection project.
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from .utils import xywh_to_xyxy

class HelmetDataset(Dataset):
    """
    Dataset class for helmet detection using YOLO format annotations.
    This class is provided for reference and custom training, though Ultralytics
    library handles dataset loading automatically.
    """
    
    def __init__(self, img_dir, label_dir=None, img_size=640, transform=None, is_train=True):
        """
        Initialize the dataset.
        
        Args:
            img_dir (str): Directory containing images
            label_dir (str, optional): Directory containing labels. If None, assume same as img_dir
            img_size (int): Target image size
            transform (callable, optional): Optional transform to be applied on a sample
            is_train (bool): Whether this is a training dataset
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir) if label_dir else self.img_dir.parent / 'labels'
        self.img_size = img_size
        self.transform = transform
        self.is_train = is_train
        
        # Get all image files
        self.img_files = sorted([f for f in self.img_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        
        # Check for empty dataset
        if len(self.img_files) == 0:
            raise ValueError(f"No images found in {img_dir}")
            
        # Map classes
        self.class_dict = {
            0: 'motorcyclist_with_helmet',
            1: 'motorcyclist_without_helmet'
        }
    
    def __len__(self):
        """Return the length of the dataset."""
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, targets, image_path, shapes)
                image (torch.Tensor): Image tensor (C, H, W)
                targets (torch.Tensor): Targets tensor (n_objects, 6) - [batch_idx, class, x, y, w, h]
                image_path (str): Path to the image
                shapes (tuple): Original image shape
        """
        # Load image
        img_path = self.img_files[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Could not read image {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get original shape
        h0, w0 = img.shape[:2]
        
        # Resize image, maintaining aspect ratio
        r = self.img_size / max(h0, w0)
        if r != 1:
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        h, w = img.shape[:2]
        
        # Get label path
        label_path = self.label_dir / f"{img_path.stem}.txt"
        
        # Initialize targets
        labels = []
        
        # Load labels if exists
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        # YOLO format: class, x_center, y_center, width, height (normalized)
                        cls = int(parts[0])
                        bbox = [float(x) for x in parts[1:5]]
                        
                        # Add to labels list (batch_idx will be added later)
                        labels.append([cls] + bbox)
        
        # Convert to numpy array
        labels = np.array(labels)
        
        # Apply transformations if available
        if self.transform:
            img, labels = self.transform(img, labels)
        
        # Convert to tensor and normalize to 0-1
        img = img.transpose((2, 0, 1))  # (H,W,C) -> (C,H,W)
        img = np.ascontiguousarray(img) / 255.0
        img = torch.from_numpy(img).float()
        
        # Create targets tensor
        targets = torch.zeros((0, 6))  # (batch_idx, class, x, y, w, h)
        if len(labels):
            # Add batch index column
            batch_idx = torch.zeros((len(labels), 1))
            labels_with_batch = np.hstack((batch_idx, labels))
            targets = torch.from_numpy(labels_with_batch).float()
        
        return img, targets, str(img_path), (h0, w0)
    
    def get_image_info(self, idx):
        """Get image info without loading the image."""
        img_path = self.img_files[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls = int(parts[0])
                        bbox = [float(x) for x in parts[1:5]]
                        labels.append([cls] + bbox)
        
        return {
            'image_path': str(img_path),
            'label_path': str(label_path),
            'labels': labels,
            'class_names': [self.class_dict.get(label[0], f"Class {label[0]}") for label in labels]
        }
    
    def visualize_sample(self, idx):
        """
        Visualize a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            numpy.ndarray: Image with drawn bounding boxes
        """
        img_path = self.img_files[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w = img.shape[:2]
        
        label_path = self.label_dir / f"{img_path.stem}.txt"
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls = int(parts[0])
                        bbox = [float(x) for x in parts[1:5]]
                        
                        # Convert normalized YOLO bbox to pixel coordinates
                        x1, y1, x2, y2 = xywh_to_xyxy(bbox, w, h)
                        
                        # Draw bounding box
                        color = (0, 255, 0) if cls == 0 else (0, 0, 255)  # Green for with helmet, Red for without
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        class_name = self.class_dict.get(cls, f"Class {cls}")
                        cv2.putText(img, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img 