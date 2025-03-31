# YOLOv11 Dataset Preparation Update

This document describes the updates made to the YOLOv11 dataset preparation scripts to include folder names in the saved image and label filenames.

## Changes Made

1. **Updated prepare_data.py**:
   - Now uses hardcoded paths: 
     - `data_dir = D:/2025/yolo/main_src/data/part_1/part_1`
     - `annotation_dir = D:/2025/yolo/main_src/data/annotation`
   - Modified `process_dataset()` function to include folder names in saved filenames
   - Added error handling for processing individual images

2. **Updated batch_prepare_data.py**:
   - Added hardcoded paths similar to prepare_data.py
   - Added `get_folder_name()` function to extract folder names from image paths
   - Modified `process_batch()` function to include folder names in saved filenames

3. **Created prepare_yolov11_dataset.py**:
   - A simple wrapper script that runs either the standard or batch preparation method
   - Uses the correct hardcoded paths
   - Provides options for memory optimization and limiting the number of samples

## Naming Convention

Images and labels are now saved with the following naming conventions:

- **Images**: `{folder_name}_{original_filename}`
  - Example: `Bago_highway_15_11.jpg`

- **Labels**: `{folder_name}_{original_filename_without_extension}.txt`
  - Example: `Bago_highway_15_11.txt`

This ensures that:
1. Image and label files have a clear origin reference
2. There are no filename collisions when copying from different folders
3. The folder structure is preserved in the filename

## How to Run

```bash
# Run with batch processing (recommended for most cases)
python scripts/prepare_yolov11_dataset.py --output_dir ./data/yolov11_dataset --method batch

# Run with standard processing and memory optimization
python scripts/prepare_yolov11_dataset.py --output_dir ./data/yolov11_dataset --method standard --memory_limit

# Limit the number of samples for testing
python scripts/prepare_yolov11_dataset.py --output_dir ./data/yolov11_dataset --max_samples 1000
```

## Dataset Structure

The prepared dataset follows the standard YOLO format:

```
data/yolov11_dataset/
├── dataset.yaml        # Dataset configuration file
├── train/              # Training data
│   ├── images/         # Training images with folder names as prefixes
│   └── labels/         # Training labels with folder names as prefixes
├── val/                # Validation data
│   ├── images/         # Validation images with folder names as prefixes
│   └── labels/         # Validation labels with folder names as prefixes
└── test/               # Test data
    ├── images/         # Test images with folder names as prefixes
    └── labels/         # Test labels with folder names as prefixes
```

## Using the Dataset with YOLOv11

To use the prepared dataset with YOLOv11, point to the `dataset.yaml` file in your training configuration:

```bash
python train.py --data ./data/yolov11_dataset/dataset.yaml --cfg ./models/yolov11n.yaml --weights '' --batch-size 16
```

For fine-tuning from a pretrained model:

```bash
python train.py --data ./data/yolov11_dataset/dataset.yaml --cfg ./models/yolov11n.yaml --weights yolov11n.pt --batch-size 16
``` 