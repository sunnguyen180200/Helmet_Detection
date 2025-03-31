@echo off
echo Starting YOLOv11 dataset preparation...

REM Check if source directories exist
if not exist "D:\2025\yolo\part_1\part_1" (
    echo ERROR: Source data directory not found.
    exit /b 1
)

if not exist "D:\2025\yolo\main_src\data\annotation" (
    echo ERROR: Annotation directory not found.
    exit /b 1
)

REM Create the output directory
set OUTPUT_DIR=data\yolov11_dataset
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "%OUTPUT_DIR%\train\images" mkdir "%OUTPUT_DIR%\train\images"
if not exist "%OUTPUT_DIR%\train\labels" mkdir "%OUTPUT_DIR%\train\labels" 
if not exist "%OUTPUT_DIR%\val\images" mkdir "%OUTPUT_DIR%\val\images"
if not exist "%OUTPUT_DIR%\val\labels" mkdir "%OUTPUT_DIR%\val\labels"
if not exist "%OUTPUT_DIR%\test\images" mkdir "%OUTPUT_DIR%\test\images"
if not exist "%OUTPUT_DIR%\test\labels" mkdir "%OUTPUT_DIR%\test\labels"

echo Created directory structure in %OUTPUT_DIR%

REM Create dataset.yaml file
echo # YOLOv11 dataset configuration > "%OUTPUT_DIR%\dataset.yaml"
echo path: %OUTPUT_DIR% >> "%OUTPUT_DIR%\dataset.yaml"
echo train: %OUTPUT_DIR%/train >> "%OUTPUT_DIR%\dataset.yaml"
echo val: %OUTPUT_DIR%/val >> "%OUTPUT_DIR%\dataset.yaml"
echo test: %OUTPUT_DIR%/test >> "%OUTPUT_DIR%\dataset.yaml"
echo. >> "%OUTPUT_DIR%\dataset.yaml"
echo # Classes >> "%OUTPUT_DIR%\dataset.yaml"
echo nc: 2 >> "%OUTPUT_DIR%\dataset.yaml"
echo names: ['helmet', 'no_helmet'] >> "%OUTPUT_DIR%\dataset.yaml"

echo Created dataset.yaml configuration file

REM First, try running our optimized script
echo Trying to prepare dataset with optimized script...
python scripts/simple_prepare_data.py --data_dir "D:\2025\yolo\part_1\part_1" --annotation_dir "D:\2025\yolo\main_src\data\annotation" --output_dir "%OUTPUT_DIR%" --max_samples 2000

REM If we have images, we're done
if exist "%OUTPUT_DIR%\train\images\*.jpg" (
    echo Dataset preparation completed successfully.
    exit /b 0
)

REM If not, let's use the fallback method - manual copy
echo Using manual file copying method...

REM Process Bago_highway_1 folder as an example
echo Processing Bago_highway_1 folder...
copy "D:\2025\yolo\part_1\part_1\Bago_highway_1\*.jpg" "%OUTPUT_DIR%\train\images\"

REM Create corresponding label files
python -c "import os, random; files = [f for f in os.listdir('%OUTPUT_DIR%\\train\\images') if f.endswith('.jpg')]; for file in files: with open('%OUTPUT_DIR%\\train\\labels\\' + os.path.splitext(file)[0] + '.txt', 'w') as f: f.write('0 ' + ' '.join([str(random.uniform(0.2, 0.8)) for _ in range(4)]) + '\n')"

echo Manual dataset preparation completed.
echo IMPORTANT: These are placeholder annotations. You should replace them with real annotations.

echo Done. 