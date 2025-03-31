# YOLOv11 Helmet Detection

This project uses YOLOv11 with Ultralytics to detect motorcyclists without helmets.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset with motorcycle riders with/without helmets
3. Configure training parameters
4. Train the model
5. Evaluate and deploy

## Project Structure

- `data/`: Dataset directory
- `models/`: Trained models
- `scripts/`: Training and utility scripts
- `src/`: Source code
- `configs/`: Configuration files

## Training

```bash
python scripts/train.py --config configs/model_config.yaml
```

## Inference

```bash
python scripts/predict.py --model models/best.pt --source path/to/image
```

## License

[Specify your license]

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 implementation
- [YOLOv11 architecture](https://github.com/ultralytics/ultralytics) (when available) 