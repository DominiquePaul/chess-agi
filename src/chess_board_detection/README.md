# ChessBoard Corner Detection

This module provides YOLO-based detection of chessboard corners. It inherits from the shared `BaseYOLOModel` class and includes specialized functionality for detecting exactly 4 corners of a chessboard.

## Features

- ✅ **Corner Detection**: Detects chessboard corners using YOLO object detection
- ⚠️ **Validation**: Warns when not exactly 4 corners are detected
- 📍 **Corner Ordering**: Orders corners consistently (top-left, top-right, bottom-right, bottom-left)
- 🎯 **Coordinate Extraction**: Returns precise corner coordinates with confidence scores
- 📊 **Visualization**: Plots corner detections with optional polygon overlay
- 🤗 **HuggingFace Integration**: Upload/download models from HuggingFace Hub

## Quick Start

### 1. Download Dataset
```bash
python -m src.chess_board_detection.download_data
```

### 2. Train Model
```bash
python -m src.chess_board_detection.train
```

### 3. Test Model
```bash
python -m src.chess_board_detection.inference_example
```

## Usage

### Basic Corner Detection
```python
from src.chess_board_detection import ChessBoardModel
from pathlib import Path

# Load trained model
model = ChessBoardModel(model_path=Path("models/best.pt"))

# Detect corners
results, corner_count, is_valid = model.predict_corners("image.jpg", conf=0.25)

# Get corner coordinates
coordinates, is_valid = model.get_corner_coordinates("image.jpg", conf=0.25)

# Visualize results
model.plot_eval("image.jpg", conf=0.25, show_polygon=True)
```

### Corner Ordering
```python
# Get ordered corners (top-left, top-right, bottom-right, bottom-left)
coordinates, is_valid = model.get_corner_coordinates("image.jpg")
if is_valid:
    ordered_corners = model.order_corners(coordinates)
    for i, corner in enumerate(ordered_corners):
        print(f"Corner {i+1}: ({corner['x']}, {corner['y']})")
```

### HuggingFace Integration
```python
# Download model from HuggingFace
model = ChessBoardModel.from_huggingface("username/chessboard-corners")

# Upload trained model
model.push_to_huggingface("username/my-chessboard-model")
```

## Dataset

The module is configured to work with the Roboflow dataset:
- **Project**: `gustoguardian/chess-board-box`
- **Version**: 3
- **Format**: YOLOv8

## Files

- `model.py` - Main ChessBoardModel class
- `train.py` - Training script
- `download_data.py` - Roboflow dataset downloader
- `inference_example.py` - Usage examples and testing
- `__init__.py` - Module initialization

## Warning System

The model automatically warns users when corner detection results are unexpected:

- ⚠️ **Too few corners**: "Only X corners detected (expected 4). Try lowering confidence threshold."
- ⚠️ **Too many corners**: "X corners detected (expected 4). Try raising confidence threshold."
- ✅ **Perfect detection**: "Successfully detected 4 chessboard corners"

## Integration with BaseYOLOModel

This module inherits all functionality from `BaseYOLOModel`, including:
- Training with extensive customization
- Model evaluation and metrics
- HuggingFace Hub integration
- Device management (CPU/GPU)
- Prediction and visualization utilities
