# Training a SO-100 robot to play chess

## Roadmap

- [ ] Chess piece detection works reliably
- [ ] Board detection works in new setting
- [ ] Board detection + piece detection works together
- [ ] Board position can be fed into chess engine and fetch next move
- [ ] User can collect data with dynamically marked points on board.
- [ ] Collected dataset of 250 samples
- [ ] Train ACT policy on dataset
- [ ] Control-loop for playing a game
- [ ] Needs to detect whether other player made a move
- [ ] Add optional voice over comment with the voice of Borat


# Setup

```
pip install uv
uv sync
```

## Get datasets

### Quick Start (Recommended)

```bash
# Download ready-to-use datasets from Hugging Face
python src/data_prep/download_from_hf.py --dataset merged

# Or download all datasets at once
python src/data_prep/prepare_and_upload_datasets.py
```

### 📖 Detailed Instructions

For comprehensive dataset options, API setup, troubleshooting, and advanced usage, see:

**[📋 Complete Dataset Guide →](src/data_prep/README.md)**

The detailed guide covers:
- Multiple download methods (Hugging Face, Roboflow, Kaggle)
- Chess piece detection datasets
- Chessboard corner detection datasets  
- API key setup and troubleshooting
- Dataset recreation from source
- Upload to Hugging Face Hub

## Important links:

- **Hugging Face Profile**: https://huggingface.co/dopaul
- **Trained Models**:
  - [Chess Board Segmentation](https://huggingface.co/dopaul/chess_board_segmentation) - YOLO segmentation model for precise board boundary detection
- **Chess Datasets**:
  - [Merged Dataset (Recommended)](https://huggingface.co/datasets/dopaul/chess-pieces-merged) - Combined dataset for training
  - [Dominique Dataset](https://huggingface.co/datasets/dopaul/chess-pieces-dominique) - Individual dataset from Roboflow
  - [Roboflow Dataset](https://huggingface.co/datasets/dopaul/chess-pieces-roboflow) - Processed dataset from Kaggle
- **Original Data Sources**:
  - [Kaggle Chess Pieces Dataset](https://www.kaggle.com/datasets/imtkaggleteam/chess-pieces-detection-image-dataset)
  - [Roboflow Chess Pieces Detection](https://universe.roboflow.com/gustoguardian/chess-piece-detection-bltvi/dataset/6)
  - [Roboflow Chessboard Corners](https://universe.roboflow.com/gustoguardian/chess-board-box/dataset/3)
  - [Roboflow Chessboard Segmentation](https://universe.roboflow.com/gustoguardian/chess-board-i0ptl/dataset/3)

## Using the models Training Models

### Quickstart for prediction 

[tbd]

### Training the models from scratch

Make sure that you've downloaded the data first. 

#### Chess Piece Detection

Train a YOLO object detection model to identify and classify chess pieces on board images.

```bash
# Basic training with default settings (YOLOv8s COCO pretrained)
python src/chess_piece_detection/train.py --epochs 100

# Choose different COCO-pretrained model size
python src/chess_piece_detection/train.py \
    --pretrained-model yolov8m.pt \
    --epochs 100

# Custom training parameters
python src/chess_piece_detection/train.py \
    --data data/chess_pieces_merged/data.yaml \
    --pretrained-model yolov8s.pt \
    --epochs 100 \
    --batch 16 \
    --imgsz 640

# Complete example with all options
python src/chess_piece_detection/train.py \
    --data data/chess_pieces_merged/data.yaml \
    --pretrained-model yolov8m.pt \
    --models-folder models/chess_piece_detection \
    --name training_v3 \
    --epochs 100 \
    --batch 32 \
    --imgsz 640 \
    --lr 0.001 \
    --patience 15 \
    --save-period 10 \
    --degrees 10.0 \
    --translate 0.1 \
    --scale 0.2 \
    --fliplr 0.5 \
    --mosaic 1.0 \
    --mixup 0.1 \
    --optimizer AdamW \
    --eval-individual \
    --push-to-hf \
    --verbose

# View all training options
python src/chess_piece_detection/train.py --help

# Available COCO-pretrained models:
#   - yolov8n.pt (nano, ~6MB, fastest)
#   - yolov8s.pt (small, ~22MB, fast, recommended)  
#   - yolov8m.pt (medium, ~52MB, balanced)
#   - yolov8l.pt (large, ~104MB, accurate)
#   - yolov8x.pt (extra large, ~136MB, most accurate)

# Run inference example
python -m src.chess_piece_detection.inference_example
```

#### Chessboard Detection (Corner Detection)

We first identify the chessboard corners which we then use to identify the borders with in a second step using classical CV or perspective transformation.

```bash
# Download corner detection dataset
export ROBOFLOW_API_KEY=your_api_key_here
python src/chess_board_detection/download_data.py

# Train corner detection model with default settings
python src/chess_board_detection/yolo/train.py

# Train with custom parameters and HuggingFace upload
python src/chess_board_detection/yolo/train.py \
    --data data/chessboard_corners/chess-board-box-3/data.yaml \
    --epochs 100 \
    --batch 32 \
    --upload-hf \
    --hf-model-name username/chessboard-detector

# View all training options
python src/chess_board_detection/train.py --help

# Test corner detection
python -m src.chess_board_detection.yolo.inference_example
```

#### Chessboard Segmentation (Polygon Detection)

For more precise chessboard boundary detection, we can use segmentation to get exact polygon coordinates.

```bash
# Download segmentation dataset
export ROBOFLOW_API_KEY=your_api_key_here
python src/chess_board_detection/download_data.py \
    --project gustoguardian/chess-board-i0ptl \
    --version 3 \
    --data-dir data/chessboard_segmentation

# Train segmentation model with default settings (YOLOv8n-seg)
python src/chess_board_detection/yolo/segmentation/train_segmentation.py \
    --data data/chessboard_segmentation/chess-board-3/data.yaml \
    --epochs 100

# Train with larger model for better accuracy
python src/chess_board_detection/yolo/segmentation/train_segmentation.py \
    --data data/chessboard_segmentation/chess-board-3/data.yaml \
    --pretrained-model yolov8s-seg.pt \
    --epochs 100 \
    --batch 16 \
    --name polygon_segmentation_training

# View all training options
python src/chess_board_detection/yolo/segmentation/train_segmentation.py --help

# Test segmentation model
python src/chess_board_detection/yolo/segmentation/test_segmentation.py \
    --model artifacts/models/chess_board_segmentation/polygon_segmentation_training/weights/best.pt \
    --image path/to/test_image.jpg
```

#### Upload Models to Hugging Face

Once you've trained your models, you can easily upload them to Hugging Face Hub for sharing and deployment.

```bash
# First, login to Hugging Face
huggingface-cli login

# Upload corner detection model
python src/chess_board_detection/upload_hf.py \
    --model models/chess_board_detection/corner_detection_training/weights/best.pt \
    --repo-name yourusername/chess-corner-detector \
    --model-task corner-detection

# Upload segmentation model
python src/chess_board_detection/upload_hf.py \
    --model artifacts/models/chess_board_segmentation/polygon_segmentation_training/weights/best.pt \
    --repo-name yourusername/chess-segmentation \
    --model-task segmentation

# Upload with custom metadata and training logs
python src/chess_board_detection/upload_hf.py \
    --model artifacts/models/chess_board_segmentation/training/weights/best.pt \
    --repo-name yourusername/chess-board-segmentation \
    --model-task segmentation \
    --description "High-accuracy YOLO segmentation model for chess board polygon detection" \
    --tags computer-vision,chess,yolo,segmentation,polygon-detection \
    --license mit \
    --include-training-dir \
    --verbose

# Test upload without actually uploading
python src/chess_board_detection/upload_hf.py \
    --model path/to/model.pt \
    --repo-name yourusername/model-name \
    --model-task segmentation \
    --dry-run
```

## Usage Examples

### Chess Board Analysis CLI

Analyze chess board images from the command line using the comprehensive analyzer:

```bash
# Basic analysis - detect corners and pieces
python examples/analyze_chess_board.py --image data/eval_images/chess_4.jpeg

# Get pixel coordinates for specific squares (useful for robotics)
python examples/analyze_chess_board.py --image chess.jpg --squares e4,d4,a1,h8

# Corners-only mode (skip piece detection)
python examples/analyze_chess_board.py --image chess.jpg --corners-only

# Verbose output with detailed analysis
python examples/analyze_chess_board.py --image chess.jpg --verbose --output results/

# JSON output for programmatic use
python examples/analyze_chess_board.py --image chess.jpg --json

# Custom models and confidence threshold
python examples/analyze_chess_board.py --image chess.jpg --conf 0.7 --segmentation-model path/to/model.pt
```

The CLI provides:
- 🎯 **Corner Detection**: Uses segmentation for precise board boundary detection
- ♟️ **Piece Detection**: Identifies and classifies chess pieces (optional)
- 📍 **Square Coordinates**: Get exact pixel coordinates for any chess square
- 🎨 **Visualizations**: Automatic generation of analysis images
- 📊 **Multiple Formats**: Human-readable output or JSON for automation

### Chess Piece Detection
```python
from src.chess_piece_detection.model import ChessModel

# Load model from Hugging Face
model = ChessModel.from_huggingface("username/chess-piece-detector")

# Or create new model with pretrained checkpoint (for training/transfer learning)
model = ChessModel(pretrained_checkpoint="yolov8s.pt")  # Uses YOLOv8s by default
model = ChessModel(pretrained_checkpoint="yolov8m.pt")  # Use larger model

# Or load your own trained model
model = ChessModel(model_path="models/chess_piece_detection/training_v3/weights/best.pt")

# Predict pieces on an image
results = model.predict("chess_board_image.jpg", conf=0.5)

# Plot evaluation with bounding boxes and labels
model.plot_eval("chess_board_image.jpg", conf=0.25)

# Get detected piece information
for box in results.boxes:
    class_name = results.names[int(box.cls)]
    confidence = float(box.conf)
    coordinates = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
    print(f"Detected {class_name} with confidence {confidence:.2f} at {coordinates}")
```

### Chessboard Corner Detection
```python
from src.chess_board_detection import ChessBoardModel

# Load model from Hugging Face
corner_model = ChessBoardModel.from_huggingface("dopaul/chess-corner-detection-training")

# Or load from local path
corner_model = ChessBoardModel(model_path="models/corner_detection.pt")

# Detect corners (warns if not exactly 4 found)
results, corner_count, is_valid = corner_model.predict_corners("image.jpg")

# Get precise corner coordinates
coordinates, is_valid = corner_model.get_corner_coordinates("image.jpg")

# Visualize with corner outline
corner_model.plot_eval("image.jpg", show_polygon=True)

# Order corners consistently (top-left, top-right, bottom-right, bottom-left)
if is_valid:
    ordered_corners = corner_model.order_corners(coordinates)
```

### Chessboard Segmentation
```python
from src.chess_board_detection.yolo.segmentation.segmentation_model import ChessBoardSegmentationModel
from ultralytics import YOLO

# Load segmentation model from local path
seg_model = ChessBoardSegmentationModel(model_path="artifacts/models/chess_board_segmentation/training/weights/best.pt")

# Or load directly from Hugging Face
yolo_model = YOLO("dopaul/chess_board_segmentation")

# Get precise polygon coordinates
polygon_info, is_valid = seg_model.get_polygon_coordinates("image.jpg")

# Visualize segmentation with polygon outline
seg_model.plot_eval("image.jpg", show_polygon=True, show_mask=True)

# Extract polygon points for perspective transformation
if is_valid:
    coordinates = polygon_info['coordinates']
    print(f"Detected {len(coordinates)} polygon points")
    for i, point in enumerate(coordinates):
        print(f"Point {i}: ({point['x']:.1f}, {point['y']:.1f})")

# Extract corners from segmentation for board alignment
corners = seg_model.extract_corners_from_segmentation("image.jpg", polygon_info)
print(f"Extracted corners: {corners}")
```

## 🔧 Troubleshooting

### Common Issues

#### Dataset Download Problems
- **API Key Error**: Follow [Roboflow API key instructions](#-how-to-get-free-roboflow-api-key) above or use manual download
- **Network Issues**: Check internet connection and try different download methods
- **Missing Dependencies**: Run `uv add roboflow huggingface_hub`

#### Training Issues
- **Dataset Not Found**: Verify data path with `--data` flag
- **GPU Memory**: Reduce `--batch` size if running out of memory
- **HuggingFace Upload**: Login with `huggingface-cli login` first

See detailed troubleshooting guide: [src/data_prep/README.md](src/data_prep/README.md#troubleshooting)

### Notes

- I used Roboflow to label my data. You can download data you labelled there like this: