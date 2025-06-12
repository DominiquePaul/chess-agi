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
```bash
# Train chess piece detection model
python -m src.chess_piece_detection.train

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

# Train segmentation model
python src/chess_board_detection/yolo/segmentation/train_segmentation.py \
    --data data/chessboard_segmentation/chess-board-i0ptl-3/data.yaml \
    --epochs 100 \
    --batch 16

# Test segmentation model
python src/chess_board_detection/yolo/segmentation/test_segmentation.py \
    --model models/chess_board_segmentation/polygon_segmentation_training/weights/best.pt \
    --image path/to/test_image.jpg
```

## Usage Examples

### Chess Piece Detection
```python
from src.chess_piece_detection import ChessModel

# Load model
model = ChessModel.from_huggingface("username/chess-piece-detector")

# Predict pieces on an image
results = model.predict("chess_board_image.jpg")

# Plot evaluation with bounding boxes
model.plot_eval("chess_board_image.jpg")
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

# Load segmentation model
seg_model = ChessBoardSegmentationModel(model_path="models/segmentation.pt")

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