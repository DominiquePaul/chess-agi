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


## Download chess datasets

### From Hugging Face (recommended)

The easiest way to get chess datasets is to download them directly from Hugging Face:

#### Chess Piece Detection

```bash
# Download merged dataset (recommended for training)
python src/data_prep/download_from_hf.py --dataset merged

# Download all available datasets  
python src/data_prep/download_from_hf.py --dataset all

# Download specific datasets
python src/data_prep/download_from_hf.py --dataset dominique roboflow

# List available datasets
python src/data_prep/download_from_hf.py --list

# Download to custom directory
python src/data_prep/download_from_hf.py --dataset merged --output-dir ./my_data
```

Or use the Hugging Face datasets directly in your code:

```python
from datasets import load_dataset

# Load individual datasets
dominique_dataset = load_dataset("dopaul/chess-pieces-dominique")
roboflow_dataset = load_dataset("dopaul/chess-pieces-roboflow") 

# Load merged dataset (recommended for training)
merged_dataset = load_dataset("dopaul/chess-pieces-merged")

# Access different splits
train_data = merged_dataset["train"]
valid_data = merged_dataset["valid"]
test_data = merged_dataset["test"]
```

#### Chessboard Corner Detection

For corner detection datasets, download directly:

```bash
# Download chessboard corner detection dataset
python -m src.chess_board_detection.download_data

# This downloads from Roboflow: gustoguardian/chess-board-box/3
# Contains images with 4 corners per chessboard labeled
```

### Recreate datasets from source

If you want to recreate the datasets from scratch (for development or to upload your own version), follow these steps:

#### Prerequisites

1. **Kaggle API**: Create an API key at [www.kaggle.com/settings](https://www.kaggle.com/settings) and place the `kaggle.json` file in `~/.kaggle/kaggle.json`

2. **Roboflow CLI**: Install if not already available:
   ```bash
   uv add roboflow
   ```

3. **Environment setup**: Make sure your `.env` file contains:
   ```bash
   DATA_FOLDER_PATH=/path/to/your/data/folder
   HF_USERNAME=your-huggingface-username  # Set this if you want to upload
   ```

4. **Hugging Face authentication** (if uploading):
   ```bash
   huggingface-cli login
   ```

#### Step-by-step recreation

```bash
# Complete pipeline: download ALL datasets (chess pieces + chessboard_corners_dominique) and upload
python src/data_prep/prepare_and_upload_datasets.py

# Download only specific datasets:
python src/data_prep/prepare_and_upload_datasets.py --datasets roboflow chesspieces_dominique  # Chess pieces only
python src/data_prep/prepare_and_upload_datasets.py --datasets chessboard_corners_dominique             # Corners only
python src/data_prep/prepare_and_upload_datasets.py --datasets chesspieces_dominique # Single dataset

# Alternative options:
python src/data_prep/prepare_and_upload_datasets.py --dry-run           # See what would be done
python src/data_prep/prepare_and_upload_datasets.py --no-individual    # Only upload merged dataset
python src/data_prep/prepare_and_upload_datasets.py --skip-download     # Only upload existing datasets
```

**What the script does:**

**Chess Piece Detection Datasets:**
- **`roboflow`**: Downloads chess pieces dataset from Kaggle, processes it to remove the extra "bishop" class, and standardizes the labels for YOLOv8 format
- **`dominique`**: Downloads the complementary dataset from Roboflow with additional chess piece images  

**Chessboard Corner Detection Dataset:**
- **`chessboard_corners_dominique`**: Downloads chessboard corner detection dataset from Roboflow (`gustoguardian/chess-board-box/3`) with 4 corners per board (Dominique)

**Upload Phase:**
- Uploads datasets individually and creates a merged version (for chess pieces) on Hugging Face for easy access

The script automatically checks prerequisites and provides clear status updates. You can inspect `src/data_prep/prepare_and_upload_datasets.py` to see exactly what it does - it imports functions from the existing processing scripts to keep the code clean and readable.

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

## Training Models

### Chess Piece Detection
```bash
# Train chess piece detection model
python -m src.chess_piece_detection.train

# Run inference example
python -m src.chess_piece_detection.inference_example
```

### Chessboard Corner Detection
```bash
# Download corner detection dataset
python -m src.chess_board_detection.download_data

# Train corner detection model
python -m src.chess_board_detection.train

# Test corner detection
python -m src.chess_board_detection.inference_example
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

# Load model
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

### Notes

- I used Roboflow to label my data. You can download data you labelled there like this: