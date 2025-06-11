# Chess Dataset Preparation

This module provides a complete pipeline for downloading, processing, and uploading chess-related datasets to Hugging Face Hub. It supports both chess piece detection and chessboard corner detection datasets.

## 🎯 Supported Datasets

### Chess Piece Detection
- **`roboflow`** - Chess pieces from Roboflow (via Kaggle)
  - 12 chess piece classes
  - Source: Kaggle (originally from Roboflow)
  - Purpose: Training chess piece detection models

- **`chesspieces_dominique`** - Chess pieces from Dominique's Roboflow project
  - 12 chess piece classes (enhanced dataset)
  - Source: Roboflow (`gustoguardian/chess-piece-detection-bltvi/6`)
  - Purpose: Training enhanced chess piece detection models

### Chessboard Corner Detection
- **`chessboard_corners_dominique`** - Chessboard corner detection dataset (Dominique)
  - 4 corners per chessboard
  - Source: Roboflow (`gustoguardian/chess-board-box/3`)
  - Purpose: Training chessboard corner detection models

## 🚀 Quick Start

### 1. Prerequisites

Set up your environment:
```bash
# Set environment variables (add to .env file)
export DATA_FOLDER_PATH="/path/to/your/data"
export HF_USERNAME="your_huggingface_username"

# Install dependencies
uv add roboflow
uv add huggingface_hub

# Set up Kaggle API (for roboflow dataset)
# Create ~/.kaggle/kaggle.json with your API credentials
```

### 2. Download All Datasets
```bash
# Download and upload all datasets
python src/data_prep/prepare_and_upload_datasets.py

# Just download without uploading
python src/data_prep/prepare_and_upload_datasets.py --skip-upload
```

### 3. Download Specific Datasets
```bash
# Download only chess piece datasets
python src/data_prep/prepare_and_upload_datasets.py --datasets roboflow chesspieces_dominique

# Download only corner detection dataset
python src/data_prep/prepare_and_upload_datasets.py --datasets chessboard_corners_dominique

# Download just Dominique's dataset
python src/data_prep/prepare_and_upload_datasets.py --datasets chesspieces_dominique
```

## 📋 Command Line Options

### Dataset Selection
```bash
--datasets [roboflow] [chesspieces_dominique] [chessboard_corners_dominique]
```
Specify which datasets to download. Default: all datasets.

### Download Control
```bash
--skip-download        # Only upload existing datasets, don't download
--dry-run             # Show what would be done without doing it
```

### Upload Control
```bash
--no-individual       # Skip individual dataset uploads, only merged
--no-merged          # Skip merged dataset upload, only individual  
```

## 💡 Usage Examples

### Basic Usage
```bash
# Full pipeline - download all and upload
python src/data_prep/prepare_and_upload_datasets.py

# See what would happen without doing it
python src/data_prep/prepare_and_upload_datasets.py --dry-run
```

### Selective Downloads
```bash
# Only chess piece datasets
python src/data_prep/prepare_and_upload_datasets.py --datasets roboflow chesspieces_dominique

# Only corner detection
python src/data_prep/prepare_and_upload_datasets.py --datasets chessboard_corners_dominique

# Single dataset
python src/data_prep/prepare_and_upload_datasets.py --datasets chesspieces_dominique
```

### Upload Control
```bash
# Download but don't upload
python src/data_prep/prepare_and_upload_datasets.py --skip-upload

# Upload existing datasets only
python src/data_prep/prepare_and_upload_datasets.py --skip-download

# Only upload individual datasets (no merged)
python src/data_prep/prepare_and_upload_datasets.py --no-merged

# Only upload merged dataset
python src/data_prep/prepare_and_upload_datasets.py --no-individual
```

## 📁 Output Structure

After running the pipeline, your data directory will contain:

```
$DATA_FOLDER_PATH/
├── chess_pieces_roboflow/          # Chess pieces (Kaggle/Roboflow)
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
├── chess_pieces_dominique/         # Chess pieces (Dominique)
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
└── chessboard_corners/            # Chessboard corners
    └── chess-board-box-3/
        ├── train/
        ├── valid/
        ├── test/
        └── data.yaml
```

## 🤗 Hugging Face Integration

### Uploaded Datasets
The pipeline uploads datasets to your Hugging Face profile:

- `{username}/chess-pieces-roboflow` - Chess pieces from Roboflow/Kaggle
- `{username}/chess-pieces-dominique` - Chess pieces from Dominique
- `{username}/chess-pieces-merged` - Combined chess pieces dataset

### Downloading from Hugging Face
```bash
# List available datasets
python src/data_prep/download_from_hf.py --list

# Download specific dataset
python src/data_prep/download_from_hf.py --dataset chess-pieces-merged
```

## 🎯 Next Steps After Download

### Chess Piece Detection
```bash
# Train on individual datasets
python -m src.chess_piece_detection.train

# Use merged dataset for better performance
# (Update data.yaml path in training script)
```

### Chessboard Corner Detection
```bash
# Train corner detection model
python -m src.chess_board_detection.train

# Test corner detection
python -m src.chess_board_detection.inference_example
```

## 🔧 Individual Scripts

If you prefer to use individual scripts instead of the main pipeline:

### Chess Piece Detection
```bash
# Download Roboflow dataset via Kaggle
python src/data_prep/roboflow_chess.py

# Upload datasets to HF
python src/data_prep/upload_to_hf.py
```

### Chessboard Corner Detection
```bash
# Download corner detection dataset
python -m src.chess_board_detection.download_data
```

## 📊 Pipeline Output Example

```
🏗️  Chess Dataset Preparation Pipeline
============================================================
📊 Target datasets: roboflow, chesspieces_dominique, chessboard_corners_dominique
📁 Data directory: /path/to/data
🔄 Mode: Download + Upload

✅ All prerequisites met!

========================================
📥 DATASET DOWNLOAD PHASE
========================================

📦 Downloading Chess Pieces Dataset (Roboflow/Kaggle)...
   📊 Dataset: Chess piece detection with 12 classes
   📍 Source: Kaggle (originally from Roboflow)
   🎯 Purpose: Training chess piece detection models
✅ Chess pieces dataset (Roboflow/Kaggle) processed successfully

📦 Downloading Chess Pieces Dataset (Dominique)...
   📊 Dataset: Chess piece detection with 12 classes
   📍 Source: Roboflow (gustoguardian/chess-piece-detection-bltvi/6)
   🎯 Purpose: Training chess piece detection models (enhanced)
✅ Chess pieces dataset (Dominique) downloaded successfully

📦 Downloading Chessboard Corners Dataset (Dominique)...
   📊 Dataset: Chessboard corner detection (4 corners per board)
   📍 Source: Roboflow (gustoguardian/chess-board-box/3)
   🎯 Purpose: Training chessboard corner detection models
✅ Chessboard corners dataset (Dominique) downloaded successfully

========================================
🚀 HUGGING FACE UPLOAD PHASE
========================================

🚀 Uploading datasets to Hugging Face Hub...
  📤 Uploading individual datasets...
    📋 Uploading Chess Pieces (Dominique)...
    📋 Uploading Chess Pieces (Roboflow/Kaggle)...
    📋 Chessboard Corners dataset (Dominique) found but upload not yet implemented
  📤 Uploading merged chess pieces dataset...
    🔄 Merging datasets: chess_pieces_dominique, chess_pieces_roboflow
✅ Dataset uploads completed!

============================================================
🎉 Pipeline completed successfully!
📁 Datasets available locally at: /path/to/data
🌐 Datasets uploaded to: https://huggingface.co/username

🎯 Next steps:
1. Check your Hugging Face profile for uploaded datasets
2. Use download script: python src/data_prep/download_from_hf.py --list
3. Start training your models:
   - Chess piece detection: python -m src.chess_piece_detection.train
   - Chessboard corners: python -m src.chess_board_detection.train
```

## 🔗 Related Documentation

- [Chess Piece Detection](../chess_piece_detection/README.md)
- [Chessboard Corner Detection](../chess_board_detection/README.md)
- [Model Training and Inference](../../README.md) 