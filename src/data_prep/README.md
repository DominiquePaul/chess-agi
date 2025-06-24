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
  - Source: Roboflow (`gustoguardian/chess-piece-detection-bltvi/8`)
  - Purpose: Training enhanced chess piece detection models

### Chessboard Corner Detection
- **`chessboard_corners_dominique`** - Chessboard corner detection dataset (Dominique)
  - 4 corners per chessboard
  - Source: Roboflow (`gustoguardian/chess-board-box/3`)
  - Purpose: Training chessboard corner detection models

### Chessboard Segmentation
- **`chessboard_segmentation_dominique`** - Chessboard segmentation dataset (Dominique)
  - Polygon boundaries for precise chessboard detection
  - Source: Roboflow (`gustoguardian/chess-board-i0ptl/3`)
  - Purpose: Training chessboard segmentation models for precise polygon detection

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

### 2. Get Free Roboflow API Key (Required)

To download datasets, get a free Roboflow API key and set it as an environment variable:

1. **Visit**: https://roboflow.com/
2. **Sign up** for a free account (takes 30 seconds)
3. **Navigate** to Settings → API
4. **Copy** your API key
5. **Set environment variable**:
   ```bash
   export ROBOFLOW_API_KEY=your_api_key_here
   # Add to your .env file or shell profile for persistence
   ```

**Why use environment variables?**
- ✅ **Secure** - no API keys in command history
- ✅ **Convenient** - set once, use everywhere
- ✅ **Safe** - won't accidentally commit keys to git
- ✅ **Standard practice** for API credentials

### 3. Download All Datasets
```bash
# Download and upload all datasets
python src/data_prep/prepare_and_upload_datasets.py

# Just download without uploading
python src/data_prep/prepare_and_upload_datasets.py --skip-upload
```

### 4. Download Specific Datasets
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
--datasets [roboflow] [chesspieces_dominique] [chessboard_corners_dominique] [chessboard_segmentation_dominique]
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

# Only segmentation dataset
python src/data_prep/prepare_and_upload_datasets.py --datasets chessboard_segmentation_dominique

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
├── chessboard_corners/            # Chessboard corners
│   └── chess-board-box-3/         # Downloaded dataset folder
│       ├── train/
│       ├── valid/
│       ├── test/
│       └── data.yaml              # Use this path for training
└── chessboard_segmentation/       # Chessboard segmentation
    └── chess-board-i0ptl-3/       # Downloaded dataset folder
        ├── train/
        ├── valid/
        ├── test/
        └── data.yaml              # Use this path for segmentation training
```

## 🎯 Training Models

### Chessboard Corner Detection Training Options

The training script now supports comprehensive command-line configuration:

```bash
# View all available options
python src/chess_board_detection/train.py --help

# Basic training with dataset validation
python src/chess_board_detection/train.py --data data/chessboard_corners/chess-board-box-3/data.yaml

# Full training with all options
python src/chess_board_detection/train.py \
    --data data/chessboard_corners/chess-board-box-3/data.yaml \
    --models-folder models/chess_corners \
    --name corner_detection_v1 \
    --epochs 100 \
    --batch 32 \
    --imgsz 640 \
    --lr 0.001 \
    --patience 20 \
    --save-period 5 \
    --degrees 10.0 \
    --translate 0.1 \
    --scale 0.3 \
    --fliplr 0.5 \
    --flipud 0.0 \
    --mosaic 0.9 \
    --upload-hf \
    --hf-model-name username/chessboard-corner-detector \
    --verbose
```

### Key Training Features
- **Dataset validation** before training starts
- **HuggingFace integration** for automatic model upload
- **Comprehensive parameter control** for training customization
- **Enhanced error handling** with helpful troubleshooting
- **Progress tracking** with detailed output

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
# Train corner detection model with default settings
python src/chess_board_detection/train.py

# Train with custom parameters and HuggingFace upload
python src/chess_board_detection/train.py \
    --data data/chessboard_corners/chess-board-box-3/data.yaml \
    --epochs 100 \
    --batch 32 \
    --upload-hf \
    --hf-model-name username/chessboard-corner-detector

# Test corner detection
python -m src.chess_board_detection.inference_example
```

### Chessboard Segmentation
```bash
# Train segmentation model with default settings
python src/chess_board_detection/yolo/segmentation/train_segmentation.py \
    --data data/chessboard_segmentation/chess-board-i0ptl-3/data.yaml

# Train with custom parameters
python src/chess_board_detection/yolo/segmentation/train_segmentation.py \
    --data data/chessboard_segmentation/chess-board-i0ptl-3/data.yaml \
    --epochs 100 \
    --batch 16 \
    --pretrained-model yolov8m-seg.pt \
    --name polygon_detection_v1

# Test segmentation model
python src/chess_board_detection/yolo/segmentation/test_segmentation.py \
    --model models/chess_board_segmentation/polygon_segmentation_training/weights/best.pt \
    --image path/to/test_image.jpg
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
# Set API key (required)
export ROBOFLOW_API_KEY=your_api_key_here

# Download corner detection dataset
python src/chess_board_detection/download_data.py

# Download to custom directory
python src/chess_board_detection/download_data.py --data-dir data/my_corners

# Download with verbose output
python src/chess_board_detection/download_data.py --verbose
```

### Chessboard Segmentation
```bash
# Set API key (required)
export ROBOFLOW_API_KEY=your_api_key_here

# Download segmentation dataset
python src/chess_board_detection/download_data.py \
    --project gustoguardian/chess-board-i0ptl \
    --version 3 \
    --data-dir data/chessboard_segmentation

# Download with verbose output
python src/chess_board_detection/download_data.py \
    --project gustoguardian/chess-board-i0ptl \
    --version 3 \
    --data-dir data/chessboard_segmentation \
    --verbose
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
   📍 Source: Roboflow (gustoguardian/chess-piece-detection-bltvi/8)
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

## 🔧 Troubleshooting

### Dataset Download Issues

#### Roboflow API Key Required
For automated downloads, you need a free Roboflow API key:

```bash
# Get your free API key:
# 1. Visit https://roboflow.com/ and sign up (free)
# 2. Get API key from Settings → API
# 3. Set environment variable:
export ROBOFLOW_API_KEY=your_api_key_here
python src/chess_board_detection/download_data.py

# Manual download alternative:
# 1. Visit https://universe.roboflow.com/gustoguardian/chess-board-box
# 2. Download YOLO v8 format
# 3. Extract to data/chessboard_corners/
```

#### Dataset Path Issues
Update your training data path based on download location:

```bash
# If downloaded to default location
python src/chess_board_detection/train.py --data data/chessboard_corners/chess-board-box-3/data.yaml

# If downloaded to custom location
python src/chess_board_detection/train.py --data path/to/your/data.yaml
```

#### HuggingFace Upload Issues
```bash
# Make sure you're logged in
huggingface-cli login

# Check your username and model name format
python src/chess_board_detection/train.py \
    --upload-hf \
    --hf-model-name your-username/model-name
```

### Common Solutions
1. **Missing dependencies**: `uv add roboflow huggingface_hub`
2. **Network issues**: Check internet connection and firewall
3. **Disk space**: Ensure sufficient space for datasets (~1-5GB each)
4. **Permissions**: Make sure you have write access to data directory

## 🔗 Related Documentation

- [Chess Piece Detection](../chess_piece_detection/README.md)
- [Chessboard Corner Detection](../chess_board_detection/README.md)
- [Model Training and Inference](../../README.md)
