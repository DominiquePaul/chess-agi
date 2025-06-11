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

### 2. Get Free Roboflow API Key (Recommended)

To download datasets reliably, get a free Roboflow API key:

1. **Visit**: https://roboflow.com/
2. **Sign up** for a free account (takes 30 seconds)
3. **Navigate** to Settings → API 
4. **Copy** your API key
5. **Use** the key with download commands:
   ```bash
   python src/chess_board_detection/download_data.py --api-key YOUR_API_KEY
   ```

**Why get an API key?**
- ✅ **Free** and takes 2 minutes
- ✅ **Reliable downloads** without rate limiting
- ✅ **Better error handling** 
- ✅ **Works for all public datasets**
- ✅ **Required for private datasets** (if needed later)

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
    └── chess-board-box-3/         # Downloaded dataset folder
        ├── train/
        ├── valid/
        ├── test/
        └── data.yaml              # Use this path for training
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
# Download with API key (required)
python src/chess_board_detection/download_data.py --api-key YOUR_ROBOFLOW_API_KEY

# Download to custom directory
python src/chess_board_detection/download_data.py --api-key YOUR_API_KEY --data-dir data/my_corners

# Download with verbose output
python src/chess_board_detection/download_data.py --api-key YOUR_API_KEY --verbose
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

## 🔧 Troubleshooting

### Dataset Download Issues

#### Roboflow API Key Required
For automated downloads, you need a free Roboflow API key:

```bash
# Get your free API key:
# 1. Visit https://roboflow.com/ and sign up (free)
# 2. Get API key from Settings → API
# 3. Use with download:
python src/chess_board_detection/download_data.py --api-key YOUR_API_KEY

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