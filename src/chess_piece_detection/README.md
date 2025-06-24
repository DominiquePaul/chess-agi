# Chess Piece Detection

This module provides YOLO11-based chess piece detection and classification. It can detect and classify all 12 types of chess pieces (6 piece types × 2 colors) using the latest and most advanced YOLO11 architecture.

## 🚀 Quick Start

### Basic Training (Auto-downloads from HuggingFace)
```bash
# Simple training with auto-download from HuggingFace
python src/chess_piece_detection/train.py --epochs 100

# The script will automatically:
# 1. Check for local dataset
# 2. Download from HuggingFace if not found locally
# 3. Cache locally for future training runs
```

### Training Options

#### Automatic HuggingFace Integration
```bash
# Use default merged dataset from HuggingFace (recommended)
python src/chess_piece_detection/train.py \
    --epochs 100 \
    --batch 32 \
    --auto-download-hf

# Use specific HuggingFace dataset
python src/chess_piece_detection/train.py \
    --hf-dataset dopaul/chess-pieces-dominique \
    --epochs 100

# Force re-download from HuggingFace (refresh dataset)
python src/chess_piece_detection/train.py \
    --force-download-hf \
    --epochs 100
```

#### Local Dataset Training
```bash
# Use local dataset (traditional approach)
python src/chess_piece_detection/train.py \
    --data data/chess_pieces_merged/data.yaml \
    --epochs 100 \
    --auto-download-hf false
```

#### Complete Training Example
```bash
python src/chess_piece_detection/train.py \
    --hf-dataset dopaul/chess-pieces-merged \
    --pretrained-model yolo11s.pt \
    --epochs 100 \
    --batch 32 \
    --lr 0.001 \
    --name training_v1 \
    --models-folder models/chess_piece_detection \
    --auto-download-hf
```

## 📊 Dataset Options

The training script supports multiple dataset sources:

### HuggingFace Datasets (Recommended)
- `dopaul/chess-pieces-merged` - Combined dataset (default, best performance)
- `dopaul/chess-pieces-dominique` - Dominique's enhanced dataset
- `dopaul/chess-pieces-roboflow` - Original Roboflow dataset

### Local Datasets
- `data/chess_pieces_merged/data.yaml` - Local merged dataset
- `data/chess_pieces_dominique/data.yaml` - Local Dominique dataset
- `data/chess_pieces_roboflow/data.yaml` - Local Roboflow dataset

## 🔄 How It Works

### Smart Dataset Loading
1. **Check Local First**: Looks for dataset locally
2. **Auto-Download**: Downloads from HuggingFace if not found
3. **Cache Locally**: Saves downloaded dataset for future use
4. **No Re-Download**: Uses cached version unless forced

### YOLO11 Benefits
- ✅ **Latest Architecture**: Uses cutting-edge YOLO11 with improved accuracy
- ✅ **Better Efficiency**: 22% fewer parameters than YOLOv8 for similar accuracy
- ✅ **Enhanced Features**: Improved backbone and neck architectures
- ✅ **Always up-to-date**: Uses latest HuggingFace datasets
- ✅ **No redundant downloads**: Caches locally after first download
- ✅ **Fast subsequent training**: Uses local cache
- ✅ **Flexible**: Can override with local datasets or force refresh

## 🎯 Training Results

After training, you'll find:
```
models/chess_piece_detection/
└── training_v1/
    ├── weights/
    │   ├── best.pt      # Best model checkpoint
    │   └── last.pt      # Last epoch checkpoint
    ├── results.png      # Training metrics plots
    └── confusion_matrix.png
```

## 🧪 Testing Your Model

```bash
# Test the trained model
python src/chess_piece_detection/inference_example.py \
    --model models/chess_piece_detection/training_v1/weights/best.pt \
    --image path/to/chess_image.jpg
```

## 🎛️ Advanced Configuration

### Model Sizes
- `yolo11n.pt` - Nano (fastest, least accurate, ~2.6M params)
- `yolo11s.pt` - Small (default, good balance, ~9.4M params)
- `yolo11m.pt` - Medium (more accurate, slower, ~20.1M params)
- `yolo11l.pt` - Large (high accuracy, slow, ~25.3M params)
- `yolo11x.pt` - Extra Large (best accuracy, slowest, ~56.9M params)

### Training Parameters
```bash
python src/chess_piece_detection/train.py \
    --epochs 100 \
    --batch 32 \
    --lr 0.001 \
    --patience 15 \
    --save-period 10 \
    --optimizer AdamW \
    --weight-decay 0.0005
```

### Data Augmentation
```bash
python src/chess_piece_detection/train.py \
    --degrees 10.0 \
    --translate 0.1 \
    --scale 0.2 \
    --fliplr 0.5 \
    --mosaic 1.0 \
    --mixup 0.1
```

## 🔧 Environment Setup

```bash
# Set data folder path
export DATA_FOLDER_PATH="/path/to/your/data"

# Install dependencies
uv add ultralytics datasets huggingface_hub

# Optional: Set up Weights & Biases for tracking (auto-enabled if WANDB_API_KEY is set)
export WANDB_API_KEY="your_wandb_key"
python src/chess_piece_detection/train.py --wandb-project chess-detection

# Or disable W&B explicitly
python src/chess_piece_detection/train.py --disable-wandb
```

## 📈 Class Labels

The model detects 12 chess piece classes:
```
0: black-bishop    6: white-bishop
1: black-king      7: white-king
2: black-knight    8: white-knight
3: black-pawn      9: white-pawn
4: black-queen    10: white-queen
5: black-rook     11: white-rook
```
