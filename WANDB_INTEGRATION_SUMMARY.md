# 🔍 Smart Weights & Biases Integration

The chess piece detection training has been enhanced with **intelligent Weights & Biases (wandb) auto-detection** that automatically enables logging when wandb is properly configured.

## 🎯 What Changed

### Before (Manual Activation)
- Required `--use-wandb` flag to enable tracking
- Users had to remember to add the flag
- Easy to forget W&B logging and lose experiment data

### After (Smart Auto-Detection)
- **Automatically detects** wandb environment
- **Auto-enables** logging when environment is configured
- **Optional `--disable-wandb`** flag to explicitly turn off

## 🔧 How Smart Detection Works

The system automatically checks for:

### Environment Variables
- `WANDB_API_KEY` - Your wandb API key
- `WANDB_PROJECT` - Default project name
- `WANDB_ENTITY` - Your wandb organization/username
- `WANDB_BASE_URL` - Custom wandb server URL

### Authentication Status
- Checks if you're logged in via `wandb login`
- Validates if wandb library is installed and accessible

### Auto-Detection Logic
```python
wandb_enabled = wandb_installed AND (has_env_vars OR logged_in) AND NOT --disable-wandb
```

## 🚀 Usage Examples

### Auto-Enabled (Recommended)
```bash
# Set your API key once
export WANDB_API_KEY="your_api_key_here"

# Training automatically enables W&B logging
python src/chess_piece_detection/train.py --epochs 100
# ✅ W&B environment detected - automatic logging enabled

# Customize W&B settings
python src/chess_piece_detection/train.py \
    --epochs 100 \
    --wandb-project "my-chess-project" \
    --wandb-name "experiment-1" \
    --wandb-tags chess yolo11 baseline
```

### Explicit Control
```bash
# Force disable even if environment is detected
python src/chess_piece_detection/train.py \
    --epochs 100 \
    --disable-wandb
# 🔕 W&B environment detected but explicitly disabled via --disable-wandb

# No environment detected (W&B disabled automatically)  
python src/chess_piece_detection/train.py --epochs 100
# ℹ️  W&B environment not detected - logging disabled
```

## 📊 Status Messages

The training script provides clear feedback about W&B detection:

### Environment Detected & Enabled
```
✅ W&B environment detected - automatic logging enabled
```

### Environment Detected but Disabled
```
🔕 W&B environment detected but explicitly disabled via --disable-wandb
```

### No Environment Detected
```
ℹ️  W&B environment not detected - logging disabled
```

## 🎛️ W&B Configuration Options

All existing W&B options remain available:

```bash
python src/chess_piece_detection/train.py \
    --wandb-project "chess-piece-detection" \
    --wandb-name "yolo11s-experiment-1" \
    --wandb-tags chess yolo11 object-detection \
    --wandb-notes "Training chess piece detector with YOLO11s"
```

### Available Options
- `--wandb-project` - Project name (default: "chess-piece-detection")
- `--wandb-name` - Run name (default: uses training name)
- `--wandb-tags` - Space-separated tags for organization
- `--wandb-notes` - Text description of the run
- `--disable-wandb` - Explicitly disable W&B logging

## 🔮 Benefits

### For Users
- **Zero configuration** - Just set WANDB_API_KEY once
- **Never miss experiments** - Auto-enabled when environment is ready
- **Still controllable** - Can disable when needed
- **Clear feedback** - Always know if W&B is active

### For Teams
- **Consistent tracking** - All team members get W&B by default
- **Shared environment** - Team environment variables work automatically
- **Centralized control** - Admins can set organization-wide defaults

### For Development
- **Smart defaults** - Production environments auto-enable tracking
- **Flexible overrides** - Development can disable as needed
- **Environment aware** - Adapts to different deployment contexts

## 🔧 Setup Guide

### 1. One-Time Setup
```bash
# Install wandb
pip install wandb

# Login (stores credentials locally)
wandb login

# Or set API key as environment variable
export WANDB_API_KEY="your_key_here"
```

### 2. Training (Auto-Enabled)
```bash
# W&B will automatically start tracking
python src/chess_piece_detection/train.py --epochs 100
```

### 3. View Results
- Dashboard automatically opens in browser
- Or visit: https://wandb.ai/your-username/chess-piece-detection

## 🎯 Migration Guide

### For Existing Users
- **Remove `--use-wandb` flags** - no longer needed
- **Set WANDB_API_KEY** environment variable
- **Everything else stays the same**

### Code Examples

#### Old Way (Manual)
```bash
python src/chess_piece_detection/train.py \
    --epochs 100 \
    --use-wandb \  # ← Remove this
    --wandb-project chess-detection
```

#### New Way (Auto)
```bash
export WANDB_API_KEY="your_key"  # ← Add this once

python src/chess_piece_detection/train.py \
    --epochs 100 \
    --wandb-project chess-detection  # ← Works automatically
```

The enhanced wandb integration makes experiment tracking effortless while maintaining full control when needed! 🎯 