#!/usr/bin/env python3
"""
Training script for ChessBoardModel - Corner Detection

This script trains a YOLO model to detect the 4 corners of a chessboard.

Usage:
    # Basic training with default settings
    python src/chess_board_detection/train.py
    
    # Run from project root as module
    python -m src.chess_board_detection.train
    
    # Custom training parameters
    python src/chess_board_detection/train.py --data data/my_dataset.yaml --epochs 100 --batch 32
    
    # Complete example with all options and HuggingFace upload
    python src/chess_board_detection/train.py \
        --data data/chessboard_corners/data.yaml \
        --models-folder models/chess_board_detection \
        --name my_corner_training \
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
        --no-plots \
        --verbose \
        --upload-hf \
        --hf-model-name username/chessboard-corner-detector
"""

import argparse
from pathlib import Path
from src.chess_board_detection.model import ChessBoardModel

def parse_args():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Train ChessBoard Corner Detection Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ========================================
    # Dataset and Model Configuration
    # ========================================
    parser.add_argument(
        "--data", 
        type=str, 
        default="data/chessboard_corners/data.yaml",
        help="Path to YOLO dataset YAML file containing train/val/test splits and class names"
    )
    parser.add_argument(
        "--models-folder", 
        type=str, 
        default="models/chess_board_detection",
        help="Root folder where trained models will be saved"
    )
    parser.add_argument(
        "--name", 
        type=str, 
        default="corner_detection_training",
        help="Name for this training run (creates subfolder in models-folder)"
    )
    
    # ========================================
    # Core Training Parameters
    # ========================================
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=50,
        help="Number of training epochs (complete passes through dataset)"
    )
    parser.add_argument(
        "--batch", 
        type=int, 
        default=16,
        help="Batch size (number of images processed together). Reduce if GPU memory issues"
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=640,
        help="Input image size in pixels (square). Common values: 416, 640, 832"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.001,
        help="Initial learning rate. Lower values (0.0001) for fine-tuning, higher (0.01) for fresh training"
    )
    parser.add_argument(
        "--patience", 
        type=int, 
        default=15,
        help="Early stopping patience (epochs without improvement before stopping)"
    )
    parser.add_argument(
        "--save-period", 
        type=int, 
        default=10,
        help="Save model checkpoint every N epochs (set to -1 to disable periodic saves)"
    )
    
    # ========================================
    # Data Augmentation Parameters
    # ========================================
    parser.add_argument(
        "--degrees", 
        type=float, 
        default=5.0,
        help="Rotation augmentation range in degrees (±degrees). Use small values for corner detection"
    )
    parser.add_argument(
        "--translate", 
        type=float, 
        default=0.05,
        help="Translation augmentation as fraction of image size (0.0-1.0)"
    )
    parser.add_argument(
        "--scale", 
        type=float, 
        default=0.2,
        help="Scale augmentation range (0.0-1.0). Higher values = more scale variation"
    )
    parser.add_argument(
        "--fliplr", 
        type=float, 
        default=0.5,
        help="Horizontal flip probability (0.0-1.0). Good for most detection tasks"
    )
    parser.add_argument(
        "--flipud", 
        type=float, 
        default=0.0,
        help="Vertical flip probability (0.0-1.0). Usually 0.0 for corner detection"
    )
    parser.add_argument(
        "--mosaic", 
        type=float, 
        default=0.8,
        help="Mosaic augmentation probability (0.0-1.0). Combines 4 images into 1"
    )
    
    # ========================================
    # HuggingFace Model Hub Configuration
    # ========================================
    parser.add_argument(
        "--upload-hf", 
        action="store_true",
        help="Upload trained model to HuggingFace Model Hub after training completes"
    )
    parser.add_argument(
        "--hf-model-name", 
        type=str, 
        default="",
        help="HuggingFace model repository name (format: username/model-name). Required if --upload-hf is used"
    )
    
    # ========================================
    # Control Flags
    # ========================================
    parser.add_argument(
        "--no-plots", 
        action="store_true",
        help="Disable training plots generation (faster training, less disk space)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show verbose output including full error tracebacks"
    )
    
    return parser.parse_args()

def validate_args(args):
    """Validate command line arguments and show helpful error messages."""
    # Check if HuggingFace upload is requested but model name is missing
    if args.upload_hf and not args.hf_model_name:
        print("❌ Error: --hf-model-name is required when --upload-hf is specified")
        print("💡 Example: --hf-model-name username/chessboard-corner-detector")
        return False
        
    # Check if dataset file exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Error: Dataset file '{data_path}' does not exist!")
        print(f"💡 Please create the dataset or specify a different path with --data")
        print(f"💡 Expected YOLO format with train/val/test image folders and labels")
        return False
        
    return True

def main():
    """Main training function for chessboard corner detection."""
    
    # Parse and validate command line arguments
    args = parse_args()
    if not validate_args(args):
        return
    
    # ========================================
    # Configuration Setup
    # ========================================
    DATA_YAML = Path(args.data)
    MODELS_FOLDER = Path(args.models_folder)
    MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # ========================================
    # Training Information Display
    # ========================================
    print("🏁 Starting ChessBoard Corner Detection Training")
    print("=" * 70)
    print(f"📁 Models will be saved to: {MODELS_FOLDER}")
    print(f"📊 Dataset: {DATA_YAML}")
    print(f"🏷️  Training name: {args.name}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"📦 Batch size: {args.batch}")
    print(f"🖼️  Image size: {args.imgsz}")
    print(f"📈 Learning rate: {args.lr}")
    print(f"⏰ Patience: {args.patience}")
    print(f"💾 Save period: {args.save_period}")
    print(f"🎲 Augmentation: degrees={args.degrees}, translate={args.translate}, scale={args.scale}")
    print(f"🔄 Flips: horizontal={args.fliplr}, vertical={args.flipud}")
    print(f"🧩 Mosaic: {args.mosaic}")
    print(f"📊 Plots: {'Disabled' if args.no_plots else 'Enabled'}")
    
    # HuggingFace upload information
    if args.upload_hf:
        print(f"🤗 HuggingFace Upload: ENABLED")
        print(f"📤 Model will be uploaded to: {args.hf_model_name}")
        print(f"⚠️  Make sure you're logged in with: huggingface-cli login")
    else:
        print(f"🤗 HuggingFace Upload: Disabled")
    
    print("=" * 70)
    
    # ========================================
    # Model Training
    # ========================================
    print("🚀 Initializing ChessBoardModel...")
    model = ChessBoardModel()
    
    print("🎯 Starting training process...")
    try:
        results = model.train(
            data_path=DATA_YAML,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            lr0=args.lr,
            project=MODELS_FOLDER,
            name=args.name,
            plots=not args.no_plots,
            # Corner detection specific parameters
            patience=args.patience,
            save_period=args.save_period,
            # Data augmentation parameters
            degrees=args.degrees,
            translate=args.translate,
            scale=args.scale,
            fliplr=args.fliplr,
            flipud=args.flipud,
            mosaic=args.mosaic,
        )
        
        # Print training summary
        print("📊 Printing training summary...")
        model.print_training_summary(results)
        
        # ========================================
        # Model Saving and Validation
        # ========================================
        best_model_path = MODELS_FOLDER / args.name / "weights" / "best.pt"
        if best_model_path.exists():
            print(f"✅ Best model saved at: {best_model_path}")
            print(f"🎯 Training completed successfully!")
            
            # Load the trained model for validation
            print("🔍 Loading trained model for validation...")
            trained_model = ChessBoardModel(model_path=best_model_path)
            print(f"✅ Model loaded successfully!")
            
            # ========================================
            # HuggingFace Upload (if requested)
            # ========================================
            if args.upload_hf:
                print("=" * 70)
                print(f"🤗 Uploading model to HuggingFace: {args.hf_model_name}")
                try:
                    trained_model.push_to_huggingface(args.hf_model_name)
                    print(f"✅ Model successfully uploaded to HuggingFace!")
                    print(f"🔗 View at: https://huggingface.co/{args.hf_model_name}")
                except Exception as e:
                    print(f"❌ HuggingFace upload failed: {e}")
                    if args.verbose:
                        raise
            
        else:
            print(f"⚠️  Warning: Best model not found at expected path: {best_model_path}")
            
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        if args.verbose:
            raise
        return
    
    # ========================================
    # Training Complete Summary
    # ========================================
    print("=" * 70)
    print("🏁 Training Complete!")
    print("Next steps:")
    print("1. Test the model: model.plot_eval('path/to/test_image.jpg')")
    print("2. Get corner coordinates: model.get_corner_coordinates('image.jpg')")
    if not args.upload_hf:
        print("3. Upload to HuggingFace: model.push_to_huggingface('username/chessboard-corners')")
    print("=" * 70)

if __name__ == "__main__":
    main() 