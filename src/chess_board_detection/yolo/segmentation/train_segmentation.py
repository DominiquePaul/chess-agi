#!/usr/bin/env python3
"""
Training script for ChessBoardSegmentationModel - Polygon Segmentation

This script trains a YOLO segmentation model to learn the exact polygon boundaries
of chessboards from polygon annotation data.

Usage:
    # Basic training with default settings (YOLOv8n-seg COCO pretrained)
    python src/chess_board_detection/yolo/segmentation/train_segmentation.py --epochs 100
    
    # Choose different COCO-pretrained model size
    python src/chess_board_detection/yolo/segmentation/train_segmentation.py \
        --pretrained-model yolov8s-seg.pt \
        --epochs 100
    
    # Custom training parameters
    python src/chess_board_detection/yolo/segmentation/train_segmentation.py \
        --data data/chessboard_corners/chess-board-box-3/data.yaml \
        --pretrained-model yolov8s-seg.pt \
        --epochs 100 \
        --batch 16 \
        --imgsz 640
    
    # Complete example with all options
    python src/chess_board_detection/yolo/train_segmentation.py \
        --data data/chessboard_corners/chess-board-box-3/data.yaml \
        --pretrained-model yolov8l-seg.pt \
        --models-folder models/chess_board_segmentation \
        --name polygon_segmentation_training \
        --epochs 100 \
        --batch 16 \
        --imgsz 640 \
        --lr 0.001 \
        --patience 20 \
        --save-period 5 \
        --degrees 10.0 \
        --translate 0.1 \
        --scale 0.3 \
        --fliplr 0.5 \
        --mosaic 0.9 \
        --verbose
        
Available COCO-pretrained models:
    - yolov8n-seg.pt (nano, ~6MB, fastest)
    - yolov8s-seg.pt (small, ~22MB, fast)  
    - yolov8m-seg.pt (medium, ~52MB, balanced)
    - yolov8l-seg.pt (large, ~104MB, accurate)
    - yolov8x-seg.pt (extra large, ~136MB, most accurate)
"""

import argparse
from pathlib import Path
from dotenv import load_dotenv
from src.chess_board_detection.yolo.segmentation.segmentation_model import ChessBoardSegmentationModel

# Load environment variables from .env file
load_dotenv()

def parse_args():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Train ChessBoard Polygon Segmentation Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ========================================
    # Dataset and Model Configuration
    # ========================================
    parser.add_argument(
        "--data", 
        type=str, 
        default="data/chessboard_segmentation/chessboard-3/data.yaml",
        help="Path to YOLO dataset YAML file containing polygon annotations"
    )
    parser.add_argument(
        "--pretrained-model", 
        type=str, 
        default="yolov8n-seg.pt",
        choices=["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt"],
        help="COCO-pretrained model to start training from (n=nano, s=small, m=medium, l=large, x=extra large)"
    )
    parser.add_argument(
        "--models-folder", 
        type=str, 
        default="models/chess_board_segmentation",
        help="Root folder where trained models will be saved"
    )
    parser.add_argument(
        "--name", 
        type=str, 
        default="polygon_segmentation_training",
        help="Name for this training run (creates subfolder in models-folder)"
    )
    
    # ========================================
    # Core Training Parameters
    # ========================================
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch", 
        type=int, 
        default=16,
        help="Batch size (reduce if GPU memory issues)"
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=640,
        help="Input image size in pixels (square)"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.001,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--patience", 
        type=int, 
        default=20,
        help="Early stopping patience (epochs without improvement)"
    )
    parser.add_argument(
        "--save-period", 
        type=int, 
        default=10,
        help="Save model checkpoint every N epochs"
    )
    
    # ========================================
    # Data Augmentation Parameters
    # ========================================
    parser.add_argument(
        "--degrees", 
        type=float, 
        default=10.0,
        help="Rotation augmentation range in degrees"
    )
    parser.add_argument(
        "--translate", 
        type=float, 
        default=0.1,
        help="Translation augmentation as fraction of image size"
    )
    parser.add_argument(
        "--scale", 
        type=float, 
        default=0.3,
        help="Scale augmentation range"
    )
    parser.add_argument(
        "--fliplr", 
        type=float, 
        default=0.5,
        help="Horizontal flip probability"
    )
    parser.add_argument(
        "--flipud", 
        type=float, 
        default=0.0,
        help="Vertical flip probability"
    )
    parser.add_argument(
        "--mosaic", 
        type=float, 
        default=0.9,
        help="Mosaic augmentation probability"
    )
    
    # ========================================
    # Control Flags
    # ========================================
    parser.add_argument(
        "--no-plots", 
        action="store_true",
        help="Disable training plots generation"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show verbose output including full error tracebacks"
    )
    
    return parser.parse_args()

def validate_args(args):
    """Validate command line arguments and show helpful error messages."""
    # Check if dataset file exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Error: Dataset file '{data_path}' does not exist!")
        print(f"💡 Please create the dataset or specify a different path with --data")
        return False
        
    return True

def main():
    """Main training function for chessboard polygon segmentation."""
    
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
    # Dataset Validation
    # ========================================
    print("🔍 Validating dataset format...")
    model = ChessBoardSegmentationModel()
    is_valid, message = model.validate_dataset_format(DATA_YAML)
    
    if not is_valid:
        print(f"❌ Dataset validation failed: {message}")
        print("💡 Please ensure your dataset has polygon annotations in YOLO segmentation format")
        return
    else:
        print(f"✅ Dataset validation passed: {message}")
    
    # ========================================
    # Training Information Display
    # ========================================
    print("\n🏁 Starting ChessBoard Polygon Segmentation Training")
    print("=" * 70)
    print(f"📁 Models will be saved to: {MODELS_FOLDER}")
    print(f"📊 Dataset: {DATA_YAML}")
    print(f"🏷️  Training name: {args.name}")
    print(f"🎯 COCO Pretrained Model: {args.pretrained_model}")
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
    print("🎯 Model Type: YOLOv8 Segmentation (polygon learning)")
    print("=" * 70)
    
    # ========================================
    # Model Training
    # ========================================
    print("🚀 Initializing ChessBoardSegmentationModel...")
    model = ChessBoardSegmentationModel(pretrained_model=args.pretrained_model)
    
    print("🎯 Starting segmentation training process...")
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
            # Segmentation specific parameters
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
        model.get_training_summary(results)
        
        # ========================================
        # Model Saving and Validation
        # ========================================
        # Handle the case where YOLO auto-increments the training name
        base_name = args.name
        potential_paths = [
            MODELS_FOLDER / base_name / "weights" / "best.pt",
            MODELS_FOLDER / f"{base_name}1" / "weights" / "best.pt",
            MODELS_FOLDER / f"{base_name}2" / "weights" / "best.pt",
            MODELS_FOLDER / f"{base_name}3" / "weights" / "best.pt",
            MODELS_FOLDER / f"{base_name}4" / "weights" / "best.pt",
            MODELS_FOLDER / f"{base_name}5" / "weights" / "best.pt",
        ]
        
        best_model_path = None
        for path in potential_paths:
            if path.exists():
                best_model_path = path
                break
        
        if not best_model_path:
            # Default to original path for error message
            best_model_path = MODELS_FOLDER / args.name / "weights" / "best.pt"
            
        if best_model_path.exists():
            print(f"✅ Best segmentation model saved at: {best_model_path}")
            print(f"🎯 Training completed successfully!")
            
            # Load the trained model for validation
            print("🔍 Loading trained segmentation model for validation...")
            trained_model = ChessBoardSegmentationModel(model_path=best_model_path)
            print(f"✅ Segmentation model loaded successfully!")
            
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
    print("🏁 Segmentation Training Complete!")
    print("\n🎯 Next Steps:")
    print("\n1️⃣  Test the segmentation model:")
    print("   Python: model.plot_eval('path/to/test_image.jpg')")
    print("   Python: polygon = model.get_polygon_coordinates('image.jpg')")
    if best_model_path and best_model_path.exists():
        print(f"   CLI:    python src/chess_board_detection/test_segmentation.py --model {best_model_path} --image path/to/image.jpg")
    
    print("\n2️⃣  Get polygon coordinates as JSON:")
    print("   Python: polygon, is_valid = model.get_polygon_coordinates('image.jpg')")
    print("   Python: print(f'Polygon has {polygon[\"num_points\"]} points')")
    
    print("\n3️⃣  Batch test multiple images:")
    if best_model_path and best_model_path.exists():
        print(f"   CLI:    python src/chess_board_detection/test_segmentation.py --model {best_model_path} --input-dir path/to/images/")
    
    print("\n💡 Key Differences from Detection Model:")
    print("   🔹 Learns exact polygon boundaries (not just corner points)")
    print("   🔹 Provides precise segmentation masks")
    print("   🔹 Can handle irregular chessboard shapes")
    print("   🔹 Better for perspective correction and board extraction")
    
    print("\n💡 COCO Pretrained Models Available:")
    print("   🔹 yolov8n-seg.pt (nano, ~6MB, fastest)")
    print("   🔹 yolov8s-seg.pt (small, ~22MB, fast)")
    print("   🔹 yolov8m-seg.pt (medium, ~52MB, balanced)")
    print("   🔹 yolov8l-seg.pt (large, ~104MB, accurate)")
    print("   🔹 yolov8x-seg.pt (extra large, ~136MB, most accurate)")
    print("   Use: --pretrained-model yolov8m-seg.pt")
    
    print("\n💡 More CLI options:")
    print("   • Add --help to any command for detailed options")
    print("   • Compare with detection model for different use cases")
    print("=" * 70)

if __name__ == "__main__":
    main() 