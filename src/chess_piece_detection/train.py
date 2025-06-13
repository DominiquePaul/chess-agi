#!/usr/bin/env python3
"""
Training script for Chess Piece Detection Model using YOLO

This script trains a YOLO object detection model to detect and classify chess pieces
from annotated chess board images.

Usage:
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
        --verbose
        
Available COCO-pretrained models:
    - yolov8n.pt (nano, ~6MB, fastest)
    - yolov8s.pt (small, ~22MB, fast, recommended)  
    - yolov8m.pt (medium, ~52MB, balanced)
    - yolov8l.pt (large, ~104MB, accurate)
    - yolov8x.pt (extra large, ~136MB, most accurate)
"""

import argparse
import os
from pathlib import Path
from src.chess_piece_detection.model import ChessModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def parse_args():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Train Chess Piece Detection Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ========================================
    # Dataset and Model Configuration
    # ========================================
    parser.add_argument(
        "--data", 
        type=str, 
        default=None,
        help="Path to YOLO dataset YAML file. If not provided, uses DATA_FOLDER_PATH env var + chess_pieces_merged/data.yaml"
    )
    parser.add_argument(
        "--pretrained-model", 
        type=str, 
        default="yolov8s.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="COCO-pretrained model to start training from (n=nano, s=small, m=medium, l=large, x=extra large)"
    )
    parser.add_argument(
        "--models-folder", 
        type=str, 
        default="models/chess_piece_detection",
        help="Root folder where trained models will be saved"
    )
    parser.add_argument(
        "--name", 
        type=str, 
        default="training_yolov8s",
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
        default=32,
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
        "--lrf", 
        type=float, 
        default=0.01,
        help="Final learning rate (lr * lrf)"
    )
    parser.add_argument(
        "--patience", 
        type=int, 
        default=15,
        help="Early stopping patience (epochs without improvement)"
    )
    parser.add_argument(
        "--save-period", 
        type=int, 
        default=10,
        help="Save model checkpoint every N epochs"
    )
    parser.add_argument(
        "--optimizer", 
        type=str, 
        default="AdamW",
        choices=["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"],
        help="Optimizer to use for training"
    )
    parser.add_argument(
        "--weight-decay", 
        type=float, 
        default=0.0005,
        help="Weight decay for regularization"
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
        default=0.2,
        help="Scale augmentation range"
    )
    parser.add_argument(
        "--shear", 
        type=float, 
        default=2.0,
        help="Shear augmentation range in degrees"
    )
    parser.add_argument(
        "--perspective", 
        type=float, 
        default=0.0001,
        help="Perspective augmentation (range 0-0.001)"
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
        default=1.0,
        help="Mosaic augmentation probability"
    )
    parser.add_argument(
        "--mixup", 
        type=float, 
        default=0.1,
        help="MixUp augmentation probability"
    )
    parser.add_argument(
        "--copy-paste", 
        type=float, 
        default=0.1,
        help="Copy-paste augmentation probability"
    )
    
    # ========================================
    # Advanced Training Parameters
    # ========================================
    parser.add_argument(
        "--warmup-epochs", 
        type=float, 
        default=3.0,
        help="Warmup epochs"
    )
    parser.add_argument(
        "--warmup-momentum", 
        type=float, 
        default=0.8,
        help="Warmup initial momentum"
    )
    parser.add_argument(
        "--warmup-bias-lr", 
        type=float, 
        default=0.1,
        help="Warmup initial bias learning rate"
    )
    parser.add_argument(
        "--box-loss", 
        type=float, 
        default=7.5,
        help="Box loss gain"
    )
    parser.add_argument(
        "--cls-loss", 
        type=float, 
        default=0.5,
        help="Classification loss gain"
    )
    parser.add_argument(
        "--dfl-loss", 
        type=float, 
        default=1.5,
        help="Distribution focal loss gain"
    )
    
    # ========================================
    # Evaluation and Output Options
    # ========================================
    parser.add_argument(
        "--eval-individual", 
        action="store_true",
        help="Evaluate on individual datasets (dominique, roboflow) if available"
    )
    parser.add_argument(
        "--push-to-hf", 
        action="store_true",
        help="Push trained model to Hugging Face Hub"
    )
    parser.add_argument(
        "--hf-repo-id", 
        type=str, 
        default="chess-piece-detector",
        help="Hugging Face repository ID for model upload"
    )
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

def get_default_data_path():
    """Get default data path from environment variable."""
    data_folder = os.environ.get("DATA_FOLDER_PATH")
    if not data_folder:
        return None
    return Path(data_folder) / "chess_pieces_merged" / "data.yaml"

def validate_args(args):
    """Validate command line arguments and show helpful error messages."""
    # Handle default data path
    if args.data is None:
        default_data = get_default_data_path()
        if default_data and default_data.exists():
            args.data = str(default_data)
            print(f"🔍 Using default dataset: {args.data}")
        else:
            print("❌ Error: No dataset specified and default not found!")
            if not os.environ.get("DATA_FOLDER_PATH"):
                print("💡 Please set DATA_FOLDER_PATH environment variable or use --data")
            else:
                print(f"💡 Please create the default dataset at: {default_data}")
                print("💡 Or specify a different dataset with --data")
            return False
    
    # Check if dataset file exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Error: Dataset file '{data_path}' does not exist!")
        print(f"💡 Please create the dataset or specify a different path with --data")
        return False
        
    return True

def main():
    """Main training function for chess piece detection."""
    
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
    print("\n🎯 Starting Chess Piece Detection Training")
    print("=" * 70)
    print(f"📁 Models will be saved to: {MODELS_FOLDER}")
    print(f"📊 Dataset: {DATA_YAML}")
    print(f"🏷️  Training name: {args.name}")
    print(f"🎯 COCO Pretrained Model: {args.pretrained_model}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"📦 Batch size: {args.batch}")
    print(f"🖼️  Image size: {args.imgsz}")
    print(f"📈 Learning rate: {args.lr} -> {args.lr * args.lrf}")
    print(f"⚙️  Optimizer: {args.optimizer}")
    print(f"⏰ Patience: {args.patience}")
    print(f"💾 Save period: {args.save_period}")
    print(f"🎲 Augmentation: degrees={args.degrees}, translate={args.translate}, scale={args.scale}")
    print(f"🔄 Flips: horizontal={args.fliplr}, vertical={args.flipud}")
    print(f"🧩 Advanced: mosaic={args.mosaic}, mixup={args.mixup}, copy-paste={args.copy_paste}")
    print(f"📊 Plots: {'Disabled' if args.no_plots else 'Enabled'}")
    print("🎯 Model Type: YOLOv8 Object Detection (chess piece classification)")
    print("=" * 70)
    
    # ========================================
    # Model Training
    # ========================================
    print("🚀 Initializing Chess Piece Detection Model...")
    print(f"📦 Using {args.pretrained_model} pretrained checkpoint for transfer learning")
    
    model = ChessModel(pretrained_checkpoint=args.pretrained_model)
    
    print("🎯 Starting detection training process...")
    try:
        results = model.train(
            data_path=DATA_YAML,
            epochs=args.epochs,
            batch=args.batch,
            lr0=args.lr,
            lrf=args.lrf,
            imgsz=args.imgsz,
            plots=not args.no_plots,
            project=MODELS_FOLDER,
            name=args.name,
            save_period=args.save_period,
            patience=args.patience,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            # Data augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=args.degrees,
            translate=args.translate,
            scale=args.scale,
            shear=args.shear,
            perspective=args.perspective,
            flipud=args.flipud,
            fliplr=args.fliplr,
            mosaic=args.mosaic,
            mixup=args.mixup,
            copy_paste=args.copy_paste,
            # Advanced parameters
            warmup_epochs=args.warmup_epochs,
            warmup_momentum=args.warmup_momentum,
            warmup_bias_lr=args.warmup_bias_lr,
            box=args.box_loss,
            cls=args.cls_loss,
            dfl=args.dfl_loss,
        )
        
        # Print training summary
        print("📊 Printing training summary...")
        model.print_training_summary(results)
        
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
            print(f"✅ Best detection model saved at: {best_model_path}")
            print(f"🎯 Training completed successfully!")
            
            # Load the trained model for evaluation
            print("🔍 Loading trained model for evaluation...")
            trained_model = ChessModel(model_path=best_model_path)
            print(f"✅ Model loaded successfully!")
            
            # ========================================
            # Model Evaluation
            # ========================================
            print("📊 Evaluating model on merged dataset...")
            trained_model.evaluate(DATA_YAML)
            
            # Optional: Evaluate on individual datasets
            if args.eval_individual:
                data_folder = os.environ.get("DATA_FOLDER_PATH")
                if data_folder:
                    dominique_yaml = Path(data_folder) / "chess_pieces_dominique" / "data.yaml"
                    roboflow_yaml = Path(data_folder) / "chess_pieces_roboflow" / "data.yaml"
                    
                    if dominique_yaml.exists():
                        print("\n📊 Evaluating model on Dominique dataset...")
                        trained_model.evaluate(dominique_yaml)
                    
                    if roboflow_yaml.exists():
                        print("\n📊 Evaluating model on Roboflow dataset...")
                        trained_model.evaluate(roboflow_yaml)
            
            # ========================================  
            # Optional: Push to Hugging Face
            # ========================================
            if args.push_to_hf:
                print(f"🤗 Pushing model to Hugging Face Hub: {args.hf_repo_id}")
                try:
                    trained_model.push_to_huggingface(
                        repo_id=args.hf_repo_id,
                        commit_message=f"Upload chess piece detection model ({args.pretrained_model} -> {args.epochs} epochs, {args.optimizer} optimizer)",
                        private=False
                    )
                    print("✅ Model successfully pushed to Hugging Face!")
                except Exception as e:
                    print(f"⚠️  Failed to push to Hugging Face: {e}")
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
    print("🏁 Chess Piece Detection Training Complete!")
    print("\n🎯 Next Steps:")
    print("\n1️⃣  Test the detection model:")
    print("   Python: model.plot_eval('path/to/test_image.jpg')")
    print("   Python: results = model.predict('image.jpg', conf=0.5)")
    if best_model_path and best_model_path.exists():
        print(f"   CLI:    python -c \"from src.chess_piece_detection.model import ChessModel; m=ChessModel('{best_model_path}'); m.plot_eval('image.jpg')\"")
    
    print("\n2️⃣  Make predictions:")
    print("   Python: results = model.predict('image.jpg', conf=0.25)")
    print("   Python: for box in results.boxes: print(f'{results.names[int(box.cls)]}: {box.conf:.2f}')")
    
    print("\n3️⃣  Compare model versions:")
    print("   Create comparison plots between different trained models")
    
    print("\n💡 Available Chess Piece Classes:")
    print("   Usually includes: King, Queen, Rook, Bishop, Knight, Pawn (for both colors)")
    
    print("\n💡 COCO Pretrained Models Available:")
    print("   🔹 yolov8n.pt (nano, ~6MB, fastest)")
    print("   🔹 yolov8s.pt (small, ~22MB, fast, recommended)")
    print("   🔹 yolov8m.pt (medium, ~52MB, balanced)")
    print("   🔹 yolov8l.pt (large, ~104MB, accurate)")
    print("   🔹 yolov8x.pt (extra large, ~136MB, most accurate)")
    print("   Use: --pretrained-model yolov8m.pt")
    
    print("\n💡 More CLI options:")
    print("   • Add --help to see all available parameters")
    print("   • Use --eval-individual to test on separate datasets")
    print("   • Use --push-to-hf to upload model to Hugging Face")
    print("   • Adjust augmentation parameters for better generalization")
    print("=" * 70)

if __name__ == "__main__":
    main()
