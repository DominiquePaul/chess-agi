#!/usr/bin/env python3
"""
Training script for ChessBoardSegmentationModel - Polygon Segmentation

This script trains a YOLO segmentation model to learn the exact polygon boundaries
of chessboards from polygon annotation data.

Usage:
    # Basic training with default settings (YOLO11s-seg COCO pretrained)
    python src/chess_board_detection/yolo/segmentation/train_segmentation.py --epochs 100

    # Choose different COCO-pretrained model size
    python src/chess_board_detection/yolo/segmentation/train_segmentation.py \
        --pretrained-model yolo11m-seg.pt \
        --epochs 100

    # Custom training parameters
    python src/chess_board_detection/yolo/segmentation/train_segmentation.py \
        --data data/chessboard_segmentation/chess-board-4/data.yaml \
        --pretrained-model yolo11s-seg.pt \
        --epochs 100 \
        --batch 16 \
        --imgsz 640

    # Complete example with all options (including HF upload)
    python src/chess_board_detection/yolo/segmentation/train_segmentation.py \
        --data data/chessboard_segmentation/chess-board-4/data.yaml \
        --pretrained-model yolo11l-seg.pt \
        --models-folder artifacts/models/chess_board_segmentation \
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
        --hf-repo-id "username/chess-board-segmentation" \
        --verbose

Available COCO-pretrained models:
    - yolo11n-seg.pt (nano, ~6MB, fastest)
    - yolo11s-seg.pt (small, ~22MB, fast, recommended)
    - yolo11m-seg.pt (medium, ~52MB, balanced)
    - yolo11l-seg.pt (large, ~104MB, accurate)
    - yolo11x-seg.pt (extra large, ~136MB, most accurate)
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.chess_board_detection.yolo.segmentation.segmentation_model import (
    ChessBoardSegmentationModel,
)

# Load environment variables from .env file
load_dotenv()


def parse_args():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Train ChessBoard Polygon Segmentation Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ========================================
    # Dataset and Model Configuration
    # ========================================
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to YOLO dataset YAML file containing polygon annotations. If not provided, uses DATA_FOLDER_PATH env var + chessboard_segmentation/data.yaml, or auto-downloads from HF",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="dopaul/chess-board-segmentation",
        help="HuggingFace dataset to use if local data not found (default: dopaul/chess-board-segmentation)",
    )
    parser.add_argument(
        "--auto-download-hf",
        action="store_true",
        default=True,
        help="Automatically download from HuggingFace if local dataset not found (default: True)",
    )
    parser.add_argument(
        "--force-download-hf",
        action="store_true",
        help="Force re-download from HuggingFace even if local dataset exists",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="yolo11s-seg.pt",
        choices=[
            "yolo11n-seg.pt",
            "yolo11s-seg.pt",
            "yolo11m-seg.pt",
            "yolo11l-seg.pt",
            "yolo11x-seg.pt",
        ],
        help="COCO-pretrained YOLO11 model to start training from (n=nano, s=small, m=medium, l=large, x=extra large)",
    )
    parser.add_argument(
        "--models-folder",
        type=str,
        default="artifacts/models/chess_board_segmentation",
        help="Root folder where trained models will be saved",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="polygon_segmentation_training",
        help="Name for this training run (creates subfolder in models-folder)",
    )

    # ========================================
    # Core Training Parameters
    # ========================================
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (reduce if GPU memory issues)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size in pixels (square)")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (epochs without improvement)",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=10,
        help="Save model checkpoint every N epochs",
    )

    # ========================================
    # Data Augmentation Parameters
    # ========================================
    parser.add_argument(
        "--degrees",
        type=float,
        default=10.0,
        help="Rotation augmentation range in degrees",
    )
    parser.add_argument(
        "--translate",
        type=float,
        default=0.1,
        help="Translation augmentation as fraction of image size",
    )
    parser.add_argument("--scale", type=float, default=0.3, help="Scale augmentation range")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip probability")
    parser.add_argument("--flipud", type=float, default=0.0, help="Vertical flip probability")
    parser.add_argument("--mosaic", type=float, default=0.9, help="Mosaic augmentation probability")

    # ========================================
    # Control Flags
    # ========================================
    parser.add_argument("--no-plots", action="store_true", help="Disable training plots generation")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output including full error tracebacks",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help="Hugging Face repository ID for model upload (e.g., 'username/chess-board-segmentation'). If specified, model will be automatically pushed to HF Hub after training.",
    )

    return parser.parse_args()


def get_default_data_path():
    """Get the default data path based on environment variable."""
    data_folder = os.environ.get("DATA_FOLDER_PATH")
    if data_folder:
        return Path(data_folder) / "chessboard_segmentation" / "data.yaml"
    return None


def auto_download_hf_dataset(hf_repo: str, force_download: bool = False) -> Path | None:
    """
    Auto-download chess board segmentation dataset from HuggingFace if not available locally.

    Args:
        hf_repo: HuggingFace repository name (e.g., "dopaul/chess-board-segmentation-yolo")
        force_download: Force re-download even if local dataset exists

    Returns:
        Path to data.yaml file if successful, None if failed
    """
    try:
        # Import here to avoid dependency issues if not needed
        from src.data_prep.download_from_hf import DATASET_LOCAL_NAMES, download_dataset

        # Get local path
        data_folder = os.environ.get("DATA_FOLDER_PATH")
        if not data_folder:
            print("❌ DATA_FOLDER_PATH environment variable not set")
            return None

        data_path = Path(data_folder)

        # Determine local dataset name
        local_name = DATASET_LOCAL_NAMES.get(hf_repo)
        if not local_name:
            # Create local name from repo name
            local_name = hf_repo.split("/")[-1].replace("-", "_")

        local_dataset_path = data_path / local_name
        data_yaml_path = local_dataset_path / "data.yaml"

        # Check if dataset already exists
        if data_yaml_path.exists() and not force_download:
            print(f"✅ Found existing dataset at: {data_yaml_path}")
            return data_yaml_path

        # Download dataset
        print(f"📥 Auto-downloading dataset from HuggingFace: {hf_repo}")
        print(f"📁 Saving to: {local_dataset_path}")

        success = download_dataset(hf_repo, local_name, data_path, convert_to_yolo=True, task_type="segmentation")

        if success and data_yaml_path.exists():
            print(f"✅ Successfully downloaded and converted dataset to: {data_yaml_path}")
            return data_yaml_path
        else:
            print(f"❌ Failed to download dataset from {hf_repo}")
            return None

    except ImportError:
        print("❌ Cannot import HuggingFace download functionality")
        print("💡 Please ensure 'datasets' and 'huggingface_hub' are installed")
        return None
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return None


def validate_args(args):
    """Validate command line arguments and show helpful error messages."""

    # Handle force download first
    if args.force_download_hf:
        print(f"🔄 Force downloading dataset from HuggingFace: {args.hf_dataset}")
        downloaded_path = auto_download_hf_dataset(args.hf_dataset, force_download=True)
        if downloaded_path:
            args.data = str(downloaded_path)
            print(f"✅ Using force-downloaded dataset: {args.data}")
            return True
        else:
            print("❌ Failed to force download dataset from HuggingFace")
            return False

    # Handle default data path
    if args.data is None:
        default_data = get_default_data_path()
        if default_data and default_data.exists():
            args.data = str(default_data)
            print(f"🔍 Using default dataset: {args.data}")
        else:
            # Try auto-download from HuggingFace
            if args.auto_download_hf:
                print(f"🔍 Local dataset not found, attempting auto-download from HuggingFace: {args.hf_dataset}")
                downloaded_path = auto_download_hf_dataset(args.hf_dataset, force_download=False)
                if downloaded_path:
                    args.data = str(downloaded_path)
                    print(f"✅ Using auto-downloaded dataset: {args.data}")
                    return True

            print("❌ Error: No dataset specified and default not found!")
            if not os.environ.get("DATA_FOLDER_PATH"):
                print("💡 Please set DATA_FOLDER_PATH environment variable or use --data")
            else:
                print(f"💡 Please create the default dataset at: {default_data}")
                print("💡 Or specify a different dataset with --data")
                print(f"💡 Or use --auto-download-hf to download from HuggingFace: {args.hf_dataset}")
            return False

    # Check if dataset file exists
    data_path = Path(args.data)
    if not data_path.exists():
        # Try auto-download from HuggingFace if enabled
        if args.auto_download_hf:
            print(
                f"🔍 Dataset file '{data_path}' not found, attempting auto-download from HuggingFace: {args.hf_dataset}"
            )
            downloaded_path = auto_download_hf_dataset(args.hf_dataset, force_download=False)
            if downloaded_path:
                args.data = str(downloaded_path)
                print(f"✅ Using auto-downloaded dataset: {args.data}")
                return True

        print(f"❌ Error: Dataset file '{data_path}' does not exist!")
        print("💡 Please create the dataset or specify a different path with --data")
        print(f"💡 Or use --auto-download-hf to download from HuggingFace: {args.hf_dataset}")
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
    print("🎯 Model Type: YOLO11 Segmentation (polygon learning)")
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
            print("🎯 Training completed successfully!")

            # Load the trained model for validation
            print("🔍 Loading trained segmentation model for validation...")
            trained_model = ChessBoardSegmentationModel(model_path=best_model_path)
            print("✅ Segmentation model loaded successfully!")

            # ========================================
            # Optional: Push to Hugging Face
            # ========================================
            if args.hf_repo_id:
                print(f"🤗 Pushing model to Hugging Face Hub: {args.hf_repo_id}")
                try:
                    trained_model.push_to_huggingface(
                        repo_id=args.hf_repo_id,
                        commit_message=f"Upload chess board segmentation model ({args.pretrained_model} -> {args.epochs} epochs)",
                        private=False,
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
    print("🏁 Segmentation Training Complete!")
    print("\n🎯 Next Steps:")
    print("\n1️⃣  Test the segmentation model:")
    print("   Python: model.plot_eval('path/to/test_image.jpg')")
    print("   Python: polygon = model.get_polygon_coordinates('image.jpg')")
    if best_model_path and best_model_path.exists():
        print(
            f"   CLI:    python src/chess_board_detection/yolo/segmentation/test_segmentation.py --model {best_model_path} --image path/to/image.jpg"
        )

    print("\n2️⃣  Get polygon coordinates as JSON:")
    print("   Python: polygon, is_valid = model.get_polygon_coordinates('image.jpg')")
    print("   Python: print(f'Polygon has {polygon[\"num_points\"]} points')")

    print("\n3️⃣  Batch test multiple images:")
    if best_model_path and best_model_path.exists():
        print(
            f"   CLI:    python src/chess_board_detection/yolo/segmentation/test_segmentation.py --model {best_model_path} --input-dir path/to/images/"
        )

    print("\n💡 Key Differences from Detection Model:")
    print("   🔹 Learns exact polygon boundaries (not just corner points)")
    print("   🔹 Provides precise segmentation masks")
    print("   🔹 Can handle irregular chessboard shapes")
    print("   🔹 Better for perspective correction and board extraction")

    print("\n💡 COCO Pretrained Models Available:")
    print("   🔹 yolo11n-seg.pt (nano, ~6MB, fastest)")
    print("   🔹 yolo11s-seg.pt (small, ~22MB, fast, recommended)")
    print("   🔹 yolo11m-seg.pt (medium, ~52MB, balanced)")
    print("   🔹 yolo11l-seg.pt (large, ~104MB, accurate)")
    print("   🔹 yolo11x-seg.pt (extra large, ~136MB, most accurate)")
    print("   Use: --pretrained-model yolo11m-seg.pt")

    print("\n💡 More CLI options:")
    print("   • Add --help to any command for detailed options")
    print("   • Use --hf-repo-id username/model-name to upload model to Hugging Face")
    print("   • Compare with detection model for different use cases")
    print("=" * 70)


if __name__ == "__main__":
    main()
