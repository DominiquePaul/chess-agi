#!/usr/bin/env python3
"""
Training script for ChessBoardModel - Corner Detection

This script trains a YOLO model to detect the 4 corners of a chessboard.

Usage:
    # Basic training with default settings (YOLO11s COCO pretrained)
    python src/chess_board_detection/yolo/train.py --epochs 100

    # Choose different COCO-pretrained model size
    python src/chess_board_detection/yolo/train.py \
        --pretrained-model yolo11m.pt \
        --epochs 100

    # Run from project root as module
    python -m src.chess_board_detection.yolo.train

    # Custom training parameters
    python src/chess_board_detection/yolo/train.py --data data/my_dataset.yaml --epochs 100 --batch 32

    # Complete example with all options and HuggingFace upload
    python src/chess_board_detection/yolo/train.py \
        --data data/chessboard_corners/chess-board-box-3/data.yaml \
        --pretrained-model yolo11l.pt \
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

Available COCO-pretrained models:
    - yolo11n.pt (nano, ~2.6MB, fastest)
    - yolo11s.pt (small, ~9.4MB, fast, recommended)
    - yolo11m.pt (medium, ~20.1MB, balanced)
    - yolo11l.pt (large, ~25.3MB, accurate)
    - yolo11x.pt (extra large, ~56.9MB, most accurate)
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.chess_board_detection.yolo.model import ChessBoardModel

# Load environment variables from .env file
load_dotenv()


def parse_args():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Train ChessBoard Corner Detection Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ========================================
    # Dataset and Model Configuration
    # ========================================
    parser.add_argument(
        "--data",
        type=str,
        default="data/chessboard_corners/chess-board-box-3/data.yaml",
        help="Path to YOLO dataset YAML file containing train/val/test splits and class names",
    )
    parser.add_argument(
        "--models-folder",
        type=str,
        default="models/chess_board_detection",
        help="Root folder where trained models will be saved",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="corner_detection_training",
        help="Name for this training run (creates subfolder in models-folder)",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="yolo11s.pt",
        choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
        help="COCO-pretrained YOLO11 model to start training from (n=nano, s=small, m=medium, l=large, x=extra large)",
    )

    # ========================================
    # Core Training Parameters
    # ========================================
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (complete passes through dataset)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (number of images processed together). Reduce if GPU memory issues",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size in pixels (square). Common values: 416, 640, 832",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate. Lower values (0.0001) for fine-tuning, higher (0.01) for fresh training",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (epochs without improvement before stopping)",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=10,
        help="Save model checkpoint every N epochs (set to -1 to disable periodic saves)",
    )

    # ========================================
    # Data Augmentation Parameters
    # ========================================
    parser.add_argument(
        "--degrees",
        type=float,
        default=5.0,
        help="Rotation augmentation range in degrees (±degrees). Use small values for corner detection",
        # Note: For small datasets, consider increasing to 10-15 degrees to reduce overfitting
    )
    parser.add_argument(
        "--translate",
        type=float,
        default=0.05,
        help="Translation augmentation as fraction of image size (0.0-1.0)",
        # Note: For small datasets, consider increasing to 0.1-0.2 to reduce overfitting
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.2,
        help="Scale augmentation range (0.0-1.0). Higher values = more scale variation",
        # Note: For small datasets, consider increasing to 0.3-0.5 to reduce overfitting
    )
    parser.add_argument(
        "--fliplr",
        type=float,
        default=0.5,
        help="Horizontal flip probability (0.0-1.0). Good for most detection tasks",
    )
    parser.add_argument(
        "--flipud",
        type=float,
        default=0.0,
        help="Vertical flip probability (0.0-1.0). Usually 0.0 for corner detection",
    )
    parser.add_argument(
        "--mosaic",
        type=float,
        default=0.8,
        help="Mosaic augmentation probability (0.0-1.0). Combines 4 images into 1",
    )

    # ========================================
    # HuggingFace Model Hub Configuration
    # ========================================
    parser.add_argument(
        "--upload-hf",
        action="store_true",
        help="Upload trained model to HuggingFace Model Hub after training completes",
    )
    parser.add_argument(
        "--hf-model-name",
        type=str,
        default="",
        help="HuggingFace model repository name (format: username/model-name). Required if --upload-hf is used",
    )

    # ========================================
    # Control Flags
    # ========================================
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable training plots generation (faster training, less disk space)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output including full error tracebacks",
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
        print("💡 Please create the dataset or specify a different path with --data")
        print("💡 Expected YOLO format with train/val/test image folders and labels")
        return False

    return True


def validate_and_fix_dataset_paths(data_yaml_path: Path):
    """Validate and fix dataset paths in data.yaml file.

    This is a fallback to handle cases where the download script didn't fix
    the paths properly, ensuring the training script is robust.
    """
    import yaml

    try:
        # Read the data.yaml file
        with open(data_yaml_path) as f:
            data_config = yaml.safe_load(f)

        paths_fixed = False

        # Check and fix each split
        for split in ["train", "val", "test"]:
            if split in data_config:
                path = data_config[split]

                # Check if path exists
                if not Path(path).exists():
                    # Try to guess the correct path
                    if path.startswith("../"):
                        # This is likely a broken relative path from Roboflow
                        # Try to find the dataset directory
                        data_yaml_parent = data_yaml_path.parent

                        # Look for directories that might contain the dataset
                        potential_paths = []
                        for item in data_yaml_parent.iterdir():
                            if item.is_dir() and not item.name.startswith("."):
                                # Check if this directory has the expected structure
                                folder_name = path.split("/")[-2]  # e.g., 'train' from '../train/images'
                                candidate_path = item / folder_name / "images"
                                if candidate_path.exists():
                                    potential_paths.append(str(candidate_path.relative_to(Path.cwd())))

                        if potential_paths:
                            # Use the first valid path found
                            new_path = potential_paths[0]
                            data_config[split] = new_path
                            paths_fixed = True
                            print(f"🔧 Auto-fixed {split} path: {path} → {new_path}")

        # If we fixed any paths, write the updated config back
        if paths_fixed:
            with open(data_yaml_path, "w") as f:
                yaml.dump(data_config, f, default_flow_style=False)
            print("✅ Updated data.yaml with corrected paths")

    except Exception as e:
        print(f"⚠️  Warning: Could not validate/fix dataset paths: {e}")


def get_huggingface_username():
    """Try to get the current HuggingFace username if logged in."""
    try:
        from huggingface_hub import whoami

        user_info = whoami()
        if user_info and "name" in user_info:
            return user_info["name"]
    except ImportError:
        # HuggingFace Hub not installed
        pass
    except Exception:
        # Not logged in or other error
        pass
    return None


def generate_upload_commands(best_model_path, training_name):
    """Generate ready-to-run upload commands with actual values filled in."""
    # Try to get HuggingFace username
    hf_username = get_huggingface_username()

    # Generate a repository name based on training name
    # Convert training name to a valid repo name (lowercase, hyphens)
    repo_suffix = training_name.lower().replace("_", "-").replace(" ", "-")
    if not repo_suffix.startswith("chess"):
        repo_suffix = f"chess-{repo_suffix}"

    if hf_username:
        # We have a username, generate complete command
        repo_name = f"{hf_username}/{repo_suffix}"
        return {
            "python_cmd": f"model.push_to_huggingface('{repo_name}')",
            "cli_cmd": f"python src/chess_board_detection/upload_hf.py --model {best_model_path} --repo-name {repo_name}",
            "has_username": True,
            "repo_name": repo_name,
        }
    else:
        # No username detected, provide template with placeholder
        repo_name = f"username/{repo_suffix}"
        return {
            "python_cmd": f"model.push_to_huggingface('{repo_name}')",
            "cli_cmd": f"python src/chess_board_detection/upload_hf.py --model {best_model_path} --repo-name {repo_name}",
            "has_username": False,
            "repo_name": repo_name,
            "suggested_repo": repo_suffix,
        }


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

    # Validate and fix dataset paths if needed (fallback safety check)
    validate_and_fix_dataset_paths(DATA_YAML)

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
        print("🤗 HuggingFace Upload: ENABLED")
        print(f"📤 Model will be uploaded to: {args.hf_model_name}")
        print("⚠️  Make sure you're logged in with: huggingface-cli login")
    else:
        print("🤗 HuggingFace Upload: Disabled")

    print("=" * 70)

    # ========================================
    # Model Training
    # ========================================
    print("🚀 Initializing ChessBoardModel...")
    model = ChessBoardModel(pretrained_checkpoint=args.pretrained_model)

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
        # Handle the case where YOLO auto-increments the training name
        base_name = args.name
        potential_paths = [
            MODELS_FOLDER / base_name / "weights" / "best.pt",
            MODELS_FOLDER / f"{base_name}1" / "weights" / "best.pt",
            MODELS_FOLDER / f"{base_name}2" / "weights" / "best.pt",
            MODELS_FOLDER / f"{base_name}3" / "weights" / "best.pt",
            MODELS_FOLDER / f"{base_name}4" / "weights" / "best.pt",
            MODELS_FOLDER / f"{base_name}5" / "weights" / "best.pt",
            MODELS_FOLDER / f"{base_name}6" / "weights" / "best.pt",
            MODELS_FOLDER / f"{base_name}7" / "weights" / "best.pt",
            MODELS_FOLDER / f"{base_name}8" / "weights" / "best.pt",
            MODELS_FOLDER / f"{base_name}9" / "weights" / "best.pt",
            MODELS_FOLDER / f"{base_name}10" / "weights" / "best.pt",
            MODELS_FOLDER / f"{base_name}11" / "weights" / "best.pt",
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
            print(f"✅ Best model saved at: {best_model_path}")
            print("🎯 Training completed successfully!")

            # Load the trained model for validation
            print("🔍 Loading trained model for validation...")
            trained_model = ChessBoardModel(model_path=best_model_path)
            print("✅ Model loaded successfully!")

            # ========================================
            # HuggingFace Upload (if requested)
            # ========================================
            if args.upload_hf:
                print("=" * 70)
                print(f"🤗 Uploading model to HuggingFace: {args.hf_model_name}")
                try:
                    trained_model.push_to_huggingface(args.hf_model_name)
                    print("✅ Model successfully uploaded to HuggingFace!")
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
    print("\n🎯 Next Steps:")
    print("\n1️⃣  Test the model:")
    print("   Python: model.plot_eval('path/to/test_image.jpg')")
    print("   Python: model.get_corner_coordinates('image.jpg')")
    if best_model_path and best_model_path.exists():
        print(
            f"   CLI:    python src/chess_board_detection/test_model.py --model {best_model_path} --image path/to/image.jpg"
        )
    else:
        print(
            "   CLI:    python src/chess_board_detection/test_model.py --model <MODEL_PATH> --image path/to/image.jpg"
        )

    print("\n2️⃣  Get coordinates as JSON:")
    print("   Python: corners = model.get_corner_coordinates('image.jpg'); print(corners)")
    if best_model_path and best_model_path.exists():
        print(
            f"   CLI:    python src/chess_board_detection/test_model.py --model {best_model_path} --image image.jpg --json-output"
        )
    else:
        print(
            "   CLI:    python src/chess_board_detection/test_model.py --model <MODEL_PATH> --image image.jpg --json-output"
        )

    print("\n3️⃣  Batch test multiple images:")
    if best_model_path and best_model_path.exists():
        print(
            f"   CLI:    python src/chess_board_detection/test_model.py --model {best_model_path} --input-dir path/to/images/"
        )
    else:
        print(
            "   CLI:    python src/chess_board_detection/test_model.py --model <MODEL_PATH> --input-dir path/to/images/"
        )

    if not args.upload_hf:
        # Generate ready-to-run upload commands
        if best_model_path and best_model_path.exists():
            upload_cmds = generate_upload_commands(best_model_path, args.name)

            print("\n4️⃣  Upload to HuggingFace:")
            print(f"   Python: {upload_cmds['python_cmd']}")
            print(
                f"   CLI:    python src/chess_board_detection/upload_hf.py --model {best_model_path} --repo-name {upload_cmds['repo_name']}"
            )

            if upload_cmds["has_username"]:
                print(f"   ✅ Ready to run! Detected HuggingFace user: {upload_cmds['repo_name'].split('/')[0]}")
            else:
                print("   ⚠️  Replace 'username' with your HuggingFace username")
                print(f"   💡 Suggested repo name: {upload_cmds['suggested_repo']}")
                print("   💡 Login first: huggingface-cli login")
        else:
            print("\n4️⃣  Upload to HuggingFace:")
            print("   ❌ Model path not found - check training output for actual model location")
            print("   💡 Look for 'Results saved to...' in training output")

    print("\n💡 More CLI options:")
    print("   • Add --help to any command for detailed options")
    print("   • View all commands: python src/chess_board_detection/cli.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
