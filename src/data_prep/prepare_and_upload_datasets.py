"""
Prepare and upload chess datasets to Hugging Face Hub.
This script orchestrates the complete dataset preparation pipeline for both
chess piece detection and chessboard corner detection.
"""

import argparse
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

# Import our existing functions
from src.data_prep.roboflow_chess import (
    clean_roboflow_data,
    download_and_move_roboflow_data,
    update_data_yaml,
)
from src.data_prep.upload_to_hf import ChessDatasetUploader

load_dotenv()

DATA_FOLDER_PATH = Path(os.environ["DATA_FOLDER_PATH"])
HF_USERNAME = os.environ.get("HF_USERNAME")


def check_prerequisites():
    """Check if prerequisites are met"""
    print("🔍 Checking prerequisites...")

    # Check environment variables
    if not DATA_FOLDER_PATH:
        print("❌ DATA_FOLDER_PATH not set in environment")
        return False

    if not HF_USERNAME:
        print("❌ HF_USERNAME not set in environment")
        return False

    # Check Kaggle API
    kaggle_path = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_path.exists():
        print("❌ Kaggle API key not found at ~/.kaggle/kaggle.json")
        print("   Please create an API key at www.kaggle.com/settings")
        return False

    # Check Roboflow CLI
    try:
        subprocess.run(["roboflow", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Roboflow CLI not found. Install with: pip install roboflow")
        return False

    print("✅ All prerequisites met!")
    return True


def download_roboflow_chess_pieces():
    """Download and process chess pieces dataset from Kaggle (Roboflow source)"""
    print("\n📦 Downloading Chess Pieces Dataset (Roboflow/Kaggle)...")
    print("   📊 Dataset: Chess piece detection with 12 classes")
    print("   📍 Source: Kaggle (originally from Roboflow)")
    print("   🎯 Purpose: Training chess piece detection models")

    try:
        dataset_name = Path("chess_pieces_roboflow")

        # Download and move data
        download_and_move_roboflow_data(DATA_FOLDER_PATH, dataset_name)

        # Clean labels (remove extra bishop class)
        clean_roboflow_data(DATA_FOLDER_PATH, dataset_name)

        # Update data.yaml
        update_data_yaml(DATA_FOLDER_PATH, dataset_name)

        print("✅ Chess pieces dataset (Roboflow/Kaggle) processed successfully")
        return True

    except Exception as e:
        print(f"❌ Error processing chess pieces dataset (Roboflow/Kaggle): {e}")
        return False


def download_dominique_chess_pieces():
    """Download Dominique's chess pieces dataset from Roboflow"""
    print("\n📦 Downloading Chess Pieces Dataset (Dominique)...")
    print("   📊 Dataset: Chess piece detection with 12 classes")
    print("   📍 Source: Roboflow (gustoguardian/chess-piece-detection-bltvi/8)")
    print("   🎯 Purpose: Training chess piece detection models (enhanced)")

    try:
        cmd = [
            "roboflow",
            "download",
            "-f",
            "yolov8",
            "-l",
            str(DATA_FOLDER_PATH / "chess_pieces_dominique"),
            "gustoguardian/chess-piece-detection-bltvi/8",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Chess pieces dataset (Dominique) downloaded successfully")
            return True
        else:
            print(f"❌ Error downloading chess pieces dataset (Dominique): {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error downloading chess pieces dataset (Dominique): {e}")
        return False


def download_chessboard_corners_dominique():
    """Download chessboard corner detection dataset from Roboflow (Dominique)"""
    print("\n📦 Downloading Chessboard Corners Dataset (Dominique)...")
    print("   📊 Dataset: Chessboard corner detection (4 corners per board)")
    print("   📍 Source: Roboflow (gustoguardian/chess-board-box/3)")
    print("   🎯 Purpose: Training chessboard corner detection models")

    try:
        # Import the download function and arguments class
        from argparse import Namespace

        from data_prep.download_data import download_roboflow_dataset

        # Create mock args for the download function
        args = Namespace(
            data_dir=str(DATA_FOLDER_PATH / "chessboard_corners"),
            project="gustoguardian/chess-board-box",
            version=3,
            format="yolov8",
            verbose=False,
        )

        # Call the download function
        dataset_folder = download_roboflow_dataset(args)

        if dataset_folder:
            print("✅ Chessboard corners dataset (Dominique) downloaded successfully")
            return True
        else:
            print("❌ Download failed - no dataset folder returned")
            return False

    except Exception as e:
        print(f"❌ Error downloading chessboard corners dataset (Dominique): {e}")
        return False


def download_chessboard_segmentation_dominique():
    """Download chessboard segmentation dataset from Roboflow (Dominique)"""
    print("\n📦 Downloading Chessboard Segmentation Dataset (Dominique)...")
    print("   📊 Dataset: Chessboard segmentation with polygon boundaries")
    print("   📍 Source: Roboflow (gustoguardian/chess-board-i0ptl/3)")
    print("   🎯 Purpose: Training chessboard segmentation models for precise polygon detection")

    try:
        # Import the download function and arguments class
        from argparse import Namespace

        from data_prep.download_data import download_roboflow_dataset

        # Create mock args for the download function
        args = Namespace(
            data_dir=str(DATA_FOLDER_PATH / "chessboard_segmentation"),
            project="gustoguardian/chess-board-i0ptl",
            version=3,
            format="yolov8",
            verbose=False,
        )

        # Call the download function
        dataset_folder = download_roboflow_dataset(args)

        if dataset_folder:
            print("✅ Chessboard segmentation dataset (Dominique) downloaded successfully")
            return True
        else:
            print("❌ Download failed - no dataset folder returned")
            return False

    except Exception as e:
        print(f"❌ Error downloading chessboard segmentation dataset (Dominique): {e}")
        return False


def upload_to_huggingface(upload_individual=True, upload_merged=True, datasets_to_upload=None):
    """Upload datasets to Hugging Face Hub"""
    print("\n🚀 Uploading datasets to Hugging Face Hub...")

    try:
        if not HF_USERNAME:
            print("❌ HF_USERNAME not set, cannot upload to Hugging Face")
            return False

        uploader = ChessDatasetUploader(DATA_FOLDER_PATH, HF_USERNAME)

        if upload_individual:
            print("  📤 Uploading individual datasets...")

            # Upload chess piece datasets if they exist and are requested
            if not datasets_to_upload or "chesspieces_dominique" in datasets_to_upload:
                dominique_path = DATA_FOLDER_PATH / "chess_pieces_dominique"
                if dominique_path.exists():
                    print("    📋 Uploading Chess Pieces (Dominique)...")
                    uploader.upload_individual_dataset(
                        dataset_name="chess_pieces_dominique",
                        repo_name="chess-pieces-dominique",
                        description="Chess piece detection dataset from Dominique/Roboflow with 12 classes of chess pieces, optimized for YOLOv8 training.",
                    )
                else:
                    print("    ⚠️  Chess Pieces (Dominique) dataset not found, skipping upload")

            if not datasets_to_upload or "roboflow" in datasets_to_upload:
                roboflow_path = DATA_FOLDER_PATH / "chess_pieces_roboflow"
                if roboflow_path.exists():
                    print("    📋 Uploading Chess Pieces (Roboflow/Kaggle)...")
                    uploader.upload_individual_dataset(
                        dataset_name="chess_pieces_roboflow",
                        repo_name="chess-pieces-roboflow",
                        description="Chess piece detection dataset from Roboflow with processed labels, cleaned and standardized for YOLOv8 format.",
                    )
                else:
                    print("    ⚠️  Chess Pieces (Roboflow/Kaggle) dataset not found, skipping upload")

            # Note: Chessboard corners dataset upload would need separate handling
            # as it has different structure and uploader logic
            if not datasets_to_upload or "chessboard_corners_dominique" in datasets_to_upload:
                corners_path = DATA_FOLDER_PATH / "chessboard_corners"
                if corners_path.exists():
                    print("    📋 Chessboard Corners dataset (Dominique) found but upload not yet implemented")
                    print("    💡 Use HuggingFace Hub directly or implement custom uploader")
                else:
                    print("    ⚠️  Chessboard Corners dataset (Dominique) not found, skipping")

        if upload_merged:
            # Only attempt to merge if chess piece datasets are being processed
            chess_piece_datasets = ["roboflow", "chesspieces_dominique"]
            has_chess_pieces = any(dataset in chess_piece_datasets for dataset in (datasets_to_upload or []))

            if has_chess_pieces:
                print("  📤 Uploading merged chess pieces dataset...")

                available_datasets = []
                if (DATA_FOLDER_PATH / "chess_pieces_dominique").exists():
                    available_datasets.append("chess_pieces_dominique")
                if (DATA_FOLDER_PATH / "chess_pieces_roboflow").exists():
                    available_datasets.append("chess_pieces_roboflow")

                if len(available_datasets) >= 2:
                    print(f"    🔄 Merging datasets: {', '.join(available_datasets)}")
                    uploader.merge_and_upload_datasets(
                        dataset_names=available_datasets,
                        repo_name="chess-pieces-merged",
                        description="Comprehensive chess piece detection dataset combining multiple high-quality sources. This merged dataset provides more training data and better generalization for chess piece detection models.",
                    )
                else:
                    print(f"    ⚠️  Need at least 2 chess piece datasets to merge, found {len(available_datasets)}")
            else:
                print("  ⏭️  Skipping merged dataset upload (no chess piece datasets in target list)")

        print("✅ Dataset uploads completed!")
        return True

    except Exception as e:
        print(f"❌ Error uploading datasets: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and upload chess datasets to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset Types:
  roboflow    - Chess pieces from Roboflow (via Kaggle)
  chesspieces_dominique   - Chess pieces from Dominique's Roboflow project
  chessboard_corners_dominique     - Chessboard corner detection dataset (Dominique)
  chessboard_segmentation_dominique - Chessboard segmentation dataset (Dominique)

Examples:
  %(prog)s                                    # Download all datasets and upload
  %(prog)s --datasets roboflow chesspieces_dominique     # Download only chess piece datasets
  %(prog)s --datasets chessboard_corners_dominique                # Download only corner detection dataset
  %(prog)s --datasets chessboard_segmentation_dominique           # Download only segmentation dataset
  %(prog)s --datasets chesspieces_dominique           # Single dataset
  %(prog)s --skip-download                   # Only upload existing datasets
  %(prog)s --no-individual                   # Skip individual uploads, only merged
  %(prog)s --no-merged                       # Skip merged upload, only individual
  %(prog)s --dry-run                         # Show what would be done
        """,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[
            "roboflow",
            "chesspieces_dominique",
            "chessboard_corners_dominique",
            "chessboard_segmentation_dominique",
        ],
        help="Specify which datasets to download (default: all)",
    )

    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading datasets, only upload existing ones",
    )

    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="Don't upload individual datasets, only merged",
    )

    parser.add_argument(
        "--no-merged",
        action="store_true",
        help="Don't upload merged dataset, only individual",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it",
    )

    args = parser.parse_args()

    # Default to all datasets if none specified
    if args.datasets is None:
        args.datasets = [
            "roboflow",
            "chesspieces_dominique",
            "chessboard_corners_dominique",
            "chessboard_segmentation_dominique",
        ]

    print("🏗️  Chess Dataset Preparation Pipeline")
    print("=" * 60)
    print(f"📊 Target datasets: {', '.join(args.datasets)}")
    print(f"📁 Data directory: {DATA_FOLDER_PATH}")
    if not args.skip_download:
        print("🔄 Mode: Download + Upload")
    else:
        print("📤 Mode: Upload only")

    if args.dry_run:
        print("🔍 DRY RUN MODE - No actual changes will be made")
    print()

    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed. Please fix the issues above.")
        return 1

    success_count = 0
    total_steps = 0

    # Download datasets
    if not args.skip_download:
        print("\n" + "=" * 40)
        print("📥 DATASET DOWNLOAD PHASE")
        print("=" * 40)

        download_functions = {
            "roboflow": (
                "Chess Pieces (Roboflow/Kaggle)",
                download_roboflow_chess_pieces,
            ),
            "chesspieces_dominique": (
                "Chess Pieces (Dominique)",
                download_dominique_chess_pieces,
            ),
            "chessboard_corners_dominique": (
                "Chessboard Corners (Dominique)",
                download_chessboard_corners_dominique,
            ),
            "chessboard_segmentation_dominique": (
                "Chessboard Segmentation (Dominique)",
                download_chessboard_segmentation_dominique,
            ),
        }

        for dataset_key in args.datasets:
            total_steps += 1
            dataset_name, download_func = download_functions[dataset_key]

            if args.dry_run:
                print(f"\n📦 Would download: {dataset_name}")
                success_count += 1
            else:
                if download_func():
                    success_count += 1

    # Upload to Hugging Face
    if not (args.no_individual and args.no_merged):
        total_steps += 1

        print("\n" + "=" * 40)
        print("🚀 HUGGING FACE UPLOAD PHASE")
        print("=" * 40)

        if args.dry_run:
            if not args.no_individual:
                print("\n📤 Would upload individual datasets:")
                for dataset in args.datasets:
                    if dataset != "chessboard_corners_dominique":  # chessboard_corners_dominique needs special handling
                        print(f"   - {dataset}")
            if not args.no_merged:
                print("\n📤 Would upload merged chess pieces dataset")
            success_count += 1
        else:
            upload_individual = not args.no_individual
            upload_merged = not args.no_merged

            if upload_to_huggingface(upload_individual, upload_merged, args.datasets):
                success_count += 1

    # Summary
    print("\n" + "=" * 60)
    if args.dry_run:
        print("🔍 Dry run completed successfully!")
        print("Run without --dry-run to execute the pipeline.")
    else:
        if success_count == total_steps:
            print("🎉 Pipeline completed successfully!")
            print(f"📁 Datasets available locally at: {DATA_FOLDER_PATH}")
            if HF_USERNAME:
                print(f"🌐 Datasets uploaded to: https://huggingface.co/{HF_USERNAME}")

            print("\n🎯 Next steps:")
            print("1. Check your Hugging Face profile for uploaded datasets")
            print("2. Use download script: python src/data_prep/download_from_hf.py --list")
            print("3. Start training your models:")
            if "roboflow" in args.datasets or "chesspieces_dominique" in args.datasets:
                print("   - Chess piece detection: python -m src.chess_piece_detection.train")
            if "chessboard_corners_dominique" in args.datasets:
                print("   - Chessboard corners: python -m src.chess_board_detection.train")
        else:
            print(f"⚠️  Pipeline completed with issues: {success_count}/{total_steps} steps successful")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
