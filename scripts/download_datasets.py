#!/usr/bin/env python3
"""
Unified Chess Dataset Downloader

This script downloads both chess piece detection and chess board segmentation datasets
from their respective sources (HuggingFace and Roboflow).

Usage Examples:
    # Download detection datasets only
    python scripts/download_datasets.py --detection

    # Download segmentation datasets only
    python scripts/download_datasets.py --segmentation

    # Download both (default)
    python scripts/download_datasets.py --both

    # Download all with custom data folder
    python scripts/download_datasets.py --both --data-dir /custom/path
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.data_prep.download_from_hf import AVAILABLE_DATASETS, download_dataset

load_dotenv()


def download_detection_datasets(data_path: Path, datasets: list[str] | None = None):
    """Download chess piece detection datasets from HuggingFace"""
    print("🎯 Downloading Chess Piece Detection Datasets")
    print("=" * 50)

    if datasets is None:
        datasets = ["merged"]  # Default to merged dataset

    success_count = 0

    for dataset_name in datasets:
        if dataset_name in AVAILABLE_DATASETS:
            if dataset_name == "all":
                # Download all individual datasets
                for repo in AVAILABLE_DATASETS["all"]:
                    local_name = repo.split("/")[-1].replace("-", "_")
                    print(f"\n📥 Downloading {repo}...")
                    success = download_dataset(repo, local_name, data_path)
                    if success:
                        success_count += 1
            else:
                repo = AVAILABLE_DATASETS[dataset_name]
                local_name = repo.split("/")[-1].replace("-", "_")
                print(f"\n📥 Downloading {repo}...")
                success = download_dataset(repo, local_name, data_path)
                if success:
                    success_count += 1
        else:
            print(f"⚠️  Unknown dataset: {dataset_name}")

    return success_count


def download_segmentation_datasets(data_path: Path, version: int = 4):
    """Download chess board segmentation datasets from Roboflow"""
    print("\n🎯 Downloading Chess Board Segmentation Datasets")
    print("=" * 50)

    project_id = "gustoguardian/chess-board-i0ptl"
    output_dir = data_path / "chessboard_segmentation"

    print(f"\n📥 Downloading {project_id} (version {version})...")
    print("💡 Using existing download_data.py script...")

    try:
        # Use subprocess to call the existing download script
        import subprocess

        cmd = [
            "python",
            "src/data_prep/download_data.py",
            "--project",
            project_id,
            "--version",
            str(version),
            "--data-dir",
            str(output_dir),
            "--format",
            "yolov8",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Segmentation dataset downloaded successfully")
            return 1
        else:
            print(f"❌ Download failed: {result.stderr}")
            return 0
    except Exception as e:
        print(f"❌ Error downloading segmentation dataset: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Download chess datasets for detection and/or segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--detection", action="store_true", help="Download only chess piece detection datasets")
    group.add_argument("--segmentation", action="store_true", help="Download only chess board segmentation datasets")
    group.add_argument(
        "--both", action="store_true", default=True, help="Download both detection and segmentation datasets (default)"
    )

    # Detection dataset options
    parser.add_argument(
        "--detection-datasets",
        nargs="+",
        choices=list(AVAILABLE_DATASETS.keys()),
        default=["merged"],
        help="Which detection datasets to download",
    )

    # Segmentation dataset options
    parser.add_argument(
        "--segmentation-version", type=int, default=4, help="Version of segmentation dataset to download"
    )

    # Path options
    parser.add_argument(
        "--data-dir", type=str, default=None, help="Data directory (uses DATA_FOLDER_PATH env var if not specified)"
    )

    args = parser.parse_args()

    # Determine data path
    if args.data_dir:
        data_path = Path(args.data_dir)
    else:
        data_folder = os.environ.get("DATA_FOLDER_PATH")
        if not data_folder:
            print("❌ Please set DATA_FOLDER_PATH environment variable or use --data-dir")
            return
        data_path = Path(data_folder)

    data_path.mkdir(parents=True, exist_ok=True)

    print("🏗️  Chess Dataset Downloader")
    print("=" * 60)
    print(f"📁 Data directory: {data_path}")

    total_downloaded = 0

    # Download detection datasets
    if args.detection or args.both:
        downloaded = download_detection_datasets(data_path, args.detection_datasets)
        total_downloaded += downloaded

    # Download segmentation datasets
    if args.segmentation or args.both:
        downloaded = download_segmentation_datasets(data_path, args.segmentation_version)
        total_downloaded += downloaded

    # Summary
    print("\n" + "=" * 60)
    print("✅ Download Summary")
    print(f"📦 Total datasets downloaded: {total_downloaded}")

    if args.detection or args.both:
        print("\n🎯 Detection datasets available for:")
        print("   • Chess piece classification (12 classes)")
        print("   • Training: uv run python src/chess_piece_detection/train.py")

    if args.segmentation or args.both:
        print("\n🎯 Segmentation datasets available for:")
        print("   • Chess board localization (polygon masks)")
        print("   • Training: uv run python src/chess_board_detection/yolo/segmentation/train_segmentation.py")

    print("\n📋 Next steps:")
    print("1. Train your models using the training scripts")
    print("2. Upload datasets to HuggingFace: python scripts/upload_datasets_hf.py")
    print("3. Use trained models for chess analysis")


if __name__ == "__main__":
    main()
