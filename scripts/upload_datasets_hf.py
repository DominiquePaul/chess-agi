"""
Example script showing how to upload chess datasets to Hugging Face Hub.

Make sure to set up your environment first:
1. Set HF_USERNAME in your .env file
2. Login to Hugging Face: huggingface-cli login
3. Make sure your datasets are in the correct format
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from src.data_prep.upload_to_hf import ChessDatasetUploader

load_dotenv()


def main():
    # Configuration
    DATA_FOLDER_PATH = Path(os.environ["DATA_FOLDER_PATH"])
    HF_USERNAME = os.environ.get("HF_USERNAME", "your-username")

    # Initialize uploader
    uploader = ChessDatasetUploader(DATA_FOLDER_PATH, HF_USERNAME)

    print("🏗️  Chess Dataset Uploader")
    print("=" * 60)

    # Option 1: Upload individual detection datasets
    print("\n🎯 Option 1: Upload Detection Datasets (Bounding Boxes)")
    print("-" * 50)

    # Upload Dominique dataset
    print("Uploading Dominique dataset...")
    uploader.upload_individual_dataset(
        dataset_name="chess_pieces_dominique",
        repo_name="chess-pieces-dominique",
        description="Chess piece detection dataset from Dominique with 12 classes of chess pieces, optimized for YOLOv8 training.",
        task_type="detection",
    )

    # Upload Roboflow dataset
    print("Uploading Roboflow dataset...")
    uploader.upload_individual_dataset(
        dataset_name="chess_pieces_roboflow",
        repo_name="chess-pieces-roboflow",
        description="Chess piece detection dataset from Roboflow with processed labels, cleaned and standardized for YOLOv8 format.",
        task_type="detection",
    )

    # Option 2: Upload merged detection dataset
    print("\n🔄 Option 2: Upload Merged Detection Dataset")
    print("-" * 50)

    uploader.merge_and_upload_datasets(
        dataset_names=["chess_pieces_dominique", "chess_pieces_roboflow"],
        repo_name="chess-pieces-merged",
        description="Comprehensive chess piece detection dataset combining multiple high-quality sources. This merged dataset provides more training data and better generalization for chess piece detection models.",
        task_type="detection",
    )

    # Option 3: Upload chess board segmentation datasets
    print("\n🎯 Option 3: Upload Segmentation Datasets (Polygons)")
    print("-" * 50)

    # Upload chess board segmentation dataset
    if (DATA_FOLDER_PATH / "chessboard_segmentation" / "chess-board-4").exists():
        print("Uploading chess board segmentation dataset (v4)...")
        uploader.upload_individual_dataset(
            dataset_name="chessboard_segmentation/chess-board-4",
            repo_name="chess-board-segmentation",
            description="Chess board segmentation dataset with polygon annotations for precise board detection and localization. Optimized for YOLOv8 segmentation training with chess-board class.",
            task_type="segmentation",
        )
    elif (DATA_FOLDER_PATH / "chessboard_segmentation" / "chess-board-3").exists():
        print("Uploading chess board segmentation dataset (v3)...")
        uploader.upload_individual_dataset(
            dataset_name="chessboard_segmentation/chess-board-3",
            repo_name="chess-board-segmentation",
            description="Chess board segmentation dataset with polygon annotations for precise board detection and localization. Optimized for YOLOv8 segmentation training with chess-board class.",
            task_type="segmentation",
        )
    else:
        print("⚠️  Chess board segmentation dataset not found at data/chessboard_segmentation")
        print(
            "💡 Download it first using: uv run python src/data_prep/download_data.py --project 'gustoguardian/chess-board-i0ptl' --version 4 --data-dir data/chessboard_segmentation"
        )

    print("\n✅ Upload process completed!")
    print("\n📋 Next steps:")
    print("1. Check your Hugging Face profile for the uploaded datasets")
    print("2. Test loading the datasets using the provided code examples")
    print("3. Use the datasets for training your chess detection and segmentation models")
    print("4. Detection datasets: bounding box annotations for piece classification")
    print("5. Segmentation datasets: polygon annotations for precise chess board localization")


def upload_single_dataset_example():
    """Example: Upload just one dataset"""
    DATA_FOLDER_PATH = Path(os.environ["DATA_FOLDER_PATH"])
    HF_USERNAME = os.environ.get("HF_USERNAME", "your-username")

    uploader = ChessDatasetUploader(DATA_FOLDER_PATH, HF_USERNAME)

    # Upload only the Roboflow dataset
    uploader.upload_individual_dataset(
        dataset_name="chess_pieces_roboflow",
        repo_name="chess-pieces-roboflow-v2",
        description="Updated chess piece detection dataset with improved annotations.",
    )


def upload_merged_only_example():
    """Example: Upload only the merged dataset"""
    DATA_FOLDER_PATH = Path(os.environ["DATA_FOLDER_PATH"])
    HF_USERNAME = os.environ.get("HF_USERNAME", "your-username")

    uploader = ChessDatasetUploader(DATA_FOLDER_PATH, HF_USERNAME)

    # Upload merged dataset only
    uploader.merge_and_upload_datasets(
        dataset_names=["chess_pieces_dominique", "chess_pieces_roboflow"],
        repo_name="chess-pieces-complete",
        description="Complete chess piece detection dataset for production use.",
        task_type="detection",
    )


def upload_segmentation_only_example():
    """Example: Upload only chess board segmentation dataset"""
    DATA_FOLDER_PATH = Path(os.environ["DATA_FOLDER_PATH"])
    HF_USERNAME = os.environ.get("HF_USERNAME", "your-username")

    uploader = ChessDatasetUploader(DATA_FOLDER_PATH, HF_USERNAME)

    # Upload chess board segmentation dataset only (latest version)
    uploader.upload_individual_dataset(
        dataset_name="chessboard_segmentation/chess-board-4",
        repo_name="chess-board-segmentation",
        description="Chess board segmentation dataset with polygon annotations for precise board detection and localization. Optimized for YOLOv8 segmentation training.",
        task_type="segmentation",
    )


def upload_custom_segmentation_example():
    """Example: Upload custom chess board segmentation with different naming"""
    DATA_FOLDER_PATH = Path(os.environ["DATA_FOLDER_PATH"])
    HF_USERNAME = os.environ.get("HF_USERNAME", "your-username")

    uploader = ChessDatasetUploader(DATA_FOLDER_PATH, HF_USERNAME)

    # Upload with custom naming
    uploader.upload_individual_dataset(
        dataset_name="chessboard_segmentation/chess-board-4",  # Specific version path
        repo_name="chess-board-segmentation-v4",
        description="Chess board segmentation dataset v4 with improved polygon annotations for chess board detection and localization.",
        task_type="segmentation",
    )


if __name__ == "__main__":
    # Run the full example (detection + segmentation)
    main()

    # Uncomment to run individual examples:
    # upload_single_dataset_example()
    # upload_merged_only_example()
    # upload_segmentation_only_example()
    # upload_custom_segmentation_example()
