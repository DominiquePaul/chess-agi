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
    print("=" * 50)

    # Option 1: Upload individual datasets
    print("\n📦 Option 1: Upload Individual Datasets")
    print("-" * 40)

    # Upload Dominique dataset
    print("Uploading Dominique dataset...")
    uploader.upload_individual_dataset(
        dataset_name="chess_pieces_dominique",
        repo_name="chess-pieces-dominique",
        description="Chess piece detection dataset from Dominique with 12 classes of chess pieces, optimized for YOLOv8 training.",
    )

    # Upload Roboflow dataset  to
    print("Uploading Roboflow dataset...")
    uploader.upload_individual_dataset(
        dataset_name="chess_pieces_roboflow",
        repo_name="chess-pieces-roboflow",
        description="Chess piece detection dataset from Roboflow with processed labels, cleaned and standardized for YOLOv8 format.",
    )

    # Option 2: Upload merged dataset
    print("\n🔄 Option 2: Upload Merged Dataset")
    print("-" * 40)

    uploader.merge_and_upload_datasets(
        dataset_names=["chess_pieces_dominique", "chess_pieces_roboflow"],
        repo_name="chess-pieces-merged",
        description="Comprehensive chess piece detection dataset combining multiple high-quality sources. This merged dataset provides more training data and better generalization for chess piece detection models.",
    )

    print("\n✅ Upload process completed!")
    print("\n📋 Next steps:")
    print("1. Check your Hugging Face profile for the uploaded datasets")
    print("2. Test loading the datasets using the provided code examples")
    print("3. Use the datasets for training your chess piece detection models")


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
    )


if __name__ == "__main__":
    # Run the full example
    main()

    # Uncomment to run individual examples:
    # upload_single_dataset_example()
    # upload_merged_only_example()
