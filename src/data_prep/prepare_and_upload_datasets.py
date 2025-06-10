"""
Prepare and upload chess piece datasets to Hugging Face Hub.
This script orchestrates the complete dataset preparation pipeline.
"""

import os
import subprocess
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Import our existing functions
from src.data_prep.roboflow_chess import (
    download_and_move_roboflow_data,
    clean_roboflow_data, 
    update_data_yaml
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


def download_roboflow_data():
    """Download and process Roboflow dataset from Kaggle"""
    print("\n📦 Step 1: Downloading and processing Roboflow dataset from Kaggle...")
    
    try:
        dataset_name = Path("chess_pieces_roboflow")
        
        # Download and move data
        download_and_move_roboflow_data(DATA_FOLDER_PATH, dataset_name)
        
        # Clean labels (remove extra bishop class)
        clean_roboflow_data(DATA_FOLDER_PATH, dataset_name)
        
        # Update data.yaml
        update_data_yaml(DATA_FOLDER_PATH, dataset_name)
        
        print("✅ Roboflow dataset processed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error processing Roboflow dataset: {e}")
        return False


def download_dominique_data():
    """Download Dominique dataset from Roboflow"""
    print("\n📦 Step 2: Downloading Dominique dataset from Roboflow...")
    
    try:
        cmd = [
            "roboflow", "download", 
            "-f", "yolov8",
            "-l", str(DATA_FOLDER_PATH / "chess_pieces_dominique"),
            "gustoguardian/chess-piece-detection-bltvi/6"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dominique dataset downloaded successfully")
            return True
        else:
            print(f"❌ Error downloading Dominique dataset: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading Dominique dataset: {e}")
        return False


def upload_to_huggingface(upload_individual=True, upload_merged=True):
    """Upload datasets to Hugging Face Hub"""
    print("\n🚀 Step 3: Uploading datasets to Hugging Face Hub...")
    
    try:
        uploader = ChessDatasetUploader(DATA_FOLDER_PATH, HF_USERNAME)
        
        if upload_individual:
            print("  📤 Uploading individual datasets...")
            
            # Upload Dominique dataset
            uploader.upload_individual_dataset(
                dataset_name="chess_pieces_dominique",
                repo_name="chess-pieces-dominique",
                description="Chess piece detection dataset from Dominique/Roboflow with 12 classes of chess pieces, optimized for YOLOv8 training."
            )
            
            # Upload Roboflow dataset
            uploader.upload_individual_dataset(
                dataset_name="chess_pieces_roboflow", 
                repo_name="chess-pieces-roboflow",
                description="Chess piece detection dataset from Roboflow with processed labels, cleaned and standardized for YOLOv8 format."
            )
        
        if upload_merged:
            print("  📤 Uploading merged dataset...")
            
            uploader.merge_and_upload_datasets(
                dataset_names=["chess_pieces_dominique", "chess_pieces_roboflow"],
                repo_name="chess-pieces-merged", 
                description="Comprehensive chess piece detection dataset combining multiple high-quality sources. This merged dataset provides more training data and better generalization for chess piece detection models."
            )
        
        print("✅ All datasets uploaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error uploading datasets: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and upload chess piece datasets to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Full pipeline: download, process, and upload all
  %(prog)s --skip-download           # Only upload existing datasets
  %(prog)s --no-individual           # Skip individual dataset uploads, only merged
  %(prog)s --no-merged               # Skip merged dataset upload, only individual
        """
    )
    
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading datasets, only upload existing ones"
    )
    
    parser.add_argument(
        "--no-individual", 
        action="store_true",
        help="Don't upload individual datasets, only merged"
    )
    
    parser.add_argument(
        "--no-merged",
        action="store_true", 
        help="Don't upload merged dataset, only individual"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it"
    )
    
    args = parser.parse_args()
    
    print("🏗️  Chess Dataset Preparation Pipeline")
    print("=" * 50)
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No actual changes will be made")
        print()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed. Please fix the issues above.")
        return 1
    
    success_count = 0
    total_steps = 0
    
    # Step 1 & 2: Download datasets
    if not args.skip_download:
        total_steps += 2
        
        if not args.dry_run:
            # Download Roboflow data
            if download_roboflow_data():
                success_count += 1
            
            # Download Dominique data  
            if download_dominique_data():
                success_count += 1
        else:
            print("\n📦 Would download and process Roboflow dataset from Kaggle")
            print("📦 Would download Dominique dataset from Roboflow")
            success_count += 2
    
    # Step 3: Upload to Hugging Face
    if not (args.no_individual and args.no_merged):
        total_steps += 1
        
        if not args.dry_run:
            upload_individual = not args.no_individual
            upload_merged = not args.no_merged
            
            if upload_to_huggingface(upload_individual, upload_merged):
                success_count += 1
        else:
            if not args.no_individual:
                print("\n🚀 Would upload individual datasets to HF")
            if not args.no_merged:
                print("🚀 Would upload merged dataset to HF")
            success_count += 1
    
    # Summary
    print("\n" + "=" * 50)
    if args.dry_run:
        print("🔍 Dry run completed successfully!")
        print("Run without --dry-run to execute the pipeline.")
    else:
        if success_count == total_steps:
            print("🎉 Pipeline completed successfully!")
            print(f"📁 Datasets available locally at: {DATA_FOLDER_PATH}")
            print(f"🌐 Datasets uploaded to: https://huggingface.co/{HF_USERNAME}")
            print("\n🎯 Next steps:")
            print("1. Check your Hugging Face profile for the uploaded datasets")
            print("2. Use the download script to get datasets: python src/data_prep/download_from_hf.py --list")
            print("3. Start training your chess piece detection models!")
        else:
            print(f"⚠️  Pipeline completed with issues: {success_count}/{total_steps} steps successful")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 