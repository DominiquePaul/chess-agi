#!/usr/bin/env python3
"""
Data download script for ChessBoard Corner Detection

Downloads the chessboard corner dataset from Roboflow.
Dataset: gustoguardian/chess-board-box/3
"""

import os
import sys
from pathlib import Path
import yaml

def download_roboflow_dataset():
    """Download the chessboard corner dataset from Roboflow."""
    
    try:
        # Try to import roboflow
        from roboflow import Roboflow
    except ImportError:
        print("❌ Roboflow library not found!")
        print("Please install it with: uv add roboflow")
        sys.exit(1)
    
    # Configuration
    PROJECT_NAME = "gustoguardian/chess-board-box"
    VERSION = 3
    FORMAT = "yolov8"  # YOLO format
    DATA_DIR = Path("data/chessboard_corners")
    
    print("📥 Downloading ChessBoard Corner Dataset from Roboflow")
    print("=" * 60)
    print(f"📊 Project: {PROJECT_NAME}")
    print(f"🔢 Version: {VERSION}")
    print(f"📁 Download directory: {DATA_DIR}")
    print(f"📋 Format: {FORMAT}")
    print("=" * 60)
    
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize Roboflow
        rf = Roboflow()
        
        # Get the project
        print(f"🔍 Accessing project: {PROJECT_NAME}")
        project = rf.workspace().project(PROJECT_NAME.split('/')[-1])
        
        # Get the specific version
        print(f"📦 Getting version {VERSION}")
        dataset = project.version(VERSION)
        
        # Download the dataset
        print(f"⬇️  Downloading dataset to: {DATA_DIR}")
        dataset.download(
            location=str(DATA_DIR),
            overwrite=True
        )
        
        # Find the downloaded dataset folder
        downloaded_folders = [d for d in DATA_DIR.iterdir() if d.is_dir()]
        if not downloaded_folders:
            raise Exception("No dataset folder found after download")
        
        dataset_folder = downloaded_folders[0]  # Usually named like "chess-board-box-3"
        print(f"✅ Dataset downloaded to: {dataset_folder}")
        
        # Check for data.yaml file
        data_yaml_path = dataset_folder / "data.yaml"
        if data_yaml_path.exists():
            print(f"✅ Found data.yaml at: {data_yaml_path}")
            
            # Read and display dataset info
            with open(data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            
            print("\n📊 Dataset Information:")
            print(f"  Classes: {data_config.get('nc', 'Unknown')} ({', '.join(data_config.get('names', []))})")
            if 'train' in data_config:
                print(f"  Train path: {data_config['train']}")
            if 'val' in data_config:
                print(f"  Validation path: {data_config['val']}")
            if 'test' in data_config:
                print(f"  Test path: {data_config['test']}")
        
        else:
            print(f"⚠️  Warning: data.yaml not found at {data_yaml_path}")
        
        # Create a symlink or copy to make it easier to find
        easy_access_path = DATA_DIR / "data.yaml"
        if data_yaml_path.exists() and not easy_access_path.exists():
            try:
                easy_access_path.symlink_to(data_yaml_path.relative_to(DATA_DIR))
                print(f"🔗 Created symlink: {easy_access_path} -> {data_yaml_path}")
            except Exception as e:
                print(f"⚠️  Could not create symlink: {e}")
        
        print("\n✅ Download completed successfully!")
        print("\n🎯 Next steps:")
        print(f"1. Update training script to use: {data_yaml_path}")
        print("2. Run training: python -m src.chess_board_detection.train")
        print("3. Test the trained model on your images")
        
        return dataset_folder
        
    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Make sure you have internet connection")
        print("2. Check if the Roboflow project exists and is public")
        print("3. Try logging into Roboflow: rf = Roboflow(api_key='your_key')")
        raise

def check_dataset_structure(dataset_path: Path):
    """Check and display the structure of the downloaded dataset."""
    
    print(f"\n📁 Dataset structure for: {dataset_path}")
    print("-" * 40)
    
    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return
    
    # Check main folders
    for item in sorted(dataset_path.iterdir()):
        if item.is_dir():
            print(f"📁 {item.name}/")
            # Count files in each directory
            try:
                file_count = len(list(item.glob("*")))
                print(f"   ({file_count} files)")
            except:
                print("   (could not count files)")
        else:
            print(f"📄 {item.name}")

def main():
    """Main function to download and setup the chessboard corner dataset."""
    
    try:
        dataset_folder = download_roboflow_dataset()
        check_dataset_structure(dataset_folder)
        
    except Exception as e:
        print(f"❌ Script failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 