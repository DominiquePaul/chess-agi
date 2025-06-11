#!/usr/bin/env python3
"""
Data download script for ChessBoard Corner Detection

Downloads the chessboard corner dataset from Roboflow.
Dataset: gustoguardian/chess-board-box/3

Usage:
    # Set API key as environment variable (required)
    export ROBOFLOW_API_KEY=your_api_key_here
    python src/chess_board_detection/download_data.py
    
    # Download to custom directory
    python src/chess_board_detection/download_data.py --data-dir data/my_corners
    
    # Get help with all options
    python src/chess_board_detection/download_data.py --help
"""

import argparse
import os
import sys
from pathlib import Path
import yaml

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download ChessBoard Corner Detection Dataset from Roboflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/chessboard_corners",
        help="Directory to download the dataset to"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="gustoguardian/chess-board-box",
        help="Roboflow project name"
    )
    parser.add_argument(
        "--version",
        type=int,
        default=3,
        help="Dataset version to download"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="yolov8",
        choices=["yolov8", "yolov5", "coco", "pascal_voc"],
        help="Dataset format for download"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output"
    )
    
    return parser.parse_args()

def get_api_key():
    """Get API key from environment variables."""
    api_key = os.environ.get('ROBOFLOW_API_KEY')
    
    if not api_key:
        print("❌ ROBOFLOW_API_KEY environment variable not set!")
        print("\n💡 Set your API key:")
        print("   export ROBOFLOW_API_KEY=your_api_key_here")
        print("   python src/chess_board_detection/download_data.py")
        print()
        print("🔑 Get free API key at: https://roboflow.com/ → Settings → API")
        sys.exit(1)
    
    return api_key

def download_roboflow_dataset(args):
    """Download the chessboard corner dataset from Roboflow."""
    
    try:
        # Try to import roboflow
        from roboflow import Roboflow
    except ImportError:
        print("❌ Roboflow library not found!")
        print("💡 Install it with: uv add roboflow")
        print("💡 Or try manual download from: https://universe.roboflow.com/gustoguardian/chess-board-box")
        sys.exit(1)
    
    # Get API key from environment
    api_key = get_api_key()
    
    # Configuration from arguments
    PROJECT_NAME = args.project
    VERSION = args.version
    FORMAT = args.format
    DATA_DIR = Path(args.data_dir)
    
    print("📥 Downloading ChessBoard Corner Dataset from Roboflow")
    print("=" * 60)
    print(f"📊 Project: {PROJECT_NAME}")
    print(f"🔢 Version: {VERSION}")
    print(f"📁 Download directory: {DATA_DIR}")
    print(f"📋 Format: {FORMAT}")
    print(f"🔑 Using API key: {api_key[:8]}...")
    print("=" * 60)
    
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize Roboflow with API key
        print(f"🔑 Authenticating with Roboflow...")
        rf = Roboflow(api_key=api_key)
        
        # Get the project
        print(f"🔍 Accessing project: {PROJECT_NAME}")
        project = rf.workspace().project(PROJECT_NAME.split('/')[-1])
        
        # Get the specific version
        print(f"📦 Getting version {VERSION}")
        dataset = project.version(VERSION)
        
        # Download the dataset
        print(f"⬇️  Downloading dataset to: {DATA_DIR}")
        dataset.download(
            location=str(DATA_DIR)
        )
        print("✅ Dataset downloaded successfully!")
        
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
        print(f"1. Train the model: python src/chess_board_detection/train.py --data {data_yaml_path}")
        print("2. View training options: python src/chess_board_detection/train.py --help")
        print("3. Test the trained model on your images")
        
        return dataset_folder
        
    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")
        print("\n💡 Troubleshooting Options:")
        print("1. Check your API key:")
        print("   • Visit: https://roboflow.com/")
        print("   • Go to Settings → API")
        print("   • Verify your API key is correct")
        print()
        print("2. Manual download:")
        print("   • Visit: https://universe.roboflow.com/gustoguardian/chess-board-box")
        print("   • Download YOLO v8 format")
        print("   • Extract to: data/chessboard_corners/")
        print()
        print("3. Check network connection:")
        print("   • Ensure you have internet access")
        print("   • Try again in a few minutes")
        
        if args.verbose:
            raise
        else:
            print("\n💡 Add --verbose flag for detailed error information")
            return None

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
    
    # Parse command line arguments
    try:
        args = parse_args()
    except SystemExit as e:
        if e.code == 2:  # argparse error
            print("\n💡 Quick start:")
            print("1. Get free API key: https://roboflow.com/ → Settings → API")
            print("2. Set environment variable: export ROBOFLOW_API_KEY=your_key_here")
            print("3. Run: python src/chess_board_detection/download_data.py")
            print("4. Or use manual download from: https://universe.roboflow.com/gustoguardian/chess-board-box")
        raise
    
    try:
        dataset_folder = download_roboflow_dataset(args)
        if dataset_folder:
            check_dataset_structure(dataset_folder)
        else:
            print("❌ Download failed - no dataset folder to check")
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ Script failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 