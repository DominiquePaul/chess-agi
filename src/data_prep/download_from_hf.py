"""
Download chess piece datasets from Hugging Face Hub.
This script downloads the processed datasets and saves them locally in YOLOv8 format.
"""

import os
import yaml
import argparse
from pathlib import Path
from typing import List, Optional
from datasets import load_dataset
from dotenv import load_dotenv
from PIL import Image
import shutil

load_dotenv()

# Default datasets available
AVAILABLE_DATASETS = {
    "dominique": "dopaul/chess-pieces-dominique",
    "roboflow": "dopaul/chess-pieces-roboflow", 
    "merged": "dopaul/chess-pieces-merged",
    "all": ["dopaul/chess-pieces-dominique", "dopaul/chess-pieces-roboflow", "dopaul/chess-pieces-merged"]
}

DATASET_LOCAL_NAMES = {
    "dopaul/chess-pieces-dominique": "chess_pieces_dominique",
    "dopaul/chess-pieces-roboflow": "chess_pieces_roboflow",
    "dopaul/chess-pieces-merged": "chess_pieces_merged"
}


def download_dataset(hf_repo: str, local_name: str, data_path: Path, convert_to_yolo: bool = True):
    """Download a single dataset from Hugging Face and optionally convert to YOLOv8 format"""
    print(f"📦 Downloading {hf_repo}...")
    
    try:
        dataset = load_dataset(hf_repo)
        print(f"✅ Successfully loaded {hf_repo}")
    except Exception as e:
        print(f"❌ Error loading {hf_repo}: {e}")
        return False
    
    # Create local directory
    local_dir = data_path / local_name
    local_dir.mkdir(exist_ok=True, parents=True)
    
    if convert_to_yolo:
        # Convert to YOLOv8 format
        print(f"🔄 Converting {local_name} to YOLOv8 format...")
        success = convert_hf_to_yolo(dataset, local_dir, local_name)
        if not success:
            print(f"❌ Failed to convert {local_name} to YOLOv8 format")
            return False
    else:
        # Save in HF format
        for split_name, split_data in dataset.items():
            split_dir = local_dir / split_name
            split_dir.mkdir(exist_ok=True)
            split_data.save_to_disk(str(split_dir))
    
    print(f"✅ Saved {hf_repo} to {local_dir}")
    return True


def convert_hf_to_yolo(dataset, output_dir: Path, dataset_name: str):
    """Convert Hugging Face dataset to YOLOv8 format"""
    try:
        # Create YOLOv8 directory structure
        for split_name, split_data in dataset.items():
            print(f"  Converting {split_name} split ({len(split_data)} examples)...")
            
            # Create directories
            images_dir = output_dir / split_name / "images"
            labels_dir = output_dir / split_name / "labels"
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each example
            for i, example in enumerate(split_data):
                # Save image
                image = example["image"]
                image_id = example["image_id"]
                image_path = images_dir / f"{image_id}.jpg"
                
                # Convert to RGB if needed and save
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(image_path, "JPEG", quality=95)
                
                # Save labels
                label_path = labels_dir / f"{image_id}.txt"
                with open(label_path, 'w') as f:
                    annotations = example["annotations"]
                    # Handle the case where annotations is a dict with lists
                    if isinstance(annotations, dict) and "class_id" in annotations:
                        num_objects = len(annotations["class_id"])
                        for j in range(num_objects):
                            class_id = annotations["class_id"][j]
                            x_center = annotations["x_center"][j]
                            y_center = annotations["y_center"][j]
                            width = annotations["width"][j]
                            height = annotations["height"][j]
                            
                            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                    else:
                        # Handle the case where annotations is a list of dicts (fallback)
                        for annotation in annotations:
                            class_id = annotation["class_id"]
                            x_center = annotation["x_center"] 
                            y_center = annotation["y_center"]
                            width = annotation["width"]
                            height = annotation["height"]
                            
                            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                
                # Progress indicator
                if (i + 1) % 100 == 0:
                    print(f"    Processed {i + 1}/{len(split_data)} examples...")
        
        # Create data.yaml file
        create_data_yaml(output_dir, dataset_name)
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting dataset: {e}")
        return False


def create_data_yaml(output_dir: Path, dataset_name: str):
    """Create data.yaml file for YOLOv8"""
    data_yaml = {
        "train": "train/images",
        "val": "valid/images", 
        "test": "test/images",
        "names": {
            0: "black-bishop",
            1: "black-king", 
            2: "black-knight",
            3: "black-pawn",
            4: "black-queen",
            5: "black-rook",
            6: "white-bishop",
            7: "white-king",
            8: "white-knight", 
            9: "white-pawn",
            10: "white-queen",
            11: "white-rook"
        },
        "nc": 12
    }
    
    # Add source information for merged datasets
    if "merged" in dataset_name:
        data_yaml["source_datasets"] = ["chess_pieces_dominique", "chess_pieces_roboflow"]
    
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    
    print(f"✅ Created {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download chess piece datasets from Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dataset merged              # Download merged dataset (recommended)
  %(prog)s --dataset all                 # Download all available datasets
  %(prog)s --dataset dominique roboflow  # Download specific datasets
  %(prog)s --list                        # List available datasets
  %(prog)s --no-convert                  # Download in HF format (no YOLOv8 conversion)
        """
    )
    
    parser.add_argument(
        "--dataset", "-d",
        nargs="+",
        choices=list(AVAILABLE_DATASETS.keys()) + ["custom"],
        default=["merged"],
        help="Dataset(s) to download (default: merged)"
    )
    
    parser.add_argument(
        "--custom-repo",
        help="Custom HuggingFace repository (use with --dataset custom)"
    )
    
    parser.add_argument(
        "--custom-name", 
        help="Custom local name for custom repository"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("DATA_FOLDER_PATH", "./data")),
        help="Output directory for datasets"
    )
    
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Don't convert to YOLOv8 format, keep as HF dataset"
    )
    
    parser.add_argument(
        "--list",
        action="store_true", 
        help="List available datasets and exit"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("📋 Available datasets:")
        for key, value in AVAILABLE_DATASETS.items():
            if key != "all":
                print(f"  {key:12} -> {value}")
            else:
                print(f"  {key:12} -> Downloads all individual datasets")
        return
    
    print("🏗️  Chess Dataset Downloader")
    print("=" * 50)
    print(f"Output directory: {args.output_dir}")
    print(f"Convert to YOLOv8: {not args.no_convert}")
    print()
    
    # Ensure output directory exists
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    # Determine which datasets to download
    datasets_to_download = []
    
    for dataset in args.dataset:
        if dataset == "all":
            for repo in AVAILABLE_DATASETS["all"]:
                datasets_to_download.append((repo, DATASET_LOCAL_NAMES[repo]))
        elif dataset == "custom":
            if not args.custom_repo:
                print("❌ --custom-repo required when using --dataset custom")
                return
            local_name = args.custom_name or args.custom_repo.split("/")[-1]
            datasets_to_download.append((args.custom_repo, local_name))
        else:
            repo = AVAILABLE_DATASETS[dataset]
            datasets_to_download.append((repo, DATASET_LOCAL_NAMES[repo]))
    
    # Remove duplicates
    datasets_to_download = list(set(datasets_to_download))
    
    print(f"📦 Downloading {len(datasets_to_download)} dataset(s)...")
    print()
    
    # Download each dataset
    success_count = 0
    for hf_repo, local_name in datasets_to_download:
        success = download_dataset(
            hf_repo, 
            local_name, 
            args.output_dir,
            convert_to_yolo=not args.no_convert
        )
        if success:
            success_count += 1
        print()
    
    print("=" * 50)
    print(f"✅ Successfully downloaded {success_count}/{len(datasets_to_download)} datasets")
    
    if success_count > 0:
        print(f"📁 Datasets saved to: {args.output_dir}")
        print("\n🎯 Next steps:")
        print("1. Check the downloaded datasets in your data folder")
        print("2. Use them for training your chess piece detection models")
        if not args.no_convert:
            print("3. Each dataset includes a data.yaml file for YOLOv8 training")


if __name__ == "__main__":
    main() 