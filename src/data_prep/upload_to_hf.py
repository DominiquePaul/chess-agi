"""
Upload chess piece datasets to Hugging Face Hub.
Supports uploading individual datasets and merged datasets.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import yaml
from datasets import Dataset, DatasetDict, Image, Value, Sequence, Features
from huggingface_hub import HfApi, upload_folder, create_repo
from dotenv import load_dotenv
import pandas as pd
from PIL import Image as PILImage

load_dotenv()

DATA_FOLDER_PATH = Path(os.environ["DATA_FOLDER_PATH"])
HF_USERNAME = os.environ.get("HF_USERNAME", "your-username")  # Set in .env file


class ChessDatasetUploader:
    def __init__(self, data_folder_path: Path, hf_username: str):
        self.data_folder_path = data_folder_path
        self.hf_username = hf_username
        self.api = HfApi()
    
    def _load_dataset_info(self, dataset_path: Path) -> Dict[str, Any]:
        """Load dataset configuration from data.yaml"""
        data_yaml_path = dataset_path / "data.yaml"
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found in {dataset_path}")
        
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        return data_config
    
    def _prepare_dataset_for_hf(self, dataset_path: Path, split: str) -> List[Dict]:
        """Prepare a single split for Hugging Face format"""
        images_dir = dataset_path / split / "images"
        labels_dir = dataset_path / split / "labels"
        
        if not images_dir.exists():
            print(f"Warning: {images_dir} does not exist, skipping {split} split")
            return []
        
        data = []
        
        for img_file in images_dir.glob("*"):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                label_file = labels_dir / f"{img_file.stem}.txt"
                
                # Load image
                image = PILImage.open(img_file)
                
                # Load annotations if they exist
                annotations = []
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                
                                annotations.append({
                                    "class_id": class_id,
                                    "x_center": x_center,
                                    "y_center": y_center,
                                    "width": width,
                                    "height": height
                                })
                
                data.append({
                    "image": image,
                    "image_id": img_file.stem,
                    "annotations": annotations,
                    "image_width": image.width,
                    "image_height": image.height
                })
        
        return data
    
    def create_hf_dataset(self, dataset_path: Path) -> DatasetDict:
        """Create a Hugging Face Dataset from YOLOv8 format"""
        data_config = self._load_dataset_info(dataset_path)
        
        # Prepare data for each split
        dataset_dict = {}
        
        for split in ["train", "valid", "test"]:
            # Handle different naming conventions
            split_variants = [split, "val" if split == "valid" else split]
            
            split_data = []
            for variant in split_variants:
                split_data = self._prepare_dataset_for_hf(dataset_path, variant)
                if split_data:
                    break
            
            if split_data:
                # Define features
                features = Features({
                    "image": Image(),
                    "image_id": Value("string"),
                    "annotations": Sequence({
                        "class_id": Value("int32"),
                        "x_center": Value("float32"),
                        "y_center": Value("float32"),
                        "width": Value("float32"),
                        "height": Value("float32")
                    }),
                    "image_width": Value("int32"),
                    "image_height": Value("int32")
                })
                
                dataset_dict[split] = Dataset.from_list(split_data, features=features)
        
        return DatasetDict(dataset_dict)
    
    def upload_individual_dataset(self, dataset_name: str, repo_name: str, description: str | None = None):
        """Upload a single dataset to Hugging Face Hub"""
        dataset_path = self.data_folder_path / dataset_name
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset {dataset_name} not found at {dataset_path}")
        
        print(f"Processing dataset: {dataset_name}")
        
        # Create HF dataset
        hf_dataset = self.create_hf_dataset(dataset_path)
        
        # Create repository
        repo_id = f"{self.hf_username}/{repo_name}"
        try:
            create_repo(repo_id, repo_type="dataset", exist_ok=True)
            print(f"Created/found repository: {repo_id}")
        except Exception as e:
            print(f"Error creating repository: {e}")
            return
        
        # Upload dataset
        try:
            hf_dataset.push_to_hub(repo_id)
            print(f"Successfully uploaded {dataset_name} to {repo_id}")
            
            # Upload additional files (data.yaml, README files)
            self._upload_additional_files(dataset_path, repo_id, description)
            
        except Exception as e:
            print(f"Error uploading dataset: {e}")
    
    def merge_and_upload_datasets(self, dataset_names: List[str], repo_name: str, description: str | None = None):
        """Merge multiple datasets and upload to Hugging Face Hub"""
        print(f"Merging datasets: {dataset_names}")
        
        all_datasets = {}
        
        # Load all datasets
        for dataset_name in dataset_names:
            dataset_path = self.data_folder_path / dataset_name
            if not dataset_path.exists():
                print(f"Warning: Dataset {dataset_name} not found, skipping")
                continue
            
            hf_dataset = self.create_hf_dataset(dataset_path)
            all_datasets[dataset_name] = hf_dataset
        
        if not all_datasets:
            print("No valid datasets found to merge")
            return
        
        # Merge datasets by split
        merged_dict = {}
        
        for split in ["train", "valid", "test"]:
            split_datasets = []
            
            for dataset_name, dataset in all_datasets.items():
                if split in dataset:
                    # Add source information to each example
                    dataset_with_source = dataset[split].map(
                        lambda x: {**x, "source_dataset": dataset_name}
                    )
                    split_datasets.append(dataset_with_source)
            
            if split_datasets:
                # Concatenate datasets for this split
                from datasets import concatenate_datasets
                merged_split = concatenate_datasets(split_datasets)
                merged_dict[split] = merged_split
        
        merged_dataset = DatasetDict(merged_dict)
        
        # Create repository
        repo_id = f"{self.hf_username}/{repo_name}"
        try:
            create_repo(repo_id, repo_type="dataset", exist_ok=True)
            print(f"Created/found repository: {repo_id}")
        except Exception as e:
            print(f"Error creating repository: {e}")
            return
        
        # Upload merged dataset
        try:
            merged_dataset.push_to_hub(repo_id)
            print(f"Successfully uploaded merged dataset to {repo_id}")
            
            # Create and upload merged data.yaml and README
            self._create_merged_config(dataset_names, repo_id, description)
            
        except Exception as e:
            print(f"Error uploading merged dataset: {e}")
    
    def _upload_additional_files(self, dataset_path: Path, repo_id: str, description: str | None = None):
        """Upload additional files like data.yaml and README"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Copy data.yaml
            if (dataset_path / "data.yaml").exists():
                shutil.copy2(dataset_path / "data.yaml", temp_path / "data.yaml")
            
            # Copy README files
            for readme_file in dataset_path.glob("README*"):
                shutil.copy2(readme_file, temp_path / readme_file.name)
            
            # Create dataset card
            self._create_dataset_card(temp_path, dataset_path.name, description)
            
            # Upload files
            upload_folder(
                folder_path=temp_path,
                repo_id=repo_id,
                repo_type="dataset"
            )
    
    def _create_merged_config(self, dataset_names: List[str], repo_id: str, description: str | None = None):
        """Create configuration files for merged dataset"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create merged data.yaml
            merged_config = {
                "names": {
                    i: name for i, name in enumerate([
                        "black-bishop", "black-king", "black-knight", "black-pawn",
                        "black-queen", "black-rook", "white-bishop", "white-king",
                        "white-knight", "white-pawn", "white-queen", "white-rook"
                    ])
                },
                "nc": 12,
                "source_datasets": dataset_names
            }
            
            with open(temp_path / "data.yaml", 'w') as f:
                yaml.dump(merged_config, f, sort_keys=False)
            
            # Create dataset card for merged dataset
            self._create_dataset_card(temp_path, f"merged-{'-'.join(dataset_names)}", description, is_merged=True)
            
            # Upload files
            upload_folder(
                folder_path=temp_path,
                repo_id=repo_id,
                repo_type="dataset"
            )
    
    def _create_dataset_card(self, output_path: Path, dataset_name: str, description: str | None = None, is_merged: bool = False):
        """Create a dataset card (README.md) for the Hugging Face dataset"""
        card_content = f"""---
license: cc-by-4.0
task_categories:
- object-detection
tags:
- chess
- computer-vision
- yolo
- object-detection
size_categories:
- 1K<n<10K
---

# Chess Piece Detection Dataset{'s' if is_merged else ''}: {dataset_name}

## Dataset Description

{'This is a merged dataset combining multiple chess piece detection datasets.' if is_merged else 'This dataset contains chess piece detection annotations in YOLOv8 format.'}

{description or 'A dataset for detecting chess pieces on a chessboard using computer vision techniques.'}

## Dataset Structure

The dataset follows the YOLOv8 format with the following structure:
- `train/`: Training images and labels
- `valid/`: Validation images and labels  
- `test/`: Test images and labels

## Classes

The dataset contains 12 classes of chess pieces:

0. black-bishop
1. black-king
2. black-knight
3. black-pawn
4. black-queen
5. black-rook
6. white-bishop
7. white-king
8. white-knight
9. white-pawn
10. white-queen
11. white-rook

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{self.hf_username}/{dataset_name}")

# Access different splits
train_data = dataset["train"]
valid_data = dataset["valid"]
test_data = dataset["test"]

# Example: Access first training image and annotations
example = train_data[0]
image = example["image"]
annotations = example["annotations"]
```

## Citation

If you use this dataset, please consider citing the original sources and this repository.

## License

This dataset is released under the CC BY 4.0 license.
"""
        
        with open(output_path / "README.md", 'w') as f:
            f.write(card_content)


def main():
    """Main function to demonstrate usage"""
    uploader = ChessDatasetUploader(DATA_FOLDER_PATH, HF_USERNAME)
    
    # Upload individual datasets
    print("=== Uploading Individual Datasets ===")
    
    uploader.upload_individual_dataset(
        dataset_name="chess_pieces_dominique",
        repo_name="chess-pieces-dominique",
        description="Chess piece detection dataset from Dominique/Roboflow with 12 classes of chess pieces."
    )
    
    uploader.upload_individual_dataset(
        dataset_name="chess_pieces_roboflow",
        repo_name="chess-pieces-roboflow", 
        description="Chess piece detection dataset from Roboflow with processed labels and 12 classes of chess pieces."
    )
    
    # Upload merged dataset
    print("\n=== Uploading Merged Dataset ===")
    
    uploader.merge_and_upload_datasets(
        dataset_names=["chess_pieces_dominique", "chess_pieces_roboflow"],
        repo_name="chess-pieces-merged",
        description="Merged chess piece detection dataset combining multiple sources for improved model training."
    )


if __name__ == "__main__":
    main() 