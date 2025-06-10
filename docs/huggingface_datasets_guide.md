# Uploading Chess Datasets to Hugging Face Hub

This guide shows you how to upload your chess piece detection datasets to Hugging Face Hub, both individually and as merged datasets.

## 🚀 Quick Setup

### 1. Environment Setup

First, make sure you have the required environment variables set in your `.env` file:

```bash
# Add to your .env file
HF_USERNAME=your-huggingface-username
DATA_FOLDER_PATH=/path/to/your/data/folder
```

### 2. Authentication

Login to Hugging Face CLI:

```bash
huggingface-cli login
```

Enter your Hugging Face token when prompted. You can get your token from [Hugging Face Settings](https://huggingface.co/settings/tokens).

### 3. Install Dependencies

The required dependencies should already be installed, but if needed:

```bash
uv add datasets huggingface_hub pillow
```

## 📊 Dataset Formats Supported

The uploader supports YOLOv8 format datasets with the following structure:

```
dataset_name/
├── data.yaml          # Dataset configuration
├── train/
│   ├── images/        # Training images
│   └── labels/        # Training labels (.txt files)
├── valid/
│   ├── images/        # Validation images
│   └── labels/        # Validation labels
└── test/
    ├── images/        # Test images
    └── labels/        # Test labels
```

## 🎯 Usage Examples

### Option 1: Upload Individual Datasets

```python
from pathlib import Path
from src.data_prep.upload_to_hf import ChessDatasetUploader

# Initialize uploader
uploader = ChessDatasetUploader(DATA_FOLDER_PATH, HF_USERNAME)

# Upload individual dataset
uploader.upload_individual_dataset(
    dataset_name="chess_pieces_roboflow",
    repo_name="my-chess-dataset",
    description="Chess piece detection dataset with 12 classes"
)
```

### Option 2: Upload Merged Dataset

```python
# Merge and upload multiple datasets
uploader.merge_and_upload_datasets(
    dataset_names=["chess_pieces_dominique", "chess_pieces_roboflow"],
    repo_name="chess-pieces-merged",
    description="Combined chess dataset for better model training"
)
```

### Option 3: Run the Complete Example

```bash
python examples/upload_datasets_example.py
```

## 📋 What Gets Uploaded

### For Individual Datasets:
- ✅ Training, validation, and test images with annotations
- ✅ Original `data.yaml` configuration file
- ✅ Existing README files from the dataset
- ✅ Generated dataset card with metadata and usage examples
- ✅ Proper Hugging Face Dataset format with features schema

### For Merged Datasets:
- ✅ Combined training, validation, and test splits
- ✅ Source dataset tracking (each example includes `source_dataset` field)
- ✅ Unified `data.yaml` configuration
- ✅ Comprehensive dataset card explaining the merge
- ✅ Statistics from all contributing datasets

## 🏷️ Dataset Features Schema

Each uploaded dataset includes:

```python
{
    "image": Image(),           # The actual image
    "image_id": str,           # Unique identifier
    "annotations": [           # List of bounding box annotations
        {
            "class_id": int,        # Chess piece class (0-11)
            "x_center": float,      # Normalized x center (0-1)
            "y_center": float,      # Normalized y center (0-1) 
            "width": float,         # Normalized width (0-1)
            "height": float,        # Normalized height (0-1)
        }
    ],
    "image_width": int,        # Original image width
    "image_height": int,       # Original image height
    "source_dataset": str,     # Only for merged datasets
}
```

## 🎭 Chess Piece Classes

All datasets use these 12 standardized classes:

| ID | Class Name    | Description |
|----|---------------|-------------|
| 0  | black-bishop  | Black bishop piece |
| 1  | black-king    | Black king piece |
| 2  | black-knight  | Black knight piece |
| 3  | black-pawn    | Black pawn piece |
| 4  | black-queen   | Black queen piece |
| 5  | black-rook    | Black rook piece |
| 6  | white-bishop  | White bishop piece |
| 7  | white-king    | White king piece |
| 8  | white-knight  | White knight piece |
| 9  | white-pawn    | White pawn piece |
| 10 | white-queen   | White queen piece |
| 11 | white-rook    | White rook piece |

## 💻 Loading Datasets from Hugging Face

Once uploaded, you can easily load your datasets:

```python
from datasets import load_dataset

# Load your dataset
dataset = load_dataset("your-username/chess-pieces-merged")

# Access different splits
train_data = dataset["train"]
valid_data = dataset["valid"] 
test_data = dataset["test"]

# Example: View first training example
example = train_data[0]
image = example["image"]
annotations = example["annotations"]

print(f"Image size: {example['image_width']}x{example['image_height']}")
print(f"Number of pieces detected: {len(annotations)}")

# For merged datasets, check source
if "source_dataset" in example:
    print(f"Original source: {example['source_dataset']}")
```

## 🔄 Advanced Usage

### Custom Dataset Names and Descriptions

```python
uploader.upload_individual_dataset(
    dataset_name="my_custom_dataset",
    repo_name="chess-pieces-custom-v1",
    description="Custom chess dataset with specific camera angles and lighting conditions"
)
```

### Selective Merging

```python
# Merge only specific datasets
uploader.merge_and_upload_datasets(
    dataset_names=["chess_pieces_roboflow"],  # Only one dataset
    repo_name="chess-pieces-single-source",
    description="Single-source dataset for controlled experiments"
)
```

## 🛠️ Troubleshooting

### Common Issues:

1. **Authentication Error**: Make sure you're logged in with `huggingface-cli login`
2. **Dataset Not Found**: Check that your dataset paths are correct in `.env`
3. **Upload Timeout**: Large datasets may take time; the process will show progress
4. **Permission Denied**: Ensure your HF token has write permissions

### Checking Upload Status:

```python
from huggingface_hub import HfApi

api = HfApi()
files = api.list_repo_files("your-username/your-dataset-name", repo_type="dataset")
print(f"Uploaded files: {files}")
```

## 📈 Best Practices

1. **Use Descriptive Names**: Choose clear repository names that indicate the dataset purpose
2. **Add Good Descriptions**: Help others understand your dataset with detailed descriptions
3. **Version Control**: Use version tags in repo names for dataset updates
4. **Test Loading**: Always test loading your uploaded dataset before sharing
5. **Document Sources**: Include information about original data sources in descriptions

## 🎉 Benefits of Hugging Face Datasets

- **Easy Sharing**: Share datasets with the community or keep them private
- **Version Control**: Track changes and maintain dataset versions
- **Integration**: Works seamlessly with popular ML frameworks
- **Metadata**: Rich metadata and documentation capabilities
- **Streaming**: Support for large datasets with streaming
- **Preprocessing**: Built-in data processing and transformation tools

## 📞 Need Help?

If you encounter issues:

1. Check the [Hugging Face Datasets documentation](https://huggingface.co/docs/datasets/)
2. Review the error messages carefully
3. Ensure your dataset structure matches the expected YOLOv8 format
4. Verify your environment variables are set correctly

---

*This guide covers the essential steps for uploading chess piece detection datasets to Hugging Face Hub. The provided tools handle the conversion from YOLOv8 format to Hugging Face Dataset format automatically.* 