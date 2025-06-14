#!/usr/bin/env python3
"""
HuggingFace Upload CLI for Chess AI Models

Upload trained chess AI models (board corner detection, board segmentation, or piece detection) to HuggingFace Model Hub.

Usage:
    # Corner Detection Models
    python scripts/upload_hf.py --model models/chess_board_detection/corner_detection_training/weights/best.pt --repo-name username/chessboard-corner-detector --model-task corner-detection
    
    # Segmentation Models
    python scripts/upload_hf.py --model artifacts/models/chess_board_segmentation/polygon_segmentation_training/weights/best.pt --repo-name username/chessboard-segmentation --model-task segmentation
    
    # Piece Detection Models
    python scripts/upload_hf.py --model models/chess_piece_detection/training_yolo11s/weights/best.pt --repo-name username/chess-piece-detector --model-task detection
    
    # Upload with custom description and tags
    python scripts/upload_hf.py --model path/to/model.pt --repo-name username/model-name --model-task segmentation --description "My custom model" --tags computer-vision,chess,detection
    
    # Upload and make repository private
    python scripts/upload_hf.py --model path/to/model.pt --repo-name username/my-chess-model --model-task corner-detection --private

Examples:
    # Corner detection model
    python scripts/upload_hf.py --model models/chess_board_detection/corner_detection_training/weights/best.pt --repo-name myusername/chess-corner-detector --model-task corner-detection
    
    # Segmentation model
    python scripts/upload_hf.py --model artifacts/models/chess_board_segmentation/training/weights/best.pt --repo-name myusername/chess-segmentation --model-task segmentation
    
    # Piece detection model
    python scripts/upload_hf.py --model models/chess_piece_detection/training_yolo11s/weights/best.pt --repo-name myusername/chess-piece-detector --model-task detection
    
    # Complete upload with metadata
    python scripts/upload_hf.py \
        --model artifacts/models/chess_board_segmentation/training/weights/best.pt \
        --repo-name myusername/chess-segmentation \
        --model-task segmentation \
        --description "YOLO-based chessboard segmentation model" \
        --tags computer-vision,chess,yolo,segmentation \
        --license mit \
        --dataset-name "Chess Board Segmentation Dataset"

Prerequisites:
    1. Install HuggingFace CLI: pip install huggingface_hub
    2. Login to HuggingFace: huggingface-cli login
    3. Have a trained model file (.pt)
"""

import argparse
import sys
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv()

def parse_args():
    """Parse command line arguments for HuggingFace upload."""
    parser = argparse.ArgumentParser(
        description="Upload ChessBoard Corner Detection Model to HuggingFace Hub",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ========================================
    # Required Arguments
    # ========================================
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Path to trained model file (.pt file). Usually in models/chess_board_detection/[training_name]/weights/best.pt"
    )
    parser.add_argument(
        "--repo-name", 
        type=str, 
        required=True,
        help="HuggingFace repository name in format 'username/model-name' (e.g., 'myusername/chess-corner-detector')"
    )
    parser.add_argument(
        "--model-task", 
        type=str, 
        required=True,
        choices=["corner-detection", "segmentation", "detection"],
        help="Type of model task: 'corner-detection' for corner detection models, 'segmentation' for segmentation models, 'detection' for chess piece detection models"
    )
    
    # ========================================
    # Repository Configuration
    # ========================================
    parser.add_argument(
        "--description", 
        type=str, 
        default=None,
        help="Description for the model repository (auto-generated based on model-task if not provided)"
    )
    parser.add_argument(
        "--tags", 
        type=str, 
        default=None,
        help="Comma-separated tags for the model (auto-generated based on model-task if not provided)"
    )
    parser.add_argument(
        "--license", 
        type=str, 
        default="apache-2.0",
        choices=["apache-2.0", "mit", "bsd-3-clause", "gpl-3.0", "lgpl-3.0", "cc-by-4.0", "cc-by-sa-4.0", "other"],
        help="License for the model"
    )
    parser.add_argument(
        "--private", 
        action="store_true",
        help="Make the repository private (default is public)"
    )
    
    # ========================================
    # Model Metadata
    # ========================================
    parser.add_argument(
        "--dataset-name", 
        type=str, 
        default=None,
        help="HuggingFace dataset ID used for training (format: 'username/dataset-name'). If not a valid HF dataset ID, will be omitted from model card metadata."
    )
    parser.add_argument(
        "--model-type", 
        type=str, 
        default="yolo",
        help="Type of model architecture"
    )
    parser.add_argument(
        "--task", 
        type=str, 
        default=None,
        help="Primary task the model performs (auto-generated based on model-task if not provided)"
    )
    
    # ========================================
    # Additional Files
    # ========================================
    parser.add_argument(
        "--include-training-dir", 
        action="store_true",
        help="Include the entire training directory (with logs, plots, etc.)"
    )
    parser.add_argument(
        "--example-images", 
        type=str, 
        help="Path to directory containing example images to include in the repo"
    )
    
    # ========================================
    # Control Options
    # ========================================
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be uploaded without actually uploading"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show detailed output during upload process"
    )
    
    return parser.parse_args()

def set_task_defaults(args):
    """Set default values based on the model task."""
    if args.model_task == "corner-detection":
        if args.description is None:
            args.description = "ChessBoard corner detection model trained with YOLO"
        if args.tags is None:
            args.tags = "computer-vision,chess,yolo,object-detection"
        if args.task is None:
            args.task = "object-detection"
    elif args.model_task == "segmentation":
        if args.description is None:
            args.description = "ChessBoard segmentation model trained with YOLO"
        if args.tags is None:
            args.tags = "computer-vision,chess,yolo,segmentation,instance-segmentation"
        if args.task is None:
            args.task = "instance-segmentation"
    elif args.model_task == "detection":
        if args.description is None:
            args.description = "Chess piece detection model trained with YOLO"
        if args.tags is None:
            args.tags = "computer-vision,chess,yolo,object-detection,piece-detection"
        if args.task is None:
            args.task = "object-detection"
    
    # Note: dataset_name is left as None by default since we need a valid HF dataset ID

def validate_args(args) -> bool:
    """Validate command line arguments."""
    # Check if model file exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Error: Model file '{model_path}' does not exist!")
        print("💡 Train a model first with:")
        if args.model_task == "corner-detection":
            print("   python src/chess_board_detection/train.py")
        elif args.model_task == "segmentation":
            print("   python src/chess_board_detection/yolo/segmentation/train_segmentation.py")
        elif args.model_task == "detection":
            print("   python src/chess_piece_detection/train.py")
        return False
    
    # Validate repository name format
    if "/" not in args.repo_name:
        print(f"❌ Error: Repository name must be in format 'username/model-name'")
        print(f"💡 Example: 'myusername/chess-corner-detector'")
        return False
    
    # Check example images directory if specified
    if args.example_images:
        example_dir = Path(args.example_images)
        if not example_dir.exists():
            print(f"❌ Error: Example images directory '{example_dir}' does not exist!")
            return False
        if not example_dir.is_dir():
            print(f"❌ Error: '{example_dir}' is not a directory!")
            return False
    
    return True

def check_huggingface_auth() -> bool:
    """Check if user is authenticated with HuggingFace."""
    import os
    
    try:
        from huggingface_hub import whoami
        
        # First try CLI authentication
        try:
            user_info = whoami()
            if user_info is not None:
                return True
        except:
            pass
        
        # If CLI auth fails, try environment variables
        hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
        if hf_token:
            try:
                from huggingface_hub import login
                login(token=hf_token)
                user_info = whoami()
                if user_info is not None:
                    print("✅ Authenticated using environment variable token")
                    return True
            except Exception as e:
                print(f"❌ Error authenticating with environment token: {e}")
        
        # If both methods fail
        print("❌ Error: Not logged in to HuggingFace!")
        print("💡 Option 1: Login with CLI: huggingface-cli login")
        print("💡 Option 2: Set environment variable: HF_TOKEN or HUGGINGFACE_HUB_TOKEN")
        print("💡 Get your token from: https://huggingface.co/settings/tokens")
        return False
        
    except ImportError:
        print("❌ Error: HuggingFace Hub not installed!")
        print("💡 Install with: pip install huggingface_hub")
        return False

def load_model_for_upload(model_path: Path, model_task: str, verbose: bool = False):
    """Load the model to prepare for upload."""
    try:
        if model_task == "corner-detection":
            from src.chess_board_detection.model import ChessBoardModel
            if verbose:
                print(f"🔄 Loading corner detection model from: {model_path}")
            model = ChessBoardModel(model_path=model_path)
        elif model_task == "segmentation":
            from src.chess_board_detection.yolo.segmentation.segmentation_model import ChessBoardSegmentationModel
            if verbose:
                print(f"🔄 Loading segmentation model from: {model_path}")
            model = ChessBoardSegmentationModel(model_path=model_path)
        elif model_task == "detection":
            from src.chess_piece_detection.model import ChessModel
            if verbose:
                print(f"🔄 Loading piece detection model from: {model_path}")
            model = ChessModel(model_path=model_path)
        else:
            raise ValueError(f"Unknown model task: {model_task}")
        
        if verbose:
            print("✅ Model loaded successfully")
            
        return model
        
    except Exception as e:
        print(f"❌ Failed to load {model_task} model: {e}")
        print("💡 Make sure the model file is a valid YOLO .pt file")
        return None

def parse_tags(tags_string: str) -> List[str]:
    """Parse comma-separated tags string into a list."""
    return [tag.strip() for tag in tags_string.split(",") if tag.strip()]

def create_model_card(args, model_path: Path) -> str:
    """Create a model card (README.md) for the repository."""
    tags = parse_tags(args.tags)
    
    # Check if dataset_name looks like a valid HF dataset ID (contains /)
    datasets_section = ""
    if args.dataset_name and "/" in args.dataset_name:
        datasets_section = f"""datasets:
- {args.dataset_name}"""
    
    # Common model card header
    model_card = f"""---
license: {args.license}
tags:
{chr(10).join(f"- {tag}" for tag in tags)}
{datasets_section}
model-index:
- name: {args.repo_name.split('/')[-1]}
  results: []
---

# {args.repo_name.split('/')[-1]}

{args.description}

## Model Details

- **Model Type**: {args.model_type.upper()} {args.model_task.title().replace('-', ' ')}
- **Task**: {args.task}
- **License**: {args.license}

## Usage

"""
    
    # Add task-specific usage examples
    if args.model_task == "corner-detection":
        model_card += """### Loading the Model

```python
from src.chess_board_detection.model import ChessBoardModel

# Load the model
model = ChessBoardModel(model_path="path/to/downloaded/model.pt")

# Detect corners in an image
corners = model.get_corner_coordinates("path/to/chessboard_image.jpg")
print(f"Detected corners: {corners}")

# Visualize results
model.plot_eval("path/to/chessboard_image.jpg", show=True)
```

### Direct YOLO Usage

```python
from ultralytics import YOLO

# Load the model
model = YOLO("path/to/downloaded/model.pt")

# Run detection
results = model("path/to/chessboard_image.jpg")

# Get bounding boxes and keypoints
for result in results:
    boxes = result.boxes
    keypoints = result.keypoints
    print(f"Detected {len(boxes)} chessboard(s)")
```

### Training Data Format

This model expects YOLO detection format with keypoint annotations:

```yaml
# data.yaml
train: path/to/train/images
val: path/to/val/images
nc: 1
names: ['chessboard']
```

With corresponding label files containing keypoint coordinates:
```
# labels/image.txt
0 x_center y_center width height x1 y1 v1 x2 y2 v2 ...  # normalized coordinates
```

## Training

This model was trained using the ChessBoard Corner Detection training pipeline:

```bash
python src/chess_board_detection/train.py \\
    --data data/chessboard_corners/data.yaml \\
    --epochs 50 \\
    --batch 16
```
"""
    
    elif args.model_task == "segmentation":
        model_card += """### Loading the Model

```python
from src.chess_board_detection.yolo.segmentation.segmentation_model import ChessBoardSegmentationModel

# Load the model
model = ChessBoardSegmentationModel(model_path="path/to/downloaded/model.pt")

# Get polygon coordinates for a chessboard
polygon_info, is_valid = model.get_polygon_coordinates("path/to/chessboard_image.jpg")

if is_valid:
    print(f"Detected chessboard polygon: {polygon_info}")
    
    # Extract corners from the segmentation
    corners = model.extract_corners_from_segmentation(
        "path/to/chessboard_image.jpg", 
        polygon_info
    )
    print(f"Extracted corners: {corners}")

# Visualize results
model.plot_eval("path/to/chessboard_image.jpg", show=True)
```

### Direct YOLO Usage

```python
from ultralytics import YOLO

# Load the model
model = YOLO("path/to/downloaded/model.pt")

# Run segmentation
results = model("path/to/chessboard_image.jpg")

# Get masks and polygons
for result in results:
    if result.masks is not None:
        for mask in result.masks:
            polygon = mask.xy[0]  # Polygon coordinates
            print(f"Polygon points: {polygon}")
```

### Training Data Format

This model expects YOLO segmentation format with polygon annotations:

```yaml
# data.yaml
train: path/to/train/images
val: path/to/val/images
nc: 1
names: ['chessboard']
```

With corresponding label files containing polygon coordinates:
```
# labels/image.txt
0 x1 y1 x2 y2 x3 y3 x4 y4 ...  # normalized coordinates
```

## Training

This model was trained using the ChessBoard Segmentation training pipeline:

```bash
python src/chess_board_detection/yolo/segmentation/train_segmentation.py \\
    --data data/chessboard_segmentation/chess-board-3/data.yaml \\
    --epochs 100 \\
    --batch 16 \\
    --pretrained-model yolov8s-seg.pt
```
"""
    
    elif args.model_task == "detection":
        model_card += """### Loading the Model

```python
from src.chess_piece_detection.model import ChessModel

# Load the model
model = ChessModel(model_path="path/to/downloaded/model.pt")

# Detect chess pieces in an image
detected_pieces = model.detect_pieces("path/to/chessboard_image.jpg")

for square_num, piece_class in detected_pieces:
    piece_name = model.get_piece_name(piece_class)
    print(f"Square {square_num}: {piece_name}")

# Visualize results
model.visualize_detections("path/to/chessboard_image.jpg", show=True)
```

### Direct YOLO Usage

```python
from ultralytics import YOLO

# Load the model
model = YOLO("path/to/downloaded/model.pt")

# Run detection on chess piece images
results = model("path/to/chessboard_image.jpg")

# Get bounding boxes and classifications
for result in results:
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            print(f"Detected piece class {class_id} with confidence {confidence:.2f}")
```

### Training Data Format

This model expects YOLO detection format with chess piece annotations:

```yaml
# data.yaml
train: path/to/train/images
val: path/to/val/images
nc: 12  # Number of chess piece classes
names: ['white-king', 'white-queen', 'white-rook', 'white-bishop', 'white-knight', 'white-pawn', 
        'black-king', 'black-queen', 'black-rook', 'black-bishop', 'black-knight', 'black-pawn']
```

With corresponding label files containing piece bounding boxes:
```
# labels/image.txt
class_id x_center y_center width height  # normalized coordinates
```

## Training

This model was trained using the Chess Piece Detection training pipeline:

```bash
python src/chess_piece_detection/train.py \\
    --data data/chess_pieces/data.yaml \\
    --epochs 100 \\
    --batch 16 \\
    --img-size 640
```
"""
    
    # Common footer
    model_card += f"""
## Model Performance

<!-- Add performance metrics here after training -->

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{{{args.repo_name.replace('/', '_').replace('-', '_')},
    title={{{args.repo_name.split('/')[-1]}}},
    author={{{args.repo_name.split('/')[0]}}},
    year={{2024}},
    publisher={{Hugging Face}},
    url={{https://huggingface.co/{args.repo_name}}}
}}
```
"""
    
    return model_card

def upload_model(model_path: Path, args) -> bool:
    """Upload the model to HuggingFace Hub using HF Hub API."""
    try:
        from huggingface_hub import HfApi
        import tempfile
        import shutil
        
        if args.dry_run:
            print("🔍 DRY RUN - Would upload model with following configuration:")
            print(f"   Repository: {args.repo_name}")
            print(f"   Description: {args.description}")
            print(f"   Tags: {args.tags}")
            print(f"   License: {args.license}")
            print(f"   Private: {args.private}")
            return True
        
        if args.verbose:
            print(f"🚀 Uploading model to: {args.repo_name}")
        
        # Initialize HuggingFace API
        api = HfApi()
        
        # Create repository if it doesn't exist
        try:
            api.create_repo(
                repo_id=args.repo_name,
                private=args.private,
                exist_ok=True
            )
            print(f"✅ Repository {args.repo_name} created/verified")
        except Exception as e:
            print(f"❌ Failed to create repository: {e}")
            return False
        
        # Create temporary directory for files to upload
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Copy model file
            model_dest = temp_path / "model.pt"
            shutil.copy2(model_path, model_dest)
            print(f"📁 Model copied to temporary directory")
            
            # Create model card
            model_card_content = create_model_card(args, model_path)
            readme_path = temp_path / "README.md"
            readme_path.write_text(model_card_content)
            print(f"📝 Model card created")
            
            # Copy additional files if specified
            if args.include_training_dir:
                training_dir = model_path.parent.parent
                if training_dir.exists():
                    training_dest = temp_path / "training"
                    shutil.copytree(training_dir, training_dest)
                    print(f"📂 Training directory copied")
            
            if args.example_images:
                example_dir = Path(args.example_images)
                if example_dir.exists():
                    examples_dest = temp_path / "examples"
                    shutil.copytree(example_dir, examples_dest)
                    print(f"🖼️ Example images copied")
            
            # Upload all files
            try:
                api.upload_folder(
                    folder_path=temp_dir,
                    repo_id=args.repo_name,
                    commit_message=f"Upload {args.model_type} {args.model_task} model"
                )
                print(f"✅ Files uploaded successfully")
                return True
                
            except Exception as e:
                print(f"❌ Upload failed: {e}")
                return False
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def main():
    """Main function for HuggingFace upload CLI."""
    args = parse_args()
    
    # Set task-specific defaults
    set_task_defaults(args)
    
    if not validate_args(args):
        sys.exit(1)
    
    # ========================================
    # Pre-flight Checks
    # ========================================
    task_name = args.model_task.title().replace('-', ' ')
    print(f"🤗 ChessBoard {task_name} - HuggingFace Upload")
    print("=" * 60)
    
    if not args.dry_run:
        if not check_huggingface_auth():
            sys.exit(1)
        print("✅ HuggingFace authentication verified")
    
    # ========================================
    # Load Model
    # ========================================
    model = load_model_for_upload(Path(args.model), args.model_task, args.verbose)
    if not model:
        print("⚠️  Model validation failed, but proceeding with upload...")
        model = None
    
    # ========================================
    # Upload Configuration
    # ========================================
    print(f"\n📋 Upload Configuration:")
    print(f"   🎯 Repository: {args.repo_name}")
    print(f"   📝 Description: {args.description}")
    print(f"   🏷️  Tags: {args.tags}")
    print(f"   📄 License: {args.license}")
    print(f"   🔒 Private: {args.private}")
    print(f"   📁 Model file: {args.model}")
    
    if args.include_training_dir:
        training_dir = Path(args.model).parent.parent
        print(f"   📂 Training directory: {training_dir}")
        
    if args.example_images:
        print(f"   🖼️  Example images: {args.example_images}")
    
    # ========================================
    # Upload Model
    # ========================================
    print(f"\n🚀 Starting upload...")
    
    success = upload_model(Path(args.model), args)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Upload completed successfully!")
        print(f"🔗 View your model at: https://huggingface.co/{args.repo_name}")
        print(f"💡 Share with others: {args.repo_name}")
        
        if not args.dry_run:
            print(f"\n🎯 Next steps:")
            print(f"1. Test your uploaded model:")
            if args.model_task == "corner-detection":
                print(f"   python src/chess_board_detection/test_model.py --model {args.repo_name} --image path/to/image.jpg")
            elif args.model_task == "segmentation":
                print(f"   python src/chess_board_detection/yolo/segmentation/test_segmentation.py --model {args.repo_name} --image path/to/image.jpg")
            elif args.model_task == "detection":
                print(f"   python src/chess_piece_detection/test_model.py --model {args.repo_name} --image path/to/image.jpg")
            print(f"2. Update model card with performance metrics")
            print(f"3. Add example images to the repository")
    else:
        print("\n❌ Upload failed!")
        print("💡 Check your internet connection and HuggingFace authentication")
        sys.exit(1)

if __name__ == "__main__":
    main() 