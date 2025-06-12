#!/usr/bin/env python3
"""
HuggingFace Upload CLI for ChessBoard Corner Detection

Upload trained chessboard corner detection models to HuggingFace Model Hub.

Usage:
    # Basic upload
    python src/chess_board_detection/upload_hf.py --model models/chess_board_detection/corner_detection_training/weights/best.pt --repo-name username/chessboard-corner-detector
    
    # Upload with custom description and tags
    python src/chess_board_detection/upload_hf.py --model models/chess_board_detection/corner_detection_training/weights/best.pt --repo-name username/chessboard-corners --description "My custom corner detector" --tags computer-vision,chess,detection
    
    # Upload and make repository private
    python src/chess_board_detection/upload_hf.py --model models/chess_board_detection/corner_detection_training/weights/best.pt --repo-name username/my-chess-model --private
    
    # Run from project root as module
    python -m src.chess_board_detection.upload_hf --help

Examples:
    # Simple upload
    python src/chess_board_detection/upload_hf.py --model models/chess_board_detection/corner_detection_training/weights/best.pt --repo-name myusername/chess-corner-detector
    
    # Complete upload with metadata
    python src/chess_board_detection/upload_hf.py \
        --model models/chess_board_detection/corner_detection_training/weights/best.pt \
        --repo-name myusername/chess-corner-detector \
        --description "YOLO-based chessboard corner detection model" \
        --tags computer-vision,chess,yolo,detection \
        --license mit \
        --dataset-name "Chess Board Corners Dataset"

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
    
    # ========================================
    # Repository Configuration
    # ========================================
    parser.add_argument(
        "--description", 
        type=str, 
        default="ChessBoard corner detection model trained with YOLO",
        help="Description for the model repository"
    )
    parser.add_argument(
        "--tags", 
        type=str, 
        default="computer-vision,chess,yolo,object-detection",
        help="Comma-separated tags for the model (e.g., 'computer-vision,chess,detection')"
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
        default="Chess Board Corner Detection Dataset",
        help="Name of the dataset used for training"
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
        default="object-detection",
        help="Primary task the model performs"
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

def validate_args(args) -> bool:
    """Validate command line arguments."""
    # Check if model file exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Error: Model file '{model_path}' does not exist!")
        print("💡 Train a model first with:")
        print("   python src/chess_board_detection/train.py")
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

def load_model_for_upload(model_path: Path, verbose: bool = False):
    """Load the model to prepare for upload."""
    try:
        from src.chess_board_detection.model import ChessBoardModel
        
        if verbose:
            print(f"🔄 Loading model from: {model_path}")
        
        model = ChessBoardModel(model_path=model_path)
        
        if verbose:
            print("✅ Model loaded successfully")
            
        return model
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("💡 Make sure the model file is a valid YOLO .pt file")
        return None

def parse_tags(tags_string: str) -> List[str]:
    """Parse comma-separated tags string into a list."""
    return [tag.strip() for tag in tags_string.split(",") if tag.strip()]

def create_model_card(args, model_path: Path) -> str:
    """Create a model card (README.md) for the repository."""
    tags = parse_tags(args.tags)
    
    model_card = f"""---
license: {args.license}
tags:
{chr(10).join(f"- {tag}" for tag in tags)}
datasets:
- {args.dataset_name}
model-index:
- name: {args.repo_name.split('/')[-1]}
  results: []
---

# {args.repo_name.split('/')[-1]}

{args.description}

## Model Details

- **Model Type**: {args.model_type.upper()}
- **Task**: {args.task}
- **Dataset**: {args.dataset_name}
- **License**: {args.license}

## Usage

### Loading the Model

```python
from src.chess_board_detection.model import ChessBoardModel

# Load the model
model = ChessBoardModel.from_pretrained("{args.repo_name}")

# Detect corners in an image
corners = model.get_corner_coordinates("path/to/chessboard_image.jpg")
print(f"Detected corners: {{corners}}")

# Visualize results
model.plot_eval("path/to/chessboard_image.jpg", show=True)
```

### CLI Usage

```bash
# Test the model on an image
python src/chess_board_detection/test_model.py \\
    --model {args.repo_name} \\
    --image path/to/image.jpg

# Get corner coordinates as JSON
python src/chess_board_detection/test_model.py \\
    --model {args.repo_name} \\
    --image path/to/image.jpg \\
    --json-output
```

## Model Performance

<!-- Add performance metrics here after training -->

## Training

This model was trained using the ChessBoard Corner Detection training pipeline:

```bash
python src/chess_board_detection/train.py \\
    --data data/chessboard_corners/data.yaml \\
    --epochs 50 \\
    --batch 16
```

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

def upload_model(model, args) -> bool:
    """Upload the model to HuggingFace Hub."""
    try:
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
        
        # Upload using the model's built-in method
        model.push_to_huggingface(
            repo_id=args.repo_name,
            private=args.private,
            create_model_card=True
        )
        
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def main():
    """Main function for HuggingFace upload CLI."""
    args = parse_args()
    
    if not validate_args(args):
        sys.exit(1)
    
    # ========================================
    # Pre-flight Checks
    # ========================================
    print("🤗 ChessBoard Corner Detection - HuggingFace Upload")
    print("=" * 60)
    
    if not args.dry_run:
        if not check_huggingface_auth():
            sys.exit(1)
        print("✅ HuggingFace authentication verified")
    
    # ========================================
    # Load Model
    # ========================================
    model = load_model_for_upload(Path(args.model), args.verbose)
    if not model:
        sys.exit(1)
    
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
    
    success = upload_model(model, args)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Upload completed successfully!")
        print(f"🔗 View your model at: https://huggingface.co/{args.repo_name}")
        print(f"💡 Share with others: {args.repo_name}")
        
        if not args.dry_run:
            print(f"\n🎯 Next steps:")
            print(f"1. Test your uploaded model:")
            print(f"   python src/chess_board_detection/test_model.py --model {args.repo_name} --image path/to/image.jpg")
            print(f"2. Update model card with performance metrics")
            print(f"3. Add example images to the repository")
    else:
        print("\n❌ Upload failed!")
        print("💡 Check your internet connection and HuggingFace authentication")
        sys.exit(1)

if __name__ == "__main__":
    main() 