import os
from pathlib import Path

import torch
from huggingface_hub import HfApi, login


def is_notebook():
    """
    Check if we're running in a Jupyter notebook environment (including VSCode Jupyter extension)
    """
    try:
        # Check if we're in an IPython environment
        from IPython.core.getipython import get_ipython

        ipython = get_ipython()

        if ipython is None:
            return False

        # Check if it's a Jupyter notebook (including VSCode Jupyter extension)
        if ipython.__class__.__name__ in ["ZMQInteractiveShell", "TerminalIPythonApp"]:
            return True

        # Additional check for VSCode Jupyter extension
        if hasattr(ipython, "kernel"):
            return True

        return False
    except ImportError:
        return False


def get_best_torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def _check_hf_environment():
    """Check if required Hugging Face environment variables are set."""
    # Check for HF token
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        raise OSError(
            "Hugging Face authentication token not found. Please set one of these environment variables:\n"
            "- HF_TOKEN=your_token_here\n"
            "- HUGGINGFACE_HUB_TOKEN=your_token_here\n\n"
            "You can get your token from: https://huggingface.co/settings/tokens\n"
            "Add it to your .env file or set it as an environment variable."
        )

    # Check for HF username (optional but recommended)
    hf_username = os.getenv("HF_USERNAME") or os.getenv("HUGGINGFACE_USERNAME")
    if not hf_username:
        print("Warning: HF_USERNAME not set. You'll need to specify the full repo_id (username/model-name)")

    return hf_token, hf_username


def push_model_to_huggingface(
    model,
    repo_id: str,
    model_path: Path | None = None,
    commit_message: str | None = None,
    token: str | None = None,
    private: bool = False,
    create_model_card: bool = True,
):
    """
    Push a trained model to Hugging Face Hub.

    Args:
        model: The model object (should have a save method and model_path attribute)
        repo_id: Repository ID on Hugging Face Hub (e.g., "username/chess-piece-detector")
        model_path: Path to the model to upload (auto-detected if not provided)
        commit_message: Commit message for the upload
        token: Hugging Face authentication token (uses env var if not provided)
        private: Whether to create a private repository
        create_model_card: Whether to create a basic model card

    Returns:
        str: URL of the uploaded model

    Raises:
        EnvironmentError: If required HF environment variables are not set
        ValueError: If model path is invalid
        FileNotFoundError: If model files are not found
    """
    # Check environment variables if token not provided
    if not token:
        hf_token, hf_username = _check_hf_environment()
        token = hf_token

        # Auto-prepend username if repo_id doesn't contain one
        if "/" not in repo_id and hf_username:
            repo_id = f"{hf_username}/{repo_id}"
            print(f"Using full repo ID: {repo_id}")

    # Auto-detect model path if not provided
    upload_path = model_path
    if upload_path is None:
        upload_path = _auto_detect_model_path(model)

    if upload_path is None:
        raise ValueError(
            "Could not auto-detect model path. Please provide model_path explicitly, "
            "or ensure the model has been trained/saved recently."
        )

    if not upload_path.exists():
        raise FileNotFoundError(f"Model not found at {upload_path}")

    print(f"📦 Uploading model from: {upload_path}")

    try:
        # Initialize Hugging Face API and login
        api = HfApi()
        login(token=token)
        print("✓ Successfully authenticated with Hugging Face")

        # Create repository if it doesn't exist
        try:
            api.create_repo(repo_id=repo_id, exist_ok=True, private=private)
            print(f"✓ Repository {repo_id} is ready")
        except Exception as e:
            print(f"Note: Repository creation returned: {e}")

        # Determine what files to upload
        files_to_upload = []

        if upload_path.is_file():
            # Single model file
            files_to_upload.append(upload_path)
        elif upload_path.is_dir():
            # Model directory - upload all relevant files
            for pattern in ["*.pt", "*.yaml", "*.yml", "args.yaml"]:
                files_to_upload.extend(upload_path.glob(pattern))
                files_to_upload.extend(upload_path.glob(f"**/{pattern}"))

        if not files_to_upload:
            raise ValueError(f"No model files found at {upload_path}")

        # Upload files
        uploaded_files = []
        for file_path in files_to_upload:
            try:
                relative_path = file_path.relative_to(upload_path.parent) if upload_path.is_dir() else file_path.name

                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=str(relative_path),
                    repo_id=repo_id,
                    commit_message=commit_message or f"Upload {file_path.name}",
                )
                uploaded_files.append(file_path.name)
                print(f"✓ Uploaded {file_path.name}")
            except Exception as e:
                print(f"✗ Failed to upload {file_path.name}: {e}")

        # Create basic model card if requested
        if create_model_card:
            try:
                model_card_content = generate_chess_model_card(repo_id, uploaded_files)
                api.upload_file(
                    path_or_fileobj=model_card_content.encode("utf-8"),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    commit_message=commit_message or "Add model card",
                )
                print("✓ Created model card (README.md)")
            except Exception as e:
                print(f"Note: Could not create model card: {e}")

        # Return the model URL
        model_url = f"https://huggingface.co/{repo_id}"
        print(f"\n🎉 Model successfully uploaded to: {model_url}")
        return model_url

    except Exception as e:
        print(f"❌ Failed to upload model: {e}")
        raise


def _auto_detect_model_path(model) -> Path | None:
    """
    Automatically detect the best model weights path from a trained model.

    Args:
        model: The trained model object

    Returns:
        Path to the best model weights, or None if not found
    """
    # Method 1: Check if model has a model_path attribute
    if hasattr(model, "model_path") and model.model_path:
        if model.model_path.exists():
            return model.model_path

    # Method 2: Check if model has trainer with best weights
    if hasattr(model, "model") and hasattr(model.model, "trainer"):
        trainer = model.model.trainer
        if hasattr(trainer, "best") and trainer.best:
            best_path = Path(trainer.best)
            if best_path.exists():
                return best_path

    # Method 3: Look for recent YOLO training runs
    # Check common YOLO output directories
    yolo_runs_dirs = [
        Path("runs/detect"),
        Path("runs/train"),
        Path("./runs/detect"),
        Path("./runs/train"),
    ]

    # Also check if MODELS_FOLDER env var is set
    if "MODELS_FOLDER" in os.environ:
        models_folder = Path(os.environ["MODELS_FOLDER"])
        yolo_runs_dirs.extend(
            [
                models_folder / "runs/detect",
                models_folder / "runs/train",
                models_folder / "chess_piece_detection",
            ]
        )

    # Find the most recent training directory
    latest_run = None
    latest_time = 0

    for runs_dir in yolo_runs_dirs:
        if not runs_dir.exists():
            continue

        # Look for training directories (train, train2, train3, etc.)
        for run_dir in runs_dir.glob("train*"):
            if not run_dir.is_dir():
                continue

            # Check for best.pt weights
            best_weights = run_dir / "weights" / "best.pt"
            if best_weights.exists():
                # Use modification time to find the most recent
                mod_time = best_weights.stat().st_mtime
                if mod_time > latest_time:
                    latest_time = mod_time
                    latest_run = best_weights

    if latest_run:
        print(f"🔍 Auto-detected model weights: {latest_run}")
        return latest_run

    # Method 4: Check current working directory for model files
    cwd = Path.cwd()
    for pattern in ["best.pt", "last.pt", "*.pt"]:
        model_files = list(cwd.glob(f"**/{pattern}"))
        if model_files:
            # Return the most recent one
            latest_file = max(model_files, key=lambda p: p.stat().st_mtime)
            print(f"🔍 Auto-detected model file: {latest_file}")
            return latest_file

    return None


def generate_chess_model_card(repo_id: str, uploaded_files: list) -> str:
    """Generate a basic model card for a chess piece detection model."""
    return f"""---
library_name: ultralytics
tags:
- object-detection
- chess
- computer-vision
- yolo
datasets:
- chess-pieces
pipeline_tag: object-detection
---

# Chess Piece Detection Model

This is a YOLO model trained to detect chess pieces on a chessboard.

## Model Details

- **Model Type**: YOLOv8/YOLOv11 Object Detection
- **Task**: Chess piece detection and classification
- **Framework**: Ultralytics YOLO
- **Repository**: {repo_id}

## Files

The following files are included in this model:

{chr(10).join(f"- `{file}`" for file in uploaded_files)}

## Usage

```python
from ultralytics import YOLO

# Load the model
model = YOLO('path/to/best.pt')

# Run inference
results = model('path/to/chess_image.jpg')

# Display results
results[0].show()
```

## Model Performance

This model can detect and classify various chess pieces including:
- Pawns
- Rooks
- Knights
- Bishops
- Queens
- Kings

For both black and white pieces.

## Training Data

The model was trained on chess piece datasets to achieve robust detection across different chess sets and lighting conditions.
"""
