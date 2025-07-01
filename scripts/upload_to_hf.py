#!/usr/bin/env python3

"""
Upload a trained LeRobot model to HuggingFace Hub.

Usage examples:

# Upload a checkpoint (description auto-generated)
python scripts/upload_to_hf.py \
    --model_path outputs/train/2025-07-01/18-42-46_act \
    --checkpoint 020000 \
    --hf_model_name dopaul/100_rooks_unclipped_act_20k 

# Upload with custom description and architecture
python scripts/upload_to_hf.py \
    --model_path outputs/train/2025-07-01/18-42-46_act \
    --checkpoint 020000 \
    --hf_model_name dopaul/100_rooks_unclipped_act_20k \
    --description "ACT policy trained on un-clipped 100 rooks dataset for 20k steps" \
    --architecture "ACT Policy"

# Upload a specific checkpoint as private
python scripts/upload_to_hf.py \
    --model_path outputs/train/2025-07-01/18-42-46_act \
    --checkpoint 040000 \
    --hf_model_name your_username/my_robot_policy \
    --private \
    --tags robot-learning manipulation-policy
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, create_repo

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_latest_checkpoint(model_path: Path) -> Optional[str]:
    """Find the latest checkpoint in the model directory."""
    checkpoints_dir = model_path / "checkpoints"
    if not checkpoints_dir.exists():
        return None
    
    # Check if there's a 'last' symlink
    last_link = checkpoints_dir / "last"
    if last_link.exists() and last_link.is_symlink():
        return last_link.readlink().name
    
    # Find the highest numbered checkpoint
    checkpoints = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if not checkpoints:
        return None
    
    return max(checkpoints, key=lambda x: int(x.name)).name


def validate_model_files(pretrained_model_path: Path) -> bool:
    """Validate that required model files exist."""
    required_files = ["config.json", "model.safetensors"]
    
    for file in required_files:
        if not (pretrained_model_path / file).exists():
            logger.error(f"Required file missing: {file}")
            return False
    
    logger.info("✅ All required model files found")
    return True


def generate_model_description(model_path: Path, checkpoint: str, architecture: Optional[str] = None) -> str:
    """Generate an automatic model description based on config files."""
    
    # Try to read config to get model info
    config_path = model_path / "checkpoints" / checkpoint / "pretrained_model" / "config.json"
    train_config_path = model_path / "checkpoints" / checkpoint / "pretrained_model" / "train_config.json"
    
    # Use provided architecture or auto-detect
    if architecture is not None:
        model_type = architecture
    else:
        model_type = "Robot Learning Policy"
        
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                
                # Extract model type from config
                if "_target_" in config:
                    target = config["_target_"]
                    if "diffusion" in target.lower():
                        model_type = "Diffusion Policy"
                    elif "act" in target.lower():
                        model_type = "ACT Policy"  
                    elif "vq_bet" in target.lower():
                        model_type = "VQ-BeT Policy"
                    else:
                        model_type = target.split(".")[-1].replace("Policy", "").replace("_", " ").title() + " Policy"
                        
            except Exception as e:
                logger.warning(f"Could not read config: {e}")
    
    task_info = ""
    if train_config_path.exists():
        try:
            with open(train_config_path) as f:
                train_config = json.load(f)
            
            # Extract task information
            if "dataset" in train_config and "repo_id" in train_config["dataset"]:
                dataset_name = train_config["dataset"]["repo_id"].split("/")[-1]
                task_info = f" trained on {dataset_name} dataset"
                
        except Exception as e:
            logger.warning(f"Could not read training config: {e}")
    
    return f"{model_type}{task_info}, trained using the LeRobot framework."


def generate_model_card(
    hf_model_name: str, 
    description: str, 
    tags: list[str], 
    license_name: str,
    checkpoint: str,
    model_path: Path,
    architecture: Optional[str] = None
) -> str:
    """Generate a basic model card for the uploaded model."""
    
    # Try to read config to get model info
    config_path = model_path / "checkpoints" / checkpoint / "pretrained_model" / "config.json"
    train_config_path = model_path / "checkpoints" / checkpoint / "pretrained_model" / "train_config.json"
    
    model_info = f"- **Checkpoint**: {checkpoint}\n"
    
    # Use provided architecture or auto-detect from config
    if architecture is not None:
        model_info += f"- **Architecture**: {architecture}\n"
    elif config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            model_info += f"- **Architecture**: {config.get('_target_', 'Unknown')}\n"
        except Exception as e:
            logger.warning(f"Could not read config: {e}")
    
    # Add other config details
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            if "n_obs_steps" in config:
                model_info += f"- **Observation Steps**: {config['n_obs_steps']}\n"
            if "n_action_steps" in config:
                model_info += f"- **Action Steps**: {config['n_action_steps']}\n"
        except Exception as e:
            logger.warning(f"Could not read config: {e}")
    
    if train_config_path.exists():
        try:
            with open(train_config_path) as f:
                train_config = json.load(f)
            if "dataset" in train_config and "repo_id" in train_config["dataset"]:
                model_info += f"- **Training Dataset**: {train_config['dataset']['repo_id']}\n"
        except Exception as e:
            logger.warning(f"Could not read training config: {e}")
    
    tags_str = "\n".join(f"- {tag}" for tag in tags)
    
    return f"""---
license: {license_name}
tags:
{tags_str}
---

# {hf_model_name.split('/')[-1]}

{description}

## Model Details

{model_info}

## Usage

```python
from lerobot.common.policies.factory import make_policy

# Load the policy
policy = make_policy.from_pretrained("{hf_model_name}")

# Use for inference
action = policy.select_action(observation)
```

## Training Details

This model was trained using the LeRobot framework and uploaded from checkpoint `{checkpoint}`.

For more information about LeRobot, visit: https://github.com/huggingface/lerobot
"""


def upload_model_to_hf(
    model_path: Path,
    hf_model_name: str,
    checkpoint: str,
    description: Optional[str] = None,
    architecture: Optional[str] = None,
    private: bool = False,
    tags: Optional[list[str]] = None,
    license_name: str = "apache-2.0",
    include_training_config: bool = True,
    create_model_card: bool = True,
) -> None:
    """Upload a model to HuggingFace Hub."""
    
    if tags is None:
        tags = ["robotics", "lerobot", "robot-learning"]
    
    logger.info(f"Using checkpoint: {checkpoint}")
    if architecture is not None:
        logger.info(f"Using specified architecture: {architecture}")
    
    # Validate paths
    pretrained_model_path = model_path / "checkpoints" / checkpoint / "pretrained_model"
    if not pretrained_model_path.exists():
        raise FileNotFoundError(f"Pretrained model not found: {pretrained_model_path}")
    
    if not validate_model_files(pretrained_model_path):
        raise ValueError("Model validation failed")
    
    # Use provided description or generate automatic one
    if description is None:
        description = generate_model_description(model_path, checkpoint, architecture)
        logger.info(f"Auto-generated description: {description}")
    else:
        logger.info(f"Using provided description: {description}")
    
    # Initialize HuggingFace API
    api = HfApi()
    
    # Create repository
    try:
        repo_url = create_repo(
            repo_id=hf_model_name,
            private=private,
            exist_ok=True,
        )
        logger.info(f"✅ Repository ready: {repo_url}")
    except Exception as e:
        logger.error(f"Failed to create repository: {e}")
        raise
    
    # Upload model files
    try:
        logger.info("📤 Uploading model files...")
        
        # Upload required model files
        api.upload_file(
            path_or_fileobj=pretrained_model_path / "config.json",
            path_in_repo="config.json",
            repo_id=hf_model_name,
        )
        
        api.upload_file(
            path_or_fileobj=pretrained_model_path / "model.safetensors", 
            path_in_repo="model.safetensors",
            repo_id=hf_model_name,
        )
        
        # Upload training config if requested
        if include_training_config:
            train_config_path = pretrained_model_path / "train_config.json"
            if train_config_path.exists():
                api.upload_file(
                    path_or_fileobj=train_config_path,
                    path_in_repo="train_config.json", 
                    repo_id=hf_model_name,
                )
                logger.info("✅ Training config uploaded")
        
        logger.info("✅ Model files uploaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to upload model files: {e}")
        raise
    
    # Create and upload model card
    if create_model_card:
        try:
            logger.info("📝 Creating model card...")
            model_card_content = generate_model_card(
                hf_model_name, description, tags, license_name, checkpoint, model_path, architecture
            )
            
            api.upload_file(
                path_or_fileobj=model_card_content.encode(),
                path_in_repo="README.md",
                repo_id=hf_model_name,
            )
            logger.info("✅ Model card uploaded")
            
        except Exception as e:
            logger.warning(f"Failed to upload model card: {e}")
    
    logger.info(f"🎉 Model successfully uploaded to: https://huggingface.co/{hf_model_name}")


def print_examples():
    """Print usage examples."""
    print("""
Example usage:

# Upload latest checkpoint (public by default, auto-generated description)
python scripts/upload_to_hf.py \\
    --model_path outputs/train/2025-07-01/18-42-46_act \\
    --checkpoint last \\
    --hf_model_name your_username/my_robot_policy

# Upload with custom description
python scripts/upload_to_hf.py \\
    --model_path outputs/train/2025-07-01/18-42-46_act \\
    --checkpoint 040000 \\
    --hf_model_name your_username/my_robot_policy \\
    --description "ACT policy trained on custom dataset for manipulation tasks"

# Upload with custom architecture name
python scripts/upload_to_hf.py \\
    --model_path outputs/train/2025-07-01/18-42-46_act \\
    --checkpoint 040000 \\
    --hf_model_name your_username/my_robot_policy \\
    --architecture "Custom ACT Policy v2"

# Upload specific checkpoint as private
python scripts/upload_to_hf.py \\
    --model_path outputs/train/2025-07-01/18-42-46_act \\
    --checkpoint 040000 \\
    --hf_model_name your_username/my_robot_policy \\
    --private \\
    --tags robot-learning manipulation-policy act-model

# Upload without training config
python scripts/upload_to_hf.py \\
    --model_path outputs/train/2025-07-01/18-42-46_act \\
    --checkpoint 040000 \\
    --hf_model_name your_username/my_robot_policy \\
    --no-training-config
""")


def main():
    parser = argparse.ArgumentParser(description="Upload LeRobot model to HuggingFace Hub")
    
    # Required arguments
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to the training output directory (e.g., outputs/train/2025-07-01/18-42-46_act)"
    )
    parser.add_argument(
        "--hf_model_name", 
        type=str,
        required=True,
        help="HuggingFace model name (e.g., username/model_name)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint to upload (e.g., '040000', 'last'). Use 'last' for latest checkpoint."
    )
    
    # Optional arguments
    parser.add_argument(
        "--description",
        type=str,
        help="Custom description for the model card (if not provided, auto-generated from config files)"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        help="Model architecture name (e.g., 'ACT Policy', 'Diffusion Policy'). If not provided, auto-detected from config files"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private (default: public)"
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=["robotics", "lerobot", "robot-learning"],
        help="Tags for the model (space-separated)"
    )
    parser.add_argument(
        "--license",
        type=str,
        default="apache-2.0",
        help="License for the model"
    )
    parser.add_argument(
        "--no-training-config",
        action="store_true",
        help="Do not include training configuration in upload"
    )
    parser.add_argument(
        "--no-model-card",
        action="store_true", 
        help="Skip creating model card"
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Show usage examples and exit"
    )
    
    args = parser.parse_args()
    
    if args.example:
        print_examples()
        sys.exit(0)
    
    # Handle 'last' checkpoint
    if args.checkpoint == "last":
        latest_checkpoint = find_latest_checkpoint(args.model_path)
        if latest_checkpoint is None:
            logger.error(f"No checkpoints found in {args.model_path}/checkpoints")
            sys.exit(1)
        args.checkpoint = latest_checkpoint
        logger.info(f"Resolved 'last' to checkpoint: {args.checkpoint}")
    
    # Validate model path
    if not args.model_path.exists():
        logger.error(f"Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    # Validate HF model name format
    if "/" not in args.hf_model_name:
        logger.error("HuggingFace model name must be in format 'username/model_name'")
        sys.exit(1)
    
    try:
        upload_model_to_hf(
            model_path=args.model_path,
            hf_model_name=args.hf_model_name,
            checkpoint=args.checkpoint,
            description=args.description,
            architecture=args.architecture,
            private=args.private,
            tags=args.tags,
            license_name=args.license,
            include_training_config=not args.no_training_config,
            create_model_card=not args.no_model_card,
        )
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 