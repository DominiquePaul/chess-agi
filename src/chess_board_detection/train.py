#!/usr/bin/env python3
"""
Training script for ChessBoardModel - EfficientNet B0 Keypoint Detection

This script trains an EfficientNet B0 model to detect the 4 corners of a chessboard
using keypoint regression from YOLO format labels.

Expected label format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized coordinates 0-1)
Example: 0 0.235 0.848 0.764 0.859 0.705 0.295 0.309 0.298

Usage:
    # Download data first (if needed)
    python src/chess_board_detection/download_data.py
    
    # Basic training with default settings (auto-detects dataset from download_data.py)
    python src/chess_board_detection/train.py
    
    # Run from project root as module
    python -m src.chess_board_detection.train
    
    # Custom training parameters
    python src/chess_board_detection/train.py --epochs 100 --batch 32
    
    # With Weights & Biases tracking
    python src/chess_board_detection/train.py --wandb --wandb-project chess-board-detection
    
    # Complete example with all options
    python src/chess_board_detection/train.py \
        --data data/chessboard_corners \
        --models-folder models/chess_board_detection \
        --name efficientnet_keypoints \
        --epochs 100 \
        --batch 32 \
        --lr 0.001 \
        --patience 20 \
        --val-split 0.2 \
        --augmentation \
        --loss-type smooth_l1 \
        --wandb \
        --wandb-project chess-detection \
        --verbose
"""

import os
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import wandb
from dotenv import load_dotenv
load_dotenv()


from src.chess_board_detection.model import ChessBoardModel


class ChessboardKeypointDataset(Dataset):
    """Dataset for chessboard keypoint regression from YOLO format labels."""
    
    def __init__(self, images_dir: Path, labels_dir: Path, transform=None, model_transform=None):
        """
        Initialize dataset.
        
        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing YOLO format label files (.txt)
            transform: Data augmentation transforms
            model_transform: Model preprocessing transforms
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.model_transform = model_transform
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.images_dir.glob(f'*{ext}'))
            image_files.extend(self.images_dir.glob(f'*{ext.upper()}'))
        
        # Create samples list
        self.samples = []
        for img_path in image_files:
            # Find corresponding label file
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                self.samples.append({
                    'image_path': img_path,
                    'label_path': label_path
                })
        
        print(f"Loaded {len(self.samples)} samples")
        print(f"Images from: {self.images_dir}")
        print(f"Labels from: {self.labels_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Load YOLO label
        with open(sample['label_path'], 'r') as f:
            label_line = f.readline().strip()
        
        # Parse YOLO format: class_id x1 y1 x2 y2 x3 y3 x4 y4 [x5 y5] [x1 y1] (variable corners, may have repeated first point)
        parts = label_line.split()
        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]  # All coordinate values
        
        if len(parts) == 9:  # 4 corners: class_id + 8 coordinates
            keypoints = coords  # Already normalized [0-1]
        elif len(parts) == 11:  # Either 4 corners + repeated first, OR 5 corners
            # Check if last 2 values match first 2 (repeated first point)
            if len(coords) >= 4 and abs(coords[-2] - coords[0]) < 1e-6 and abs(coords[-1] - coords[1]) < 1e-6:
                # 4 corners with repeated first point - take first 8 coordinates
                keypoints = coords[:8]
            else:
                # 5 corners - take first 8 coordinates (first 4 corners only)
                keypoints = coords[:8]
        elif len(parts) == 13:  # 5 corners + repeated first point
            # Take first 8 coordinates (first 4 corners only)
            keypoints = coords[:8]
        else:
            # For any other number of coordinates, take first 8 or pad to 8
            if len(coords) >= 8:
                keypoints = coords[:8]  # Take first 4 corners
            else:
                raise ValueError(f"Invalid label format in {sample['label_path']}: expected at least 8 coordinates, got {len(coords)}")
        
        # Ensure we have exactly 8 coordinates for 4 corners
        if len(keypoints) != 8:
            raise ValueError(f"Expected 8 keypoint coordinates for 4 corners, got {len(keypoints)} in {sample['label_path']}")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Apply model preprocessing
        if self.model_transform:
            image = self.model_transform(image)
        
        return {
            'image': image,
            'keypoints': torch.tensor(keypoints, dtype=torch.float32),
            'image_path': str(sample['image_path']),
            'class_id': class_id
        }


def detect_dataset_structure(data_dir: Path):
    """
    Detect and return the dataset structure from download_data.py.
    
    Args:
        data_dir: Base data directory (e.g., data/chessboard_corners)
    
    Returns:
        tuple: (dataset_folder, has_splits) or (None, False) if not found
    """
    # Look for dataset folder (e.g., chess-board-box-3)
    dataset_folders = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not dataset_folders:
        print(f"⚠️  No dataset folders found in {data_dir}")
        return None, False
    
    # Use the first dataset folder found
    dataset_folder = dataset_folders[0]
    print(f"📁 Found dataset folder: {dataset_folder}")
    
    # Check if it already has train/val splits (try both 'val' and 'valid')
    train_images = dataset_folder / "train" / "images"
    train_labels = dataset_folder / "train" / "labels"
    val_images = dataset_folder / "val" / "images"
    val_labels = dataset_folder / "val" / "labels"
    valid_images = dataset_folder / "valid" / "images"
    valid_labels = dataset_folder / "valid" / "labels"
    
    # Debug output
    print(f"🔍 Checking paths:")
    print(f"  train_images: {train_images} -> exists: {train_images.exists()}")
    print(f"  train_labels: {train_labels} -> exists: {train_labels.exists()}")
    print(f"  val_images: {val_images} -> exists: {val_images.exists()}")
    print(f"  val_labels: {val_labels} -> exists: {val_labels.exists()}")
    print(f"  valid_images: {valid_images} -> exists: {valid_images.exists()}")
    print(f"  valid_labels: {valid_labels} -> exists: {valid_labels.exists()}")
    
    # Check for val structure
    val_has_splits = all([
        train_images.exists() and train_images.is_dir(),
        train_labels.exists() and train_labels.is_dir(),
        val_images.exists() and val_images.is_dir(),
        val_labels.exists() and val_labels.is_dir()
    ])
    
    # Check for valid structure
    valid_has_splits = all([
        train_images.exists() and train_images.is_dir(),
        train_labels.exists() and train_labels.is_dir(),
        valid_images.exists() and valid_images.is_dir(),
        valid_labels.exists() and valid_labels.is_dir()
    ])
    
    has_splits = val_has_splits or valid_has_splits
    
    print(f"  val_has_splits: {val_has_splits}")
    print(f"  valid_has_splits: {valid_has_splits}")
    print(f"  has_splits: {has_splits}")
    
    if has_splits:
        if valid_has_splits:
            print("✅ Found existing train/valid splits")
        else:
            print("✅ Found existing train/val splits")
        return dataset_folder, True
    else:
        print("⚠️  No existing train/val splits found")
        return dataset_folder, False


def create_train_val_split(images_dir: Path, labels_dir: Path, val_split: float = 0.2, seed: int = 42):
    """
    Create train/validation split from YOLO dataset.
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing labels
        val_split: Fraction of data to use for validation
        seed: Random seed for reproducible splits
    
    Returns:
        tuple: (train_images_dir, train_labels_dir, val_images_dir, val_labels_dir)
    """
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f'*{ext}'))
        image_files.extend(images_dir.glob(f'*{ext.upper()}'))
    
    # Filter to only include images that have corresponding labels
    valid_images = []
    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            valid_images.append(img_path)
    
    if not valid_images:
        raise FileNotFoundError(f"No matching image-label pairs found in {images_dir} and {labels_dir}")
    
    # Split data
    random.seed(seed)
    random.shuffle(valid_images)
    
    split_idx = int(len(valid_images) * (1 - val_split))
    train_images = valid_images[:split_idx]
    val_images = valid_images[split_idx:]
    
    # Create split directories
    train_images_dir = images_dir.parent / "train" / "images"
    train_labels_dir = images_dir.parent / "train" / "labels"
    val_images_dir = images_dir.parent / "val" / "images"
    val_labels_dir = images_dir.parent / "val" / "labels"
    
    # Create directories
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy or link files (using symlinks to save space)
    import shutil
    
    # Train split
    for img_path in train_images:
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        train_img_dest = train_images_dir / img_path.name
        train_label_dest = train_labels_dir / label_path.name
        
        if not train_img_dest.exists():
            shutil.copy2(img_path, train_img_dest)
        if not train_label_dest.exists():
            shutil.copy2(label_path, train_label_dest)
    
    # Val split
    for img_path in val_images:
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        val_img_dest = val_images_dir / img_path.name
        val_label_dest = val_labels_dir / label_path.name
        
        if not val_img_dest.exists():
            shutil.copy2(img_path, val_img_dest)
        if not val_label_dest.exists():
            shutil.copy2(label_path, val_label_dest)
    
    print(f"Created train split: {len(train_images)} samples")
    print(f"  Images: {train_images_dir}")
    print(f"  Labels: {train_labels_dir}")
    print(f"Created val split: {len(val_images)} samples")
    print(f"  Images: {val_images_dir}")
    print(f"  Labels: {val_labels_dir}")
    
    return train_images_dir, train_labels_dir, val_images_dir, val_labels_dir


def parse_args():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Train ChessBoard Keypoint Detection Model (EfficientNet B0)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset and Model Configuration
    default_data_path = os.getenv('DATA_FOLDER_PATH', 'data') + "/chessboard_corners"
    parser.add_argument(
        "--data", 
        type=str, 
        default=default_data_path,
        help="Path to dataset directory (will auto-detect YOLO structure from download_data.py)"
    )
    parser.add_argument(
        "--models-folder", 
        type=str, 
        default="models/chess_board_detection",
        help="Root folder where trained models will be saved"
    )
    parser.add_argument(
        "--name", 
        type=str, 
        default="efficientnet_keypoints",
        help="Name for this training run"
    )
    
    # Training Parameters
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch", 
        type=int, 
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--patience", 
        type=int, 
        default=15,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--val-split", 
        type=float, 
        default=0.2,
        help="Validation split fraction"
    )
    
    # Model Configuration
    parser.add_argument(
        "--loss-type", 
        type=str, 
        default="mse",
        choices=["mse", "smooth_l1", "huber"],
        help="Loss function type"
    )
    parser.add_argument(
        "--dropout", 
        type=float, 
        default=0.2,
        help="Dropout rate"
    )
    parser.add_argument(
        "--freeze-backbone", 
        action="store_true",
        help="Freeze EfficientNet backbone and only train keypoint head layers"
    )
    
    # Data Augmentation
    parser.add_argument(
        "--augmentation", 
        action="store_true",
        help="Enable data augmentation"
    )
    
    # Control Flags
    parser.add_argument(
        "--no-plots", 
        action="store_true",
        help="Disable training plots"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show verbose output"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default="",
        help="Path to checkpoint to resume training from"
    )
    
    # Weights & Biases
    parser.add_argument(
        "--wandb", 
        action="store_true",
        help="Enable Weights & Biases tracking"
    )
    parser.add_argument(
        "--wandb-project", 
        type=str, 
        default="chess-detection",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb-entity", 
        type=str, 
        default="",
        help="Weights & Biases entity (username or team name)"
    )
    parser.add_argument(
        "--wandb-name", 
        type=str, 
        default="",
        help="Weights & Biases run name (defaults to training name)"
    )
    parser.add_argument(
        "--wandb-tags", 
        type=str, 
        nargs="+",
        default=[],
        help="Weights & Biases tags for this run"
    )
    
    return parser.parse_args()


def create_data_augmentation():
    """Create data augmentation transforms."""
    from torchvision import transforms
    
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),
    ])


def setup_wandb_auth():
    """Check and provide guidance for W&B authentication following official quickstart guide."""
    
    wandb_username = os.getenv('WANDB_USERNAME')
    
    # Check if WANDB_API_KEY is set in environment (as per official quickstart)
    wandb_api_key = os.getenv('WANDB_API_KEY')
    
    if not wandb_api_key:
        print("⚠️  W&B Authentication Issue:")
        print("   Following the official W&B quickstart guide:")
        print("   1. Set WANDB_API_KEY environment variable")
        print("   2. Run: wandb login")
        print("   3. Get your API key from: https://wandb.ai/authorize")
        return False, None
    
    try:
        # Use wandb.login() without parameters - it will use WANDB_API_KEY from environment
        # This is the official recommended method from the quickstart guide
        wandb.login()
        print("✅ W&B authenticated using WANDB_API_KEY environment variable")
        
        if wandb_username:
            print(f"✅ Using WANDB_USERNAME from environment: {wandb_username}")
            return True, wandb_username
        else:
            print("ℹ️  No WANDB_USERNAME set - using default entity")
            return True, None
            
    except Exception as e:
        print(f"⚠️  W&B authentication failed: {e}")
        print("   Try running: wandb login --relogin")
        return False, None


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, total_epochs, log_wandb=False):
    """Train for one epoch."""
    model.train_mode()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")
    
    for i, batch in enumerate(progress_bar):
        images = batch['image'].to(device)
        targets = batch['keypoints'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model.get_model()(images)
        loss = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        avg_loss = total_loss / (i + 1)
        
        progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}'})
        
        # Log batch metrics to wandb (every 10 batches to avoid too much logging)
        if log_wandb and i % 10 == 0:
            wandb.log({
                "batch_loss": loss.item(),
                "epoch": epoch,
                "batch": i + epoch * num_batches
            })
    
    return total_loss / num_batches


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval_mode()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            targets = batch['keypoints'].to(device)
            
            predictions = model.get_model()(images)
            loss = criterion(predictions, targets)
            
            total_loss += loss.item()
    
    return total_loss / num_batches


def save_training_plots(train_losses, val_losses, save_dir):
    """Save training plots."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    if len(val_losses) > 1:
        best_epoch = np.argmin(val_losses)
        plt.axvline(x=best_epoch, color='red', linestyle='--', label=f'Best Epoch: {best_epoch+1}')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss Progress')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_plots.png', dpi=150, bbox_inches='tight')
    plt.close()


def log_sample_predictions(model, dataset, device, num_samples=4):
    """Log sample predictions to wandb."""
    model.eval_mode()
    samples_logged = []
    
    # Get random samples
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(device)
            target_keypoints = sample['keypoints'].cpu().numpy()
            
            # Get prediction
            pred_keypoints = model.get_model()(image).cpu().numpy()[0]
            
            # Load original image for visualization
            img_path = sample['image_path']
            original_img = Image.open(img_path).convert('RGB')
            img_array = np.array(original_img)
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Ground truth
            ax1.imshow(img_array)
            ax1.set_title('Ground Truth')
            # Plot keypoints (convert from normalized to pixel coordinates)
            h, w = img_array.shape[:2]
            gt_points = target_keypoints.reshape(4, 2) * [w, h]
            ax1.scatter(gt_points[:, 0], gt_points[:, 1], c='red', s=50, marker='o')
            for i, (x, y) in enumerate(gt_points):
                ax1.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', color='red')
            ax1.axis('off')
            
            # Prediction
            ax2.imshow(img_array)
            ax2.set_title('Prediction')
            pred_points = pred_keypoints.reshape(4, 2) * [w, h]
            ax2.scatter(pred_points[:, 0], pred_points[:, 1], c='blue', s=50, marker='x')
            for i, (x, y) in enumerate(pred_points):
                ax2.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', color='blue')
            ax2.axis('off')
            
            plt.tight_layout()
            
            # Log to wandb
            samples_logged.append(wandb.Image(plt, caption=f"Sample {idx}"))
            plt.close()
    
    if samples_logged:
        wandb.log({"sample_predictions": samples_logged})


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup paths
    data_dir = Path(args.data)
    models_folder = Path(args.models_folder)
    models_folder.mkdir(parents=True, exist_ok=True)
    
    save_dir = models_folder / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate data directory
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print(f"💡 Download data first: python src/chess_board_detection/download_data.py")
        return
    
    # Detect dataset structure
    dataset_folder, has_splits = detect_dataset_structure(data_dir)
    if dataset_folder is None:
        print("❌ No valid dataset found")
        print(f"💡 Expected structure: {data_dir}/[dataset-name]/...")
        print(f"💡 Download data first: python src/chess_board_detection/download_data.py")
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize Weights & Biases
    use_wandb = False
    wandb_entity = None
    if args.wandb:
        # Check authentication before attempting to use W&B
        auth_ok, wandb_entity = setup_wandb_auth()
        use_wandb = auth_ok
    
    if use_wandb:
        try:
            wandb_config = {
                "epochs": args.epochs,
                "batch_size": args.batch,
                "learning_rate": args.lr,
                "patience": args.patience,
                "val_split": args.val_split,
                "loss_type": args.loss_type,
                "dropout": args.dropout,
                "augmentation": args.augmentation,
                "freeze_backbone": args.freeze_backbone,
                "architecture": "EfficientNet-B0",
                "task": "keypoint_detection",
                "dataset": str(data_dir),
            }
            
            wandb_kwargs = {
                "project": args.wandb_project,
                "name": args.wandb_name or args.name,
                "config": wandb_config,
                "tags": args.wandb_tags,
            }
            
            # Use entity from command line args, environment var, or authenticated user
            entity = args.wandb_entity or wandb_entity
            if entity:
                wandb_kwargs["entity"] = entity
                print(f"🏷️  Using W&B entity: {entity}")
            
            wandb.init(**wandb_kwargs)
            print(f"🔗 Weights & Biases: {wandb.run.url}")
            
        except Exception as e:
            print(f"⚠️  Failed to initialize Weights & Biases: {e}")
            print("🔄 Continuing training without W&B logging...")
            use_wandb = False
    
    # Print configuration
    print("🏁 Starting ChessBoard Keypoint Detection Training")
    print("=" * 70)
    print(f"📁 Data directory: {data_dir}")
    print(f"📁 Dataset folder: {dataset_folder}")
    print(f"💾 Models folder: {models_folder}")
    print(f"🏷️  Training name: {args.name}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"📦 Batch size: {args.batch}")
    print(f"📈 Learning rate: {args.lr}")
    print(f"⏰ Patience: {args.patience}")
    print(f"📊 Validation split: {args.val_split}")
    print(f"🔧 Loss type: {args.loss_type}")
    print(f"🎲 Augmentation: {'Enabled' if args.augmentation else 'Disabled'}")
    print(f"📊 Weights & Biases: {'Enabled' if use_wandb else 'Disabled'}")
    print("=" * 70)
    
    # Get or create train/val splits
    if has_splits:
        # Use existing splits from download_data.py
        train_images_dir = dataset_folder / "train" / "images"
        train_labels_dir = dataset_folder / "train" / "labels"
        
        # Check if using 'val' or 'valid' directory
        val_images_dir = dataset_folder / "val" / "images"
        val_labels_dir = dataset_folder / "val" / "labels"
        valid_images_dir = dataset_folder / "valid" / "images"
        valid_labels_dir = dataset_folder / "valid" / "labels"
        
        if valid_images_dir.exists() and valid_labels_dir.exists():
            val_images_dir = valid_images_dir
            val_labels_dir = valid_labels_dir
            print("🔍 Using existing train/valid splits from dataset")
        else:
            print("🔍 Using existing train/val splits from dataset")
    else:
        # Look for all images/labels in the dataset folder
        images_dir = None
        labels_dir = None
        
        # Common patterns for single folder datasets
        possible_structures = [
            (dataset_folder / "images", dataset_folder / "labels"),
            (dataset_folder, dataset_folder),  # Images and labels in same folder
        ]
        
        for img_dir, lbl_dir in possible_structures:
            if img_dir.exists() and lbl_dir.exists():
                # Check if there are actually images and labels
                image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
                label_files = list(lbl_dir.glob("*.txt"))
                if image_files and label_files:
                    images_dir = img_dir
                    labels_dir = lbl_dir
                    break
        
        if images_dir is None or labels_dir is None:
            print("❌ Could not find images and labels directories")
            print(f"💡 Expected structures:")
            print(f"   - {dataset_folder}/images/ and {dataset_folder}/labels/")
            print(f"   - Images and labels in {dataset_folder}/")
            return
        
        print(f"📁 Found images in: {images_dir}")
        print(f"📁 Found labels in: {labels_dir}")
        
        # Create train/val split
        try:
            train_images_dir, train_labels_dir, val_images_dir, val_labels_dir = create_train_val_split(
                images_dir, labels_dir, args.val_split
            )
        except Exception as e:
            print(f"❌ Error creating train/val split: {e}")
            return
    
    # Create datasets
    augmentation_transform = create_data_augmentation() if args.augmentation else None
    
    # Initialize model to get transform
    model = ChessBoardModel(device=device)
    model_transform = model.transform
    
    train_dataset = ChessboardKeypointDataset(
        train_images_dir, train_labels_dir,
        transform=augmentation_transform,
        model_transform=model_transform
    )
    val_dataset = ChessboardKeypointDataset(
        val_images_dir, val_labels_dir,
        transform=None,
        model_transform=model_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch, shuffle=False, num_workers=4
    )
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    
    # Setup training
    optimizer = optim.Adam(model.get_model().parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.patience//2
    )
    
    # Loss function
    if args.loss_type == 'mse':
        criterion = nn.MSELoss()
    elif args.loss_type == 'smooth_l1':
        criterion = nn.SmoothL1Loss()
    elif args.loss_type == 'huber':
        criterion = nn.HuberLoss()
    
    # Training state
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    start_epoch = 0
    
    # Resume training if checkpoint provided
    if args.resume and Path(args.resume).exists():
        print(f"📂 Resuming training from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.get_model().load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        print(f"✅ Resumed from epoch {start_epoch}")
    
    # Training loop
    print("\n🚀 Starting training...")
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.epochs, log_wandb=use_wandb
        )
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Log metrics to wandb
        if use_wandb:
            metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
            }
            wandb.log(metrics)
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"LR: {current_lr:.6f}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            best_model_path = save_dir / 'best.pt'
            model.save_model(
                best_model_path, optimizer, epoch, val_loss
            )
            print(f"✅ New best model saved: {best_model_path}")
            
            # Log sample predictions for best model
            if use_wandb and (epoch + 1) % 10 == 0:  # Every 10 epochs
                log_sample_predictions(model, val_dataset, device)
        else:
            patience_counter += 1
        
        # Save checkpoint
        checkpoint_path = save_dir / 'checkpoint.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.get_model().state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
        }, checkpoint_path)
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"🛑 Early stopping triggered after {patience_counter} epochs without improvement")
            break
    
    # Save final plots
    if not args.no_plots:
        save_training_plots(train_losses, val_losses, save_dir)
        print(f"📊 Training plots saved to {save_dir}/training_plots.png")
        
        # Log plots to wandb
        if use_wandb:
            wandb.log({"training_plots": wandb.Image(str(save_dir / 'training_plots.png'))})
    
    # Log final metrics to wandb
    if use_wandb:
        final_metrics = {
            "final_train_loss": train_losses[-1] if train_losses else 0,
            "final_val_loss": val_losses[-1] if val_losses else 0,
            "best_val_loss": best_val_loss,
            "total_epochs": len(train_losses),
        }
        wandb.log(final_metrics)
        
        # Log model artifact
        best_model_path = save_dir / 'best.pt'
        if best_model_path.exists():
            artifact = wandb.Artifact(
                name=f"{args.name}_model",
                type="model",
                description=f"ChessBoard Keypoint Detection Model - Best Val Loss: {best_val_loss:.4f}"
            )
            artifact.add_file(str(best_model_path))
            wandb.log_artifact(artifact)
            print(f"📦 Model artifact logged to wandb")
    
    # Training complete
    print("=" * 70)
    print("🏁 Training Complete!")
    print(f"✅ Best validation loss: {best_val_loss:.4f}")
    print(f"💾 Best model saved to: {save_dir}/best.pt")
    print(f"📊 Final training loss: {train_losses[-1]:.4f}")
    print(f"📊 Final validation loss: {val_losses[-1]:.4f}")
    
    # Test the trained model
    best_model_path = save_dir / 'best.pt'
    if best_model_path.exists():
        print("\n🧪 Testing trained model...")
        try:
            # Load the best model
            trained_model = ChessBoardModel(model_path=best_model_path, device=device)
            
            # Test on a validation sample
            sample = val_dataset[0]
            test_image_path = sample['image_path']
            
            corners, is_valid = trained_model.predict_keypoints(test_image_path)
            if is_valid:
                print("✅ Model test successful!")
                print("🎯 Predicted corner coordinates:")
                for corner_name, corner in corners.items():
                    if isinstance(corner, dict) and 'x' in corner:
                        print(f"  {corner_name}: ({corner['x']:.1f}, {corner['y']:.1f})")
            else:
                print("⚠️  Model test failed")
                
        except Exception as e:
            print(f"❌ Model test error: {e}")
    
    print("\n🎯 Next Steps:")
    print(f"1. Test: model = ChessBoardModel(model_path=Path('{best_model_path}'))")
    print("2. Predict: corners, valid = model.predict_keypoints('image.jpg')")
    print("3. Visualize: model.plot_eval('image.jpg')")
    print("=" * 70)
    
    # Finish wandb run
    if use_wandb:
        wandb.finish()
        print("🔗 Weights & Biases run completed")


if __name__ == "__main__":
    main() 