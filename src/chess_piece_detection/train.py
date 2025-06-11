import os
from pathlib import Path
from src.chess_piece_detection.model import ChessModel
from dotenv import load_dotenv

load_dotenv()


model = ChessModel()

# Use the merged dataset that combines both Dominique and Roboflow data
data_merged_yaml_path = (
    Path(os.environ["DATA_FOLDER_PATH"]) / "chess_pieces_merged" / "data.yaml"
)

# Keep individual dataset paths for evaluation comparison
data_dominique_yaml_path = (
    Path(os.environ["DATA_FOLDER_PATH"]) / "chess_pieces_dominique" / "data.yaml"
)
data_roboflow_yaml_path = (
    Path(os.environ["DATA_FOLDER_PATH"]) / "chess_pieces_roboflow" / "data.yaml"
)
single_eval_img_path = Path(os.environ["DATA_FOLDER_PATH"]) / "eval_images/chess_2.jpeg"

print("🚀 Training model on merged dataset (Dominique + Roboflow)...")
print(f"📊 Dataset path: {data_merged_yaml_path}")

# Train on the merged dataset with improved hyperparameters
results = model.train(
    data_path=data_merged_yaml_path,
    epochs=100,  # Increased from 20 - your losses were still decreasing
    batch=32,    # Reduced from 64 to allow higher learning rate and better gradients
    lr0=0.001,   # Increased from 0.0005 for faster initial learning
    lrf=0.01,    # Reduced from 0.1 for gentler learning rate decay
    imgsz=640,
    plots=True,
    project=Path("models/chess_piece_detection"),
    name="training_merged_v2",
    save_period=10,  # Save checkpoint every 10 epochs
    patience=15,     # Early stopping if no improvement for 15 epochs
    
    # Data Augmentation - Enable these to improve generalization
    hsv_h=0.015,     # HSV hue augmentation (fraction)
    hsv_s=0.7,       # HSV saturation augmentation (fraction) 
    hsv_v=0.4,       # HSV value augmentation (fraction)
    degrees=10.0,    # Image rotation (+/- degrees)
    translate=0.1,   # Image translation (+/- fraction)
    scale=0.2,       # Image scale (+/- gain)
    shear=2.0,       # Image shear (+/- degrees)
    perspective=0.0001,  # Image perspective (+/- fraction), range 0-0.001
    flipud=0.0,      # Vertical flip (probability)
    fliplr=0.5,      # Horizontal flip (probability)
    mosaic=1.0,      # Mosaic augmentation (probability)
    mixup=0.1,       # MixUp augmentation (probability)
    copy_paste=0.1,  # Copy-paste augmentation (probability)
    
    # Advanced training parameters
    warmup_epochs=3,     # Warmup epochs
    warmup_momentum=0.8, # Warmup initial momentum
    warmup_bias_lr=0.1,  # Warmup initial bias learning rate
    box=7.5,             # Box loss gain
    cls=0.5,             # Classification loss gain
    dfl=1.5,             # Distribution focal loss gain
    
    # Optimizer settings
    optimizer='AdamW',   # Use AdamW optimizer (often better than SGD)
    weight_decay=0.0005, # Weight decay for regularization
)

# Print training summary with final validation metrics
model.print_training_summary(results)

# Evaluate model on the merged dataset
print("\n📊 Evaluating model on merged dataset...")
model.evaluate(data_merged_yaml_path)

# Optional: Evaluate on individual datasets to see performance breakdown
# (only if those datasets are available)
if data_dominique_yaml_path.exists():
    print("\n📊 Evaluating model on Dominique dataset...")
    model.evaluate(data_dominique_yaml_path)

if data_roboflow_yaml_path.exists():
    print("\n📊 Evaluating model on Roboflow dataset...")
    model.evaluate(data_roboflow_yaml_path)

# Push model to Hugging Face Hub
model.push_to_huggingface(
    repo_id="chess-piece-detector-merged-v2",
    commit_message="Upload improved chess piece detection model with enhanced training (100 epochs, data augmentation, AdamW optimizer)",
    private=False
)











