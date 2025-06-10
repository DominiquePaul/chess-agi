import os
from pathlib import Path
from src.chess_piece_detection.model import ChessModel


model = ChessModel()

data_dominique_yaml_path = (
    Path(os.environ["DATA_FOLDER_PATH"]) / "chess_pieces_dominique" / "data.yaml"
)
data_roboflow_yaml_path = (
    Path(os.environ["DATA_FOLDER_PATH"]) / "chess_pieces_roboflow" / "data.yaml"
)
single_eval_img_path = Path(os.environ["DATA_FOLDER_PATH"]) / "eval_images/chess_2.jpeg"


results = model.train(
    data_path=data_roboflow_yaml_path,
    epochs=15,
    batch=32,
    lr0=0.0001,
    lrf=0.1,
    imgsz=640,
    plots=True,
    project=Path(os.environ["MODELS_FOLDER"]) / "chess_piece_detection",
    name="training_v1",
    ### Augmentation parameters
    # hsv_h=0.015,        # HSV hue augmentation (fraction)
    # hsv_s=0.7,          # HSV saturation augmentation (fraction)
    # hsv_v=0.4,          # HSV value augmentation (fraction)
    # scale=0.2,          # Image scale (+/- gain)
    # flipud=0.0,         # Vertical flip (probability)
    # fliplr=0.5,         # Horizontal flip (probability)
    ### Exposure augmentations
    # exposure=0.2,       # Exposure adjustment (+/- fraction)
    # gamma=0.5,          # Gamma correction range (0.5-1.5)
    # contrast=0.4,       # Contrast adjustment (+/- fraction)
)

# Print training summary with final validation metrics
model.print_training_summary(results)

# Evaluate model on test data
print("\nEvaluating model on Roboflow dataset...")
model.evaluate(data_roboflow_yaml_path)

print("\nEvaluating model on OUR dataset...")
model.evaluate(data_dominique_yaml_path)


results_v2 = model.train(
    data_path=data_dominique_yaml_path,
    epochs=15,
    batch=32,
    lr0=0.0001,
    lrf=0.1,
    imgsz=640,
    plots=True,
)

# Print training summary for second training run
model.print_training_summary(results_v2)

# Evaluate model on both datasets after second training
print("\nEvaluating model on Dominique dataset...")
model.evaluate(data_dominique_yaml_path)

print("\nEvaluating model on Roboflow dataset (after additional training)...")
model.evaluate(data_roboflow_yaml_path)



# Push models to Hugging Face Hub (model_path is auto-detected!)
model.push_to_huggingface(
    repo_id="chess-piece-detector-v1",
    commit_message="Upload chess piece detection model v1 (trained on Roboflow data)",
    private=False
)

model.push_to_huggingface(
    repo_id="chess-piece-detector-v2", 
    commit_message="Upload chess piece detection model v2 (trained on Roboflow + Dominique data)",
    private=False
)











