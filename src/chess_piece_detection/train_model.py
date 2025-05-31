import os
import shutil
import torch
import matplotlib

matplotlib.use("Agg")  # Change to non-interactive backend
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle  # Import Rectangle from patches
import ultralytics
from ultralytics import YOLO

# Check if MPS is available
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Download latest version
dominique_chess_piece_path = (
    Path(os.environ["DATA_FOLDER_PATH"]) / "chess_pieces_dominique"
)
roboflow_chess_piece_path = (
    Path(os.environ["DATA_FOLDER_PATH"]) / "chess_pieces_roboflow"
)
single_eval_img_path = Path(os.environ["DATA_FOLDER_PATH"]) / "eval_images/chess_2.jpeg"

MODELS_FOLDER = Path("models")
MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
CHESS_PIECE_MODEL_NAME = "yolo_chess_piece_detector"

ultralytics.checks()


def get_or_train_model(replace_existing=False):
    if os.path.exists(MODELS_FOLDER / CHESS_PIECE_MODEL_NAME) and not replace_existing:
        print(f"Model already exists in {MODELS_FOLDER / CHESS_PIECE_MODEL_NAME}")
        model = YOLO(
            MODELS_FOLDER / CHESS_PIECE_MODEL_NAME / "training_v1/weights/best.pt"
        )
    else:
        if os.path.exists(MODELS_FOLDER / CHESS_PIECE_MODEL_NAME) and replace_existing:
            print(
                f"Replacing existing model in {MODELS_FOLDER / CHESS_PIECE_MODEL_NAME}"
            )
            shutil.rmtree(MODELS_FOLDER / CHESS_PIECE_MODEL_NAME)
        # Initialize model with MPS device
        model = YOLO(MODELS_FOLDER / "yolov8s.pt")
        model.to(device)
        yaml_path = os.path.abspath(
            os.path.join(roboflow_chess_piece_path, "data.yaml")
        )
        results = model.train(
            data=yaml_path,
            epochs=15,
            batch=32,
            lr0=0.0001,
            lrf=0.1,
            imgsz=640,
            plots=True,
            device=device,
            project=MODELS_FOLDER / CHESS_PIECE_MODEL_NAME,
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
    return model


def eval_image(model, img_path, ax=None, conf=0.1):
    # Get prediction results
    results = model.predict(img_path, conf=conf, save=True)[0]

    # Print all possible class names (targets) that the model can detect
    print("Possible detection targets:")
    print('", "'.join(results.names.values()))

    # Use provided axes or create a new one
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))
    
    # Display original image
    ax.imshow(results.orig_img)

    # Draw bounding boxes and labels
    for box in results.boxes:
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        # Get class name and confidence
        cls = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        label = f"{results.names[cls]} {conf:.2f}"

        # Draw rectangle
        rect = Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=2
        )
        ax.add_patch(rect)

        # Add label
        ax.text(
            x1, y1 - 5, label, color="red", bbox=dict(facecolor="white", alpha=0.7)
        )

    ax.axis("off")  # Hide axes
    return ax


# Continue training on my data
updated_model = get_or_train_model()
if os.path.exists(MODELS_FOLDER / CHESS_PIECE_MODEL_NAME / "training_v2"):
    print(f"Training v2 already exists in {MODELS_FOLDER / CHESS_PIECE_MODEL_NAME / 'training_v2'}")
    updated_model = YOLO(MODELS_FOLDER / CHESS_PIECE_MODEL_NAME / "training_v2/weights/best.pt")
    # shutil.rmtree(MODELS_FOLDER / CHESS_PIECE_MODEL_NAME / "training_v2")
else:
    updated_model.train(
        data=(dominique_chess_piece_path / "data.yaml").resolve(),
        epochs=15,
        batch=32,
        lr0=0.0001,
        lrf=0.1,
        imgsz=640,
        plots=True,
        device=device,
        project=MODELS_FOLDER / CHESS_PIECE_MODEL_NAME,
        name="training_v2",
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

# Create a figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Plot default model predictions
default_model = get_or_train_model()
axes[0].set_title("Model trained on online chess piece data")
eval_image(default_model, single_eval_img_path, axes[0], conf=0.2)

# Plot updated model predictions
axes[1].set_title("Model trained on online chess piece data + my data")
eval_image(updated_model, single_eval_img_path, axes[1], conf=0.3)

plt.tight_layout()

# Create output directory if it doesn't exist
os.makedirs('output/plots', exist_ok=True)

# Save figure
plt.savefig('output/plots/chess_pieces_annotated.jpg', dpi=200)
plt.close(fig)

# ### Video
# example_video_path = "/kaggle/input/chess-pieces-detection-image-dataset/Chess_pieces/Chess_video_example.mp4"
# video_output = model.predict(source=example_video_path, conf=0.6, save=True)
# # !ffmpeg -y -loglevel panic -i /kaggle/working/runs/detect/train2/Chess_video_example.avi Chess_video_example.mp4
# Video("Chess_video_example.mp4", embed=True, width=960)
