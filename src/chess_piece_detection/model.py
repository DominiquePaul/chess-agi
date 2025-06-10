import os
import torch
import matplotlib
import tempfile
import io

matplotlib.use("Agg")  # Change to non-interactive backend
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle  # Import Rectangle from patches
import ultralytics
from ultralytics import YOLO
from PIL import Image

from src.utils import is_notebook, get_best_torch_device

class ChessModel:
    def __init__(self, model_path: Path | None = None, device: torch.device | None = None):
        self.model_path = model_path
        self.device = device or get_best_torch_device()
        if model_path and model_path.exists():
            self.load_model()
        else:
            self.model = YOLO()
        self.model.to(self.device)

    def save_model(self, path: Path | None = None):
        save_path = path or self.model_path
        self.model.save(save_path)
        return save_path

    def load_model(self):
        if self.model_path.exists():
            self.model = YOLO(self.model_path)
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")

    def train(self, data_path: Path, epochs: int = 15, batch: int = 32, lr0: float = 0.0001, lrf: float = 0.1, imgsz: int = 640, plots: bool = True, project: Path | None = None, name: str | None = None):
        results = self.model.train(
            data=data_path,
            epochs=epochs,
            batch=batch,
            lr0=lr0,
            lrf=lrf,
            imgsz=imgsz,
            plots=plots,
            device=self.device,
            project=project,
            name=name,
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
        return results

    def predict(self, img_path, conf=0.1, save=False):
        results = self.model.predict(img_path, conf=conf, save=save)[0]
        return results

    def plot_eval(self, img_path, ax=None, conf=0.1):
        results = self.predict(img_path, conf=conf)
        # Print all possible class names (targets) that the model can detect
        print("Possible detection targets:")
        print('", "'.join(results.names.values()))

        # Use provided axes or create a new one
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))
            show_inline = True
        else:
            show_inline = False

        # Display original image
        ax.imshow(results.orig_img)

        # Draw bounding boxes and labels
        if results.boxes is not None:
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
        
        if show_inline:
            if is_notebook():
                from IPython.display import display, Image as IPyImage
                # Save the plot to a bytes buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                plt.close()
                buf.seek(0)
                # Display in notebook
                display(IPyImage(data=buf.getvalue()))
                buf.close()
            else:
                # Fallback to PIL display for non-notebook environments
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    plt.savefig(tmp.name, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img = Image.open(tmp.name)
                    img.show()
                    # Clean up the temporary file
                    os.unlink(tmp.name)
            
        return ax

if __name__ == "__main__":
    # Create models folder
    MODELS_FOLDER = Path("models")
    MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
    CHESS_PIECE_MODEL_NAME = "yolo_chess_piece_detector"
    ultralytics.checks()

    single_eval_img_path = Path(os.environ["DATA_FOLDER_PATH"]) / "eval_images/chess_2.jpeg"
    
    # Load old model
    model_path = MODELS_FOLDER / CHESS_PIECE_MODEL_NAME / "training_v1/weights/best.pt"
    model = ChessModel(model_path)
    plot = model.plot_eval(single_eval_img_path)

    # Load updated model
    updated_model_path = MODELS_FOLDER / CHESS_PIECE_MODEL_NAME / "training_v2/weights/best.pt"
    updated_model = ChessModel(updated_model_path)
    updated_model.plot_eval(single_eval_img_path, conf=0.5)
    updated_model.predict(single_eval_img_path, conf=0.5)
    
    ### Create figure with old and new model predictions
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    # Plot original model predictions
    axes[0].set_title("Model trained on online chess piece data")
    model.plot_eval(single_eval_img_path, ax=axes[0], conf=0.5)
    # Plot updated model predictions  
    axes[1].set_title("Model trained on online chess piece data + my data")
    updated_model.plot_eval(single_eval_img_path, ax=axes[1], conf=0.5)
    plt.tight_layout()
    # Create output directory if it doesn't exist
    os.makedirs("output/plots", exist_ok=True)
    # Save figure
    plt.savefig("output/plots/chess_pieces_annotated.jpg", dpi=200)
    plt.close(fig)

    # ### Video
    # example_video_path = "/kaggle/input/chess-pieces-detection-image-dataset/Chess_pieces/Chess_video_example.mp4"
    # video_output = model.predict(source=example_video_path, conf=0.6, save=True)
    # # !ffmpeg -y -loglevel panic -i /kaggle/working/runs/detect/train2/Chess_video_example.avi Chess_video_example.mp4
    # Video("Chess_video_example.mp4", embed=True, width=960)
