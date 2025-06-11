import os
import matplotlib
import cv2

matplotlib.use("Agg")  # Change to non-interactive backend
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle  # Import Rectangle from patches
import ultralytics
from src.base_model import BaseYOLOModel

class ChessModel(BaseYOLOModel):
    """Model for detecting chess pieces using YOLO."""

    def plot_eval(self, img_path, ax=None, conf=0.1):
        """
        Plot evaluation results with bounding boxes for chess pieces.
        
        Args:
            img_path: Path to the image to evaluate
            ax: Matplotlib axes to plot on (creates new if None)
            conf: Confidence threshold for predictions
            
        Returns:
            ax: The matplotlib axes used for plotting
        """
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

        # Convert BGR to RGB for proper matplotlib display
        image_rgb = cv2.cvtColor(results.orig_img, cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)

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

        ax.axis("off")
        
        if show_inline:
            self._display_plot_inline(ax=ax)
            
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
