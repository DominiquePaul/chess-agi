import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics.engine.results import Results

from src.base_model import BaseYOLOModel
from src.datatypes import ChessPieceBoundingBox


class ChessModel(BaseYOLOModel):
    """Model for detecting chess pieces using YOLO."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        device=None,
        pretrained_checkpoint: str = "yolo11s.pt",
    ):
        """
        Initialize ChessModel with YOLO11s as default pretrained checkpoint.

        Args:
            model_path: Path to existing trained model or HuggingFace model name (optional)
            device: Device to run model on (auto-detected if None)
            pretrained_checkpoint: Pretrained YOLO checkpoint to use (default: yolo11s.pt)
        """
        super().__init__(model_path, device, pretrained_checkpoint)

        # Update initialization message if loaded from HuggingFace
        print(
            f"🧠 Chess Piece Detection Model initialized on {self.device} - loaded from {'cache' if self.loaded_from_cache else 'HF Hub'}"
        )

        # Store class mapping for chess pieces
        self.class_dict = dict(
            enumerate(
                [
                    "black-bishop",
                    "black-king",
                    "black-knight",
                    "black-pawn",
                    "black-queen",
                    "black-rook",
                    "white-bishop",
                    "white-king",
                    "white-knight",
                    "white-pawn",
                    "white-queen",
                    "white-rook",
                ]
            )
        )

    def detect_pieces_in_squares(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.5,
    ) -> list[ChessPieceBoundingBox]:
        """
        Detect chess pieces and map them to chess board squares.

        Args:
            image: Input image (path or numpy array)
            square_coordinates: Dictionary mapping square numbers to coordinates
            conf_threshold: Confidence threshold for detections

        Returns:
            Tuple of (Results, visualization_image)
            Results: Results object from YOLO
            visualization_image: Visualization image of the detections
        """
        # Run YOLO detection
        results = self.predict(image, conf=conf_threshold)

        return self._convert_results_to_chess_pieces(results)

    def _convert_results_to_chess_pieces(self, results: Results) -> list[ChessPieceBoundingBox]:
        """Convert YOLO results to chess piece detections."""
        chess_pieces = []
        if results.boxes is not None and results.boxes.xyxy is not None:
            boxes = results.boxes.xyxy
            classes = results.boxes.cls
            confidences = results.boxes.conf

            # Convert tensors to numpy arrays
            try:
                boxes = boxes.cpu().numpy() if hasattr(boxes, "cpu") else boxes
                classes = classes.cpu().numpy() if hasattr(classes, "cpu") else classes
                confidences = confidences.cpu().numpy() if hasattr(confidences, "cpu") else confidences
            except AttributeError:
                # Already numpy arrays
                boxes = np.array(boxes) if not isinstance(boxes, np.ndarray) else boxes
                classes = np.array(classes) if not isinstance(classes, np.ndarray) else classes
                confidences = np.array(confidences) if not isinstance(confidences, np.ndarray) else confidences

            for box, cls, confidence in zip(boxes, classes, confidences, strict=False):
                chess_pieces.append(
                    ChessPieceBoundingBox(
                        piece_class=int(cls),
                        piece_name=self.class_dict[int(cls)],
                        confidence=float(confidence),
                        x_left=float(box[0]),
                        y_top=float(box[1]),
                        x_right=float(box[2]),
                        y_bottom=float(box[3]),
                        assigned_square=None,
                    )
                )
        return chess_pieces

    def _create_detection_visualization(self, results) -> np.ndarray:
        """Create visualization image with bounding boxes."""
        # Use YOLO's built-in visualization
        vis_image = results.plot(line_width=2, font_size=12)

        # Convert BGR to RGB for consistency
        if len(vis_image.shape) == 3 and vis_image.shape[2] == 3:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

        return vis_image

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
                rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=2)
                ax.add_patch(rect)

                # Add label
                ax.text(
                    x1,
                    y1 - 5,
                    label,
                    color="red",
                    bbox={"facecolor": "white", "alpha": 0.7},
                )

        ax.axis("off")

        if show_inline:
            self._display_plot_inline(ax=ax)

        return ax

    def get_piece_name(self, class_id: int) -> str:
        """Get human-readable piece name from class ID."""
        return self.class_dict.get(class_id, f"Unknown({class_id})")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import ultralytics
    from matplotlib.patches import Rectangle

    # Create models folder
    MODELS_FOLDER = Path("models")
    MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
    CHESS_PIECE_MODEL_NAME = "yolo_chess_piece_detector"
    ultralytics.checks()

    single_eval_img_path = Path(os.environ["DATA_FOLDER_PATH"]) / "eval_images/chess_2.jpeg"

    # Create new model with YOLO11s pretrained checkpoint for training
    print("=== Creating new model with YOLO11s checkpoint ===")
    new_model = ChessModel(pretrained_checkpoint="yolo11s.pt")

    # Example: Train the model (uncomment when you have your dataset ready)
    # data_yaml_path = Path("path/to/your/chess_dataset.yaml")
    # training_results = new_model.train(
    #     data_path=data_yaml_path,
    #     epochs=100,
    #     batch=16,
    #     imgsz=640,
    #     project=MODELS_FOLDER / CHESS_PIECE_MODEL_NAME,
    #     name="training_v3_yolo11s"
    # )

    # Load old model for comparison (if it exists)
    model_path = MODELS_FOLDER / CHESS_PIECE_MODEL_NAME / "training_v1/weights/best.pt"
    if model_path.exists():
        print("=== Loading old model for comparison ===")
        model = ChessModel(model_path)
        plot = model.plot_eval(single_eval_img_path)

    # Load updated model (if it exists)
    updated_model_path = MODELS_FOLDER / CHESS_PIECE_MODEL_NAME / "training_v2/weights/best.pt"
    if updated_model_path.exists():
        print("=== Loading updated model for comparison ===")
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
    else:
        print("=== No existing trained models found ===")
        print("Run model training first to compare results")

    # Test the new YOLO11s model on sample image
    print("=== Testing YOLO11s model on sample image ===")
    new_model.plot_eval(single_eval_img_path, conf=0.1)

    # Example of HuggingFace loading (uncomment when model is available)
    # print("=== Testing HuggingFace model loading ===")
    # try:
    #     hf_model = ChessModel("dopaul/chess_piece_detector")
    #     hf_model.plot_eval(single_eval_img_path, conf=0.5)
    # except Exception as e:
    #     print(f"HuggingFace model loading failed: {e}")

    # ### Video
    # example_video_path = "/kaggle/input/chess-pieces-detection-image-dataset/Chess_pieces/Chess_video_example.mp4"
    # video_output = model.predict(source=example_video_path, conf=0.6, save=True)
    # # !ffmpeg -y -loglevel panic -i /kaggle/working/runs/detect/train2/Chess_video_example.avi Chess_video_example.mp4
    # Video("Chess_video_example.mp4", embed=True, width=960)
