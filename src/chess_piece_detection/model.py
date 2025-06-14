import os
import matplotlib
import cv2

matplotlib.use("Agg")  # Change to non-interactive backend
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle  # Import Rectangle from patches
import ultralytics
from ultralytics import YOLO
from src.base_model import BaseYOLOModel
import numpy as np
from typing import Union, List, Tuple, Optional
import tempfile

class ChessModel(BaseYOLOModel):
    """Model for detecting chess pieces using YOLO."""

    def __init__(self, model_path: Path | str | None = None, device=None, pretrained_checkpoint: str = "yolo11s.pt"):
        """
        Initialize ChessModel with YOLO11s as default pretrained checkpoint.
        
        Args:
            model_path: Path to existing trained model or HuggingFace model name (optional)
            device: Device to run model on (auto-detected if None)
            pretrained_checkpoint: Pretrained YOLO checkpoint to use (default: yolo11s.pt)
        """
        # Handle HuggingFace Hub models
        if isinstance(model_path, str) and "/" in model_path and not Path(model_path).exists():
            # This looks like a HuggingFace model name
            model_path = self._load_from_huggingface(model_path)
        
        super().__init__(model_path, device, pretrained_checkpoint)
        
        # Store class mapping for chess pieces
        self.class_dict = {
            0: "black-bishop",
            1: "black-king", 
            2: "black-knight",
            3: "black-pawn",
            4: "black-queen",
            5: "black-rook",
            6: "white-bishop",
            7: "white-king",
            8: "white-knight",
            9: "white-pawn",
            10: "white-queen",
            11: "white-rook",
        }
    
    def _load_from_huggingface(self, model_name: str) -> Path:
        """
        Load model from HuggingFace Hub.
        
        Args:
            model_name: HuggingFace model name (e.g., "dopaul/chess_piece_detector")
            
        Returns:
            Path to downloaded model
        """
        try:
            from huggingface_hub import hf_hub_download
            print(f"🤗 Loading model from HuggingFace Hub: {model_name}")
            
            # Download the model file
            # Models on HuggingFace Hub are uploaded as "model.pt"
            model_file = hf_hub_download(
                repo_id=model_name,
                filename="model.pt",
                cache_dir="models/huggingface_cache"
            )
            
            print(f"✅ Model downloaded successfully: {model_file}")
            return Path(model_file)
            
        except ImportError:
            raise ImportError(
                "huggingface_hub is required to load models from HuggingFace Hub. "
                "Install it with: pip install huggingface_hub"
            )
        except Exception as e:
            raise Exception(f"Failed to load model from HuggingFace Hub: {e}")

    def detect_pieces_in_squares(self, 
                                image: Union[str, Path, np.ndarray], 
                                square_coordinates: dict,
                                conf_threshold: float = 0.5) -> Tuple[List, np.ndarray]:
        """
        Detect chess pieces and map them to chess board squares.
        
        Args:
            image: Input image (path or numpy array)
            square_coordinates: Dictionary mapping square numbers to coordinates
            conf_threshold: Confidence threshold for detections
            
        Returns:
            Tuple of (piece_mappings, visualization_image)
            piece_mappings: List of [square_number, piece_class_id] pairs
        """
        # Handle different input types
        if isinstance(image, np.ndarray):
            # Save numpy array to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            cv2.imwrite(temp_file.name, image)
            image_path = temp_file.name
            temp_file.close()
            cleanup_temp = True
        else:
            image_path = str(image)
            cleanup_temp = False
        
        try:
            # Run YOLO detection
            results = self.predict(image_path, conf=conf_threshold)
            
            # Create visualization
            detection_image = self._create_detection_visualization(results)
            
            # Map detections to squares
            piece_mappings = self._map_detections_to_squares(results, square_coordinates)
            
            print(f"🎯 Detected {len(piece_mappings)} chess pieces")
            for square_num, piece_class in piece_mappings:
                piece_name = self.class_dict.get(piece_class, f"Unknown({piece_class})")
                print(f"   Square {square_num:2d}: {piece_name}")
            
            return piece_mappings, detection_image
            
        finally:
            # Clean up temporary file if created
            if cleanup_temp:
                try:
                    os.unlink(image_path)
                except:
                    pass
    
    def _create_detection_visualization(self, results) -> np.ndarray:
        """Create visualization image with bounding boxes."""
        # Use YOLO's built-in visualization
        vis_image = results.plot(line_width=2, font_size=12)
        
        # Convert BGR to RGB for consistency
        if len(vis_image.shape) == 3 and vis_image.shape[2] == 3:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        
        return vis_image
    
    def _map_detections_to_squares(self, results, square_coordinates: dict) -> List:
        """Map YOLO detections to chess board squares, handling duplicates by keeping highest confidence."""
        piece_mappings = []
        square_to_piece = {}  # Track best piece for each square
        
        if results.boxes is not None and results.boxes.xyxy is not None:
            boxes = results.boxes.xyxy
            classes = results.boxes.cls
            confidences = results.boxes.conf
            
            # Convert tensors to numpy arrays
            try:
                boxes = boxes.cpu().numpy()
                classes = classes.cpu().numpy()
                confidences = confidences.cpu().numpy()
            except AttributeError:
                # Already numpy arrays
                boxes = np.array(boxes) if not isinstance(boxes, np.ndarray) else boxes
                classes = np.array(classes) if not isinstance(classes, np.ndarray) else classes
                confidences = np.array(confidences) if not isinstance(confidences, np.ndarray) else confidences
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                
                # Find center of bounding box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Find which square this center belongs to
                for square_num, coords in square_coordinates.items():
                    if len(coords) >= 5:
                        # coords format: [center, bottom_right, top_right, top_left, bottom_left]
                        x_coords = [point[0] for point in coords[1:]]  # Skip center, use corners
                        y_coords = [point[1] for point in coords[1:]]
                        
                        if (min(x_coords) <= center_x <= max(x_coords) and 
                            min(y_coords) <= center_y <= max(y_coords)):
                            
                            piece_class = int(classes[i])
                            confidence = float(confidences[i])
                            
                            # Check if this square already has a piece detected
                            if square_num not in square_to_piece or confidence > square_to_piece[square_num]['conf']:
                                square_to_piece[square_num] = {'class': piece_class, 'conf': confidence}
                            break
        
        # Convert to list format (keeping only the piece class, not confidence)
        for square_num, piece_info in square_to_piece.items():
            piece_mappings.append([square_num, piece_info['class']])
        
        return piece_mappings

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

    def get_piece_name(self, class_id: int) -> str:
        """Get human-readable piece name from class ID."""
        return self.class_dict.get(class_id, f"Unknown({class_id})")

    def create_chess_position(self, piece_mappings: List) -> Optional[object]:
        """
        Create a chess.Board object from piece mappings.
        
        Args:
            piece_mappings: List of [square_number, piece_class_id] pairs
            
        Returns:
            chess.Board object or None if chess library not available
        """
        try:
            import chess
            
            # Create empty board
            board = chess.Board(None)
            
            # Piece mapping from class names to chess piece types
            piece_type_mapping = {
                'white-pawn': (chess.PAWN, chess.WHITE),
                'black-pawn': (chess.PAWN, chess.BLACK),
                'white-knight': (chess.KNIGHT, chess.WHITE),
                'black-knight': (chess.KNIGHT, chess.BLACK),
                'white-bishop': (chess.BISHOP, chess.WHITE),
                'black-bishop': (chess.BISHOP, chess.BLACK),
                'white-rook': (chess.ROOK, chess.WHITE),
                'black-rook': (chess.ROOK, chess.BLACK),
                'white-queen': (chess.QUEEN, chess.WHITE),
                'black-queen': (chess.QUEEN, chess.BLACK),
                'white-king': (chess.KING, chess.WHITE),
                'black-king': (chess.KING, chess.BLACK),
            }
            
            # Place pieces on board
            for square_num, piece_class_id in piece_mappings:
                piece_name = self.get_piece_name(piece_class_id)
                
                if piece_name in piece_type_mapping:
                    piece_type, color = piece_type_mapping[piece_name]
                    
                    # Convert square number to chess square
                    # Square numbers are 1-64, starting from bottom-left
                    # Convert to file (a-h) and rank (1-8)
                    file = (square_num - 1) % 8  # 0-7 for a-h
                    rank = 7 - ((square_num - 1) // 8)  # 0-7 for ranks 1-8 (inverted)
                    
                    chess_square = chess.square(file, rank)
                    board.set_piece_at(chess_square, chess.Piece(piece_type, color))
            
            return board
            
        except ImportError:
            print("⚠️  chess library not available - cannot create chess position")
            return None


if __name__ == "__main__":
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
    #     name="training_v3_yolov8s"
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
