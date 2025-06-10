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

from src.utils import is_notebook, get_best_torch_device, push_model_to_huggingface

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:
    hf_hub_download = None
    list_repo_files = None


class ChessModel:
    def __init__(self, model_path: Path | None = None, device: torch.device | None = None):
        self.model_path = model_path
        self.device = device or get_best_torch_device()
        if model_path and model_path.exists():
            self.load_model()
        else:
            self.model = YOLO()
        self.model.to(self.device)

    @classmethod
    def from_huggingface(cls, 
                        repo_id: str, 
                        filename: str = "best.pt",
                        device: torch.device | None = None,
                        cache_dir: Path | None = None,
                        token: str | None = None):
        """
        Load a ChessModel from a Hugging Face repository.
        
        Args:
            repo_id: Repository ID on Hugging Face Hub (e.g., "username/chess-piece-detector")
            filename: Name of the model file to download (default: "best.pt")
            device: Device to load the model on (auto-detected if None)
            cache_dir: Directory to cache downloaded files (uses HF default if None)
            token: Hugging Face authentication token (uses env var if not provided)
            
        Returns:
            ChessModel: Instantiated model loaded from HF Hub
            
        Raises:
            ImportError: If huggingface_hub is not installed
            FileNotFoundError: If the model file is not found in the repository
            Exception: If download or model loading fails
            
        Example:
            >>> model = ChessModel.from_huggingface("username/chess-detector")
            >>> model = ChessModel.from_huggingface("username/chess-detector", filename="last.pt")
        """
        if hf_hub_download is None:
            raise ImportError(
                "huggingface_hub is required to load models from HF Hub. "
                "Please install it with: uv add huggingface_hub"
            )
        
        print(f"📥 Downloading model from Hugging Face: {repo_id}")
        
        try:
            # Download the model file from HF Hub
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                token=token
            )
            
            print(f"✓ Downloaded model to: {model_path}")
            
            # Create and return ChessModel instance
            return cls(model_path=Path(model_path), device=device)
            
        except Exception as e:
            print(f"❌ Failed to download model from {repo_id}: {e}")
            raise

    @classmethod
    def list_huggingface_files(cls, repo_id: str, token: str | None = None):
        """
        List available model files in a Hugging Face repository.
        
        Args:
            repo_id: Repository ID on Hugging Face Hub
            token: Hugging Face authentication token (uses env var if not provided)
            
        Returns:
            list: List of model files (.pt) available in the repository
            
        Example:
            >>> files = ChessModel.list_huggingface_files("username/chess-detector")
            >>> print(files)  # ['best.pt', 'last.pt', 'epoch_10.pt']
        """
        if list_repo_files is None:
            raise ImportError(
                "huggingface_hub is required to list repository files. "
                "Please install it with: uv add huggingface_hub"
            )
        
        try:
            # Get all files in the repository
            all_files = list_repo_files(repo_id=repo_id, token=token)
            
            # Filter for model files (.pt)
            model_files = [f for f in all_files if f.endswith('.pt')]
            
            print(f"📋 Available model files in {repo_id}:")
            for file in model_files:
                print(f"  - {file}")
            
            return model_files
            
        except Exception as e:
            print(f"❌ Failed to list files from {repo_id}: {e}")
            raise

    def save_model(self, path: Path | None = None):
        save_path = path or self.model_path
        if save_path is None:
            raise ValueError("No path provided and no model_path set")
        self.model.save(save_path)
        return save_path

    def load_model(self):
        if self.model_path is None:
            raise ValueError("No model_path provided")
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

    def evaluate(self, data_path: Path, conf: float = 0.25, iou: float = 0.5):
        """
        Evaluate the model on test data and print accuracy metrics.
        
        Args:
            data_path: Path to the dataset yaml file
            conf: Confidence threshold for predictions
            iou: IoU threshold for NMS
        """
        print("\n" + "="*50)
        print("EVALUATING MODEL PERFORMANCE")
        print("="*50)
        
        # Run validation
        results = self.model.val(
            data=data_path,
            conf=conf,
            iou=iou,
            device=self.device
        )
        
        # Print key metrics
        if hasattr(results, 'box'):
            metrics = results.box
            print(f"Overall mAP@0.5: {metrics.map50:.4f}")
            print(f"Overall mAP@0.5:0.95: {metrics.map:.4f}")
            print(f"Precision: {metrics.p.mean():.4f}")
            print(f"Recall: {metrics.r.mean():.4f}")
            
            # Print per-class metrics if available
            if hasattr(metrics, 'ap_class_index') and len(metrics.ap_class_index) > 0:
                print("\nPer-class metrics:")
                class_names = self.model.names
                for i, class_idx in enumerate(metrics.ap_class_index):
                    class_name = class_names[int(class_idx)]
                    print(f"  {class_name}:")
                    print(f"    mAP@0.5: {metrics.ap50[i]:.4f}")
                    print(f"    mAP@0.5:0.95: {metrics.ap[i]:.4f}")
                    print(f"    Precision: {metrics.p[i]:.4f}")
                    print(f"    Recall: {metrics.r[i]:.4f}")
        
        print("="*50)
        return results

    def push_to_huggingface(self, 
                           repo_id: str, 
                           model_path: Path | None = None,
                           commit_message: str | None = None,
                           token: str | None = None,
                           private: bool = False,
                           create_model_card: bool = True):
        """
        Push the trained model to Hugging Face Hub.
        
        Args:
            repo_id: Repository ID on Hugging Face Hub (e.g., "username/chess-piece-detector")
            model_path: Path to the model to upload (defaults to self.model_path)
            commit_message: Commit message for the upload
            token: Hugging Face authentication token (if not already logged in)
            private: Whether to create a private repository
            create_model_card: Whether to create a basic model card
        
        Returns:
            str: URL of the uploaded model
        """
        return push_model_to_huggingface(
            model=self,
            repo_id=repo_id,
            model_path=model_path,
            commit_message=commit_message,
            token=token,
            private=private,
            create_model_card=create_model_card
        )

    def print_training_summary(self, training_results):
        """
        Print a summary of training results including final validation metrics.
        
        Args:
            training_results: Results returned from the train() method
        """
        print("\n" + "="*50)
        print("TRAINING COMPLETED - FINAL METRICS")
        print("="*50)
        
        # Training results contain validation metrics from the last epoch
        if hasattr(training_results, 'results_dict'):
            metrics = training_results.results_dict
            
            # Print final validation metrics
            if 'metrics/mAP50(B)' in metrics:
                print(f"Final Validation mAP@0.5: {metrics['metrics/mAP50(B)']:.4f}")
            if 'metrics/mAP50-95(B)' in metrics:
                print(f"Final Validation mAP@0.5:0.95: {metrics['metrics/mAP50-95(B)']:.4f}")
            if 'metrics/precision(B)' in metrics:
                print(f"Final Validation Precision: {metrics['metrics/precision(B)']:.4f}")
            if 'metrics/recall(B)' in metrics:
                print(f"Final Validation Recall: {metrics['metrics/recall(B)']:.4f}")
                
            # Print training losses
            if 'train/box_loss' in metrics:
                print(f"Final Training Box Loss: {metrics['train/box_loss']:.4f}")
            if 'train/cls_loss' in metrics:
                print(f"Final Training Class Loss: {metrics['train/cls_loss']:.4f}")
            if 'train/dfl_loss' in metrics:
                print(f"Final Training DFL Loss: {metrics['train/dfl_loss']:.4f}")
        
        print("="*50)

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
