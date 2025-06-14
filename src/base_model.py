import os
import torch
import matplotlib
import tempfile
import io
import cv2

matplotlib.use("Agg")  # Change to non-interactive backend
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle  # Import Rectangle from patches
import ultralytics
from ultralytics import YOLO
from PIL import Image

from src.utils import is_notebook, get_best_torch_device, push_model_to_huggingface

# W&B integration
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from huggingface_hub import hf_hub_download, list_repo_files


class BaseYOLOModel:
    """Base class for YOLO-based models with common functionality."""
    
    def __init__(self, model_path: Path | None = None, device: torch.device | None = None, pretrained_checkpoint: str | None = None):
        """
        Initialize BaseYOLOModel.
        
        Args:
            model_path: Path to existing trained model (optional)
            device: Device to run model on (auto-detected if None)
            pretrained_checkpoint: Pretrained YOLO checkpoint to use when no model_path exists (optional)
        """
        self.model_path = model_path
        self.device = device or get_best_torch_device()
        self.pretrained_checkpoint = pretrained_checkpoint
        
        if model_path and model_path.exists():
            self.load_model()
        elif pretrained_checkpoint:
            print(f"Loading pretrained checkpoint: {pretrained_checkpoint}")
            self.model = YOLO(pretrained_checkpoint)
        else:
            self.model = YOLO()
        self.model.to(self.device)

    @classmethod
    def from_huggingface(cls, 
                        repo_id: str, 
                        filename: str = "best.pt",
                        device: torch.device | None = None,
                        cache_dir: Path | None = None,
                        token: str | None = None,
                        pretrained_checkpoint: str | None = None):
        """
        Load a model from a Hugging Face repository.
        
        Args:
            repo_id: Repository ID on Hugging Face Hub (e.g., "username/chess-piece-detector")
            filename: Name of the model file to download (default: "best.pt")
            device: Device to load the model on (auto-detected if None)
            cache_dir: Directory to cache downloaded files (uses HF default if None)
            token: Hugging Face authentication token (uses env var if not provided)
            pretrained_checkpoint: Pretrained YOLO checkpoint to use as fallback (optional)
            
        Returns:
            BaseYOLOModel: Instantiated model loaded from HF Hub
            
        Raises:
            ImportError: If huggingface_hub is not installed
            FileNotFoundError: If the model file is not found in the repository
            Exception: If download or model loading fails
            
        Example:
            >>> model = cls.from_huggingface("username/chess-detector")
            >>> model = cls.from_huggingface("username/chess-detector", filename="last.pt")
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
            
            # Create and return model instance
            return cls(model_path=Path(model_path), device=device, pretrained_checkpoint=pretrained_checkpoint)
            
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
            >>> files = cls.list_huggingface_files("username/chess-detector")
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
        """Save the model to the specified path."""
        save_path = path or self.model_path
        if save_path is None:
            raise ValueError("No path provided and no model_path set")
        self.model.save(save_path)
        return save_path

    def load_model(self):
        """Load the model from the specified path."""
        if self.model_path is None:
            raise ValueError("No model_path provided")
        if self.model_path.exists():
            self.model = YOLO(self.model_path)
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")

    def train(self, data_path: Path, epochs: int = 15, batch: int = 32, lr0: float = 0.0001, lrf: float = 0.1, 
              imgsz: int = 640, plots: bool = True, project: Path | None = None, name: str | None = None,
              # Additional training parameters
              save_period: int = -1, patience: int = 50, optimizer: str = 'auto', weight_decay: float = 0.0005,
              # Data augmentation parameters
              hsv_h: float = 0.015, hsv_s: float = 0.7, hsv_v: float = 0.4,
              degrees: float = 0.0, translate: float = 0.1, scale: float = 0.5, shear: float = 0.0,
              perspective: float = 0.0, flipud: float = 0.0, fliplr: float = 0.5,
              mosaic: float = 1.0, mixup: float = 0.0, copy_paste: float = 0.0,
              # Advanced training parameters  
              warmup_epochs: float = 3.0, warmup_momentum: float = 0.8, warmup_bias_lr: float = 0.1,
              box: float = 7.5, cls: float = 0.5, dfl: float = 1.5,
              # W&B integration parameters
              use_wandb: bool = False, wandb_project: str | None = None, wandb_name: str | None = None,
              wandb_tags: list | None = None, wandb_notes: str | None = None,
              **kwargs):
        """
        Train the YOLO model with extensive customization options and optional W&B tracking.
        
        Args:
            data_path: Path to dataset YAML file
            epochs: Number of epochs to train for
            batch: Batch size for training
            lr0: Initial learning rate
            lrf: Final learning rate (lr0 * lrf)
            imgsz: Input image size
            plots: Save plots during training
            project: Project name for saving runs
            name: Run name for saving
            save_period: Save checkpoint every x epochs (-1 to disable)
            patience: Epochs to wait for no improvement before early stopping
            optimizer: Optimizer to use ('SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto')
            weight_decay: Weight decay for regularization
            
            # Data augmentation
            hsv_h: HSV hue augmentation (fraction)
            hsv_s: HSV saturation augmentation (fraction)
            hsv_v: HSV value augmentation (fraction)
            degrees: Image rotation (+/- degrees)
            translate: Image translation (+/- fraction)
            scale: Image scale (+/- gain)
            shear: Image shear (+/- degrees)
            perspective: Image perspective (+/- fraction)
            flipud: Vertical flip (probability)
            fliplr: Horizontal flip (probability)
            mosaic: Mosaic augmentation (probability)
            mixup: MixUp augmentation (probability)
            copy_paste: Copy-paste augmentation (probability)
            
            # Advanced parameters
            warmup_epochs: Warmup epochs
            warmup_momentum: Warmup initial momentum
            warmup_bias_lr: Warmup initial bias learning rate
            box: Box loss gain
            cls: Classification loss gain
            dfl: Distribution focal loss gain
            
            # W&B parameters
            use_wandb: Enable W&B tracking
            wandb_project: W&B project name (defaults to 'chess-piece-detection')
            wandb_name: W&B run name (defaults to training name)
            wandb_tags: List of tags for W&B run
            wandb_notes: Notes for W&B run
            **kwargs: Additional arguments passed to YOLO.train()
        """
        # Initialize W&B if requested
        wandb_run = None
        if use_wandb:
            
            # Set up W&B configuration
            wandb_config = {
                "epochs": epochs,
                "batch_size": batch,
                "learning_rate": lr0,
                "final_lr": lr0 * lrf,
                "image_size": imgsz,
                "optimizer": optimizer,
                "weight_decay": weight_decay,
                "patience": patience,
                "save_period": save_period,
                # Data augmentation config
                "hsv_h": hsv_h,
                "hsv_s": hsv_s,
                "hsv_v": hsv_v,
                "degrees": degrees,
                "translate": translate,
                "scale": scale,
                "shear": shear,
                "perspective": perspective,
                "flipud": flipud,
                "fliplr": fliplr,
                "mosaic": mosaic,
                "mixup": mixup,
                "copy_paste": copy_paste,
                # Advanced parameters
                "warmup_epochs": warmup_epochs,
                "warmup_momentum": warmup_momentum,
                "warmup_bias_lr": warmup_bias_lr,
                "box_loss": box,
                "cls_loss": cls,
                "dfl_loss": dfl,
                # Model info
                "pretrained_checkpoint": self.pretrained_checkpoint,
                "device": str(self.device),
                "dataset_path": str(data_path),
            }
            
            # Initialize W&B run
            try:
                # Get API key from environment if available
                import os
                wandb_api_key = os.environ.get("WANDB_API_KEY")
                
                wandb_run = wandb.init(
                    project=wandb_project or "chess-piece-detection",
                    name=wandb_name or name or "yolo-training",
                    config=wandb_config,
                    tags=wandb_tags or ["chess", "yolo", "object-detection"],
                    notes=wandb_notes,
                    reinit=True,
                    # Pass API key if available in environment
                    **({"api_key": wandb_api_key} if wandb_api_key else {})
                )
                
                # Add W&B callback to model
                add_wandb_callback(self.model, enable_model_checkpointing=True)
                print(f"✅ W&B tracking initialized: {wandb_run.url}")
                
            except Exception as e:
                print(f"⚠️  Failed to initialize W&B: {e}")
                print("   Continuing training without W&B tracking...")
                wandb_run = None
        
        try:
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
                save_period=save_period,
                patience=patience,
                optimizer=optimizer,
                weight_decay=weight_decay,
                # Data augmentation
                hsv_h=hsv_h,
                hsv_s=hsv_s,
                hsv_v=hsv_v,
                degrees=degrees,
                translate=translate,
                scale=scale,
                shear=shear,
                perspective=perspective,
                flipud=flipud,
                fliplr=fliplr,
                mosaic=mosaic,
                mixup=mixup,
                copy_paste=copy_paste,
                # Advanced parameters
                warmup_epochs=warmup_epochs,
                warmup_momentum=warmup_momentum,
                warmup_bias_lr=warmup_bias_lr,
                box=box,
                cls=cls,
                dfl=dfl,
                **kwargs
            )
            
            return results
            
        finally:
            # Clean up W&B run if it was initialized
            if wandb_run is not None:
                try:
                    wandb.finish()
                    print("✅ W&B run completed and logged")
                except Exception as e:
                    print(f"⚠️  Warning: Failed to properly finish W&B run: {e}")

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

    def predict(self, img_path, conf=0.1, save=False, imgsz=640):
        """Make predictions on an image."""
        results = self.model.predict(img_path, conf=conf, save=save, imgsz=imgsz)[0]
        return results

    def _display_plot_inline(self, ax=None, fig=None):
        """Helper method to display plots inline in notebooks or show them otherwise."""
        if is_notebook():
            from IPython.display import display, Image as IPyImage
            # Save the plot to a bytes buffer
            buf = io.BytesIO()
            if fig:
                fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            else:
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close()
            buf.seek(0)
            # Display in notebook
            display(IPyImage(data=buf.getvalue()))
            buf.close()
        else:
            # Fallback to PIL display for non-notebook environments
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                if fig:
                    fig.savefig(tmp.name, bbox_inches='tight', pad_inches=0)
                else:
                    plt.savefig(tmp.name, bbox_inches='tight', pad_inches=0)
                plt.close()
                img = Image.open(tmp.name)
                img.show()
                # Clean up the temporary file
                os.unlink(tmp.name) 