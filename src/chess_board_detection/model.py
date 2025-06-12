import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import cv2
import numpy as np
import warnings
from torchvision import transforms
from PIL import Image

matplotlib.use("Agg")  # Change to non-interactive backend
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import timm


class ChessBoardKeypointModel(nn.Module):
    """EfficientNet B0 based model for chessboard keypoint detection."""
    
    def __init__(self, num_keypoints=4, pretrained=True, dropout=0.2, freeze_backbone=False):
        """
        Initialize EfficientNet B0 model for keypoint regression.
        
        Args:
            num_keypoints: Number of keypoints to predict (4 for chessboard corners)
            pretrained: Whether to use pretrained EfficientNet weights
            dropout: Dropout rate for regularization
            freeze_backbone: If True, freeze EfficientNet backbone and only train keypoint head
        """
        super().__init__()
        
        # Load pretrained EfficientNet B0
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)
        
        # Optionally freeze the backbone to only train the new layers
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("🔒 EfficientNet backbone frozen - only training keypoint head layers")
        else:
            print("🔓 EfficientNet backbone unfrozen - fine-tuning entire network")
        
        # Get feature dimension from backbone
        feature_dim = self.backbone.num_features  # 1280 for EfficientNet B0
        
        # Regression head for keypoint prediction
        # Each keypoint has (x, y) coordinates, so 4 keypoints = 8 outputs
        self.keypoint_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_keypoints * 2)  # x, y for each keypoint
        )
        
        self.num_keypoints = num_keypoints
        self.freeze_backbone = freeze_backbone
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            keypoints: Tensor of shape (B, num_keypoints * 2) with normalized coordinates
        """
        # Extract features using backbone
        features = self.backbone(x)  # (B, feature_dim)
        
        # Predict keypoints
        keypoints = self.keypoint_head(features)  # (B, num_keypoints * 2)
        
        # Apply sigmoid to ensure coordinates are in [0, 1] range
        keypoints = torch.sigmoid(keypoints)
        
        return keypoints


class ChessBoardModel:
    """Model for detecting chessboard keypoints using EfficientNet B0."""
    
    def __init__(self, model_path: Path | None = None, device: torch.device | None = None, freeze_backbone=False):
        """
        Initialize the chessboard keypoint detection model.
        
        Args:
            model_path: Path to saved model weights
            device: Device to run model on
            freeze_backbone: If True, freeze EfficientNet backbone and only train keypoint head
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.expected_boards = 1  # Expect exactly 1 chessboard
        
        # Initialize model
        self.model = ChessBoardKeypointModel(num_keypoints=4, freeze_backbone=freeze_backbone)
        self.model.to(self.device)
        
        # Load weights if provided
        if model_path is not None and model_path.exists():
            self.load_model(model_path)
            print(f"✅ Loaded model from {model_path}")
        else:
            print("⚠️  No model weights loaded. Using pretrained EfficientNet backbone only.")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # EfficientNet B0 input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        # Set to evaluation mode
        self.model.eval()
    
    def load_model(self, model_path: Path):
        """Load model weights from file."""
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
    
    def save_model(self, model_path: Path, optimizer=None, epoch=None, loss=None):
        """Save model weights and training state."""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'num_keypoints': self.model.num_keypoints,
            }
        }
        
        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            save_dict['epoch'] = epoch
        if loss is not None:
            save_dict['loss'] = loss
            
        torch.save(save_dict, model_path)
        print(f"✅ Model saved to {model_path}")
    
    def preprocess_image(self, img_path):
        """
        Preprocess image for model input.
        
        Args:
            img_path: Path to image or PIL Image
            
        Returns:
            tuple: (preprocessed_tensor, original_image, original_size)
        """
        if isinstance(img_path, (str, Path)):
            image = Image.open(img_path).convert('RGB')
        else:
            image = img_path.convert('RGB')
        
        original_size = image.size  # (width, height)
        
        # Apply preprocessing
        preprocessed = self.transform(image).unsqueeze(0)  # Add batch dimension
        
        return preprocessed.to(self.device), image, original_size
    
    def predict_keypoints(self, img_path, conf_threshold=0.5):
        """
        Predict chessboard keypoints.
        
        Args:
            img_path: Path to the image to predict on
            conf_threshold: Confidence threshold (unused for regression, kept for compatibility)
        
        Returns:
            tuple: (keypoints_dict, is_valid)
                - keypoints_dict: Dictionary with corner coordinates
                - is_valid: Always True for single model prediction
        """
        preprocessed, original_image, original_size = self.preprocess_image(img_path)
        
        with torch.no_grad():
            # Get predictions
            keypoints = self.model(preprocessed)  # Shape: (1, 8)
            keypoints = keypoints.squeeze(0).cpu().numpy()  # Shape: (8,)
        
        # Convert normalized coordinates to absolute coordinates
        # The model outputs normalized coordinates [0,1] relative to the original image size
        # (This is standard YOLO format - coordinates are normalized to original dimensions)
        width, height = original_size
        keypoints_abs = []
        
        for i in range(0, len(keypoints), 2):
            x_norm, y_norm = keypoints[i], keypoints[i + 1]
            x_abs = x_norm * width
            y_abs = y_norm * height
            keypoints_abs.extend([x_abs, y_abs])
        
        # Create corner dictionary
        corner_names = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
        corners_dict = {}
        
        for i, corner_name in enumerate(corner_names):
            x_idx, y_idx = i * 2, i * 2 + 1
            corners_dict[corner_name] = {
                'x': float(keypoints_abs[x_idx]),
                'y': float(keypoints_abs[y_idx])
            }
        
        # Add corners array for compatibility
        corners_array = [[corners_dict[name]['x'], corners_dict[name]['y']] 
                        for name in corner_names]
        corners_dict['corners_array'] = corners_array
        corners_dict['confidence'] = 1.0  # Regression models don't have explicit confidence
        
        return corners_dict, True
    
    def predict_board(self, img_path, conf=0.5):
        """
        Predict chessboard and validate the count (compatibility method).
        
        Args:
            img_path: Path to the image to predict on
            conf: Confidence threshold (unused for regression)
        
        Returns:
            tuple: (corners_dict, board_count, is_valid)
        """
        corners_dict, is_valid = self.predict_keypoints(img_path, conf)
        board_count = 1 if is_valid else 0
        
        if is_valid:
            print(f"✅ Successfully detected {board_count} chessboard")
        else:
            print(f"⚠️  Failed to detect chessboard")
        
        return corners_dict, board_count, is_valid
    
    def get_corner_coordinates(self, img_path, conf=0.25):
        """
        Get 4-corner coordinates from the chessboard for perspective transformation.
        
        Args:
            img_path: Path to the image to analyze
            conf: Confidence threshold (unused for regression)
            
        Returns:
            tuple: (corners, is_valid)
        """
        return self.predict_keypoints(img_path, conf)
    
    def get_board_polygon(self, img_path, conf=0.25):
        """
        Get the polygon coordinates of the detected chessboard (compatibility method).
        
        Args:
            img_path: Path to the image to analyze
            conf: Confidence threshold (unused for regression)
        
        Returns:
            tuple: (polygon_info, is_valid)
        """
        corners_dict, is_valid = self.predict_keypoints(img_path, conf)
        
        if not is_valid:
            return {}, False
        
        # Convert corners to polygon format
        polygon_coords = []
        corner_names = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
        
        for corner_name in corner_names:
            corner = corners_dict[corner_name]
            polygon_coords.append({
                'x': corner['x'],
                'y': corner['y']
            })
        
        polygon_info = {
            'coordinates': polygon_coords,
            'confidence': corners_dict.get('confidence', 1.0),
            'num_points': len(polygon_coords)
        }
        
        return polygon_info, is_valid
    
    def plot_eval(self, img_path, ax=None, conf=0.25, show_polygon=True, show_centers=True):
        """
        Plot evaluation results with chessboard keypoints.
        
        Args:
            img_path: Path to the image to evaluate
            ax: Matplotlib axes to plot on (creates new if None)
            conf: Confidence threshold (unused for regression)
            show_polygon: Whether to draw the chessboard polygon
            show_centers: Whether to show corner center points
            
        Returns:
            ax: The matplotlib axes used for plotting
        """
        corners_dict, is_valid = self.predict_keypoints(img_path, conf)
        
        # Print detection info
        board_count = 1 if is_valid else 0
        print(f"Detected chessboards: {board_count}/1")
        print("Detection target: chessboard keypoints")

        # Use provided axes or create a new one
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))
            show_inline = True
        else:
            show_inline = False

        # Load and display image
        if isinstance(img_path, (str, Path)):
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Could not load image: {img_path}")
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = np.array(img_path)
        
        ax.imshow(image_rgb)

        if is_valid and show_centers:
            # Draw the 4 corners
            corner_names = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
            colors = ['red', 'blue', 'green', 'orange']
            
            for i, (corner_name, color) in enumerate(zip(corner_names, colors)):
                if corner_name in corners_dict:
                    corner = corners_dict[corner_name]
                    ax.plot(corner['x'], corner['y'], 'o', color=color, markersize=12, 
                           markerfacecolor=color, markeredgecolor='white', markeredgewidth=2)
                    ax.text(corner['x'], corner['y'] - 20, f"{corner_name.replace('_', ' ').title()}", 
                           color=color, fontsize=10, ha='center', weight='bold',
                           bbox=dict(facecolor='white', alpha=0.8, edgecolor=color))
            
            # Draw lines connecting the corners to show the quadrilateral
            if show_polygon and len(corners_dict.get('corners_array', [])) == 4:
                corner_points = corners_dict['corners_array']
                corner_polygon = Polygon(corner_points, fill=False, edgecolor='blue', 
                                       linewidth=3, linestyle='-', alpha=0.8)
                ax.add_patch(corner_polygon)
                print("✅ Drew chessboard 4-corner outline")
            
            # Add confidence info
            if 'confidence' in corners_dict:
                conf_val = corners_dict['confidence']
                ax.text(10, 30, f"Model: EfficientNet B0", color="white", fontsize=12, weight='bold',
                       bbox=dict(facecolor="black", alpha=0.7))

        # Set title based on detection quality
        if is_valid:
            ax.set_title("✅ Chessboard Keypoint Detection - Corners detected", color='green', fontsize=14, weight='bold')
        else:
            ax.set_title("⚠️  Chessboard Keypoint Detection - Failed to detect corners", 
                        color='orange', fontsize=14, weight='bold')

        ax.axis("off")
        
        if show_inline:
            plt.tight_layout()
            plt.show()
            
        return ax
    
    def plot_eval_with_results(self, img_path, corners_dict, is_valid, ax=None, conf=0.25, show_polygon=True, show_centers=True):
        """
        Plot evaluation results using pre-computed corner coordinates.
        
        Args:
            img_path: Path to the image to evaluate
            corners_dict: Pre-computed corner coordinates dictionary
            is_valid: Whether detection was valid
            ax: Matplotlib axes to plot on (creates new if None)
            conf: Confidence threshold (unused for regression)
            show_polygon: Whether to draw the chessboard polygon
            show_centers: Whether to show corner center points
            
        Returns:
            ax: The matplotlib axes used for plotting
        """
        # Print detection info (using cached results)
        board_count = 1 if corners_dict else 0
        print(f"Detected chessboards: {board_count}/1")
        print("Detection target: chessboard keypoints")

        # Use provided axes or create a new one
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))
            show_inline = True
        else:
            show_inline = False

        # Load and display image
        if isinstance(img_path, (str, Path)):
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Could not load image: {img_path}")
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = np.array(img_path)
        
        ax.imshow(image_rgb)

        if corners_dict and show_centers:
            # Draw the 4 corners using cached results
            corner_names = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
            colors = ['red', 'blue', 'green', 'orange']
            
            for i, (corner_name, color) in enumerate(zip(corner_names, colors)):
                if corner_name in corners_dict:
                    corner = corners_dict[corner_name]
                    ax.plot(corner['x'], corner['y'], 'o', color=color, markersize=12, 
                           markerfacecolor=color, markeredgecolor='white', markeredgewidth=2)
                    ax.text(corner['x'], corner['y'] - 20, f"{corner_name.replace('_', ' ').title()}", 
                           color=color, fontsize=10, ha='center', weight='bold',
                           bbox=dict(facecolor='white', alpha=0.8, edgecolor=color))
            
            # Draw lines connecting the corners to show the quadrilateral
            if show_polygon and len(corners_dict.get('corners_array', [])) == 4:
                corner_points = corners_dict['corners_array']
                corner_polygon = Polygon(corner_points, fill=False, edgecolor='blue', 
                                       linewidth=3, linestyle='-', alpha=0.8)
                ax.add_patch(corner_polygon)
                print("✅ Drew chessboard 4-corner outline")
            
            # Add model info
            if 'confidence' in corners_dict:
                ax.text(10, 30, f"Model: EfficientNet B0", color="white", fontsize=12, weight='bold',
                       bbox=dict(facecolor="black", alpha=0.7))

        # Set title based on detection quality
        if is_valid:
            ax.set_title("✅ Chessboard Keypoint Detection - Corners detected", color='green', fontsize=14, weight='bold')
        else:
            board_count = 1 if corners_dict else 0
            ax.set_title("⚠️  Chessboard Keypoint Detection - Failed to detect corners", 
                        color='orange', fontsize=14, weight='bold')

        ax.axis("off")
        
        if show_inline:
            plt.tight_layout()
            plt.show()
            
        return ax
    
    def train_mode(self):
        """Set model to training mode."""
        self.model.train()
    
    def eval_mode(self):
        """Set model to evaluation mode."""
        self.model.eval()
    
    def get_model(self):
        """Get the underlying PyTorch model for training."""
        return self.model
    
    def calculate_keypoint_loss(self, predictions, targets, loss_type='mse'):
        """
        Calculate loss for keypoint regression.
        
        Args:
            predictions: Predicted keypoints (B, 8) - normalized coordinates
            targets: Target keypoints (B, 8) - normalized coordinates
            loss_type: Type of loss ('mse', 'smooth_l1', 'huber')
        
        Returns:
            loss: Calculated loss value
        """
        if loss_type == 'mse':
            loss = F.mse_loss(predictions, targets)
        elif loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(predictions, targets)
        elif loss_type == 'huber':
            loss = F.huber_loss(predictions, targets)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        return loss
    
    def polygon_to_keypoints(self, polygon_coords, image_size):
        """
        Convert polygon coordinates to ordered keypoint format.
        
        Args:
            polygon_coords: List of polygon coordinates
            image_size: (width, height) of the image
        
        Returns:
            numpy array: Normalized keypoints in order [top_left, top_right, bottom_right, bottom_left]
        """
        # Convert polygon to numpy array
        if isinstance(polygon_coords[0], dict):
            points = np.array([[p['x'], p['y']] for p in polygon_coords])
        else:
            points = np.array(polygon_coords)
        
        # Order the points: top-left, top-right, bottom-right, bottom-left
        ordered_points = self._order_corners(points)
        
        # Normalize coordinates
        width, height = image_size
        normalized_points = ordered_points.copy()
        normalized_points[:, 0] /= width  # Normalize x
        normalized_points[:, 1] /= height  # Normalize y
        
        # Flatten to match model output format
        keypoints = normalized_points.flatten()  # Shape: (8,)
        
        return keypoints
    
    def _order_corners(self, corners):
        """Order corners in clockwise order: top-left, top-right, bottom-right, bottom-left."""
        # Calculate centroid
        center = np.mean(corners, axis=0)
        
        # Calculate angle from center to each corner
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        
        # Sort corners by angle (this gives us a consistent ordering)
        sorted_indices = np.argsort(angles)
        sorted_corners = corners[sorted_indices]
        
        # Find which corner is top-left (minimum sum of coordinates)
        sums = np.sum(sorted_corners, axis=1)
        top_left_idx = np.argmin(sums)
        
        # Reorder starting from top-left, going clockwise
        ordered_corners = np.roll(sorted_corners, -top_left_idx, axis=0)
        
        # Verify the order is correct (top-left should have min x+y, bottom-right should have max x+y)
        sums = np.sum(ordered_corners, axis=1)
        if sums[0] > sums[2]:  # If our "top-left" has a larger sum than "bottom-right", we need to reverse
            ordered_corners = ordered_corners[::-1]
            # Also need to shift to start from correct top-left
            sums = np.sum(ordered_corners, axis=1)
            top_left_idx = np.argmin(sums)
            ordered_corners = np.roll(ordered_corners, -top_left_idx, axis=0)
        
        return ordered_corners


# Compatibility functions that might be called by other parts of the codebase
def create_chessboard_model(model_path: Path | None = None, device: torch.device | None = None):
    """Create a ChessBoardModel instance (factory function)."""
    return ChessBoardModel(model_path=model_path, device=device)


if __name__ == "__main__":
    # Example usage
    print("ChessBoardModel - EfficientNet B0 Keypoint Detection Example")
    print("=" * 60)
    
    # Initialize model
    model = ChessBoardModel()
    
    print("To use this model:")
    print("1. Prepare training data with polygon annotations")
    print("2. Train the model: see training script for details")
    print("3. Load trained model: model = ChessBoardModel(model_path=Path('best.pt'))")
    print("4. Predict keypoints: corners, is_valid = model.predict_keypoints('image.jpg')")
    print("5. Get coordinates: coords, is_valid = model.get_corner_coordinates('image.jpg')")
    print("6. Visualize: model.plot_eval('image.jpg')")
    
    # Test with the example image if available
    example_image = Path("data/eval_images/chess_4.jpeg")
    if example_image.exists():
        print(f"\n🧪 Testing with example image: {example_image}")
        try:
            corners, is_valid = model.predict_keypoints(example_image)
            if is_valid:
                print("✅ Successfully predicted keypoints!")
                for corner_name, corner in corners.items():
                    if isinstance(corner, dict) and 'x' in corner:
                        print(f"  {corner_name}: ({corner['x']:.1f}, {corner['y']:.1f})")
            else:
                print("⚠️  Keypoint prediction failed")
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
    else:
        print(f"\n⚠️  Example image not found: {example_image}") 