#!/usr/bin/env python3
"""
ChessBoard Segmentation Model using YOLOv8

This model learns to segment the exact polygon boundaries of chessboards
from polygon annotation data.
"""

import torch
import matplotlib
import cv2
import numpy as np
import warnings
from ultralytics import YOLO

matplotlib.use("Agg")  # Change to non-interactive backend
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import yaml

from src.base_model import BaseYOLOModel


class ChessBoardSegmentationModel(BaseYOLOModel):
    """Model for segmenting chessboard polygons using YOLOv8 segmentation."""
    
    def __init__(self, model_path: Path | None = None, device: torch.device | None = None, pretrained_model: str = "yolov8n-seg.pt"):
        """
        Initialize ChessBoard Segmentation Model.
        
        Args:
            model_path: Path to trained model weights. If None, loads pretrained YOLOv8 segmentation model
            device: Device to run model on. If None, auto-detects
            pretrained_model: Pretrained model to use when model_path is None. Options:
                - 'yolov8n-seg.pt' (nano, fastest, least accurate)
                - 'yolov8s-seg.pt' (small)
                - 'yolov8m-seg.pt' (medium)  
                - 'yolov8l-seg.pt' (large)
                - 'yolov8x-seg.pt' (extra large, slowest, most accurate)
        """
        # Initialize with segmentation model instead of detection
        if model_path is None:
            # Load pretrained YOLOv8 segmentation model (COCO pretrained)
            valid_models = ['yolov8n-seg.pt', 'yolov8s-seg.pt', 'yolov8m-seg.pt', 'yolov8l-seg.pt', 'yolov8x-seg.pt']
            if pretrained_model not in valid_models:
                print(f"⚠️  Invalid pretrained_model '{pretrained_model}'. Using default 'yolov8n-seg.pt'")
                pretrained_model = 'yolov8n-seg.pt'
            
            print(f"📥 Loading COCO-pretrained model: {pretrained_model}")
            self.model = YOLO(pretrained_model)
        else:
            self.model = YOLO(str(model_path))
            
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model.to(self.device)
        print(f"🎯 ChessBoard Segmentation Model initialized on {self.device}")

    def predict_segments(self, img_path, conf=0.25, iou=0.7):
        """
        Predict chessboard segmentation masks.
        
        Args:
            img_path: Path to the image to analyze
            conf: Confidence threshold for predictions
            iou: IoU threshold for NMS
            
        Returns:
            tuple: (results, segment_count, is_valid)
                - results: YOLO prediction results with masks
                - segment_count: Number of detected segments
                - is_valid: True if exactly 1 chessboard segment detected
        """
        results = self.model.predict(img_path, conf=conf, iou=iou)[0]
        
        # Count detected segments
        segment_count = 0
        if results.masks is not None:
            segment_count = len(results.masks)
        
        # For chessboard detection, we expect exactly 1 segment
        is_valid = segment_count == 1
        
        if not is_valid:
            if segment_count == 0:
                warnings.warn(
                    f"⚠️  No chessboard segments detected. "
                    f"Try lowering the confidence threshold or check if the chessboard is visible.",
                    UserWarning
                )
            elif segment_count > 1:
                warnings.warn(
                    f"⚠️  {segment_count} segments detected (expected 1). "
                    f"Try raising the confidence threshold or check for multiple boards.",
                    UserWarning
                )
        else:
            print(f"✅ Successfully detected {segment_count} chessboard segment")
        
        return results, segment_count, is_valid

    def get_polygon_coordinates(self, img_path, conf=0.25, iou=0.7):
        """
        Get the polygon coordinates of the detected chessboard.
        
        Args:
            img_path: Path to the image to analyze
            conf: Confidence threshold for predictions
            iou: IoU threshold for NMS
            
        Returns:
            tuple: (polygon_info, is_valid)
                - polygon_info: Dictionary with polygon coordinates and metadata
                - is_valid: True if exactly 1 chessboard detected
        """
        results, segment_count, is_valid = self.predict_segments(img_path, conf=conf, iou=iou)
        
        polygon_info = {}
        
        if results.masks is not None and len(results.masks) > 0:
            # Get the first (and ideally only) mask
            mask = results.masks[0]
            
            # Get the polygon coordinates from the mask
            if hasattr(mask, 'xy') and len(mask.xy) > 0:
                # YOLOv8 provides polygon coordinates directly
                polygon_coords = mask.xy[0]  # First contour
                
                # Convert to list of coordinate dictionaries
                coordinates = []
                for point in polygon_coords:
                    coordinates.append({
                        'x': float(point[0]),
                        'y': float(point[1])
                    })
                
                # Get confidence if available
                confidence = 1.0
                if results.boxes is not None and len(results.boxes) > 0:
                    confidence = float(results.boxes[0].conf[0].cpu().numpy())
                
                polygon_info = {
                    'coordinates': coordinates,
                    'confidence': confidence,
                    'num_points': len(coordinates),
                    'area': self._calculate_polygon_area(coordinates)
                }
            else:
                # Fallback: extract contours from mask
                mask_array = mask.data[0].cpu().numpy().astype(np.uint8)
                contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Get the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Convert contour to coordinates
                    coordinates = []
                    for point in largest_contour.reshape(-1, 2):
                        coordinates.append({
                            'x': float(point[0]),
                            'y': float(point[1])
                        })
                    
                    confidence = 1.0
                    if results.boxes is not None and len(results.boxes) > 0:
                        confidence = float(results.boxes[0].conf[0].cpu().numpy())
                    
                    polygon_info = {
                        'coordinates': coordinates,
                        'confidence': confidence,
                        'num_points': len(coordinates),
                        'area': cv2.contourArea(largest_contour)
                    }
        
        return polygon_info, is_valid

    def _calculate_polygon_area(self, coordinates):
        """Calculate the area of a polygon using the shoelace formula."""
        if len(coordinates) < 3:
            return 0.0
        
        x = [coord['x'] for coord in coordinates]
        y = [coord['y'] for coord in coordinates]
        
        return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(x) - 1)))

    def plot_eval(self, img_path, ax=None, conf=0.25, iou=0.7, show_polygon=True, show_mask=True, alpha=0.5):
        """
        Plot evaluation results with segmentation masks and polygon outlines.
        
        Args:
            img_path: Path to the image to evaluate
            ax: Matplotlib axes to plot on (creates new if None)
            conf: Confidence threshold for predictions
            iou: IoU threshold for NMS
            show_polygon: Whether to draw polygon outline
            show_mask: Whether to show segmentation mask
            alpha: Transparency for mask overlay
            
        Returns:
            ax: The matplotlib axes used for plotting
        """
        results, segment_count, is_valid = self.predict_segments(img_path, conf=conf, iou=iou)
        polygon_info, _ = self.get_polygon_coordinates(img_path, conf=conf, iou=iou)
        
        # Print detection info
        print(f"Detected segments: {segment_count}")
        if results.names:
            print(f"Detection target: {results.names[0] if 0 in results.names else 'chessboard'}")

        # Use provided axes or create a new one
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 10))
            show_inline = True
        else:
            show_inline = False

        # Convert BGR to RGB for proper matplotlib display
        if hasattr(results, 'orig_img'):
            image_rgb = cv2.cvtColor(results.orig_img, cv2.COLOR_BGR2RGB)
        else:
            # Fallback: load image directly
            image = cv2.imread(str(img_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        ax.imshow(image_rgb)

        # Draw segmentation results
        if results.masks is not None and show_mask:
            for i, mask in enumerate(results.masks):
                # Get mask array
                mask_array = mask.data[0].cpu().numpy()
                
                # Create colored mask
                colored_mask = np.zeros((*mask_array.shape, 3))
                colored_mask[mask_array > 0] = [0, 1, 0]  # Green mask
                
                # Overlay mask
                ax.imshow(colored_mask, alpha=alpha)
                
                # Get confidence if available
                conf_val = polygon_info.get('confidence', 1.0)
                ax.text(10, 30 + i * 30, f"Chessboard Segment {i+1} ({conf_val:.3f})", 
                       color="green", fontsize=12, weight='bold',
                       bbox=dict(facecolor="white", alpha=0.8))

        # Draw polygon outline
        if show_polygon and polygon_info and 'coordinates' in polygon_info:
            coordinates = polygon_info['coordinates']
            if len(coordinates) >= 3:
                # Create polygon points for matplotlib
                polygon_points = [[coord['x'], coord['y']] for coord in coordinates]
                
                # Draw polygon outline
                polygon = Polygon(polygon_points, fill=False, edgecolor='red', 
                                linewidth=3, linestyle='-', alpha=0.9)
                ax.add_patch(polygon)
                
                # Add area information
                area = polygon_info.get('area', 0)
                ax.text(10, image_rgb.shape[0] - 40, f"Polygon Area: {area:.0f} pixels²", 
                       color="red", fontsize=10, weight='bold',
                       bbox=dict(facecolor="white", alpha=0.8))
                
                print(f"✅ Drew chessboard polygon outline ({len(coordinates)} points)")

        # Set title based on detection quality
        if is_valid:
            ax.set_title("✅ Chessboard Segmentation - Successfully detected", color='green', fontsize=14)
        else:
            ax.set_title(f"⚠️  Chessboard Segmentation - {segment_count} segments detected", 
                        color='orange', fontsize=14)

        ax.axis("off")
        
        if show_inline:
            self._display_plot_inline(ax=ax)
            
        return ax

    def train(self, data_path, epochs=100, batch=16, imgsz=640, **kwargs):
        """
        Train the segmentation model.
        
        Args:
            data_path: Path to dataset YAML file
            epochs: Number of training epochs
            batch: Batch size
            imgsz: Input image size
            **kwargs: Additional training parameters
            
        Returns:
            Training results
        """
        print(f"🚀 Starting chessboard segmentation training...")
        print(f"📊 Dataset: {data_path}")
        print(f"🔄 Epochs: {epochs}, Batch: {batch}, Image size: {imgsz}")
        
        # Train the model
        results = self.model.train(
            data=str(data_path),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            **kwargs
        )
        
        return results

    def validate_dataset_format(self, data_yaml_path: Path):
        """
        Validate that the dataset is in proper segmentation format.
        
        Args:
            data_yaml_path: Path to the dataset YAML file
            
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            with open(data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            
            # Check required fields
            required_fields = ['train', 'val', 'nc', 'names']
            for field in required_fields:
                if field not in data_config:
                    return False, f"Missing required field: {field}"
            
            # Check if train directory exists
            train_path = Path(data_yaml_path.parent) / data_config['train']
            if not train_path.exists():
                return False, f"Training images directory not found: {train_path}"
            
            # Check labels directory
            labels_dir = train_path.parent / 'labels'
            if not labels_dir.exists():
                return False, f"Labels directory not found: {labels_dir}"
            
            # Sample a few label files to check format
            label_files = list(labels_dir.glob('*.txt'))[:5]
            polygon_count = 0
            
            for label_file in label_files:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 7:  # class_id + at least 3 coordinate pairs
                            # Count actual coordinate pairs (excluding padding)
                            coords = []
                            for i in range(1, len(parts), 2):
                                if i + 1 < len(parts):
                                    try:
                                        x, y = float(parts[i]), float(parts[i + 1])
                                        # Filter out padding values (like 2.0)
                                        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                                            coords.append((x, y))
                                    except ValueError:
                                        continue
                            
                            if len(coords) >= 3:  # Valid polygon needs at least 3 points
                                polygon_count += 1
            
            if polygon_count == 0:
                return False, "No valid polygon annotations found in sample files"
            
            return True, f"Valid segmentation dataset with {polygon_count} sample polygons found"
            
        except Exception as e:
            return False, f"Error validating dataset: {e}"

    def get_training_summary(self, results):
        """Print a summary of training results."""
        print("\n" + "="*70)
        print("🎯 SEGMENTATION TRAINING COMPLETE")
        print("="*70)
        
        if hasattr(results, 'results_dir'):
            print(f"📁 Results saved to: {results.results_dir}")
        
        # Print key metrics if available
        if hasattr(results, 'metrics'):
            metrics = results.metrics
            print("📊 Final Metrics:")
            if hasattr(metrics, 'seg'):
                seg_metrics = metrics.seg
                print(f"   🎯 mAP50 (Segmentation): {getattr(seg_metrics, 'map50', 'N/A'):.3f}")
                print(f"   🎯 mAP50-95 (Segmentation): {getattr(seg_metrics, 'map', 'N/A'):.3f}")
            if hasattr(metrics, 'box'):
                box_metrics = metrics.box  
                print(f"   📦 mAP50 (Detection): {getattr(box_metrics, 'map50', 'N/A'):.3f}")
                print(f"   📦 mAP50-95 (Detection): {getattr(box_metrics, 'map', 'N/A'):.3f}")
        
        print("="*70) 