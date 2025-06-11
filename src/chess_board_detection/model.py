import torch
import matplotlib
import cv2
import numpy as np
import warnings

matplotlib.use("Agg")  # Change to non-interactive backend
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon  # Import Rectangle and Polygon from patches

from src.base_model import BaseYOLOModel


class ChessBoardModel(BaseYOLOModel):
    """Model for detecting chessboard using YOLO segmentation to get precise polygon coordinates."""
    
    def __init__(self, model_path: Path | None = None, device: torch.device | None = None):
        super().__init__(model_path, device)
        self.expected_boards = 1  # Expect exactly 1 chessboard
        
        # Override default model to use segmentation if no model path provided
        if model_path is None or not model_path.exists():
            from ultralytics import YOLO
            # Use YOLOv11 segmentation model as default for chessboard detection
            self.model = YOLO('yolo11n-seg.pt')  # Use segmentation model
            self.model.to(self.device)

    def predict_board(self, img_path, conf=0.25):
        """
        Predict chessboard segmentation and validate the count.
        
        Args:
            img_path: Path to the image to analyze
            conf: Confidence threshold for predictions
            
        Returns:
            tuple: (results, board_count, is_valid)
                - results: YOLO prediction results
                - board_count: Number of detected boards
                - is_valid: True if exactly 1 board detected
        """
        results = self.predict(img_path, conf=conf)
        
        # Count detected boards
        board_count = 0
        if results.masks is not None:
            board_count = len(results.masks)
        
        # Validate board count
        is_valid = board_count == self.expected_boards
        
        if not is_valid:
            if board_count < self.expected_boards:
                warnings.warn(
                    f"⚠️  Only {board_count} boards detected (expected {self.expected_boards}). "
                    f"Try lowering the confidence threshold or check if the chessboard is fully visible.",
                    UserWarning
                )
            else:
                warnings.warn(
                    f"⚠️  {board_count} boards detected (expected {self.expected_boards}). "
                    f"Try raising the confidence threshold or check for false positives.",
                    UserWarning
                )
        else:
            print(f"✅ Successfully detected {board_count} chessboard")
        
        return results, board_count, is_valid

    def get_board_polygon(self, img_path, conf=0.25):
        """
        Get the polygon coordinates of the detected chessboard.
        
        Args:
            img_path: Path to the image to analyze
            conf: Confidence threshold for predictions
            
        Returns:
            tuple: (polygon_coords, is_valid)
                - polygon_coords: List of (x, y) tuples forming the board polygon
                - is_valid: True if exactly 1 board detected
        """
        results, board_count, is_valid = self.predict_board(img_path, conf=conf)
        
        polygon_coords = []
        if results.masks is not None and len(results.masks) > 0:
            # Get the first (and hopefully only) mask
            mask = results.masks[0]
            
            # Get polygon coordinates in original image dimensions
            if hasattr(mask, 'xy') and len(mask.xy) > 0:
                # mask.xy contains the polygon coordinates
                coords = mask.xy[0]  # First polygon (there should be only one)
                
                # Convert to list of coordinate dictionaries
                for i in range(len(coords)):
                    x, y = coords[i]
                    polygon_coords.append({
                        'x': float(x),
                        'y': float(y)
                    })
                
                # Get confidence from detection
                if results.boxes is not None and len(results.boxes) > 0:
                    confidence = float(results.boxes[0].conf[0].cpu().numpy())
                else:
                    confidence = 1.0  # Default confidence
                
                # Add confidence to polygon info
                polygon_info = {
                    'coordinates': polygon_coords,
                    'confidence': confidence,
                    'num_points': len(polygon_coords)
                }
                
                return polygon_info, is_valid
        
        return {}, is_valid

    def get_corner_coordinates(self, img_path, conf=0.25):
        """
        Get 4-corner coordinates from the chessboard polygon for perspective transformation.
        
        Args:
            img_path: Path to the image to analyze
            conf: Confidence threshold for predictions
            
        Returns:
            tuple: (corners, is_valid)
                - corners: Dictionary with 4 corners in format expected by perspective transform
                - is_valid: True if exactly 1 board detected
        """
        polygon_info, is_valid = self.get_board_polygon(img_path, conf=conf)
        
        if not is_valid or not polygon_info:
            return {}, is_valid
        
        coords = polygon_info['coordinates']
        if len(coords) < 4:
            warnings.warn(f"Polygon has only {len(coords)} points, need at least 4 for corner detection", UserWarning)
            return {}, False
        
        # Convert to numpy array for processing
        points = np.array([[c['x'], c['y']] for c in coords])
        
        # Find the 4 corners using convex hull or corner detection
        corners = self._extract_four_corners(points)
        
        # Order corners: top-left, top-right, bottom-right, bottom-left  
        ordered_corners = self._order_corners(corners)
        
        # Create corner dictionary with the expected format
        corner_dict = {
            'top_left': {'x': float(ordered_corners[0][0]), 'y': float(ordered_corners[0][1])},
            'top_right': {'x': float(ordered_corners[1][0]), 'y': float(ordered_corners[1][1])},
            'bottom_right': {'x': float(ordered_corners[2][0]), 'y': float(ordered_corners[2][1])},
            'bottom_left': {'x': float(ordered_corners[3][0]), 'y': float(ordered_corners[3][1])},
            'confidence': polygon_info['confidence'],
            'corners_array': ordered_corners.tolist()  # For easy homography calculation
        }
        
        return corner_dict, is_valid

    def _extract_four_corners(self, points):
        """Extract 4 corner points from a polygon."""
        from scipy.spatial import ConvexHull
        
        try:
            # Use convex hull to get the outer boundary
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            
            # If we have exactly 4 points, return them
            if len(hull_points) == 4:
                return hull_points
            
            # If we have more than 4 points, find the 4 most extreme corners
            # Using a simpler approach: find corners based on distance from center
            center = np.mean(hull_points, axis=0)
            
            # Calculate angles from center to each point
            angles = np.arctan2(hull_points[:, 1] - center[1], hull_points[:, 0] - center[0])
            
            # Sort by angle and take 4 points that are most spread out
            angle_indices = np.argsort(angles)
            sorted_points = hull_points[angle_indices]
            
            # Select 4 points that are roughly 90 degrees apart
            n_points = len(sorted_points)
            corner_indices = [
                0,  # First point
                n_points // 4,  # Quarter way around
                n_points // 2,  # Halfway around  
                3 * n_points // 4  # Three quarters around
            ]
            
            # Ensure indices are within bounds
            corner_indices = [min(i, n_points - 1) for i in corner_indices]
            
            corners = sorted_points[corner_indices]
            return corners
            
        except Exception as e:
            warnings.warn(f"Could not extract corners using convex hull: {e}. Using extreme points.", UserWarning)
            # Fallback: use extreme points
            return self._get_extreme_points(points)
    
    def _get_extreme_points(self, points):
        """Get 4 extreme points as corners."""
        # Find extreme points
        min_x_idx = np.argmin(points[:, 0])
        max_x_idx = np.argmax(points[:, 0])
        min_y_idx = np.argmin(points[:, 1])
        max_y_idx = np.argmax(points[:, 1])
        
        # Get the 4 extreme points
        extreme_points = points[[min_x_idx, max_x_idx, min_y_idx, max_y_idx]]
        
        # Remove duplicates and ensure we have 4 points
        unique_points = []
        for point in extreme_points:
            is_duplicate = False
            for existing in unique_points:
                if np.linalg.norm(point - existing) < 5:  # 5 pixel tolerance
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append(point)
        
        # If we don't have 4 unique points, add more from the polygon
        if len(unique_points) < 4:
            for point in points:
                is_duplicate = False
                for existing in unique_points:
                    if np.linalg.norm(point - existing) < 5:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_points.append(point)
                    if len(unique_points) == 4:
                        break
        
        return np.array(unique_points[:4])

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

    def plot_eval(self, img_path, ax=None, conf=0.25, show_polygon=True, show_centers=True):
        """
        Plot evaluation results with chessboard segmentation and polygon outline.
        
        Args:
            img_path: Path to the image to evaluate
            ax: Matplotlib axes to plot on (creates new if None)
            conf: Confidence threshold for predictions
            show_polygon: Whether to draw the chessboard polygon
            show_centers: Whether to show corner center points
            
        Returns:
            ax: The matplotlib axes used for plotting
        """
        results, board_count, is_valid = self.predict_board(img_path, conf=conf)
        
        # Print detection info
        print(f"Detected chessboards: {board_count}/1")
        if results.names:
            print("Detection target: chessboard")

        # Use provided axes or create a new one
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))
            show_inline = True
        else:
            show_inline = False

        # Convert BGR to RGB for proper matplotlib display
        image_rgb = cv2.cvtColor(results.orig_img, cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)

        # Draw segmentation mask and polygon
        if results.masks is not None and len(results.masks) > 0:
            mask = results.masks[0]
            
            # Draw the segmentation mask
            if hasattr(mask, 'data'):
                mask_data = mask.data[0].cpu().numpy()
                # Create a colored overlay for the mask
                overlay = np.zeros_like(image_rgb)
                overlay[:, :, 1] = mask_data * 255  # Green channel
                # Blend with original image
                alpha = 0.3
                image_rgb = image_rgb * (1 - alpha) + overlay * alpha
                ax.imshow(image_rgb.astype(np.uint8))
            
            # Draw polygon outline
            if hasattr(mask, 'xy') and len(mask.xy) > 0:
                polygon_coords = mask.xy[0]  # Get the polygon coordinates
                
                # Draw the full polygon
                if show_polygon:
                    polygon = Polygon(polygon_coords, fill=False, edgecolor='red', 
                                    linewidth=3, linestyle='-', alpha=0.8)
                    ax.add_patch(polygon)
                
                # Get and show the 4 corners
                try:
                    corners_dict, _ = self.get_corner_coordinates(img_path, conf=conf)
                    if corners_dict and show_centers:
                        # Draw the 4 corners
                        corner_names = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
                        colors = ['red', 'blue', 'green', 'orange']
                        
                        for i, (corner_name, color) in enumerate(zip(corner_names, colors)):
                            if corner_name in corners_dict:
                                corner = corners_dict[corner_name]
                                ax.plot(corner['x'], corner['y'], 'o', color=color, markersize=10, 
                                       markerfacecolor=color, markeredgecolor='white', markeredgewidth=2)
                                ax.text(corner['x'], corner['y'] - 20, f"{corner_name.replace('_', ' ').title()}", 
                                       color=color, fontsize=10, ha='center', 
                                       bbox=dict(facecolor='white', alpha=0.7))
                        
                        # Draw lines connecting the corners to show the quadrilateral
                        if show_polygon and len(corners_dict.get('corners_array', [])) == 4:
                            corner_points = corners_dict['corners_array']
                            corner_polygon = Polygon(corner_points, fill=False, edgecolor='blue', 
                                                   linewidth=2, linestyle='--', alpha=0.7)
                            ax.add_patch(corner_polygon)
                            print("✅ Drew chessboard 4-corner outline")
                        
                except Exception as e:
                    print(f"⚠️  Could not extract 4 corners: {e}")
                
                # Add confidence info
                if results.boxes is not None and len(results.boxes) > 0:
                    conf_val = float(results.boxes[0].conf[0].cpu().numpy())
                    ax.text(10, 30, f"Confidence: {conf_val:.2f}", color="white", fontsize=12,
                           bbox=dict(facecolor="black", alpha=0.7))

        # Set title based on detection quality
        if is_valid:
            ax.set_title("✅ Chessboard Segmentation - Board detected", color='green', fontsize=14)
        else:
            ax.set_title(f"⚠️  Chessboard Segmentation - {board_count} boards detected (expected 1)", 
                        color='orange', fontsize=14)

        ax.axis("off")
        
        if show_inline:
            self._display_plot_inline(ax=ax)
            
        return ax


if __name__ == "__main__":
    # Example usage
    print("ChessBoardModel - Corner Detection Example")
    print("=" * 50)
    
    # This would be used like:
    # model = ChessBoardModel(model_path=Path("path/to/corner_detection_model.pt"))
    # results, count, is_valid = model.predict_corners("path/to/chessboard_image.jpg")
    # coordinates, is_valid = model.get_corner_coordinates("path/to/chessboard_image.jpg")
    # model.plot_eval("path/to/chessboard_image.jpg")
    
    print("To use this model:")
    print("1. Train a YOLO model on chessboard corner data")
    print("2. Load the model: model = ChessBoardModel(model_path=Path('best.pt'))")
    print("3. Predict corners: results, count, is_valid = model.predict_corners('image.jpg')")
    print("4. Get coordinates: coords, is_valid = model.get_corner_coordinates('image.jpg')")
    print("5. Visualize: model.plot_eval('image.jpg')") 