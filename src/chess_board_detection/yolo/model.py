import warnings

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")  # Change to non-interactive backend
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import (
    Polygon,
    Rectangle,
)  # Import Rectangle and Polygon from patches

from src.base_model import BaseYOLOModel


class ChessBoardModel(BaseYOLOModel):
    """Model for detecting chessboard corners using YOLO."""

    def __init__(
        self,
        model_path: Path | None = None,
        device: torch.device | None = None,
        pretrained_checkpoint: str = "yolo11s.pt",
    ):
        super().__init__(model_path, device, pretrained_checkpoint)
        self.expected_corners = 4

    def predict_corners(self, img_path, conf=0.25):
        """
        Predict chessboard corners and validate the count.

        Args:
            img_path: Path to the image to analyze
            conf: Confidence threshold for predictions

        Returns:
            tuple: (results, corner_count, is_valid)
                - results: YOLO prediction results
                - corner_count: Number of detected corners
                - is_valid: True if exactly 4 corners detected
        """
        results = self.predict(img_path, conf=conf)

        # Count detected corners
        corner_count = 0
        if results.boxes is not None:
            corner_count = len(results.boxes)

        # Validate corner count
        is_valid = corner_count == self.expected_corners

        if not is_valid:
            if corner_count < self.expected_corners:
                warnings.warn(
                    f"⚠️  Only {corner_count} corners detected (expected {self.expected_corners}). "
                    f"Try lowering the confidence threshold or check if the chessboard is fully visible.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"⚠️  {corner_count} corners detected (expected {self.expected_corners}). "
                    f"Try raising the confidence threshold or check for false positives.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            print(f"✅ Successfully detected {corner_count} chessboard corners")

        return results, corner_count, is_valid

    def get_corner_coordinates(self, img_path, conf=0.25):
        """
        Get the coordinates of detected corners.

        Args:
            img_path: Path to the image to analyze
            conf: Confidence threshold for predictions

        Returns:
            tuple: (coordinates, is_valid)
                - coordinates: List of (x, y) tuples for each corner center
                - is_valid: True if exactly 4 corners detected
        """
        results, corner_count, is_valid = self.predict_corners(img_path, conf=conf)

        coordinates = []
        if results.boxes is not None:
            for box in results.boxes:
                # Get box coordinates and calculate center
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Get confidence
                confidence = float(box.conf[0].cpu().numpy())

                coordinates.append(
                    {
                        "x": float(center_x),
                        "y": float(center_y),
                        "confidence": confidence,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    }
                )

        # Sort corners by confidence (highest first)
        coordinates.sort(key=lambda c: c["confidence"], reverse=True)

        return coordinates, is_valid

    def order_corners(self, coordinates):
        """
        Order corners in a consistent manner: top-left, top-right, bottom-right, bottom-left.

        Args:
            coordinates: List of coordinate dictionaries from get_corner_coordinates

        Returns:
            list: Ordered coordinates [top-left, top-right, bottom-right, bottom-left]
        """
        if len(coordinates) != 4:
            warnings.warn(
                f"Cannot order {len(coordinates)} corners. Need exactly 4 corners.",
                UserWarning,
                stacklevel=2,
            )
            return coordinates

        # Convert to numpy array for easier manipulation
        points = np.array([[c["x"], c["y"]] for c in coordinates])

        # Order points: top-left, top-right, bottom-right, bottom-left
        # Sum and difference will help us identify corners
        sum_coords = points.sum(axis=1)
        diff_coords = np.diff(points, axis=1).flatten()

        # Top-left has smallest sum, bottom-right has largest sum
        top_left_idx = np.argmin(sum_coords)
        bottom_right_idx = np.argmax(sum_coords)

        # Top-right has smallest difference (x-y), bottom-left has largest difference
        top_right_idx = np.argmin(diff_coords)
        bottom_left_idx = np.argmax(diff_coords)

        ordered_coordinates = [
            coordinates[top_left_idx],  # top-left
            coordinates[top_right_idx],  # top-right
            coordinates[bottom_right_idx],  # bottom-right
            coordinates[bottom_left_idx],  # bottom-left
        ]

        return ordered_coordinates

    def plot_eval(self, img_path, ax=None, conf=0.25, show_polygon=True, show_centers=True):
        """
        Plot evaluation results with corner detections and optional chessboard outline.

        Args:
            img_path: Path to the image to evaluate
            ax: Matplotlib axes to plot on (creates new if None)
            conf: Confidence threshold for predictions
            show_polygon: Whether to draw lines connecting the corners
            show_centers: Whether to show corner center points

        Returns:
            ax: The matplotlib axes used for plotting
        """
        results, corner_count, is_valid = self.predict_corners(img_path, conf=conf)

        # Print detection info
        print(f"Detected corners: {corner_count}/4")
        if results.names:
            print("Detection target: chessboard corners")

        # Use provided axes or create a new one
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))
            show_inline = True
        else:
            show_inline = False

        # Convert BGR to RGB for proper matplotlib display
        image_rgb = cv2.cvtColor(results.orig_img, cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)

        # Draw corner detections
        corner_coords = []
        if results.boxes is not None:
            for i, box in enumerate(results.boxes):
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                corner_coords.append([center_x, center_y])

                # Get confidence
                conf_val = float(box.conf[0].cpu().numpy())
                label = f"Corner {i + 1} ({conf_val:.2f})"

                # Draw bounding box
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

                # Show center point if requested
                if show_centers:
                    ax.plot(
                        center_x,
                        center_y,
                        "ro",
                        markersize=8,
                        markerfacecolor="yellow",
                        markeredgecolor="red",
                        markeredgewidth=2,
                    )

        # Draw polygon connecting corners if we have exactly 4 and show_polygon is True
        if show_polygon and len(corner_coords) == 4:
            try:
                # Order the corners properly
                coordinates, _ = self.get_corner_coordinates(img_path, conf=conf)
                ordered_corners = self.order_corners(coordinates)

                # Extract ordered points
                ordered_points = [[c["x"], c["y"]] for c in ordered_corners]

                # Create polygon
                polygon = Polygon(
                    ordered_points,
                    fill=False,
                    edgecolor="blue",
                    linewidth=3,
                    linestyle="--",
                    alpha=0.7,
                )
                ax.add_patch(polygon)

                print("✅ Drew chessboard outline connecting ordered corners")

            except Exception as e:
                # Fallback: just connect corners in detection order
                polygon = Polygon(
                    corner_coords,
                    fill=False,
                    edgecolor="orange",
                    linewidth=2,
                    linestyle="--",
                    alpha=0.5,
                )
                ax.add_patch(polygon)
                print(f"⚠️  Drew outline in detection order (couldn't order properly: {e})")

        # Set title based on detection quality
        if is_valid:
            ax.set_title("✅ Chessboard Corner Detection - 4 corners detected", color="green")
        else:
            ax.set_title(
                f"⚠️  Chessboard Corner Detection - {corner_count} corners detected (expected 4)",
                color="orange",
            )

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
