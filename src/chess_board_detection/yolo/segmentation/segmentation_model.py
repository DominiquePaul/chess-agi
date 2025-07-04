#!/usr/bin/env python3
"""
ChessBoard Segmentation Model using YOLOv11

This model learns to segment the exact polygon boundaries of chessboards
from polygon annotation data.
"""

import warnings

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")  # Change to non-interactive backend
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from matplotlib.patches import Polygon

from src.base_model import BaseYOLOModel


class ChessBoardSegmentationModel(BaseYOLOModel):
    """Model for segmenting chessboard polygons using YOLOv11 segmentation."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        device: torch.device | None = None,
        pretrained_model: str = "yolo11n-seg.pt",
    ):
        """
        Initialize ChessBoard Segmentation Model.

        Args:
            model_path: Path to trained model weights, HuggingFace model name, or None
            device: Device to run model on. If None, auto-detects
            pretrained_model: Pretrained model to use when model_path is None. Options:
                - 'yolo11n-seg.pt' (nano, fastest, least accurate)
                - 'yolo11s-seg.pt' (small)
                - 'yolo11m-seg.pt' (medium)
                - 'yolo11l-seg.pt' (large)
                - 'yolo11x-seg.pt' (extra large, slowest, most accurate)
        """

        # Handle HuggingFace Hub models
        if isinstance(model_path, str) and "/" in model_path and not Path(model_path).exists():
            # This looks like a HuggingFace model name
            model_path = self._load_from_huggingface(model_path)

        # Initialize with segmentation model instead of detection
        if model_path is None:
            # Load pretrained YOLOv11 segmentation model (COCO pretrained)
            valid_models = [
                "yolo11n-seg.pt",
                "yolo11s-seg.pt",
                "yolo11m-seg.pt",
                "yolo11l-seg.pt",
                "yolo11x-seg.pt",
            ]
            if pretrained_model not in valid_models:
                print(f"⚠️  Invalid pretrained_model '{pretrained_model}'. Using default 'yolo11n-seg.pt'")
                pretrained_model = "yolo11n-seg.pt"

            print(f"📥 Loading COCO-pretrained model: {pretrained_model}")

        super().__init__(model_path, device, pretrained_model)

        print(
            f"🧠 ChessBoard Segmentation Model initialized on {self.device} - loaded from {'cache' if self.loaded_from_cache else 'HF Hub'}"
        )

    def predict_segments(
        self,
        img: np.ndarray | str | Path,
        conf: float = 0.25,
        iou: float = 0.7,
        max_segments: int = 1,
    ):
        """
        Predict chessboard segmentation masks.

        Args:
            img: Image as numpy array, or path to the image to analyze
            conf: Confidence threshold for predictions
            iou: IoU threshold for NMS
            max_segments: Maximum number of chessboard segments to return (keeps most confident)

        Returns:
            tuple: (results, segment_count, is_valid)
                - results: YOLO prediction results with masks (limited to max_segments)
                - segment_count: Number of segments after limiting to max_segments
                - is_valid: True if any segments detected and within max_segments limit
        """
        original_results = self.model.predict(img, conf=conf, iou=iou)[0]

        # Count originally detected segments
        original_segment_count = 0
        if original_results.masks is not None:
            original_segment_count = len(original_results.masks)

        # If no segments detected, return as-is
        if original_segment_count == 0:
            warnings.warn(
                "⚠️  No chessboard segments detected. "
                "Try lowering the confidence threshold or check if the chessboard is visible.",
                UserWarning,
                stacklevel=2,
            )
            return original_results, 0, False

        # If we have more segments than max_segments, keep only the most confident ones
        if original_segment_count > max_segments:
            print(f"🔍 Found {original_segment_count} segments, keeping top {max_segments} by confidence")

            # Get confidence scores for each detection
            confidences = []
            if original_results.boxes is not None:
                # Access boxes properly - they are iterable
                boxes_data = original_results.boxes.data
                for i in range(len(boxes_data)):
                    conf_score = float(boxes_data[i][4].cpu().numpy())  # Confidence is at index 4
                    confidences.append((i, conf_score))
            else:
                # If no boxes, use mask areas as proxy for confidence
                if original_results.masks is not None and hasattr(original_results.masks, "data"):
                    for i in range(len(original_results.masks.data)):
                        mask_array = original_results.masks.data[i].cpu().numpy()
                        area = np.sum(mask_array > 0)
                        confidences.append((i, area))

            # Sort by confidence (highest first) and keep top max_segments
            confidences.sort(key=lambda x: x[1], reverse=True)
            keep_indices = [idx for idx, _ in confidences[:max_segments]]
            keep_indices.sort()  # Keep original order

            # Create new results with filtered data
            from copy import deepcopy

            filtered_results = deepcopy(original_results)

            if original_results.masks is not None and hasattr(original_results.masks, "data"):
                # Keep only selected masks
                selected_mask_data = original_results.masks.data[keep_indices]
                if filtered_results.masks is not None:
                    filtered_results.masks.data = selected_mask_data

            if original_results.boxes is not None and hasattr(original_results.boxes, "data"):
                # Keep only selected boxes
                selected_box_data = original_results.boxes.data[keep_indices]
                if filtered_results.boxes is not None:
                    filtered_results.boxes.data = selected_box_data

            results = filtered_results
            segment_count = len(keep_indices)
        else:
            # We have max_segments or fewer, use all
            results = original_results
            segment_count = original_segment_count

        # Now validation is simple - we have valid results if we have any segments
        is_valid = segment_count > 0

        if is_valid:
            print(f"✅ Successfully detected {segment_count} chessboard segment(s) (max allowed: {max_segments})")
            if original_segment_count > max_segments:
                print(f"   Note: Selected top {segment_count} from {original_segment_count} total detections")

        return results, segment_count, is_valid

    def get_polygon_coordinates(
        self,
        img: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.7,
        max_segments: int = 1,
    ):
        """
        Get the polygon coordinates of the detected chessboard(s).

        Args:
            img: Image as numpy array, or path to the image to analyze
            conf: Confidence threshold for predictions
            iou: IoU threshold for NMS
            max_segments: Maximum number of chessboard segments to return (keeps most confident)

        Returns:
            tuple: (polygon_info, is_valid)
                - polygon_info: Dictionary with polygon coordinates and metadata (in original image space)
                  For single segment: returns dict with coordinates, confidence, etc.
                  For multiple segments: returns dict with 'segments' key containing list of segment dicts
                - is_valid: True if any segments detected (after filtering to max_segments)
        """
        results, segment_count, is_valid = self.predict_segments(img, conf=conf, iou=iou, max_segments=max_segments)

        polygon_info = {}

        if results.masks is not None and len(results.masks) > 0:
            # Get original image dimensions
            if hasattr(results, "orig_img") and results.orig_img is not None:
                orig_height, orig_width = results.orig_img.shape[:2]
            else:
                # Fallback: load image directly to get dimensions
                if isinstance(img, np.ndarray):
                    orig_height, orig_width = img.shape[:2]

            segments = []

            # Process each detected mask
            for mask_idx in range(len(results.masks.data)):
                segment_info = {}
                mask = results.masks[mask_idx]

                # Get the polygon coordinates from the mask
                if hasattr(mask, "xy") and len(mask.xy) > 0:
                    # YOLO11 provides polygon coordinates directly
                    polygon_coords = mask.xy[0]  # First contour

                    # Convert to list of coordinate dictionaries
                    coordinates = []
                    for point in polygon_coords:
                        # These coordinates should already be in original image space
                        # YOLO11 typically returns coordinates in original image dimensions
                        coordinates.append({"x": float(point[0]), "y": float(point[1])})

                    # Get confidence if available
                    confidence = 1.0
                    if results.boxes is not None and len(results.boxes) > mask_idx:
                        confidence = float(results.boxes.data[mask_idx][4].cpu().numpy())

                    segment_info = {
                        "coordinates": coordinates,
                        "confidence": confidence,
                        "num_points": len(coordinates),
                        "area": self._calculate_polygon_area(coordinates),
                        "segment_id": mask_idx,
                    }
                else:
                    # Fallback: extract contours from mask
                    mask_array = mask.data[0].cpu().numpy().astype(np.uint8)

                    # If we have original dimensions, resize mask to original size first
                    if orig_width is not None and orig_height is not None:
                        mask_array = cv2.resize(
                            mask_array,
                            (orig_width, orig_height),
                            interpolation=cv2.INTER_NEAREST,
                        )

                    contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours:
                        # Get the largest contour
                        largest_contour = max(contours, key=cv2.contourArea)

                        # Convert contour to coordinates
                        coordinates = []
                        for point in largest_contour.reshape(-1, 2):
                            coordinates.append({"x": float(point[0]), "y": float(point[1])})

                        confidence = 1.0
                        if results.boxes is not None and len(results.boxes) > mask_idx:
                            confidence = float(results.boxes.data[mask_idx][4].cpu().numpy())

                        segment_info = {
                            "coordinates": coordinates,
                            "confidence": confidence,
                            "num_points": len(coordinates),
                            "area": cv2.contourArea(largest_contour),
                            "segment_id": mask_idx,
                        }

                if segment_info:
                    segments.append(segment_info)

            # Format return value based on number of segments
            if len(segments) == 1:
                # Single segment: return as before for backward compatibility
                polygon_info = segments[0]
                # Remove segment_id for backward compatibility
                polygon_info.pop("segment_id", None)
            elif len(segments) > 1:
                # Multiple segments: return in new format
                polygon_info = {
                    "segments": segments,
                    "num_segments": len(segments),
                    "total_area": sum(s.get("area", 0) for s in segments),
                    "best_confidence": max(s.get("confidence", 0) for s in segments) if segments else 0.0,
                }

        return polygon_info, is_valid

    def _calculate_polygon_area(self, coordinates):
        """Calculate the area of a polygon using the shoelace formula."""
        if len(coordinates) < 3:
            return 0.0

        x = [coord["x"] for coord in coordinates]
        y = [coord["y"] for coord in coordinates]

        return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(x) - 1)))

    def plot_eval(
        self,
        img: np.ndarray | str | Path,
        ax=None,
        conf=0.25,
        iou=0.7,
        show_polygon=True,
        show_mask=True,
        alpha=0.5,
        max_segments=1,
    ):
        """
        Plot evaluation results with segmentation masks and polygon outlines.

        Args:
            img: Image as numpy array, or path to the image to evaluate
            ax: Matplotlib axes to plot on (creates new if None)
            conf: Confidence threshold for predictions
            iou: IoU threshold for NMS
            show_polygon: Whether to draw polygon outline
            show_mask: Whether to show segmentation mask
            alpha: Transparency for mask overlay
            max_segments: Maximum number of chessboard segments to display (keeps most confident)

        Returns:
            ax: The matplotlib axes used for plotting
        """
        results, segment_count, is_valid = self.predict_segments(img, conf=conf, iou=iou, max_segments=max_segments)
        polygon_info, _ = self.get_polygon_coordinates(img, conf=conf, iou=iou, max_segments=max_segments)

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

        # Get the original image dimensions - this is crucial for proper coordinate alignment
        if hasattr(results, "orig_img") and results.orig_img is not None:
            image_bgr = results.orig_img
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            orig_height, orig_width = image_bgr.shape[:2]
        else:
            # Fallback: load or use provided image directly
            if isinstance(img, np.ndarray):
                image_rgb = img if len(img.shape) == 3 and img.shape[2] == 3 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                orig_height, orig_width = img.shape[:2]
            else:
                image_bgr = cv2.imread(str(img))
                if image_bgr is None:
                    raise ValueError(f"Could not load image: {img}")
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                orig_height, orig_width = image_bgr.shape[:2]

        # Set up the plot with correct aspect ratio
        ax.imshow(image_rgb)
        ax.set_xlim(0, orig_width)
        ax.set_ylim(orig_height, 0)  # Invert y-axis for image coordinates
        ax.set_aspect("equal")

        # Draw segmentation results
        if results.masks is not None and show_mask:
            for i in range(len(results.masks.data)):
                # Get mask array - this is in the processed image space
                mask_array = results.masks.data[i].cpu().numpy()

                # Resize mask to match original image dimensions
                mask_resized = cv2.resize(
                    mask_array.astype(np.uint8),
                    (orig_width, orig_height),
                    interpolation=cv2.INTER_NEAREST,
                )

                # Create colored mask with better visibility - use different colors for multiple segments
                colored_mask = np.zeros((orig_height, orig_width, 3))
                if i == 0:
                    colored_mask[mask_resized > 0] = [
                        0,
                        0.8,
                        0,
                    ]  # Green for first segment
                elif i == 1:
                    colored_mask[mask_resized > 0] = [
                        0,
                        0,
                        0.8,
                    ]  # Blue for second segment
                elif i == 2:
                    colored_mask[mask_resized > 0] = [
                        0.8,
                        0.8,
                        0,
                    ]  # Yellow for third segment
                else:
                    # Cycle through colors for additional segments
                    colors = [
                        [0.8, 0, 0.8],
                        [0.8, 0.4, 0],
                        [0.4, 0.8, 0.8],
                    ]  # Purple, Orange, Cyan
                    color_idx = (i - 3) % len(colors)
                    colored_mask[mask_resized > 0] = colors[color_idx]

                # Overlay mask with better visibility
                ax.imshow(colored_mask, alpha=alpha, interpolation="nearest")

                # Get confidence - handle both single and multiple segment formats
                if "segments" in polygon_info:
                    # Multiple segments format
                    conf_val = (
                        polygon_info["segments"][i].get("confidence", 1.0) if i < len(polygon_info["segments"]) else 1.0
                    )
                else:
                    # Single segment format
                    conf_val = polygon_info.get("confidence", 1.0)

                segment_colors = ["green", "blue", "orange", "purple", "red", "cyan"]
                text_color = segment_colors[i % len(segment_colors)]

                ax.text(
                    15,
                    40 + i * 40,
                    f"Chessboard Segment {i + 1} (conf: {conf_val:.3f})",
                    color="white",
                    fontsize=14,
                    weight="bold",
                    bbox={
                        "facecolor": text_color,
                        "alpha": 0.8,
                        "edgecolor": "white",
                        "linewidth": 2,
                    },
                )

        # Draw polygon outline with better visibility
        # Handle both single and multiple segment formats
        segments_to_draw = []
        if "segments" in polygon_info:
            # Multiple segments format
            segments_to_draw = polygon_info["segments"]
        elif "coordinates" in polygon_info:
            # Single segment format
            segments_to_draw = [polygon_info]

        if show_polygon and segments_to_draw:
            polygon_colors = ["red", "blue", "orange", "purple", "green", "cyan"]

            for i, segment in enumerate(segments_to_draw):
                coordinates = segment.get("coordinates", [])
                if len(coordinates) >= 3:
                    # The polygon coordinates should now be in original image space
                    polygon_points = [[coord["x"], coord["y"]] for coord in coordinates]

                    # Use different colors for multiple segments
                    polygon_color = polygon_colors[i % len(polygon_colors)]

                    # Draw polygon outline with better visibility
                    polygon = Polygon(
                        polygon_points,
                        fill=False,
                        edgecolor=polygon_color,
                        linewidth=4,
                        linestyle="-",
                        alpha=1.0,
                    )
                    ax.add_patch(polygon)

                    # Add area information with better visibility
                    area = segment.get("area", 0)
                    ax.text(
                        15,
                        orig_height - 50 - i * 30,
                        f"Polygon {i + 1} Area: {area:.0f} pixels²",
                        color="white",
                        fontsize=12,
                        weight="bold",
                        bbox={
                            "facecolor": polygon_color,
                            "alpha": 0.8,
                            "edgecolor": "white",
                            "linewidth": 2,
                        },
                    )

                    print(f"✅ Drew chessboard polygon outline {i + 1} ({len(coordinates)} points)")

        # Set title based on detection quality
        if is_valid:
            if segment_count == 1:
                ax.set_title(
                    "DETECTED: Chessboard Segmentation - Successfully detected",
                    color="green",
                    fontsize=14,
                )
            else:
                ax.set_title(
                    f"DETECTED: Chessboard Segmentation - {segment_count} segments detected",
                    color="green",
                    fontsize=14,
                )
        else:
            ax.set_title(
                f"WARNING: Chessboard Segmentation - {segment_count} segments detected (max: {max_segments})",
                color="orange",
                fontsize=14,
            )

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
        print("🚀 Starting chessboard segmentation training...")
        print(f"📊 Dataset: {data_path}")
        print(f"🔄 Epochs: {epochs}, Batch: {batch}, Image size: {imgsz}")

        # Train the model
        results = self.model.train(data=str(data_path), epochs=epochs, batch=batch, imgsz=imgsz, **kwargs)

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
            with open(data_yaml_path) as f:
                data_config = yaml.safe_load(f)

            # Check required fields
            required_fields = ["train", "val", "nc", "names"]
            for field in required_fields:
                if field not in data_config:
                    return False, f"Missing required field: {field}"

            # Check if train directory exists
            train_path = Path(data_yaml_path.parent) / data_config["train"]
            if not train_path.exists():
                return False, f"Training images directory not found: {train_path}"

            # Check labels directory
            labels_dir = train_path.parent / "labels"
            if not labels_dir.exists():
                return False, f"Labels directory not found: {labels_dir}"

            # Sample a few label files to check format
            label_files = list(labels_dir.glob("*.txt"))[:5]
            polygon_count = 0

            for label_file in label_files:
                with open(label_file) as f:
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

            return (
                True,
                f"Valid segmentation dataset with {polygon_count} sample polygons found",
            )

        except Exception as e:
            return False, f"Error validating dataset: {e}"

    def get_training_summary(self, results):
        """Print a summary of training results."""
        print("\n" + "=" * 70)
        print("🎯 SEGMENTATION TRAINING COMPLETE")
        print("=" * 70)

        if hasattr(results, "results_dir"):
            print(f"📁 Results saved to: {results.results_dir}")

        # Print key metrics if available
        if hasattr(results, "metrics"):
            metrics = results.metrics
            print("📊 Final Metrics:")
            if hasattr(metrics, "seg"):
                seg_metrics = metrics.seg
                print(f"   🎯 mAP50 (Segmentation): {getattr(seg_metrics, 'map50', 'N/A'):.3f}")
                print(f"   🎯 mAP50-95 (Segmentation): {getattr(seg_metrics, 'map', 'N/A'):.3f}")
            if hasattr(metrics, "box"):
                box_metrics = metrics.box
                print(f"   📦 mAP50 (Detection): {getattr(box_metrics, 'map50', 'N/A'):.3f}")
                print(f"   📦 mAP50-95 (Detection): {getattr(box_metrics, 'map', 'N/A'):.3f}")

        print("=" * 70)

    def extract_corners_from_segmentation(
        self, img: np.ndarray | str | Path, polygon_info, method="hybrid", debug=False
    ):
        """
        Extract precise corner coordinates from segmentation mask using various post-processing methods.

        Args:
            img: Image as numpy array, or path to the original image
            polygon_info: Polygon information from get_polygon_coordinates
            method: Corner extraction method - 'approx', 'lines', 'hybrid', 'harris', or 'contour'
            debug: Whether to return debug information and intermediate results

        Returns:
            tuple: (corners, debug_info)
                - corners: List of 4 corner dictionaries [top_left, top_right, bottom_right, bottom_left]
                - debug_info: Dictionary with intermediate results (if debug=True)
        """
        import cv2
        import numpy as np

        # Load or use provided image
        if isinstance(img, np.ndarray):
            image = img.copy()
        else:
            image = cv2.imread(str(img))
            if image is None:
                raise ValueError(f"Could not load image: {img}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # Get segmentation mask
        mask = self._create_mask_from_polygon(polygon_info, width, height)

        debug_info = {
            "method": method,
            "original_polygon": polygon_info.get("coordinates", []),
        }

        if method == "approx":
            corners = self._extract_corners_polygon_approximation(mask, debug_info)
        elif method == "lines":
            corners = self._extract_corners_line_intersection(mask, debug_info)
        elif method == "hybrid":
            corners = self._extract_corners_hybrid(image, mask, debug_info)
        elif method == "harris":
            corners = self._extract_corners_harris(gray, mask, debug_info)
        elif method == "contour":
            corners = self._extract_corners_contour_analysis(mask, debug_info)
        elif method == "extended":
            corners = self._extract_corners_extended_lines(mask, debug_info)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Order corners consistently: top_left, top_right, bottom_right, bottom_left
        if corners and len(corners) == 4:
            corners = self._order_corners_quadrilateral(corners)

        return corners, debug_info if debug else None

    def _create_mask_from_polygon(self, polygon_info, width, height):
        """Create a binary mask from polygon coordinates."""
        mask = np.zeros((height, width), dtype=np.uint8)

        if "coordinates" in polygon_info:
            # Single segment
            coords = polygon_info["coordinates"]
            points = np.array([[int(c["x"]), int(c["y"])] for c in coords], dtype=np.int32)
            cv2.fillPoly(mask, [points], (255,))  # Fix: use tuple for color
        elif "segments" in polygon_info:
            # Multiple segments - use the first one
            coords = polygon_info["segments"][0]["coordinates"]
            points = np.array([[int(c["x"]), int(c["y"])] for c in coords], dtype=np.int32)
            cv2.fillPoly(mask, [points], (255,))  # Fix: use tuple for color

        return mask

    def _extract_corners_polygon_approximation(self, mask, debug_info):
        """Extract corners using Douglas-Peucker polygon approximation."""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate polygon - try different epsilon values to get 4 corners
        for epsilon_factor in [0.01, 0.02, 0.03, 0.05, 0.08, 0.1]:
            epsilon = epsilon_factor * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            if len(approx) == 4:
                corners = [{"x": float(p[0][0]), "y": float(p[0][1])} for p in approx]
                debug_info["epsilon_used"] = epsilon_factor
                debug_info["approximated_polygon"] = corners
                return corners

        # Fallback: use convex hull and find 4 extreme points
        hull = cv2.convexHull(largest_contour)
        if len(hull) >= 4:
            # Find 4 corner points using extreme points method
            corners = self._find_extreme_points(hull)
            debug_info["used_convex_hull"] = True
            return corners

        return []

    def _extract_corners_line_intersection(self, mask, debug_info):
        """Extract corners by fitting lines to edges and finding intersections."""
        # Edge detection on mask
        edges = cv2.Canny(mask, 50, 150)

        # Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

        if lines is None or len(lines) < 4:
            return []

        # Group lines by angle to find the 4 sides of the rectangle
        line_groups = self._group_lines_by_angle(lines)

        if len(line_groups) < 2:
            return []

        # Find intersections between perpendicular line groups
        corners = self._find_line_intersections(line_groups)
        debug_info["detected_lines"] = len(lines)
        debug_info["line_groups"] = len(line_groups)

        return corners

    def _extract_corners_hybrid(self, image: np.ndarray, mask: np.ndarray, debug_info: dict):
        """Hybrid approach: use mask to focus edge detection on chessboard region."""
        # Create region of interest from mask
        kernel = np.ones((20, 20), np.uint8)  # Fix: create explicit kernel
        roi_mask = cv2.dilate(mask, kernel, iterations=1)

        # Apply edge detection only to the ROI
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray, gray, mask=roi_mask)

        # Enhanced edge detection
        edges = cv2.Canny(masked_gray, 50, 150, apertureSize=3)

        # Combine with mask edges
        mask_edges = cv2.Canny(mask, 50, 150)
        combined_edges = cv2.bitwise_or(edges, mask_edges)

        # Hough line detection on combined edges
        lines = cv2.HoughLinesP(
            combined_edges,
            1,
            np.pi / 180,
            threshold=30,
            minLineLength=30,
            maxLineGap=15,
        )

        if lines is None or len(lines) < 4:
            # Fallback to polygon approximation
            return self._extract_corners_polygon_approximation(mask, debug_info)

        # Group lines and find intersections
        line_groups = self._group_lines_by_angle(lines)
        corners = self._find_line_intersections(line_groups)

        debug_info["hybrid_lines"] = len(lines)
        debug_info["used_hybrid"] = True

        return corners

    def _extract_corners_harris(self, gray, mask, debug_info):
        """Extract corners using Harris corner detection."""
        # Apply Harris corner detection to the masked region
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

        # Harris corner detection
        corners_harris = cv2.cornerHarris(masked_gray, 2, 3, 0.04)

        # Dilate corner image to enhance corner points
        kernel = np.ones((3, 3), np.uint8)  # Fix: create explicit kernel
        corners_harris = cv2.dilate(corners_harris, kernel)

        # Threshold for strong corners
        threshold = 0.01 * corners_harris.max()
        corner_coords = np.where(corners_harris > threshold)

        if len(corner_coords[0]) < 4:
            return []

        # Convert to (x, y) format
        corner_points = list(zip(corner_coords[1], corner_coords[0], strict=False))

        # Cluster nearby points and take cluster centers
        try:
            from sklearn.cluster import DBSCAN

            corner_points_array = np.array(corner_points)  # Fix: convert to numpy array
            clustering = DBSCAN(eps=30, min_samples=1).fit(corner_points_array)

            cluster_centers = []
            for label in set(clustering.labels_):
                cluster_points = [
                    corner_points[i] for i, label_val in enumerate(clustering.labels_) if label_val == label
                ]
                center_x = np.mean([p[0] for p in cluster_points])
                center_y = np.mean([p[1] for p in cluster_points])
                cluster_centers.append({"x": float(center_x), "y": float(center_y)})

            debug_info["harris_corners_found"] = len(corner_coords[0])
            debug_info["clustered_corners"] = len(cluster_centers)

            # Return best 4 corners if we have more than 4
            if len(cluster_centers) > 4:
                # Sort by distance from mask center and take cluster centers
                return self._select_best_4_corners(cluster_centers, mask)

            return cluster_centers
        except ImportError:
            # Fallback if sklearn is not available
            debug_info["error"] = "sklearn not available for clustering"
            return []

    def _extract_corners_contour_analysis(self, mask, debug_info):
        """Extract corners using contour curvature analysis."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return []

        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate curvature along the contour
        contour_points = largest_contour.reshape(-1, 2)

        # Find points with high curvature (corners)
        corner_indices = self._find_high_curvature_points(contour_points)

        corners = [{"x": float(contour_points[i][0]), "y": float(contour_points[i][1])} for i in corner_indices]

        debug_info["curvature_corners"] = len(corner_indices)

        return corners

    def _order_corners_quadrilateral(self, corners):
        """Order corners consistently as [top_left, top_right, bottom_right, bottom_left]."""
        if len(corners) != 4:
            return corners

        # Convert to numpy array for easier manipulation
        points = np.array([[c["x"], c["y"]] for c in corners])

        # Find center point
        center = np.mean(points, axis=0)

        # Sort by angle from center
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        sorted_indices = np.argsort(angles)

        # Reorder to start from top-left (assuming angles are sorted)
        # This gives us a consistent ordering
        ordered_points = points[sorted_indices]

        # Convert back to dictionary format with proper labels
        ordered_corners = []
        for point in ordered_points:
            ordered_corners.append({"x": float(point[0]), "y": float(point[1])})

        return ordered_corners

    def _find_extreme_points(self, hull):
        """Find 4 extreme points from convex hull."""
        hull_points = hull.reshape(-1, 2)

        # Find extreme points
        top_left = hull_points[np.argmin(hull_points[:, 0] + hull_points[:, 1])]
        top_right = hull_points[np.argmax(hull_points[:, 0] - hull_points[:, 1])]
        bottom_right = hull_points[np.argmax(hull_points[:, 0] + hull_points[:, 1])]
        bottom_left = hull_points[np.argmin(hull_points[:, 0] - hull_points[:, 1])]

        return [
            {"x": float(top_left[0]), "y": float(top_left[1])},
            {"x": float(top_right[0]), "y": float(top_right[1])},
            {"x": float(bottom_right[0]), "y": float(bottom_right[1])},
            {"x": float(bottom_left[0]), "y": float(bottom_left[1])},
        ]

    def _find_high_curvature_points(self, contour_points, window=10):
        """Find points with high curvature along contour."""
        if len(contour_points) < window * 2:
            return []

        curvatures = []
        for i in range(window, len(contour_points) - window):
            # Calculate curvature using neighboring points
            p1 = contour_points[i - window]
            p2 = contour_points[i]
            p3 = contour_points[i + window]

            # Calculate angle change
            v1 = p2 - p1
            v2 = p3 - p2

            angle1 = np.arctan2(v1[1], v1[0])
            angle2 = np.arctan2(v2[1], v2[0])

            angle_diff = abs(angle1 - angle2)
            angle_diff = min(angle_diff, 2 * np.pi - angle_diff)

            curvatures.append((i, angle_diff))

        # Find peaks in curvature
        curvatures.sort(key=lambda x: x[1], reverse=True)

        # Take top 4 curvature points that are well separated
        corner_indices = []
        min_distance = len(contour_points) // 8  # Minimum distance between corners

        for idx, _curvature in curvatures:
            if len(corner_indices) >= 4:
                break

            # Check if this point is far enough from existing corners
            too_close = False
            for existing_idx in corner_indices:
                if abs(idx - existing_idx) < min_distance:
                    too_close = True
                    break

            if not too_close:
                corner_indices.append(idx)

        return sorted(corner_indices)

    def _select_best_4_corners(self, corners, mask):
        """Select the best 4 corners from a larger set."""
        if len(corners) <= 4:
            return corners

        # Find mask centroid
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            h, w = mask.shape
            cx, cy = w / 2, h / 2

        # Calculate distances from center
        distances = []
        for corner in corners:
            dist = np.sqrt((corner["x"] - cx) ** 2 + (corner["y"] - cy) ** 2)
            distances.append((corner, dist))

        # Sort by distance and take 4 most extreme (farthest from center)
        distances.sort(key=lambda x: x[1], reverse=True)

        return [corner for corner, _ in distances[:4]]

    def _extract_corners_extended_lines(self, mask, debug_info):
        """
        Extract corners by extending lines from the approx method corners.

        This method:
        1. Uses Douglas-Peucker approximation to get 4 clean corner points
        2. Fits straight lines between adjacent corners (not to messy contour points)
        3. Extends these lines and finds their intersections for true geometric corners

        Args:
            mask (np.ndarray): Binary segmentation mask
            debug_info (dict): Dictionary to store debug information

        Returns:
            list: List of 4 corner dictionaries with 'x' and 'y' keys, or empty list if failed
        """
        try:
            # Step 1: Use polygon approximation to get 4 clean corners (same as approx method)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                debug_info["error"] = "No contours found"
                return []

            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) < 1000:
                debug_info["error"] = "Contour too small"
                return []

            # Simplify contour to 4 main corners using Douglas-Peucker
            epsilon_attempts = [0.02, 0.01, 0.03, 0.005, 0.04, 0.008, 0.05]
            simplified_corners = None

            for epsilon_ratio in epsilon_attempts:
                epsilon = epsilon_ratio * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)

                if len(approx) == 4:
                    simplified_corners = approx.reshape(-1, 2)
                    debug_info["epsilon_used"] = epsilon_ratio
                    break

            if simplified_corners is None:
                debug_info["error"] = "Could not simplify contour to 4 corners"
                return []

            # Step 2: For each side, get contour points between corners and exclude 10% on each side
            contour_points = largest_contour.reshape(-1, 2)
            side_lines = []

            for i in range(4):
                corner1 = simplified_corners[i]
                corner2 = simplified_corners[(i + 1) % 4]

                # Get all contour points between these corners
                side_points = self._get_contour_points_between(contour_points, corner1, corner2)

                if len(side_points) < 6:  # Need at least 6 points to exclude some on each side
                    debug_info["error"] = f"Not enough points on side {i}: {len(side_points)}"
                    return []

                # Decide whether to use exclusion or direct corner-to-corner line fitting
                if len(side_points) >= 20:
                    # Use your original approach: exclude a very small percentage on each side to avoid rounded corners
                    exclude_ratio = 0.02  # Just 2% - very conservative exclusion
                    exclude_count = max(1, int(len(side_points) * exclude_ratio))  # 2% on each side
                    middle_points = side_points[exclude_count:-exclude_count]
                    method_used = "contour_with_exclusion"

                    if len(middle_points) < 2:
                        debug_info["error"] = (
                            f"Not enough middle points on side {i} after excluding {exclude_ratio * 100}%"
                        )
                        return []

                    points_to_fit = middle_points
                else:
                    # Fallback: Use corner-to-corner line fitting for sides with few points
                    points_to_fit = [corner1.tolist(), corner2.tolist()]
                    method_used = "corner_to_corner"

                # Fit line to the chosen points
                line_params = self._fit_line_to_points(points_to_fit)
                if line_params is not None:
                    side_lines.append(line_params)
                    debug_info[f"side_{i}_corners"] = [
                        corner1.tolist(),
                        corner2.tolist(),
                    ]
                    debug_info[f"side_{i}_total_points"] = len(side_points)
                    debug_info[f"side_{i}_method"] = method_used
                    debug_info[f"side_{i}_points_used"] = len(points_to_fit)
                    if method_used == "contour_with_exclusion":
                        debug_info[f"side_{i}_excluded_points"] = exclude_count * 2
                        debug_info[f"side_{i}_middle_points"] = len(middle_points)
                        debug_info[f"side_{i}_exclude_ratio"] = exclude_ratio
                    debug_info[f"side_{i}_line_params"] = line_params
                else:
                    debug_info["error"] = f"Failed to fit line to side {i} using {method_used}"
                    return []

            if len(side_lines) != 4:
                debug_info["error"] = f"Expected 4 side lines, got {len(side_lines)}"
                return []

            # Step 3: Find intersections of adjacent lines
            corners = []
            mask_height, mask_width = mask.shape

            # More generous bounds since we're extending lines from good corner points
            margin = max(mask_width, mask_height) * 0.5
            min_x, max_x = -margin, mask_width + margin
            min_y, max_y = -margin, mask_height + margin

            for i in range(4):
                line1 = side_lines[i]
                line2 = side_lines[(i + 1) % 4]  # Adjacent line

                intersection = self._find_line_intersection(line1, line2)
                debug_info[f"intersection_{i}_attempt"] = {
                    "line1": line1,
                    "line2": line2,
                    "intersection": intersection.tolist() if intersection is not None else None,
                }

                if intersection is not None:
                    x, y = intersection[0], intersection[1]

                    # Check if intersection is within reasonable bounds
                    if min_x <= x <= max_x and min_y <= y <= max_y:
                        corners.append({"x": float(x), "y": float(y)})
                        debug_info[f"intersection_{i}_accepted"] = True
                    else:
                        debug_info["error"] = f"Intersection {i} out of bounds: ({x:.1f}, {y:.1f})"
                        debug_info["bounds"] = (
                            f"Valid range: X=[{min_x:.1f}, {max_x:.1f}], Y=[{min_y:.1f}, {max_y:.1f}]"
                        )
                        debug_info[f"intersection_{i}_rejected"] = f"Out of bounds: ({x:.1f}, {y:.1f})"
                        return []
                else:
                    debug_info["error"] = f"No intersection found between sides {i} and {(i + 1) % 4}"
                    return []

            # Step 4: Order corners consistently
            if len(corners) == 4:
                corners = self._order_corners_consistently(corners)
                debug_info["method_details"] = f"Extended lines from {len(simplified_corners)} approx corners"
                debug_info["bounds_used"] = f"X=[{min_x:.0f}, {max_x:.0f}], Y=[{min_y:.0f}, {max_y:.0f}]"
                debug_info["approx_corners"] = simplified_corners.tolist()

            return corners

        except Exception as e:
            debug_info["error"] = f"Extended lines method failed: {str(e)}"
            return []

    def _fit_line_to_points(self, points):
        """
        Fit a line to a set of points using least squares.

        Args:
            points (list): List of [x, y] points

        Returns:
            dict: Line parameters {'a': a, 'b': b, 'c': c} for ax + by + c = 0, or None if failed
        """
        if len(points) < 2:
            return None

        points = np.array(points)

        # For exactly 2 points, use direct line calculation
        if len(points) == 2:
            p1, p2 = points[0], points[1]

            # Check if points are identical
            if np.allclose(p1, p2):
                return None

            # Calculate line equation ax + by + c = 0
            # For line through (x1,y1) and (x2,y2):
            # (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            a = dy
            b = -dx
            c = dx * p1[1] - dy * p1[0]

            # Normalize to avoid numerical issues
            norm = np.sqrt(a * a + b * b)
            if norm > 1e-10:
                a /= norm
                b /= norm
                c /= norm

            return {"a": a, "b": b, "c": c}

        # Use SVD for robust line fitting with more than 2 points
        # Center the points
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid

        # SVD: points = U * S * V^T
        # The line direction is the first column of V (largest singular value)
        try:
            U, S, Vt = np.linalg.svd(centered_points.T)

            # Line direction vector (first column of V)
            direction = Vt[0]  # This is already normalized

            # Convert to line equation ax + by + c = 0
            # Normal vector to the line is perpendicular to direction
            normal = np.array([-direction[1], direction[0]])  # Rotate 90 degrees

            # a = normal[0], b = normal[1]
            a, b = normal[0], normal[1]

            # c = -(a*x0 + b*y0) where (x0, y0) is the centroid
            c = -(a * centroid[0] + b * centroid[1])

            return {"a": a, "b": b, "c": c}

        except np.linalg.LinAlgError:
            return None

    def _group_lines_by_angle(self, lines, angle_threshold=15):
        """Group lines by similar angles with improved clustering."""
        if lines is None:
            return []

        line_data = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate angle (normalize to 0-180 degrees)
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if angle < 0:
                angle += 180

            # Calculate line length (longer lines are more reliable)
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            line_data.append((line[0], angle, length))

        # Sort by line length (longer lines first - they're more reliable)
        line_data.sort(key=lambda x: x[2], reverse=True)

        # Group by angle using a more sophisticated approach
        groups = []
        used = [False] * len(line_data)

        for i, (line, angle, _length) in enumerate(line_data):
            if used[i]:
                continue

            # Start a new group with this line
            group = [line]
            group_angles = [angle]
            used[i] = True

            # Find all other lines with similar angles
            for j, (other_line, other_angle, _other_length) in enumerate(line_data):
                if used[j]:
                    continue

                # Calculate angle difference (handle 0-180 wrap-around)
                angle_diff = abs(angle - other_angle)
                angle_diff = min(angle_diff, 180 - angle_diff)

                if angle_diff < angle_threshold:
                    group.append(other_line)
                    group_angles.append(other_angle)
                    used[j] = True

            # Only keep groups with sufficient lines (filters noise)
            if len(group) >= 2:  # At least 2 lines to form a reliable edge
                groups.append(group)

        # Sort groups by total length (sum of all line lengths in group)
        def group_total_length(group):
            return sum(np.sqrt((line[2] - line[0]) ** 2 + (line[3] - line[1]) ** 2) for line in group)

        groups.sort(key=group_total_length, reverse=True)

        # For chessboards, we expect 4 main edge groups (top, bottom, left, right)
        # Keep the top 4 strongest groups
        return groups[:4]

    def _find_line_intersections(self, line_groups):
        """Find intersections between different line groups."""
        if len(line_groups) < 2:
            return []

        corners = []

        # Try all combinations of line groups
        for i in range(len(line_groups)):
            for j in range(i + 1, len(line_groups)):
                group1, group2 = line_groups[i], line_groups[j]

                # Find best representative lines from each group
                line1 = self._get_representative_line(group1)
                line2 = self._get_representative_line(group2)

                # Find intersection
                intersection = self._find_line_intersection(line1, line2)
                if intersection is not None:
                    corners.append({"x": float(intersection[0]), "y": float(intersection[1])})

        return corners[:4]  # Return at most 4 corners

    def _get_representative_line(self, line_group):
        """Get a representative line from a group of similar lines using robust fitting."""
        if not line_group:
            return None

        if len(line_group) == 1:
            return line_group[0]

        # Collect all points from all lines in the group
        points = []
        for line in line_group:
            x1, y1, x2, y2 = line
            points.extend([(x1, y1), (x2, y2)])

        points = np.array(points)

        # Fit a line through all points using least squares
        # This creates a more robust representative line
        if len(points) >= 2:
            # Use SVD to fit line through points
            mean_point = np.mean(points, axis=0)
            centered_points = points - mean_point

            # SVD to find principal direction
            U, S, Vt = np.linalg.svd(centered_points)
            direction = Vt[0]  # First principal component

            # Create a line segment along this direction
            # Extend far enough to ensure intersections are found
            line_length = 2000  # Large enough to extend beyond image bounds

            start_point = mean_point - direction * line_length
            end_point = mean_point + direction * line_length

            return [start_point[0], start_point[1], end_point[0], end_point[1]]

        # Fallback: use the longest line as representative
        longest_line = max(line_group, key=lambda line: np.sqrt((line[2] - line[0]) ** 2 + (line[3] - line[1]) ** 2))
        return longest_line

    def _find_line_intersection(self, line1, line2):
        """
        Find intersection point of two lines.

        Args:
            line1, line2 (dict): Line parameters {'a': a, 'b': b, 'c': c} for ax + by + c = 0

        Returns:
            np.ndarray: Intersection point [x, y], or None if lines are parallel
        """
        a1, b1, c1 = line1["a"], line1["b"], line1["c"]
        a2, b2, c2 = line2["a"], line2["b"], line2["c"]

        # Solve the system:
        # a1*x + b1*y + c1 = 0
        # a2*x + b2*y + c2 = 0

        # Using Cramer's rule
        det = a1 * b2 - a2 * b1

        if abs(det) < 1e-10:  # Lines are parallel
            return None

        x = (b1 * c2 - b2 * c1) / det
        y = (a2 * c1 - a1 * c2) / det

        return np.array([x, y])

    def _order_corners_consistently(self, corners):
        """Order corners consistently as [top_left, top_right, bottom_right, bottom_left]."""
        if len(corners) != 4:
            return corners

        # Convert to numpy array for easier manipulation
        points = np.array([[c["x"], c["y"]] for c in corners])

        # Find center point
        center = np.mean(points, axis=0)

        # Sort by angle from center
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        sorted_indices = np.argsort(angles)

        # Reorder to start from top-left (assuming angles are sorted)
        # This gives us a consistent ordering
        ordered_points = points[sorted_indices]

        # Convert back to dictionary format with proper labels
        ordered_corners = []
        for point in ordered_points:
            ordered_corners.append({"x": float(point[0]), "y": float(point[1])})

        return ordered_corners

    def _get_contour_points_between(self, contour_points, point1, point2):
        """
        Get contour points that lie between two endpoints along the contour.

        Args:
            contour_points (np.ndarray): All contour points
            point1, point2 (np.ndarray): Start and end points

        Returns:
            list: Points along the contour between point1 and point2
        """
        if len(contour_points) < 2:
            return []

        # Find indices of point1 and point2 in the contour
        distances1 = np.sum((contour_points - point1) ** 2, axis=1)
        distances2 = np.sum((contour_points - point2) ** 2, axis=1)

        idx1 = np.argmin(distances1)
        idx2 = np.argmin(distances2)

        # Get points between idx1 and idx2, handling the circular nature of contours
        if idx1 <= idx2:
            # Normal case: points are in order around the contour
            side_points = contour_points[idx1 : idx2 + 1]
        else:
            # Wrap around case: need to go from idx1 to end, then from start to idx2
            side_points = np.concatenate([contour_points[idx1:], contour_points[: idx2 + 1]])

        return side_points.tolist()
