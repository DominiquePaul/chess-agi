#!/usr/bin/env python3
"""
Inference example for ChessBoardModel - Corner Detection

This script demonstrates how to use the trained ChessBoardModel to detect
chessboard corners in images.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from src.chess_board_detection.yolo.model import ChessBoardModel


def test_corner_detection(model_path: Path, image_path: Path):
    """
    Test corner detection on a single image.

    Args:
        model_path: Path to the trained model (.pt file)
        image_path: Path to the test image
    """

    print("🎯 Testing Corner Detection")
    print("=" * 50)
    print(f"📱 Model: {model_path}")
    print(f"🖼️  Image: {image_path}")
    print("-" * 50)

    # Load the model
    try:
        model = ChessBoardModel(model_path=model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Check if image exists
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return

    # Test different confidence thresholds
    confidence_levels = [0.1, 0.25, 0.5, 0.7]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    for i, conf in enumerate(confidence_levels):
        print(f"\n🔍 Testing with confidence threshold: {conf}")

        # Predict corners
        try:
            results, corner_count, is_valid = model.predict_corners(image_path, conf=conf)
            coordinates, _ = model.get_corner_coordinates(image_path, conf=conf)

            print(f"   Detected corners: {corner_count}/4")
            print(f"   Valid detection: {'✅' if is_valid else '❌'}")

            if coordinates:
                print("   Corner coordinates:")
                for j, coord in enumerate(coordinates):
                    print(
                        f"     Corner {j + 1}: ({coord['x']:.1f}, {coord['y']:.1f}) confidence: {coord['confidence']:.3f}"
                    )

            # Plot results
            model.plot_eval(image_path, ax=axes[i], conf=conf, show_polygon=True, show_centers=True)
            axes[i].set_title(f"Confidence: {conf} | Corners: {corner_count}/4")

        except Exception as e:
            print(f"   ❌ Error during prediction: {e}")
            axes[i].text(
                0.5,
                0.5,
                f"Error: {str(e)}",
                ha="center",
                va="center",
                transform=axes[i].transAxes,
            )
            axes[i].set_title(f"Confidence: {conf} | Error")

    plt.tight_layout()
    plt.savefig("output/corner_detection_test.jpg", dpi=150, bbox_inches="tight")
    print("\n📊 Results saved to: output/corner_detection_test.jpg")
    plt.show()


def test_corner_ordering(model_path: Path, image_path: Path, conf: float = 0.25):
    """
    Test corner detection and ordering on a single image.

    Args:
        model_path: Path to the trained model
        image_path: Path to the test image
        conf: Confidence threshold to use
    """

    print("\n🎯 Testing Corner Ordering")
    print("=" * 50)

    # Load model
    model = ChessBoardModel(model_path=model_path)

    # Get corner coordinates
    coordinates, is_valid = model.get_corner_coordinates(image_path, conf=conf)

    if not is_valid or len(coordinates) != 4:
        print(f"❌ Cannot test ordering - need exactly 4 corners, got {len(coordinates)}")
        return

    # Order corners
    ordered_corners = model.order_corners(coordinates)

    print("📍 Corner ordering results:")
    corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]

    for name, corner in zip(corner_names, ordered_corners, strict=False):
        print(f"  {name}: ({corner['x']:.1f}, {corner['y']:.1f}) confidence: {corner['confidence']:.3f}")

    # Visualize ordered corners
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Original detection
    model.plot_eval(image_path, ax=ax1, conf=conf, show_polygon=False, show_centers=True)
    ax1.set_title("Original Detection Order")

    # Show corner ordering with numbered labels
    _results = model.predict(image_path, conf=conf)
    image_rgb = plt.imread(image_path) if str(image_path).lower().endswith((".png", ".jpg", ".jpeg")) else None

    if image_rgb is not None:
        ax2.imshow(image_rgb)

    # Draw ordered corners with numbers
    for i, corner in enumerate(ordered_corners):
        ax2.plot(
            corner["x"],
            corner["y"],
            "ro",
            markersize=10,
            markerfacecolor="yellow",
            markeredgecolor="red",
            markeredgewidth=2,
        )
        ax2.text(
            corner["x"],
            corner["y"],
            str(i + 1),
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="black",
        )

    ax2.set_title("Ordered Corners (1=TL, 2=TR, 3=BR, 4=BL)")
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig("output/corner_ordering_test.jpg", dpi=150, bbox_inches="tight")
    print("📊 Ordering results saved to: output/corner_ordering_test.jpg")
    plt.show()


def main():
    """Main function to run corner detection inference examples."""

    # Configuration - UPDATE THESE PATHS
    MODEL_PATH = Path("models/chess_board_detection/corner_detection_training/weights/best.pt")
    TEST_IMAGE = Path("data/test_images/chessboard_sample.jpg")  # Update this path

    print("🏁 ChessBoard Corner Detection - Inference Example")
    print("=" * 60)

    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        print("💡 Train a model first using: python -m src.chess_board_detection.yolo.train")
        sys.exit(1)

    # Check if test image exists
    if not TEST_IMAGE.exists():
        print(f"❌ Test image not found: {TEST_IMAGE}")
        print("💡 Please provide a valid test image path")
        print("   You can use any image containing a chessboard")
        sys.exit(1)

    # Create output directory
    Path("output").mkdir(exist_ok=True)

    try:
        # Test corner detection with different confidence levels
        test_corner_detection(MODEL_PATH, TEST_IMAGE)

        # Test corner ordering
        test_corner_ordering(MODEL_PATH, TEST_IMAGE)

        print("\n✅ Inference testing completed!")
        print("\n🎯 Next steps:")
        print("1. Try with your own chessboard images")
        print("2. Adjust confidence thresholds based on results")
        print("3. Use the model in your chess application")

    except Exception as e:
        print(f"❌ Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()
