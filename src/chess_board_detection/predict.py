#!/usr/bin/env python3
"""
Prediction script for ChessBoardModel - EfficientNet B0 Keypoint Detection

This script loads a trained model and makes predictions on input images,
saving visualization plots to an artifacts folder.

Usage:
    # Basic prediction
    python src/chess_board_detection/predict.py --model models/chess_board_detection/efficientnet_keypoints/best.pt --image data/eval_images/chess_4.jpeg

    # With custom output directory
    python src/chess_board_detection/predict.py --model models/chess_board_detection/efficientnet_keypoints/best.pt --image data/eval_images/chess_4.jpeg --output artifacts/predictions

    # With custom confidence
    python src/chess_board_detection/predict.py --model models/chess_board_detection/efficientnet_keypoints/best.pt --image data/eval_images/chess_4.jpeg --conf 0.5

    # Multiple images
    python src/chess_board_detection/predict.py --model models/chess_board_detection/efficientnet_keypoints/best.pt --image data/eval_images/*.jpeg
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.chess_board_detection.model import ChessBoardModel


def parse_args():
    """Parse command line arguments for prediction."""
    parser = argparse.ArgumentParser(
        description="Predict chessboard corners using trained EfficientNet B0 model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model", type=str, required=True, help="Path to trained model file (.pt)")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image or glob pattern (e.g., 'data/images/*.jpg')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/predictions",
        help="Output directory for prediction results",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (unused for regression model but kept for compatibility)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip saving visualization plots")
    parser.add_argument("--save-json", action="store_true", help="Save prediction results as JSON files")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run predictions on",
    )

    return parser.parse_args()


def setup_device(device_arg):
    """Setup computation device."""
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)

    print(f"Using device: {device}")
    return device


def get_image_files(image_path):
    """Get list of image files from path or glob pattern."""
    if "*" in image_path or "?" in image_path:
        # Glob pattern
        image_files = glob.glob(image_path)
        image_files.extend(glob.glob(image_path.replace("*", "**/*")))  # Also check subdirectories
    else:
        # Single file or directory
        path = Path(image_path)
        if path.is_file():
            image_files = [str(path)]
        elif path.is_dir():
            # Find all images in directory
            image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(str(path / ext)))
                image_files.extend(glob.glob(str(path / ext.upper())))
        else:
            image_files = []

    # Filter to only valid image files
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = [f for f in image_files if Path(f).suffix.lower() in valid_extensions]

    return sorted(image_files)


def save_prediction_json(corners_dict, is_valid, output_path):
    """Save prediction results as JSON."""
    result = {
        "valid_detection": is_valid,
        "corners": corners_dict if is_valid else {},
        "model_type": "EfficientNet_B0_Keypoint_Regression",
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)


def predict_single_image(model, image_path, output_dir, args):
    """Make prediction on a single image."""
    image_path = Path(image_path)

    print(f"\n📸 Processing: {image_path.name}")

    try:
        # Make prediction
        corners_dict, is_valid = model.predict_keypoints(image_path, conf_threshold=args.conf)

        # Print results
        if is_valid:
            print("✅ Chessboard corners detected successfully!")
            print("🎯 Corner coordinates:")
            corner_names = ["top_left", "top_right", "bottom_right", "bottom_left"]
            for corner_name in corner_names:
                if corner_name in corners_dict:
                    corner = corners_dict[corner_name]
                    print(f"  {corner_name.replace('_', ' ').title()}: ({corner['x']:.1f}, {corner['y']:.1f})")
        else:
            print("⚠️  Failed to detect chessboard corners")

        # Save visualization plot
        if not args.no_plot:
            plt.figure(figsize=(12, 8))
            model.plot_eval(image_path, conf=args.conf, show_polygon=True, show_centers=True)

            # Save plot
            plot_filename = f"{image_path.stem}_prediction.png"
            plot_path = output_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"💾 Visualization saved: {plot_path}")

        # Save JSON results
        if args.save_json:
            json_filename = f"{image_path.stem}_prediction.json"
            json_path = output_dir / json_filename
            save_prediction_json(corners_dict, is_valid, json_path)
            print(f"💾 JSON results saved: {json_path}")

        return True, is_valid

    except Exception as e:
        print(f"❌ Error processing {image_path.name}: {e}")
        return False, False


def main():
    """Main prediction function."""
    args = parse_args()

    # Setup paths
    model_path = Path(args.model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate model path
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return

    # Setup device
    device = setup_device(args.device)

    # Get image files
    image_files = get_image_files(args.image)
    if not image_files:
        print(f"❌ No image files found matching: {args.image}")
        return

    print("🚀 ChessBoard Corner Detection - Prediction")
    print("=" * 60)
    print(f"📁 Model: {model_path}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Found {len(image_files)} image(s) to process")
    print(f"🎯 Confidence threshold: {args.conf}")
    print(f"💾 Save plots: {'No' if args.no_plot else 'Yes'}")
    print(f"💾 Save JSON: {'Yes' if args.save_json else 'No'}")
    print("=" * 60)

    # Load model
    try:
        model = ChessBoardModel(model_path=model_path, device=device)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Process images
    processed_count = 0
    successful_detections = 0

    for image_file in image_files:
        success, detected = predict_single_image(model, image_file, output_dir, args)
        if success:
            processed_count += 1
            if detected:
                successful_detections += 1

    # Summary
    print("\n" + "=" * 60)
    print("🏁 Prediction Complete!")
    print(f"📊 Processed: {processed_count}/{len(image_files)} images")
    print(f"✅ Successful detections: {successful_detections}/{processed_count}")
    if processed_count > 0:
        success_rate = (successful_detections / processed_count) * 100
        print(f"📈 Success rate: {success_rate:.1f}%")
    print(f"💾 Results saved to: {output_dir}")
    print("=" * 60)

    # Show example usage
    if not args.no_plot and processed_count > 0:
        print("\n🎯 Next Steps:")
        print(f"1. Check visualization plots in: {output_dir}")
        if args.save_json:
            print(f"2. Check JSON results in: {output_dir}")
        print("3. Use predictions for perspective transformation or further analysis")


if __name__ == "__main__":
    main()
