#!/usr/bin/env python3
"""
Test/Evaluation CLI for ChessBoard Corner Detection

Test a trained chessboard corner detection model on images and save results.

Usage:
    # Test model on a single image
    python src/chess_board_detection/test_model.py --model models/chess_board_detection/corner_detection_training/weights/best.pt --image path/to/image.jpg
    
    # Test on multiple images in a directory
    python src/chess_board_detection/test_model.py --model models/chess_board_detection/corner_detection_training/weights/best.pt --input-dir path/to/images/
    
    # Test and get corner coordinates as JSON
    python src/chess_board_detection/test_model.py --model models/chess_board_detection/corner_detection_training/weights/best.pt --image image.jpg --json-output
    
    # Run from project root as module
    python -m src.chess_board_detection.test_model --help

Examples:
    # Basic test with visualization
    python src/chess_board_detection/test_model.py --model models/chess_board_detection/corner_detection_training/weights/best.pt --image examples/chessboard.jpg
    
    # Batch test all images in a folder
    python src/chess_board_detection/test_model.py --model models/chess_board_detection/corner_detection_training/weights/best.pt --input-dir examples/test_images/ --output-dir results/
    
    # Get coordinates for integration
    python src/chess_board_detection/test_model.py --model models/chess_board_detection/corner_detection_training/weights/best.pt --image board.jpg --json-output --save-coords coordinates.json
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

def parse_args():
    """Parse command line arguments for model testing."""
    parser = argparse.ArgumentParser(
        description="Test ChessBoard Corner Detection Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ========================================
    # Model and Input Configuration
    # ========================================
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Path to trained model file (.pt file). Usually in models/chess_board_detection/[training_name]/weights/best.pt"
    )
    parser.add_argument(
        "--image", 
        type=str, 
        help="Path to single image to test. Use this OR --input-dir, not both"
    )
    parser.add_argument(
        "--input-dir", 
        type=str, 
        help="Directory containing images to test. Use this OR --image, not both"
    )
    
    # ========================================
    # Output Configuration
    # ========================================
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output/corner_detection_results",
        help="Directory to save result images with predictions drawn"
    )
    parser.add_argument(
        "--json-output", 
        action="store_true",
        help="Print corner coordinates as JSON to stdout"
    )
    parser.add_argument(
        "--save-coords", 
        type=str, 
        help="Save corner coordinates to JSON file (specify filename)"
    )
    
    # ========================================
    # Detection Parameters
    # ========================================
    parser.add_argument(
        "--conf-threshold", 
        type=float, 
        default=0.25,
        help="Confidence threshold for detections (0.0-1.0). Lower = more detections"
    )
    parser.add_argument(
        "--iou-threshold", 
        type=float, 
        default=0.7,
        help="IoU threshold for non-maximum suppression (0.0-1.0)"
    )
    
    # ========================================
    # Display Options
    # ========================================
    parser.add_argument(
        "--show-image", 
        action="store_true",
        help="Display result image (requires display/GUI environment)"
    )
    parser.add_argument(
        "--no-save", 
        action="store_true",
        help="Don't save result images, only display or print coordinates"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show detailed output and processing information"
    )
    
    return parser.parse_args()

def validate_args(args) -> bool:
    """Validate command line arguments."""
    # Check that exactly one input method is specified
    if not args.image and not args.input_dir:
        print("❌ Error: Must specify either --image or --input-dir")
        return False
        
    if args.image and args.input_dir:
        print("❌ Error: Cannot specify both --image and --input-dir, choose one")
        return False
    
    # Check if model file exists (skip for HuggingFace models)
    if "/" in args.model and not Path(args.model).exists():
        # This might be a HuggingFace model ID (contains /), skip local file validation
        pass
    else:
        # Check local model file
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"❌ Error: Model file '{model_path}' does not exist!")
            print("💡 Check the path, or train a model first with:")
            print("   python src/chess_board_detection/train.py")
            return False
    
    # Check input paths
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"❌ Error: Image file '{image_path}' does not exist!")
            return False
            
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"❌ Error: Input directory '{input_dir}' does not exist!")
            return False
        if not input_dir.is_dir():
            print(f"❌ Error: '{input_dir}' is not a directory!")
            return False
    
    return True

def load_model(model_path_or_id: str, verbose: bool = False):
    """Load the trained chessboard corner detection model."""
    try:
        from src.chess_board_detection.model import ChessBoardModel
        
        if verbose:
            print(f"🔄 Loading model from: {model_path_or_id}")
        
        # Check if it's a HuggingFace model ID (contains /) or local path
        if "/" in model_path_or_id and not Path(model_path_or_id).exists():
            # Assume it's a HuggingFace model ID
            model = ChessBoardModel.from_huggingface(model_path_or_id)
        else:
            # Local path
            model = ChessBoardModel(model_path=Path(model_path_or_id))
        
        if verbose:
            print("✅ Model loaded successfully")
            
        return model
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("💡 Make sure the model file is a valid YOLO .pt file or valid HuggingFace model ID")
        return None

def test_single_image(model, image_path: Path, args) -> Optional[Dict]:
    """Test model on a single image and return results."""
    try:
        if args.verbose:
            print(f"🔍 Processing image: {image_path}")
        
        # Get corner coordinates (now returns a dictionary with 4 corners)
        corners_dict, is_valid = model.get_corner_coordinates(str(image_path))
        
        # Create results dictionary
        results = {
            "image": str(image_path),
            "corners_detected": bool(corners_dict),
            "corners": corners_dict,
            "is_valid": is_valid
        }
        
        if args.verbose:
            if corners_dict:
                print(f"✅ Detected chessboard with 4 corners")
                corner_names = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
                for corner_name in corner_names:
                    if corner_name in corners_dict:
                        corner = corners_dict[corner_name]
                        print(f"   {corner_name.replace('_', ' ').title()}: ({corner['x']:.1f}, {corner['y']:.1f})")
                if 'confidence' in corners_dict:
                    print(f"   Confidence: {corners_dict['confidence']:.2f}")
            else:
                print("⚠️  No chessboard detected")
        
        # Save visualization if requested
        if not args.no_save:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_filename = f"{image_path.stem}_corners{image_path.suffix}"
            output_path = output_dir / output_filename
            
            try:
                import matplotlib.pyplot as plt
                # Create new figure for saving
                fig, ax = plt.subplots(figsize=(10, 10))
                model.plot_eval(str(image_path), ax=ax)
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                plt.close(fig)  # Close figure to free memory
                
                if args.verbose:
                    print(f"💾 Saved result to: {output_path}")
                results["output_image"] = str(output_path)
            except Exception as e:
                if args.verbose:
                    print(f"⚠️  Could not save visualization: {e}")
        
        # Show image if requested
        if args.show_image:
            try:
                model.plot_eval(str(image_path), show=True)
            except Exception as e:
                if args.verbose:
                    print(f"⚠️  Could not display image: {e}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return None

def test_directory(model, input_dir: Path, args) -> List[Dict]:
    """Test model on all images in a directory."""
    # Common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"⚠️  No image files found in {input_dir}")
        print(f"💡 Looking for: {', '.join(image_extensions)}")
        return []
    
    if args.verbose:
        print(f"📁 Found {len(image_files)} images to process")
    
    results = []
    for i, image_path in enumerate(image_files, 1):
        if args.verbose:
            print(f"\n[{i}/{len(image_files)}] Processing {image_path.name}")
        
        result = test_single_image(model, image_path, args)
        if result:
            results.append(result)
    
    return results

def main():
    """Main function for model testing CLI."""
    args = parse_args()
    
    if not validate_args(args):
        sys.exit(1)
    
    # ========================================
    # Load Model
    # ========================================
    print("🧠 ChessBoard Corner Detection - Model Testing")
    print("=" * 50)
    
    model = load_model(args.model, args.verbose)
    if not model:
        sys.exit(1)
    
    # ========================================
    # Run Tests
    # ========================================
    all_results = []
    
    if args.image:
        # Test single image
        result = test_single_image(model, Path(args.image), args)
        if result:
            all_results.append(result)
    else:
        # Test directory
        all_results = test_directory(model, Path(args.input_dir), args)
    
    # ========================================
    # Output Results
    # ========================================
    if args.json_output:
        if len(all_results) == 1:
            # Single image - output just the corners
            print(json.dumps(all_results[0]["corners"], indent=2))
        else:
            # Multiple images - output all results
            print(json.dumps(all_results, indent=2))
    
    if args.save_coords:
        coords_path = Path(args.save_coords)
        with open(coords_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"💾 Coordinates saved to: {coords_path}")
    
    # ========================================
    # Summary
    # ========================================
    if not args.json_output:  # Don't print summary if JSON output is requested
        print("\n" + "=" * 50)
        print("📊 Testing Summary")
        total_images = len(all_results)
        images_with_corners = sum(1 for r in all_results if r.get("corners_detected", False))
        
        print(f"📁 Images processed: {total_images}")
        print(f"✅ Images with corners detected: {images_with_corners}")
        print(f"⚠️  Images without corners: {total_images - images_with_corners}")
        
        if not args.no_save:
            print(f"💾 Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 