#!/usr/bin/env python3
"""
Testing script for ChessBoardSegmentationModel

This script tests a trained segmentation model on images and shows
the polygon detection results.

Usage:
    # Test single image
    python src/chess_board_detection/yolo/test_segmentation.py --model models/best.pt --image image.jpg
    
    # Test multiple images
    python src/chess_board_detection/yolo/test_segmentation.py --model models/best.pt --input-dir test_images/
    
    # Get JSON output
    python src/chess_board_detection/yolo/test_segmentation.py --model models/best.pt --image image.jpg --json-output
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from src.chess_board_detection.yolo.segmentation.segmentation_model import ChessBoardSegmentationModel

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test ChessBoard Segmentation Model")
    
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to single image")
    group.add_argument("--input-dir", type=str, help="Directory with images")
    
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--json-output", action="store_true", help="Output as JSON")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold")
    
    return parser.parse_args()

def test_single_image(model, image_path, args):
    """Test model on a single image."""
    print(f"🔍 Testing image: {image_path}")
    
    polygon_info, is_valid = model.get_polygon_coordinates(image_path, conf=args.conf, iou=args.iou)
    
    if is_valid and polygon_info:
        print(f"✅ Successfully detected chessboard polygon")
        print(f"   Points: {polygon_info.get('num_points', 0)}")
        print(f"   Confidence: {polygon_info.get('confidence', 0):.3f}")
        print(f"   Area: {polygon_info.get('area', 0):.0f} pixels²")
    else:
        print(f"❌ No valid chessboard polygon detected")
    
    if args.json_output:
        return {
            "image_path": str(image_path),
            "is_valid": is_valid,
            "polygon_info": polygon_info,
            "parameters": {"confidence_threshold": args.conf, "iou_threshold": args.iou}
        }
    else:
        fig, ax = plt.subplots(figsize=(12, 10))
        model.plot_eval(image_path, ax=ax, conf=args.conf, iou=args.iou)
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(image_path).stem}_segmentation.jpg"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Results saved to: {output_path}")
        return str(output_path)

def main():
    """Main testing function."""
    args = parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    print(f"🚀 Loading segmentation model from: {model_path}")
    try:
        model = ChessBoardSegmentationModel(model_path=model_path)
        print(f"✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"❌ Image file not found: {image_path}")
            return
        
        result = test_single_image(model, image_path, args)
        
        if args.json_output:
            print(f"\n📄 JSON Result:")
            print(json.dumps(result, indent=2))
    
    elif args.input_dir:
        input_path = Path(args.input_dir)
        if not input_path.exists():
            print(f"❌ Input directory does not exist: {args.input_dir}")
            return
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"❌ No image files found in {args.input_dir}")
            return
        
        print(f"🔍 Found {len(image_files)} images to test")
        
        results = []
        successful_detections = 0
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n📸 Processing {i}/{len(image_files)}: {image_path.name}")
            
            try:
                result = test_single_image(model, image_path, args)
                
                if args.json_output:
                    results.append(result)
                    if result['is_valid']:
                        successful_detections += 1
                else:
                    polygon_info, is_valid = model.get_polygon_coordinates(image_path, conf=args.conf, iou=args.iou)
                    if is_valid:
                        successful_detections += 1
                        
            except Exception as e:
                print(f"❌ Error processing {image_path.name}: {e}")
        
        print(f"\n📊 Summary: {successful_detections}/{len(image_files)} successful ({successful_detections/len(image_files)*100:.1f}%)")
        
        if args.json_output:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            json_path = output_dir / "segmentation_results.json"
            
            with open(json_path, 'w') as f:
                json.dump({
                    "summary": {
                        "total_images": len(image_files),
                        "successful_detections": successful_detections,
                        "success_rate": successful_detections/len(image_files)*100
                    },
                    "results": results
                }, f, indent=2)
            
            print(f"💾 JSON results saved to: {json_path}")
    
    print(f"\n🏁 Testing complete!")

if __name__ == "__main__":
    main() 