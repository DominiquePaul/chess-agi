#!/usr/bin/env python3
"""
Testing script for ChessBoardSegmentationModel

This script tests a trained segmentation model on images and shows
the polygon detection results.

Usage:
    # Test single image
    python src/chess_board_detection/yolo/segmentation/test_segmentation.py --model models/best.pt --image image.jpg
    
    # Test multiple images
    python src/chess_board_detection/yolo/segmentation/test_segmentation.py --model models/best.pt --input-dir test_images/
    
    # Get JSON output
    python src/chess_board_detection/yolo/segmentation/test_segmentation.py --model models/best.pt --image image.jpg --json-output
    
    # Allow multiple chessboard segments (default is 1)
    python src/chess_board_detection/yolo/segmentation/test_segmentation.py --model models/best.pt --image image.jpg --max-segments 3   
    
    # Extract precise corner coordinates from segmentation
    python src/chess_board_detection/yolo/segmentation/test_segmentation.py --model models/best.pt --image image.jpg --extract-corners
    
    # Use different corner extraction methods
    python src/chess_board_detection/yolo/segmentation/test_segmentation.py --model models/best.pt --image image.jpg --extract-corners --corner-method harris
    
    # Use extended lines method for most accurate corners
    python src/chess_board_detection/yolo/segmentation/test_segmentation.py --model models/best.pt --image image.jpg --extract-corners --corner-method extended
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
from src.chess_board_detection.yolo.segmentation.segmentation_model import ChessBoardSegmentationModel

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test ChessBoard Segmentation Model")
    
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to single image")
    group.add_argument("--input-dir", type=str, help="Directory with images")
    
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Output directory")
    parser.add_argument("--json-output", action="store_true", help="Output as JSON")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold")
    parser.add_argument("--max-segments", type=int, default=1, help="Maximum number of chessboard segments to detect (default: 1)")
    parser.add_argument("--extract-corners", action="store_true", help="Extract and display corner coordinates")
    parser.add_argument("--corner-method", type=str, default="approx", 
                       choices=["approx", "lines", "hybrid", "harris", "contour", "extended"],
                       help="Corner extraction method (default: approx)")
    
    return parser.parse_args()

def test_single_image(model, image_path, args):
    """Test model on a single image."""
    print(f"🔍 Testing image: {image_path}")
    
    # Get original image dimensions for debugging
    original_image = cv2.imread(str(image_path))
    if original_image is not None:
        orig_height, orig_width = original_image.shape[:2]
        print(f"📏 Original image dimensions: {orig_width}x{orig_height}")
    
    polygon_info, is_valid = model.get_polygon_coordinates(image_path, conf=args.conf, iou=args.iou, max_segments=args.max_segments)
    
    if is_valid and polygon_info:
        print(f"✅ Successfully detected chessboard polygon(s)")
        
        # Handle both single and multiple segment formats
        if 'segments' in polygon_info:
            # Multiple segments format
            print(f"   Number of segments: {polygon_info.get('num_segments', 0)}")
            print(f"   Total area: {polygon_info.get('total_area', 0):.0f} pixels²")
            print(f"   Best confidence: {polygon_info.get('best_confidence', 0):.3f}")
            
            # Show details for each segment
            for i, segment in enumerate(polygon_info.get('segments', [])):
                print(f"   Segment {i+1}:")
                print(f"     Points: {segment.get('num_points', 0)}")
                print(f"     Confidence: {segment.get('confidence', 0):.3f}")
                print(f"     Area: {segment.get('area', 0):.0f} pixels²")
                
                # Debug coordinate ranges for first segment
                if i == 0 and 'coordinates' in segment:
                    coords = segment['coordinates']
                    if coords:
                        x_coords = [c['x'] for c in coords]
                        y_coords = [c['y'] for c in coords]
                        print(f"     Coordinate ranges: X=[{min(x_coords):.1f}, {max(x_coords):.1f}], Y=[{min(y_coords):.1f}, {max(y_coords):.1f}]")
        else:
            # Single segment format (backward compatibility)
            print(f"   Points: {polygon_info.get('num_points', 0)}")
            print(f"   Confidence: {polygon_info.get('confidence', 0):.3f}")
            print(f"   Area: {polygon_info.get('area', 0):.0f} pixels²")
            
            # Debug coordinate ranges
            if 'coordinates' in polygon_info:
                coords = polygon_info['coordinates']
                if coords:
                    x_coords = [c['x'] for c in coords]
                    y_coords = [c['y'] for c in coords]
                    print(f"   Coordinate ranges: X=[{min(x_coords):.1f}, {max(x_coords):.1f}], Y=[{min(y_coords):.1f}, {max(y_coords):.1f}]")
    else:
        print(f"❌ No valid chessboard polygon detected")
    
    # Extract corners if requested
    corners_info = None
    if args.extract_corners and is_valid and polygon_info:
        print(f"\n🔍 Extracting corners using '{args.corner_method}' method...")
        try:
            corners, debug_info = model.extract_corners_from_segmentation(
                image_path, polygon_info, method=args.corner_method, debug=True
            )
            
            if corners and len(corners) == 4:
                print(f"✅ Successfully extracted 4 corners:")
                corner_names = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
                for i, (corner, name) in enumerate(zip(corners, corner_names)):
                    print(f"   {name}: ({corner['x']:.1f}, {corner['y']:.1f})")
                
                corners_info = {
                    'corners': corners,
                    'method': args.corner_method,
                    'debug_info': debug_info
                }
                
                # Print debug information
                if debug_info:
                    print(f"   Debug info: {debug_info}")
            else:
                print(f"⚠️  Corner extraction failed - got {len(corners) if corners else 0} corners instead of 4")
                # Print debug information to understand why it failed
                if debug_info:
                    print(f"   Debug info: {debug_info}")
                
        except Exception as e:
            print(f"❌ Corner extraction failed: {e}")
    
    if args.json_output:
        result = {
            "image_path": str(image_path),
            "is_valid": is_valid,
            "polygon_info": polygon_info,
            "parameters": {"confidence_threshold": args.conf, "iou_threshold": args.iou},
            "original_dimensions": {"width": orig_width, "height": orig_height} if original_image is not None else None
        }
        
        # Add corner extraction results if available
        if corners_info:
            result["corners"] = corners_info
            
        return result
    else:
        fig, ax = plt.subplots(figsize=(12, 10))
        # Enhanced visualization with better visibility for segmentation
        model.plot_eval(image_path, ax=ax, conf=args.conf, iou=args.iou, 
                       show_polygon=True, show_mask=True, alpha=0.4, max_segments=args.max_segments)
        
        # Add corner points visualization if extracted
        if corners_info and corners_info['corners']:
            corners = corners_info['corners']
            corner_names = ['TL', 'TR', 'BR', 'BL']  # Top-Left, Top-Right, Bottom-Right, Bottom-Left
            corner_colors = ['red', 'blue', 'green', 'orange']
            
            print(f"🎯 Drawing {len(corners)} corner points on visualization...")
            
            for i, (corner, name, color) in enumerate(zip(corners, corner_names, corner_colors)):
                x, y = corner['x'], corner['y']
                
                # Draw corner point as a large circle
                ax.plot(x, y, 'o', color=color, markersize=15, 
                       markerfacecolor=color, markeredgecolor='white', markeredgewidth=3)
                
                # Add corner label
                ax.text(x + 20, y - 20, f"{name}\n({x:.0f},{y:.0f})", 
                       color='white', fontsize=12, weight='bold',
                       bbox=dict(facecolor=color, alpha=0.8, edgecolor='white', linewidth=2))
            
            # Add enhanced visualization for extended method
            if corners_info.get('method') == 'extended' and 'debug_info' in corners_info:
                debug_info = corners_info['debug_info']
                print(f"🔍 Adding extended method visualization...")
                
                # Draw the contour segments used and fitted lines
                for side_idx in range(4):
                    side_key = f'side_{side_idx}'
                    if f'{side_key}_method' in debug_info:
                        method = debug_info[f'{side_key}_method']
                        line_params = debug_info.get(f'{side_key}_line_params')
                        
                        # Choose colors for different methods
                        if method == 'contour_with_exclusion':
                            segment_color = 'yellow'
                            line_color = 'cyan'
                        else:  # corner_to_corner
                            segment_color = 'magenta'
                            line_color = 'lime'
                        
                        # If we have line parameters, draw the extended line
                        if line_params:
                            a, b, c = line_params['a'], line_params['b'], line_params['c']
                            
                            # Get image bounds for line drawing
                            original_image = cv2.imread(str(image_path))
                            if original_image is not None:
                                img_height, img_width = original_image.shape[:2]
                                
                                # Generate points along the line for visualization
                                if abs(b) > abs(a):  # More horizontal line
                                    x_vals = np.linspace(-img_width*0.2, img_width*1.2, 100)
                                    y_vals = -(a * x_vals + c) / b
                                else:  # More vertical line
                                    y_vals = np.linspace(-img_height*0.2, img_height*1.2, 100)
                                    x_vals = -(b * y_vals + c) / a
                                
                                # Draw the extended line
                                ax.plot(x_vals, y_vals, color=line_color, linewidth=3, alpha=0.7,
                                       label=f'Line {side_idx} ({method})')
                        
                        # Draw the contour segment used (if available)
                        corners_used = debug_info.get(f'{side_key}_corners', [])
                        if len(corners_used) == 2 and method == 'contour_with_exclusion':
                            # Show which contour points were used
                            total_points = debug_info.get(f'{side_key}_total_points', 0)
                            excluded_points = debug_info.get(f'{side_key}_excluded_points', 0)
                            exclude_ratio = debug_info.get(f'{side_key}_exclude_ratio', 0.05)
                            
                            # Add annotation showing the method details
                            corner1, corner2 = corners_used
                            mid_x = (corner1[0] + corner2[0]) / 2
                            mid_y = (corner1[1] + corner2[1]) / 2
                            
                            ax.text(mid_x, mid_y, f"Side {side_idx}\n{method}\n{total_points}→{total_points-excluded_points} pts\n({exclude_ratio*100:.0f}% excluded)", 
                                   color='white', fontsize=9, weight='bold', ha='center',
                                   bbox=dict(facecolor=segment_color, alpha=0.8, edgecolor='white'))
                        
                        elif len(corners_used) == 2 and method == 'corner_to_corner':
                            # Show corner-to-corner method
                            corner1, corner2 = corners_used
                            
                            # Draw line between corners
                            ax.plot([corner1[0], corner2[0]], [corner1[1], corner2[1]], 
                                   color=segment_color, linewidth=5, alpha=0.8)
                            
                            mid_x = (corner1[0] + corner2[0]) / 2
                            mid_y = (corner1[1] + corner2[1]) / 2
                            
                            ax.text(mid_x, mid_y, f"Side {side_idx}\n{method}\n2 corners", 
                                   color='white', fontsize=10, weight='bold', ha='center',
                                   bbox=dict(facecolor=segment_color, alpha=0.8, edgecolor='white'))
                
                # Add legend for the line types
                ax.legend(loc='upper left', fontsize=10, facecolor='black', edgecolor='white')
            
            # Add method info to the plot
            method_used = corners_info.get('method', 'unknown')
            ax.text(15, 100, f"Corner Method: {method_used}", 
                   color='white', fontsize=12, weight='bold',
                   bbox=dict(facecolor='purple', alpha=0.8, edgecolor='white', linewidth=2))
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update filename to indicate corner extraction
        if args.extract_corners:
            output_path = output_dir / f"{Path(image_path).stem}_segmentation_corners.jpg"
        else:
            output_path = output_dir / f"{Path(image_path).stem}_segmentation.jpg"
        
        # Use tight layout to prevent aspect ratio issues
        fig.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', pad_inches=0.1)
        plt.close()
        
        print(f"💾 Results saved to: {output_path}")
        
        # Check saved image dimensions
        saved_image = cv2.imread(str(output_path))
        if saved_image is not None:
            saved_height, saved_width = saved_image.shape[:2]
            print(f"💾 Saved image dimensions: {saved_width}x{saved_height}")
        
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
                    polygon_info, is_valid = model.get_polygon_coordinates(image_path, conf=args.conf, iou=args.iou, max_segments=args.max_segments)
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