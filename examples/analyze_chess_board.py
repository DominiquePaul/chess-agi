#!/usr/bin/env python3
"""
Chess Board Analyzer CLI

USAGE:
======
# Basic analysis
python examples/analyze_chess_board.py --image data/eval_images/chess_4.jpeg

# With custom confidence threshold
python examples/analyze_chess_board.py --image chess.jpg --conf 0.6

# Save visualization
python examples/analyze_chess_board.py --image chess.jpg --output results/

# Use custom models (local paths or HuggingFace models)
python examples/analyze_chess_board.py --image chess.jpg --segmentation-model path/to/model.pt --piece-model dopaul/chess_piece_detection

# Show only corners (no piece detection)
python examples/analyze_chess_board.py --image chess.jpg --corners-only

# Verbose output with all details
python examples/analyze_chess_board.py --image chess.jpg --verbose

# Get square coordinates for specific squares
python examples/analyze_chess_board.py --image chess.jpg --squares e4,d4,a1,h8

EXAMPLES:
=========
python examples/analyze_chess_board.py --image data/eval_images/chess_4.jpeg --verbose --output artifacts/
python examples/analyze_chess_board.py --image chess.jpg --squares e2,e4 --conf 0.7
python examples/analyze_chess_board.py --image chess.jpg --corners-only --output results/corners/
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
load_dotenv()

# Add the project root to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from src.chess_board_detection.chess_board_analyzer import ChessBoardAnalyzer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze chess board images using segmentation and piece detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --image chess.jpg
  %(prog)s --image chess.jpg --conf 0.6 --output results/
  %(prog)s --image chess.jpg --squares e4,d4,a1 --verbose
  %(prog)s --image chess.jpg --corners-only
        """
    )
    
    # Required arguments
    parser.add_argument("--image", "-i", type=str, required=True,
                       help="Path to chess board image")
    
    # Model configuration
    parser.add_argument("--segmentation-model", type=str, 
                       default="dopaul/chess_board_segmentation",
                       help="Segmentation model name or path (default: dopaul/chess_board_segmentation)")
    parser.add_argument("--piece-model", type=str,
                       default="dopaul/chess_piece_detection",
                       help="Piece detection model path or HuggingFace model name")
    parser.add_argument("--corner-method", type=str, default="approx",
                       choices=["approx", "extended"],
                       help="Corner extraction method (default: approx)")
    
    # Analysis parameters
    parser.add_argument("--conf", "-c", type=float, default=0.5,
                       help="Confidence threshold for piece detection (default: 0.5)")
    parser.add_argument("--target-size", type=int, default=512,
                       help="Target size for image processing (default: 512)")
    
    # Output options
    parser.add_argument("--output", "-o", type=str, default="artifacts",
                       help="Output directory for visualizations (default: artifacts)")
    parser.add_argument("--no-visualization", action="store_true",
                       help="Skip creating visualization images")
    parser.add_argument("--corners-only", action="store_true",
                       help="Only detect corners, skip piece detection")
    
    # Information options
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed analysis information")
    parser.add_argument("--squares", type=str,
                       help="Comma-separated list of squares to show coordinates for (e.g., e4,d4,a1)")
    parser.add_argument("--json", action="store_true",
                       help="Output results in JSON format")
    
    return parser.parse_args()


def print_analysis_results(result, args):
    """Print analysis results in a formatted way."""
    print("\n" + "="*60)
    print("🏁 CHESS BOARD ANALYSIS RESULTS")
    print("="*60)
    
    # Basic information
    info = result['processing_info']
    print(f"📏 Original image size: {info['original_dimensions'][0]}x{info['original_dimensions'][1]}")
    print(f"🎯 Segmentation method: {info['segmentation_method']}")
    print(f"🎲 Confidence threshold: {info['confidence_threshold']}")
    
    # Corner detection results
    corners = result['board_corners']
    if corners['top_left']:
        print(f"\n📍 BOARD CORNERS DETECTED:")
        for corner_name, coord in corners.items():
            if coord:
                print(f"   {corner_name:12}: ({coord[0]:7.1f}, {coord[1]:7.1f})")
    else:
        print(f"\n❌ No board corners detected")
        return
    
    # Piece detection results
    if not args.corners_only and result['detected_pieces']:
        print(f"\n♟️  CHESS PIECES DETECTED: {len(result['detected_pieces'])}")
        if args.verbose:
            from src.chess_board_detection.classical_method import class_dict
            print(f"   Detected pieces:")
            for cell_num, piece_class in result['detected_pieces']:
                piece_name = class_dict.get(piece_class, f"Unknown({piece_class})")
                print(f"   Cell {cell_num:2d}: {piece_name}")
    elif not args.corners_only:
        print(f"\n♟️  No chess pieces detected")
    
    # Chess position
    if result['chess_position']:
        print(f"\n🎯 CHESS POSITION AVAILABLE")
        if args.verbose:
            print(f"   FEN: {result['chess_position'].fen()}")
            print(f"   Turn: {'White' if result['chess_position'].turn else 'Black'}")
            print(f"   Legal moves: {len(list(result['chess_position'].legal_moves))}")
    
    # Specific square coordinates
    if args.squares:
        print(f"\n📍 SQUARE COORDINATES:")
        analyzer = ChessBoardAnalyzer()  # Create analyzer instance for helper methods
        squares = [s.strip() for s in args.squares.split(',')]
        for square in squares:
            coords = analyzer.get_square_coordinates(result, square)
            piece = analyzer.get_piece_at_square(result, square)
            if coords:
                center = coords['center']
                piece_str = f" -> {piece}" if piece else " -> empty"
                print(f"   {square:2}: center({center[0]:7.1f}, {center[1]:7.1f}){piece_str}")
            else:
                print(f"   {square:2}: Invalid square")


def save_visualizations(result, args, analyzer):
    """Save visualization images."""
    if args.no_visualization:
        return
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_name = Path(args.image).stem
    
    print(f"\n🎨 CREATING VISUALIZATIONS...")
    
    # 1. Create corners and grid visualization
    print("   📍 Creating corners and grid visualization...")
    corners_vis = analyzer.create_corners_and_grid_visualization(result)
    corners_path = output_dir / f"{image_name}_corners_and_grid.png"
    plt.figure(figsize=(12, 8))
    plt.imshow(corners_vis)
    plt.title('Board Corners and Chess Grid')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(corners_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      💾 Saved: {corners_path}")
    
    # 2. Create piece detections visualization
    if not args.corners_only and result['detected_pieces']:
        print("   ♟️  Creating piece detections visualization...")
        pieces_vis = analyzer.create_piece_detections_visualization(result)
        pieces_path = output_dir / f"{image_name}_piece_detections.png"
        plt.figure(figsize=(12, 8))
        plt.imshow(pieces_vis)
        plt.title('Chess Piece Detections')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(pieces_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      💾 Saved: {pieces_path}")
    else:
        pieces_vis = None
        print("      ⚠️  Skipping piece detections (no pieces or corners-only mode)")
    
    # 3. Create comprehensive visualization (existing)
    print("   🔍 Creating complete analysis visualization...")
    complete_vis = analyzer.visualize_result(
        result,
        show_pieces=not args.corners_only,
        show_grid=True,
        show_corners=True
    )
    complete_path = output_dir / f"{image_name}_complete_analysis.png"
    plt.figure(figsize=(12, 8))
    plt.imshow(complete_vis)
    plt.title('Complete Chess Board Analysis')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(complete_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      💾 Saved: {complete_path}")
    
    # 4. Create chess diagram PNG
    if result.get('chess_position'):
        print("   🎨 Creating chess diagram...")
        diagram_path = output_dir / f"{image_name}_chess_diagram.png"
        analyzer.create_chess_diagram_png(result, diagram_path)
        print(f"      💾 Saved: {diagram_path}")
    else:
        print("      ⚠️  Skipping chess diagram (no chess position detected)")
    
    # 5. Create combined 2x2 summary visualization
    print("   📊 Creating analysis summary...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle(f'Chess Board Analysis Summary: {Path(args.image).name}', fontsize=16)
    
    # Top-left: Corners and grid
    axes[0, 0].imshow(corners_vis)
    axes[0, 0].set_title('Board Corners & Grid')
    axes[0, 0].axis('off')
    
    # Top-right: Piece detections or message
    if pieces_vis is not None:
        axes[0, 1].imshow(pieces_vis)
        axes[0, 1].set_title('Piece Detections')
    else:
        axes[0, 1].text(0.5, 0.5, 'No Piece Detection\n(disabled or no pieces found)', 
                       ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=12)
        axes[0, 1].set_title('Piece Detections (None)')
    axes[0, 1].axis('off')
    
    # Bottom-left: Complete analysis
    axes[1, 0].imshow(complete_vis)
    axes[1, 0].set_title('Complete Analysis')
    axes[1, 0].axis('off')
    
    # Bottom-right: Chess diagram or info
    if result.get('chess_position'):
        # Create chess diagram and display it
        try:
            chess_diagram_array = analyzer.create_chess_diagram_png(result, output_dir / f"{image_name}_temp_diagram.png")
            axes[1, 1].imshow(chess_diagram_array)
            axes[1, 1].set_title('Chess Diagram')
            axes[1, 1].axis('off')
            # Clean up temp file
            temp_diagram_path = output_dir / f"{image_name}_temp_diagram.png"
            if temp_diagram_path.exists():
                temp_diagram_path.unlink()
        except Exception as e:
            # Fallback to text if diagram creation fails
            chess_board = result['chess_position']
            position_text = f"Chess Position Detected\n\n"
            position_text += f"Pieces: {len(result['detected_pieces'])}\n"
            position_text += f"Turn: {'White' if chess_board.turn else 'Black'}\n"
            position_text += f"Legal moves: {len(list(chess_board.legal_moves))}\n\n"
            position_text += f"FEN: {chess_board.fen()[:40]}..."  # Truncate long FEN
            
            axes[1, 1].text(0.1, 0.9, position_text, transform=axes[1, 1].transAxes, 
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
            axes[1, 1].set_title('Chess Position Info')
            axes[1, 1].axis('off')
    else:
        axes[1, 1].text(0.5, 0.5, 'No Chess Position\nDetected', 
                       ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Chess Position (None)')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    summary_path = output_dir / f"{image_name}_analysis_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      💾 Saved: {summary_path}")
    
    # Save legacy verbose output if requested
    if args.verbose:
        visualizations = result['visualizations']
        
        # Create multi-panel figure (old format)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Chess Board Analysis Steps: {Path(args.image).name}', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(visualizations['original_image'][:, :, ::-1])  # BGR to RGB
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Warped board
        axes[0, 1].imshow(visualizations['warped_board'])
        axes[0, 1].set_title('Extracted Chess Board')
        axes[0, 1].axis('off')
        
        # Board with grid
        axes[0, 2].imshow(visualizations['board_with_grid'])
        axes[0, 2].set_title('Board with Grid')
        axes[0, 2].axis('off')
        
        # Piece detections
        if visualizations['piece_detections'] is not None and not args.corners_only:
            axes[1, 0].imshow(visualizations['piece_detections'])
            axes[1, 0].set_title('Piece Detections')
        else:
            axes[1, 0].text(0.5, 0.5, 'No Piece Detection', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Piece Detections (Skipped)')
        axes[1, 0].axis('off')
        
        # Final visualization
        axes[1, 1].imshow(complete_vis)
        axes[1, 1].set_title('Complete Analysis')
        axes[1, 1].axis('off')
        
        # Chess position visualization
        if result.get('chess_position'):
            # Create and display chess diagram
            try:
                chess_diagram_array = analyzer.create_chess_diagram_png(result, output_dir / f"{image_name}_temp_verbose_diagram.png")
                axes[1, 2].imshow(chess_diagram_array)
                axes[1, 2].set_title('Chess Diagram')
                axes[1, 2].axis('off')
                # Clean up temp file
                temp_verbose_diagram_path = output_dir / f"{image_name}_temp_verbose_diagram.png"
                if temp_verbose_diagram_path.exists():
                    temp_verbose_diagram_path.unlink()
            except Exception as e:
                # Fallback to text if diagram creation fails
                axes[1, 2].text(0.1, 0.9, f"Chess Position Detected", transform=axes[1, 2].transAxes, fontsize=12, weight='bold')
                axes[1, 2].text(0.1, 0.8, f"Pieces: {len(result['detected_pieces'])}", transform=axes[1, 2].transAxes)
                axes[1, 2].text(0.1, 0.7, f"Turn: {'White' if result['chess_position'].turn else 'Black'}", transform=axes[1, 2].transAxes)
                axes[1, 2].text(0.1, 0.6, f"Legal moves: {len(list(result['chess_position'].legal_moves))}", transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Chess Position Info')
                axes[1, 2].axis('off')
        else:
            axes[1, 2].text(0.5, 0.5, 'No Chess Position', ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Chess Position (None)')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        steps_path = output_dir / f"{image_name}_processing_steps.png"
        plt.savefig(steps_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      💾 Processing steps saved: {steps_path}")
    
    print(f"\n📁 All visualizations saved to: {output_dir.absolute()}")


def output_json(result, args):
    """Output results in JSON format."""
    import json
    
    # Create JSON-serializable result
    json_result = {
        'image_path': args.image,
        'analysis_successful': result['board_corners']['top_left'] is not None,
        'board_corners': result['board_corners'],
        'detected_pieces_count': len(result['detected_pieces']),
        'detected_pieces': result['detected_pieces'],
        'chess_position_available': result['chess_position'] is not None,
        'processing_info': {
            'original_dimensions': result['processing_info']['original_dimensions'],
            'segmentation_method': result['processing_info']['segmentation_method'],
            'confidence_threshold': result['processing_info']['confidence_threshold'],
            'piece_detection_available': result['processing_info']['piece_detection_available']
        }
    }
    
    if result['chess_position']:
        json_result['chess_position_fen'] = result['chess_position'].fen()
        json_result['legal_moves_count'] = len(list(result['chess_position'].legal_moves))
    
    print(json.dumps(json_result, indent=2))


def main():
    """Main CLI function."""
    args = parse_args()
    
    # Validate input image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Error: Image file not found: {image_path}")
        sys.exit(1)
    
    print(f"🚀 Chess Board Analyzer CLI")
    print(f"📸 Analyzing image: {image_path}")
    
    try:
        # Initialize analyzer
        print(f"🔧 Initializing analyzer...")
        piece_model = None if args.corners_only else args.piece_model
        analyzer = ChessBoardAnalyzer(
            segmentation_model=args.segmentation_model,
            piece_detection_model=piece_model,
            corner_method=args.corner_method
        )
        
        # Analyze the image
        print(f"🔍 Running analysis...")
        result = analyzer.analyze_board(
            input_image=image_path,
            conf_threshold=args.conf,
            target_size=args.target_size
        )
        
        # Output results
        if args.json:
            output_json(result, args)
        else:
            print_analysis_results(result, args)
            
            # Save visualizations
            save_visualizations(result, args, analyzer)
            
            print(f"\n✅ Analysis completed successfully!")
            if not args.no_visualization:
                print(f"📁 Generated {4 + (1 if result.get('chess_position') else 0) + (1 if args.verbose else 0)} visualization files")
                print(f"📁 All results saved to: {Path(args.output).absolute()}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 