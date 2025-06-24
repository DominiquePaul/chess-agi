#!/usr/bin/env python3
"""
Chess Board Analyzer CLI - Simplified Version

USAGE:
======
# Basic analysis
python scripts/analyze_chess_board.py --image data/eval_images/chess_4.jpeg

# With options
python scripts/analyze_chess_board.py --image chess.jpg --conf 0.6 --output results/

# With 20% expansion from center
python scripts/analyze_chess_board.py --image chess.jpg --threshold 20

# Overwrite existing files in output directory
python scripts/analyze_chess_board.py --image chess.jpg --overwrite
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from dotenv import load_dotenv

from src.chess_board_detection.chess_board_analyzer import ChessBoardAnalyzer
from src.visualisation import (
    create_chess_diagram_png,
    create_combined_visualization,
    create_corners_and_grid_visualization,
    create_piece_bounding_boxes_visualization,
    create_piece_centers_visualization,
)

load_dotenv()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze chess board images using segmentation and piece detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to chess board image")

    # Model configuration
    parser.add_argument(
        "--segmentation-model",
        type=str,
        default="dopaul/chess_board_segmentation",
        help="Segmentation model name or path (default: dopaul/chess_board_segmentation)",
    )
    parser.add_argument(
        "--piece-model",
        type=str,
        default="dopaul/chess_piece_detection",
        help="Piece detection model path or HuggingFace model name",
    )
    parser.add_argument(
        "--corner-method",
        type=str,
        default="approx",
        choices=["approx", "extended"],
        help="Corner extraction method (default: approx)",
    )

    # Analysis parameters
    parser.add_argument(
        "--conf",
        "-c",
        type=float,
        default=0.5,
        help="Confidence threshold for piece detection (default: 0.5)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=int,
        default=0,
        help="Percentage expansion of chess board from center (0-100, default: 0)",
    )
    parser.add_argument(
        "--use-weighted-center",
        action="store_true",
        default=True,
        help="Use weighted center coordinates for piece mapping (default: True)",
    )
    parser.add_argument(
        "--use-geometric-center",
        action="store_true",
        help="Use geometric center coordinates instead of weighted center",
    )

    # Output options
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="artifacts",
        help="Output directory for visualizations (default: artifacts)",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Skip creating visualization images",
    )
    parser.add_argument(
        "--skip-piece-detection",
        action="store_true",
        help="Skip piece detection, only detect board and grid",
    )

    # Information options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed analysis information",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in output directory",
    )

    return parser.parse_args()


def print_analysis_results(chess_analysis, args):
    """Print analysis results in a formatted way."""
    print("\n" + "=" * 60)
    print("🏁 CHESS BOARD ANALYSIS RESULTS")
    print("=" * 60)

    # Basic information
    metadata = chess_analysis.metadata
    print(f"📏 Original image size: {metadata.original_dimensions[0]}x{metadata.original_dimensions[1]}")
    print(f"🎯 Segmentation method: {metadata.segmentation_method}")
    print(f"🎲 Confidence threshold: {metadata.confidence_threshold}")

    # Corner detection results
    board_corners = chess_analysis.chess_board.board_corners
    if board_corners:
        print("\n📍 BOARD CORNERS DETECTED:")
        corner_names = ["top_left", "top_right", "bottom_left", "bottom_right"]
        for i, corner in enumerate(board_corners):
            if i < len(corner_names):
                print(f"   {corner_names[i]:12}: ({corner.x:7.1f}, {corner.y:7.1f})")
    else:
        print("\n❌ No board corners detected")
        return

    # Piece detection results
    chess_pieces = chess_analysis.chess_board.chess_pieces
    if not args.skip_piece_detection and chess_pieces:
        print(f"\n♟️  CHESS PIECES DETECTED: {len(chess_pieces)}")
        if args.verbose:
            print("   Detected pieces:")
            for piece in chess_pieces:
                square_info = f" -> Square {piece.assigned_square}" if piece.assigned_square else ""
                print(f"   {piece.piece_name}: {piece.confidence:.2f}{square_info}")
    elif not args.skip_piece_detection:
        print("\n♟️  No chess pieces detected")

    # Chess position
    chess_position = chess_analysis.chess_board.board_position
    if chess_position:
        print("\n🎯 CHESS POSITION AVAILABLE")
        if args.verbose:
            print(f"   FEN: {chess_position.fen()}")
            print(f"   Turn: {'White' if chess_position.turn else 'Black'}")
            print(f"   Legal moves: {len(list(chess_position.legal_moves))}")


def save_visualizations(chess_analysis, args, use_weighted_center=True):
    """Save visualization images."""
    if args.no_visualization:
        return

    # Create subfolder named after the input file/folder
    input_path = Path(args.image)
    if input_path.is_file():
        subfolder_name = input_path.stem
    else:
        subfolder_name = input_path.name

    output_dir = Path(args.output) / subfolder_name

    # Check if output directory exists and has files
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            print(f"❌ Error: Output directory '{output_dir}' already contains files.")
            print("   Use --overwrite flag to overwrite existing files.")
            sys.exit(1)
        else:
            print(f"⚠️  Overwriting existing files in: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_name = input_path.stem

    print("\n🎨 CREATING VISUALIZATIONS...")

    # 1. Create corners and grid visualization
    print("   📍 Creating corners and grid visualization...")
    corners_vis = create_corners_and_grid_visualization(chess_analysis)
    corners_path = output_dir / f"{image_name}_corners_and_grid.png"
    plt.figure(figsize=(12, 8))
    plt.imshow(corners_vis)
    plt.title("Board Corners and Chess Grid")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(corners_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      💾 Saved: {corners_path}")

    # 2. Create piece bounding boxes visualization
    if not args.skip_piece_detection and chess_analysis.chess_board.chess_pieces:
        print("   📦 Creating piece bounding boxes visualization...")
        pieces_boxes_vis = create_piece_bounding_boxes_visualization(chess_analysis)
        pieces_boxes_path = output_dir / f"{image_name}_piece_bounding_boxes.png"
        plt.figure(figsize=(12, 8))
        plt.imshow(pieces_boxes_vis)
        plt.title("Chess Piece Bounding Boxes")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(pieces_boxes_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"      💾 Saved: {pieces_boxes_path}")

        # 3. Create piece centers visualization
        coordinate_method = "weighted_center" if use_weighted_center else "geometric_center"
        print(f"   🎯 Creating piece centers visualization ({coordinate_method})...")
        pieces_centers_vis = create_piece_centers_visualization(chess_analysis, use_weighted_center)
        pieces_centers_path = output_dir / f"{image_name}_piece_centers.png"
        plt.figure(figsize=(12, 8))
        plt.imshow(pieces_centers_vis)
        plt.title(f"Chess Piece Centers ({coordinate_method.replace('_', ' ').title()})")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(pieces_centers_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"      💾 Saved: {pieces_centers_path}")
    else:
        print("      ⚠️  Skipping piece detections (no pieces or piece detection disabled)")

    # 4. Create chess diagram PNG if chess position exists
    chess_position = chess_analysis.chess_board.board_position
    if chess_position:
        print("   🎨 Creating chess diagram...")
        diagram_path = output_dir / f"{image_name}_chess_diagram.png"
        try:
            create_chess_diagram_png(chess_analysis, diagram_path)
            print(f"      💾 Saved: {diagram_path}")
        except Exception as e:
            print(f"      ⚠️  Failed to create chess diagram: {e}")
    else:
        print("      ⚠️  Skipping chess diagram (no chess position detected)")

    # 5. Create combined visualization with all plots as subplots
    print("   🎨 Creating combined visualization...")
    create_combined_visualization(
        chess_analysis, output_dir, image_name, args.skip_piece_detection, use_weighted_center
    )

    print(f"\n📁 All visualizations saved to: {output_dir.absolute()}")


def main():
    """Main CLI function."""
    args = parse_args()

    # Validate input image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Error: Image file not found: {image_path}")
        sys.exit(1)

    print("🚀 Chess Board Analyzer CLI")

    try:
        # Determine coordinate method
        use_weighted_center = (
            not args.use_geometric_center
        )  # Default is True unless --use-geometric-center is specified
        coordinate_method = "weighted_center" if use_weighted_center else "geometric_center"
        print(f"📍 Using {coordinate_method} for piece mapping")

        # Initialize analyzer
        piece_model = None if args.skip_piece_detection else args.piece_model
        analyzer = ChessBoardAnalyzer(
            segmentation_model=args.segmentation_model,
            piece_detection_model=piece_model,
            corner_method=args.corner_method,
            threshold=args.threshold,
        )

        # Analyze the image
        chess_analysis = analyzer.analyze_board(
            input_image=image_path, conf_threshold=args.conf, use_weighted_center=use_weighted_center
        )

        # Output results
        print_analysis_results(chess_analysis, args)

        # Save visualizations
        save_visualizations(chess_analysis, args, use_weighted_center)

        print("\n✅ Analysis completed successfully!")
        if not args.no_visualization:
            visualization_count = 1  # corners
            if chess_analysis.chess_board.chess_pieces and not args.skip_piece_detection:
                visualization_count += 2  # bounding boxes + centers
            if chess_analysis.chess_board.board_position:
                visualization_count += 1  # chess diagram
            visualization_count += 1  # combined visualization
            print(f"📁 Generated {visualization_count} visualization files")

            # Show the actual output directory (with subfolder)
            input_path = Path(args.image)
            if input_path.is_file():
                subfolder_name = input_path.stem
            else:
                subfolder_name = input_path.name
            actual_output_dir = Path(args.output) / subfolder_name
            print(f"📁 All results saved to: {actual_output_dir.absolute()}")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
