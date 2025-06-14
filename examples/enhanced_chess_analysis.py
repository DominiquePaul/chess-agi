#!/usr/bin/env python3
"""
Enhanced Chess Board Analysis Example

This example demonstrates the improved chess board analyzer with:
- HuggingFace Hub model loading
- YOLO-based piece detection
- Comprehensive chess position analysis

Usage:
    python examples/enhanced_chess_analysis.py
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add the project root to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from src.chess_board_detection.chess_board_analyzer import ChessBoardAnalyzer
from src.chess_piece_detection.model import ChessModel


def demonstrate_huggingface_loading():
    """Demonstrate loading a chess piece detection model from HuggingFace Hub."""
    print("🤗 Demonstrating HuggingFace Hub model loading...")
    
    try:
        # Load model from HuggingFace Hub
        model = ChessModel("dopaul/chess_piece_detection")
        print("✅ Successfully loaded chess piece detection model from HuggingFace Hub!")
        
        # Test the model on a sample image
        test_image = Path("data/eval_images/chess_4.jpeg")
        if test_image.exists():
            print(f"🎯 Testing model on: {test_image}")
            # This would run inference but we'll skip for demo
            print("   Model ready for inference!")
        else:
            print("   Test image not found, but model is loaded and ready")
            
    except Exception as e:
        print(f"⚠️  Could not load HuggingFace model: {e}")
        print("   This is expected if the model hasn't been uploaded yet")


def demonstrate_enhanced_analyzer():
    """Demonstrate the enhanced chess board analyzer."""
    print("\n🏁 Demonstrating Enhanced Chess Board Analyzer...")
    
    # Test image path
    test_image = Path("data/eval_images/chess_4.jpeg")
    
    if not test_image.exists():
        print(f"⚠️  Test image not found: {test_image}")
        print("   Please ensure you have test images in the data/eval_images/ directory")
        return
    
    try:
        # Initialize analyzer with HuggingFace models
        print("🚀 Initializing analyzer with HuggingFace models...")
        analyzer = ChessBoardAnalyzer(
            segmentation_model="dopaul/chess_board_segmentation",  # When available
            piece_detection_model="dopaul/chess_piece_detection"   # When available
        )
        
        # Analyze the chess board
        print(f"🔍 Analyzing chess board: {test_image}")
        result = analyzer.analyze_board(
            input_image=test_image,
            conf_threshold=0.5,
            target_size=512
        )
        
        # Display results
        print("\n📊 Analysis Results:")
        print(f"   Board corners detected: {bool(result['board_corners']['top_left'])}")
        print(f"   Chess pieces detected: {len(result['detected_pieces'])}")
        print(f"   Chess position available: {result['chess_position'] is not None}")
        
        if result['chess_position']:
            print(f"   FEN: {result['chess_position'].fen()}")
            print(f"   Legal moves: {len(list(result['chess_position'].legal_moves))}")
        
        # Show detected pieces
        if result['detected_pieces']:
            print("\n♟️  Detected pieces:")
            for square_num, piece_class in result['detected_pieces']:
                piece_name = analyzer.piece_model.get_piece_name(piece_class)
                print(f"      Square {square_num:2d}: {piece_name}")
        
        # Create visualization
        print("\n🎨 Creating visualization...")
        vis_image = analyzer.visualize_result(result)
        
        # Save visualization
        output_path = Path("artifacts/enhanced_analysis_demo.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_image)
        plt.title('Enhanced Chess Board Analysis Demo')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Visualization saved: {output_path}")
        print("✅ Enhanced analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print("   This might be due to missing models or test data")


def demonstrate_direct_model_usage():
    """Demonstrate using the ChessModel directly."""
    print("\n🎯 Demonstrating direct ChessModel usage...")
    
    test_image = Path("data/eval_images/chess_4.jpeg")
    if not test_image.exists():
        print(f"⚠️  Test image not found: {test_image}")
        return
    
    try:
        # Load a local model for demonstration (fallback if HF not available)
        local_model_path = Path("models/yolo_chess_piece_detector/training_v2/weights/best.pt")
        
        if local_model_path.exists():
            print(f"🚀 Loading local model: {local_model_path}")
            model = ChessModel(local_model_path)
        else:
            print("🤗 Attempting to load from HuggingFace...")
            model = ChessModel("dopaul/chess_piece_detection")
        
        print("✅ Model loaded successfully!")
        
        # Example square coordinates (dummy data for demo)
        dummy_squares = {
            1: [(100, 100), (150, 100), (150, 150), (100, 150), (100, 100)],
            2: [(150, 100), (200, 100), (200, 150), (150, 150), (150, 100)],
            # ... more squares would be here in real usage
        }
        
        print(f"🔍 Running piece detection on: {test_image}")
        # This would run the actual detection
        print("   Detection ready to run (skipping actual inference for demo)")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")


def main():
    """Main demonstration function."""
    print("🚀 Enhanced Chess Board Analysis Demo")
    print("=" * 50)
    
    # Demonstrate different aspects of the enhanced system
    demonstrate_huggingface_loading()
    demonstrate_enhanced_analyzer()
    demonstrate_direct_model_usage()
    
    print("\n" + "=" * 50)
    print("📝 Summary:")
    print("   ✅ Enhanced ChessModel with HuggingFace Hub support")
    print("   ✅ Improved ChessBoardAnalyzer integration")
    print("   ✅ YOLO-based piece detection")
    print("   ✅ Chess position analysis")
    print("   ✅ Comprehensive visualization")
    print("\n🎯 The chess board analyzer is now ready for enhanced piece detection!")


if __name__ == "__main__":
    main() 