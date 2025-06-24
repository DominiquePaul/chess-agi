#!/usr/bin/env python3
"""
Chess Board Detection CLI Help

Shows all available CLI commands for the chess board detection pipeline.

Usage:
    python src/chess_board_detection/cli.py
    python -m src.chess_board_detection.cli
"""


def show_help():
    """Display help for all chess board detection CLI commands."""

    print("🏆 Chess Board Detection - CLI Commands")
    print("=" * 60)
    print()

    print("📥 1. DOWNLOAD DATASET")
    print("   Download chessboard corner detection dataset from Roboflow")
    print()
    print("   Basic usage:")
    print("   python src/chess_board_detection/download_data.py")
    print()
    print("   With options:")
    print("   python src/chess_board_detection/download_data.py \\")
    print("       --data-dir data/my_corners \\")
    print("       --verbose")
    print()
    print("   Help: python src/chess_board_detection/download_data.py --help")
    print()

    print("🎯 2. TRAIN MODEL")
    print("   Train a chessboard corner detection model")
    print()
    print("   Basic usage:")
    print("   python src/chess_board_detection/train.py")
    print()
    print("   Custom training:")
    print("   python src/chess_board_detection/train.py \\")
    print("       --data data/chessboard_corners/data.yaml \\")
    print("       --epochs 100 \\")
    print("       --batch 32 \\")
    print("       --name my_training")
    print()
    print("   With HuggingFace upload:")
    print("   python src/chess_board_detection/train.py \\")
    print("       --upload-hf \\")
    print("       --hf-model-name username/chess-corner-detector")
    print()
    print("   Help: python src/chess_board_detection/train.py --help")
    print()

    print("🧪 3. TEST MODEL")
    print("   Test trained model on images and save predictions")
    print()
    print("   Single image:")
    print("   python src/chess_board_detection/test_model.py \\")
    print("       --model models/chess_board_detection/corner_detection_training/weights/best.pt \\")
    print("       --image path/to/image.jpg")
    print()
    print("   Batch processing:")
    print("   python src/chess_board_detection/test_model.py \\")
    print("       --model models/chess_board_detection/corner_detection_training/weights/best.pt \\")
    print("       --input-dir path/to/images/ \\")
    print("       --output-dir results/")
    print()
    print("   Get coordinates as JSON:")
    print("   python src/chess_board_detection/test_model.py \\")
    print("       --model models/chess_board_detection/corner_detection_training/weights/best.pt \\")
    print("       --image image.jpg \\")
    print("       --json-output")
    print()
    print("   Help: python src/chess_board_detection/test_model.py --help")
    print()

    print("🤗 4. UPLOAD TO HUGGINGFACE")
    print("   Upload trained model to HuggingFace Model Hub")
    print()
    print("   Basic upload:")
    print("   python src/chess_board_detection/upload_hf.py \\")
    print("       --model models/chess_board_detection/corner_detection_training/weights/best.pt \\")
    print("       --repo-name username/chess-corner-detector")
    print()
    print("   With custom metadata:")
    print("   python src/chess_board_detection/upload_hf.py \\")
    print("       --model models/chess_board_detection/corner_detection_training/weights/best.pt \\")
    print("       --repo-name username/chess-corner-detector \\")
    print('       --description "My custom corner detector" \\')
    print("       --tags computer-vision,chess,detection \\")
    print("       --license mit")
    print()
    print("   Dry run (test without uploading):")
    print("   python src/chess_board_detection/upload_hf.py \\")
    print("       --model models/chess_board_detection/corner_detection_training/weights/best.pt \\")
    print("       --repo-name username/chess-corner-detector \\")
    print("       --dry-run")
    print()
    print("   Help: python src/chess_board_detection/upload_hf.py --help")
    print()

    print("🔧 TYPICAL WORKFLOW")
    print("=" * 60)
    print("1. Set API key: export ROBOFLOW_API_KEY=your_key_here")
    print("2. Download data: python src/chess_board_detection/download_data.py")
    print("3. Train model: python src/chess_board_detection/train.py")
    print("4. Test model: python src/chess_board_detection/test_model.py --model models/.../best.pt --image test.jpg")
    print(
        "5. Upload model: python src/chess_board_detection/upload_hf.py --model models/.../best.pt --repo-name user/model"
    )
    print()

    print("📚 MORE HELP")
    print("=" * 60)
    print("• Add --help to any command for detailed options")
    print("• Check README.md files in src/chess_board_detection/")
    print("• View examples in examples/ directory")
    print()


def main():
    """Main function for CLI help."""
    show_help()


if __name__ == "__main__":
    main()
