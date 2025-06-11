#!/usr/bin/env python3
"""
Training script for ChessBoardModel - Corner Detection

This script trains a YOLO model to detect the 4 corners of a chessboard.
"""

from pathlib import Path
from src.chess_board_detection.model import ChessBoardModel

def main():
    """Main training function for chessboard corner detection."""
    
    # Configuration
    DATA_YAML = Path("data/chessboard_corners/data.yaml")  # Path to your dataset YAML
    MODELS_FOLDER = Path("models/chess_board_detection")
    MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # Training parameters
    EPOCHS = 50
    BATCH_SIZE = 16
    IMG_SIZE = 640
    LEARNING_RATE = 0.001
    
    print("🏁 Starting ChessBoard Corner Detection Training")
    print("=" * 60)
    print(f"📁 Models will be saved to: {MODELS_FOLDER}")
    print(f"📊 Dataset: {DATA_YAML}")
    print(f"🔄 Epochs: {EPOCHS}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"🖼️  Image size: {IMG_SIZE}")
    print(f"📈 Learning rate: {LEARNING_RATE}")
    print("=" * 60)
    
    # Initialize model
    model = ChessBoardModel()
    
    # Train the model
    try:
        results = model.train(
            data_path=DATA_YAML,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMG_SIZE,
            lr0=LEARNING_RATE,
            project=MODELS_FOLDER,
            name="corner_detection_training",
            plots=True,
            # Corner detection specific parameters
            patience=15,  # Early stopping patience
            save_period=10,  # Save every 10 epochs
            # Data augmentation - less aggressive for corner detection
            degrees=5.0,    # Small rotation
            translate=0.05, # Small translation
            scale=0.2,      # Small scale variation
            fliplr=0.5,     # Horizontal flip
            flipud=0.0,     # No vertical flip for corners
            mosaic=0.8,     # Reduce mosaic for corner detection
        )
        
        # Print training summary
        model.print_training_summary(results)
        
        # Save the final model
        best_model_path = MODELS_FOLDER / "corner_detection_training" / "weights" / "best.pt"
        if best_model_path.exists():
            print(f"✅ Best model saved at: {best_model_path}")
            print(f"🎯 Training completed successfully!")
            
            # Load the trained model and test it
            trained_model = ChessBoardModel(model_path=best_model_path)
            print(f"✅ Model loaded successfully for testing")
            
        else:
            print(f"⚠️  Warning: Best model not found at expected path: {best_model_path}")
            
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        raise
    
    print("=" * 60)
    print("🏁 Training Complete!")
    print("Next steps:")
    print("1. Test the model: model.plot_eval('path/to/test_image.jpg')")
    print("2. Get corner coordinates: model.get_corner_coordinates('image.jpg')")
    print("3. Upload to HuggingFace: model.push_to_huggingface('username/chessboard-corners')")

if __name__ == "__main__":
    main() 