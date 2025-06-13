#!/usr/bin/env python3
"""
Test script to verify W&B integration with chess piece detection model.
This script runs a quick training test to ensure W&B logging works properly.
"""

import os
from pathlib import Path
from src.chess_piece_detection.model import ChessModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_wandb_integration():
    """Test W&B integration with a short training run."""
    
    print("🧪 Testing W&B Integration with Chess Piece Detection")
    print("=" * 60)
    
    # Check if W&B credentials are available
    api_key = os.environ.get("WANDB_API_KEY")
    username = os.environ.get("WANDB_USERNAME")
    
    if not api_key:
        print("❌ WANDB_API_KEY not found in environment variables")
        print("💡 Please set WANDB_API_KEY in your .env file")
        return False
        
    if not username:
        print("⚠️  WANDB_USERNAME not found in environment variables")
        print("💡 This is optional, but recommended for organization")
    
    print(f"✅ Found W&B credentials for user: {username or 'unknown'}")
    
    # Check if dataset exists
    data_folder = os.environ.get("DATA_FOLDER_PATH")
    if not data_folder:
        print("❌ DATA_FOLDER_PATH not found in environment variables")
        return False
        
    data_yaml = Path(data_folder) / "chess_pieces_merged" / "data.yaml"
    if not data_yaml.exists():
        print(f"❌ Dataset not found at: {data_yaml}")
        print("💡 Please ensure your chess dataset is properly set up")
        return False
        
    print(f"✅ Found dataset at: {data_yaml}")
    
    # Create test model
    print("🏗️  Creating test model with YOLOv8n (smallest/fastest)...")
    model = ChessModel(pretrained_checkpoint="yolov8n.pt")
    
    # Test W&B integration with minimal training
    print("🚀 Starting W&B integration test (1 epoch)...")
    
    try:
        # Run a very short training to test W&B logging
        results = model.train(
            data_path=data_yaml,
            epochs=1,  # Just 1 epoch for testing
            batch=8,   # Small batch size
            lr0=0.001,
            imgsz=320,  # Smaller image size for faster training
            plots=True,
            project=Path("test_runs"),
            name="wandb_integration_test",
            # Enable W&B with test configuration
            use_wandb=True,
            wandb_project="chess-piece-detection-test",
            wandb_name="integration-test",
            wandb_tags=["test", "integration", "chess", "yolo"],
            wandb_notes="Testing W&B integration with chess piece detection model",
        )
        
        print("✅ W&B integration test completed successfully!")
        print("🎯 Check your W&B dashboard at: https://wandb.ai")
        print(f"   Project: chess-piece-detection-test")
        print(f"   Run: integration-test")
        
        return True
        
    except Exception as e:
        print(f"❌ W&B integration test failed: {e}")
        print("💡 Common issues:")
        print("   - Run 'wandb login' in terminal")
        print("   - Check your WANDB_API_KEY is correct")
        print("   - Ensure you have internet connection")
        return False

if __name__ == "__main__":
    success = test_wandb_integration()
    
    if success:
        print("\n🎉 W&B Integration Test PASSED!")
        print("\n📈 You can now use W&B tracking in your training:")
        print("   python src/chess_piece_detection/train.py \\")
        print("       --epochs 100 \\")
        print("       --batch 32 \\")
        print("       --use-wandb \\")
        print("       --wandb-project 'chess-piece-detection' \\")
        print("       --wandb-name 'experiment-1'")
    else:
        print("\n❌ W&B Integration Test FAILED!")
        print("   Please fix the issues above before using W&B tracking") 