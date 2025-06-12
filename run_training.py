#!/usr/bin/env python3
"""Simple launcher for chess board detection training."""

import sys
import os
sys.path.append(os.getcwd())

try:
    print("🚀 Starting Chess Board Detection Training...")
    print("Working directory:", os.getcwd())
    
    # Import and run the main training function
    from src.chess_board_detection.train import main
    
    print("✅ Successfully imported training module")
    print("🏃 Running training...")
    
    main()
    
except Exception as e:
    print(f"❌ Error occurred: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("✅ Training completed successfully!") 