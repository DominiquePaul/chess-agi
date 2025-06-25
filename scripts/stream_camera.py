#!/usr/bin/env python3
"""
Simple Camera Streaming Script

Stream live video from webcam using WebcamCapture class.

USAGE:
======
# Stream from default camera (index 0)
python scripts/stream_camera.py

# Stream from specific camera
python scripts/stream_camera.py --camera 1

# Custom resolution and FPS
python scripts/stream_camera.py --camera 0 --resolution 1280x720 --fps 30
"""

import argparse
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from camera import WebcamCapture


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Simple camera streaming script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera device index (default: 0)")

    parser.add_argument(
        "--resolution", type=str, default="640x480", help="Camera resolution in WIDTHxHEIGHT format (default: 640x480)"
    )

    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")

    parser.add_argument(
        "--window-name", type=str, default="Camera Stream", help="Window name for display (default: 'Camera Stream')"
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Parse resolution
    try:
        width, height = map(int, args.resolution.split("x"))
    except ValueError:
        print(f"❌ Invalid resolution format: {args.resolution}")
        print("   Expected format: WIDTHxHEIGHT (e.g., 1280x720)")
        sys.exit(1)

    print("📹 Simple Camera Streaming")
    print("=" * 30)
    print(f"🎥 Camera: {args.camera}")
    print(f"📐 Resolution: {width}x{height}")
    print(f"🎯 FPS: {args.fps}")
    print(f"🪟 Window: '{args.window_name}'")
    print()
    print("📝 Controls:")
    print("   - Press 'q' in the video window to quit")
    print("   - Or press Ctrl+C in terminal")
    print()

    # Create camera instance
    camera = WebcamCapture(camera_index=args.camera, width=width, height=height, fps=args.fps)

    # Connect to camera
    print(f"🔌 Connecting to camera {args.camera}...")
    if not camera.connect():
        print(f"❌ Failed to connect to camera {args.camera}")
        print("💡 Try:")
        print("   - Check if camera is connected")
        print("   - Try a different camera index (--camera 1, --camera 2, etc.)")
        print("   - Make sure no other app is using the camera")
        sys.exit(1)

    print("✅ Camera connected successfully!")

    try:
        # Start streaming
        print("🚀 Starting video stream...")
        camera.display_stream(args.window_name)

    except KeyboardInterrupt:
        print("\n⏹️  Stream interrupted by user")

    except Exception as e:
        print(f"❌ Error during streaming: {e}")
        sys.exit(1)

    finally:
        print("🧹 Cleaning up...")
        camera.disconnect()
        print("✅ Done!")


if __name__ == "__main__":
    main()
