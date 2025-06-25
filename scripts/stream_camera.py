#!/usr/bin/env python3
"""
Simple Camera Streaming Script

Stream live video from webcam using WebcamCapture class.

USAGE:
======
# Stream from default camera with default settings (camera 0, 640x480, 30fps)
python scripts/stream_camera.py

# Record frames while streaming (default is that every 25th frame is saved to a new directory in artifacts/)
python scripts/stream_camera.py --record

# Stream from specific camera
python scripts/stream_camera.py --camera 1

# Custom resolution and FPS
python scripts/stream_camera.py --camera 0 --resolution 1280x720 --fps 30

# Record to custom output directory
python scripts/stream_camera.py --record --output-dir ./my_recordings
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

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

    parser.add_argument("--record", action="store_true", help="Record frames during streaming (default: False)")

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for recorded frames. If not specified and --record is set, creates a timestamped directory in artifacts/",
    )

    parser.add_argument(
        "--save-n-frames", type=int, default=25, help="Save every Nth frame when recording (default: 25)"
    )

    return parser.parse_args()


def create_output_directory(output_dir: str = None) -> Path:
    """Create and return the output directory for recording."""
    if output_dir:
        output_path = Path(output_dir)
    else:
        # Create timestamped directory in artifacts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifacts_dir = Path(__file__).parent.parent / "artifacts"
        output_path = artifacts_dir / f"camera_recording_{timestamp}"

    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def display_stream_with_recording(
    camera: WebcamCapture, window_name: str, record: bool, output_dir: Path = None, save_n_frames: int = 10
):
    """
    Display the video stream with optional recording capability.

    Args:
        camera: WebcamCapture instance
        window_name: Name of the display window
        record: Whether to record frames
        output_dir: Directory to save frames (if recording)
        save_n_frames: Save every Nth frame
    """
    if not camera.start_streaming():
        return

    frame_count = 0
    saved_count = 0
    last_frame_hash = None
    start_time = time.time() if record else None
    last_status_update = time.time() if record else None

    print(f"Displaying stream in window '{window_name}'. Press 'q' to quit.")
    if record:
        print(f"🎬 Recording enabled - saving every {save_n_frames} frame(s) to: {output_dir}")

    try:
        while True:
            frame = camera.get_latest_frame()
            if frame is not None:
                # Check if this is a new frame by computing a simple hash
                current_frame_hash = hash(frame.tobytes())

                # Only process if we have a new frame
                if current_frame_hash != last_frame_hash:
                    cv2.imshow(window_name, frame)
                    frame_count += 1
                    last_frame_hash = current_frame_hash

                    # Save frame if recording is enabled
                    if record and frame_count % save_n_frames == 0:
                        filename = output_dir / f"frame_{frame_count:06d}.jpg"
                        success = cv2.imwrite(str(filename), frame)
                        if success:
                            saved_count += 1

                    # Update status display every 0.5 seconds if recording
                    if record and start_time is not None:
                        current_time = time.time()
                        if current_time - last_status_update >= 0.5:
                            elapsed = current_time - start_time
                            minutes = int(elapsed // 60)
                            seconds = elapsed % 60

                            if minutes > 0:
                                time_str = f"{minutes}m {seconds:.1f}s"
                            else:
                                time_str = f"{seconds:.1f}s"

                            print(
                                f"\r📸 Recording: {time_str} | Frames: {frame_count} | Saved: {saved_count}",
                                end="",
                                flush=True,
                            )
                            last_status_update = current_time
                else:
                    # Still need to update display even if frame hasn't changed
                    cv2.imshow(window_name, frame)

            # Check for 'q' key press to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        if record:
            print()  # New line after the dynamic status display
        print("\nInterrupted by user")
    finally:
        if record:
            print()  # New line after the dynamic status display
        cv2.destroyWindow(window_name)
        camera.stop_streaming()

        if record and start_time is not None:
            end_time = time.time()
            duration = end_time - start_time

            # Calculate timing statistics
            minutes = int(duration // 60)
            seconds = duration % 60

            if duration > 0:
                frames_per_second = saved_count / duration
                total_frames_captured = frame_count

                # Format duration string
                if minutes > 0:
                    duration_str = f"{minutes}m {seconds:.1f}s"
                else:
                    duration_str = f"{seconds:.1f}s"

                print(f"🎬 Recording complete: {saved_count} frames saved to {output_dir}")
                print(
                    f"⏱️  Duration: {duration_str} | Total frames: {total_frames_captured} | Saved rate: {frames_per_second:.2f} frames/sec"
                )
            else:
                print(f"🎬 Recording complete: {saved_count} frames saved to {output_dir}")


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

    if args.record:
        output_dir = create_output_directory(args.output_dir)
        print("🎬 Recording: Enabled")
        print(f"📁 Output Directory: {output_dir}")
        print(f"🎞️  Save Rate: Every {args.save_n_frames} frame(s)")
    else:
        output_dir = None
        print("🎬 Recording: Disabled")

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

    # Get and display actual camera FPS
    width, height, actual_fps = camera.get_frame_info()
    print(f"📊 Actual camera settings: {width}x{height} @ {actual_fps:.1f} FPS")

    try:
        # Start streaming with optional recording
        print("🚀 Starting video stream...")
        display_stream_with_recording(camera, args.window_name, args.record, output_dir, args.save_n_frames)

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
