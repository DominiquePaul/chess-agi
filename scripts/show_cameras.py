#!/usr/bin/env python3
"""
Camera Port Scanner and Display Script

This script scans all available camera ports, captures a photo from each working camera,
and displays them in a popup window showing all available cameras.

USAGE:
======
# Scan all cameras and show popup with captured images
python scripts/show_cameras.py

# Scan with custom resolution
python scripts/show_cameras.py --resolution 640x480

# Save captured images to directory
python scripts/show_cameras.py --save-images --output-dir ./camera_captures

# Scan specific port range
python scripts/show_cameras.py --max-cameras 5
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    import matplotlib.pyplot as plt

    GUI_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not available. GUI display will be disabled.")
    GUI_AVAILABLE = False

from camera import WebcamCapture


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Scan camera ports and display available cameras",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--max-cameras", type=int, default=10, help="Maximum number of camera ports to scan (default: 10)"
    )

    parser.add_argument(
        "--resolution", type=str, default="640x480", help="Camera resolution in WIDTHxHEIGHT format (default: 640x480)"
    )

    parser.add_argument(
        "--timeout", type=float, default=2.0, help="Timeout in seconds for camera connection attempts (default: 2.0)"
    )

    parser.add_argument("--save-images", action="store_true", help="Save captured images to disk")

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for saved images. If not specified, creates timestamped directory",
    )

    parser.add_argument("--no-gui", action="store_true", help="Disable GUI popup and only print results to console")

    return parser.parse_args()


class CameraInfo:
    """Container for camera information and captured image."""

    def __init__(self, port: int, width: int, height: int, fps: float, image: np.ndarray | None = None):
        self.port = port
        self.width = width
        self.height = height
        self.fps = fps
        self.image = image
        self.timestamp = datetime.now()

    def __str__(self):
        return f"Camera {self.port}: {self.width}x{self.height} @ {self.fps:.1f}FPS"


def scan_camera_ports(max_cameras: int, resolution: str, timeout: float) -> list[CameraInfo]:
    """
    Scan for available camera ports and capture images.

    Args:
        max_cameras: Maximum number of ports to scan
        resolution: Resolution string in "WIDTHxHEIGHT" format
        timeout: Connection timeout in seconds

    Returns:
        List of CameraInfo objects for working cameras
    """
    # Parse resolution
    try:
        width, height = map(int, resolution.split("x"))
    except ValueError:
        print(f"Error: Invalid resolution format '{resolution}'. Using 640x480.")
        width, height = 640, 480

    working_cameras = []

    print(f"🔍 Scanning camera ports 0-{max_cameras - 1}...")
    print(f"📐 Testing resolution: {width}x{height}")
    print(f"⏱️  Connection timeout: {timeout}s")
    print()

    for port in range(max_cameras):
        print(f"Testing camera port {port}...", end=" ", flush=True)

        camera = WebcamCapture(camera_index=port, width=width, height=height)

        try:
            # Set a reasonable timeout for connection attempts
            start_time = time.time()
            connected = camera.connect()
            connection_time = time.time() - start_time

            if connected and connection_time < timeout:
                # Get camera info
                actual_width, actual_height, actual_fps = camera.get_frame_info()

                # Capture a test frame
                frame = camera.load_frame()
                if frame is not None:
                    camera_info = CameraInfo(port, actual_width, actual_height, actual_fps, frame)
                    working_cameras.append(camera_info)
                    print(f"✅ Found - {actual_width}x{actual_height} @ {actual_fps:.1f}FPS")
                else:
                    print("❌ Failed to capture frame")
            else:
                print("❌ Connection failed or timeout")

        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            try:
                camera.disconnect()
            except Exception:
                pass

    print()
    print(f"🎯 Found {len(working_cameras)} working camera(s)")
    return working_cameras


def save_camera_images(cameras: list[CameraInfo], output_dir: str | None = None) -> Path:
    """Save captured images to disk."""
    if output_dir:
        save_path = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifacts_dir = Path(__file__).parent.parent / "artifacts"
        save_path = artifacts_dir / f"camera_scan_{timestamp}"

    save_path.mkdir(parents=True, exist_ok=True)

    for camera in cameras:
        if camera.image is not None:
            filename = save_path / f"camera_{camera.port}_{camera.width}x{camera.height}.jpg"
            success = cv2.imwrite(str(filename), camera.image)
            if success:
                print(f"💾 Saved: {filename}")

    return save_path


def display_cameras_matplotlib(cameras: list[CameraInfo]):
    """Display all discovered cameras and their captured images using matplotlib."""
    if not cameras:
        print("❌ No cameras to display")
        return

    # Calculate grid layout
    n_cameras = len(cameras)
    if n_cameras == 1:
        rows, cols = 1, 1
    elif n_cameras == 2:
        rows, cols = 1, 2
    elif n_cameras <= 4:
        rows, cols = 2, 2
    elif n_cameras <= 6:
        rows, cols = 2, 3
    elif n_cameras <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = 4, int(np.ceil(n_cameras / 4))

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    fig.suptitle(f"🎥 Camera Scanner Results - {n_cameras} Camera(s) Found", fontsize=16, fontweight="bold")

    # Handle single subplot case
    if n_cameras == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    else:
        axes = axes.flatten()

    # Display each camera
    for i, camera in enumerate(cameras):
        ax = axes[i]

        if camera.image is not None:
            # Convert BGR to RGB for matplotlib
            rgb_image = cv2.cvtColor(camera.image, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb_image)
        else:
            # Show placeholder for missing image
            ax.text(
                0.5,
                0.5,
                "❌ No Image\nCaptured",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color="red",
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        # Set title with camera info
        title = f"Camera {camera.port}\n{camera.width}×{camera.height} @ {camera.fps:.1f}FPS\n{camera.timestamp.strftime('%H:%M:%S')}"
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    # Hide unused subplots
    for i in range(n_cameras, len(axes)):
        axes[i].set_visible(False)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for main title

    # Show the plot
    print("🖼️  Displaying camera images in matplotlib window...")
    plt.show()


def print_camera_summary(cameras: list[CameraInfo]):
    """Print a summary of found cameras to console."""
    print("=" * 60)
    print("🎥 CAMERA SCAN SUMMARY")
    print("=" * 60)

    if not cameras:
        print("❌ No working cameras found")
        return

    for i, camera in enumerate(cameras, 1):
        print(f"\n📹 Camera #{i} (Port {camera.port}):")
        print(f"   📐 Resolution: {camera.width} x {camera.height}")
        print(f"   ⚡ FPS: {camera.fps:.1f}")
        print(f"   🕒 Captured: {camera.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   🖼️  Image: {'✅ Available' if camera.image is not None else '❌ Not captured'}")

    print(f"\n🎯 Total working cameras: {len(cameras)}")


def main():
    """Main function."""
    args = parse_args()

    print("🎥 Camera Port Scanner")
    print("=" * 40)

    # Scan for cameras
    cameras = scan_camera_ports(args.max_cameras, args.resolution, args.timeout)

    # Print summary
    print_camera_summary(cameras)

    # Save images if requested
    if args.save_images and cameras:
        print("\n💾 Saving captured images...")
        save_path = save_camera_images(cameras, args.output_dir)
        print(f"📁 Images saved to: {save_path}")

    # Show GUI if available and not disabled
    if not args.no_gui and GUI_AVAILABLE and cameras:
        print("\n🖼️  Opening camera display window...")
        try:
            display_cameras_matplotlib(cameras)
        except Exception as e:
            print(f"Warning: Failed to show GUI: {e}")
    elif not GUI_AVAILABLE:
        print("\n⚠️  GUI libraries not available. Install matplotlib for visual display.")
    elif args.no_gui:
        print("\n📝 GUI disabled by user.")

    print("\n✅ Camera scan complete!")


if __name__ == "__main__":
    main()
