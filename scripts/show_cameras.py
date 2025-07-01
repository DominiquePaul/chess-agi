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
    import tkinter as tk
    from tkinter import ttk

    from PIL import Image, ImageTk

    GUI_AVAILABLE = True
except ImportError:
    print("Warning: tkinter or PIL not available. GUI display will be disabled.")
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


class CameraDisplayWindow:
    """GUI window to display all discovered cameras and their captured images."""

    def __init__(self, cameras: list[CameraInfo]):
        self.cameras = cameras
        self.root = tk.Tk()
        self.root.title(f"Camera Scanner - {len(cameras)} Camera(s) Found")
        self.root.geometry("800x600")

        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        # Title
        title_label = tk.Label(
            self.root,
            text=f"🎥 Camera Scanner Results - {len(self.cameras)} Camera(s) Found",
            font=("Arial", 16, "bold"),
            pady=10,
        )
        title_label.pack()

        if not self.cameras:
            # No cameras found
            no_camera_label = tk.Label(self.root, text="❌ No working cameras found", font=("Arial", 14), fg="red")
            no_camera_label.pack(pady=50)

            close_button = tk.Button(self.root, text="Close", command=self.root.destroy, font=("Arial", 12), pady=5)
            close_button.pack()
            return

        # Create scrollable frame for cameras
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Add cameras to scrollable frame
        for i, camera in enumerate(self.cameras):
            self.add_camera_display(scrollable_frame, camera, i)

        # Pack scrollable components
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")

        # Close button
        button_frame = tk.Frame(self.root)
        button_frame.pack(side="bottom", pady=10)

        close_button = tk.Button(button_frame, text="Close", command=self.root.destroy, font=("Arial", 12), pady=5)
        close_button.pack()

    def add_camera_display(self, parent: tk.Widget, camera: CameraInfo, index: int):
        """Add a camera display section to the parent widget."""
        # Camera frame
        camera_frame = tk.LabelFrame(parent, text=f"Camera {camera.port}", font=("Arial", 12, "bold"), padx=10, pady=10)
        camera_frame.pack(fill="x", padx=5, pady=5)

        # Create a horizontal layout
        content_frame = tk.Frame(camera_frame)
        content_frame.pack(fill="x")

        # Info section (left side)
        info_frame = tk.Frame(content_frame)
        info_frame.pack(side="left", fill="y", padx=(0, 10))

        info_text = f"""📹 Port: {camera.port}
📐 Resolution: {camera.width} x {camera.height}
⚡ FPS: {camera.fps:.1f}
🕒 Captured: {camera.timestamp.strftime("%H:%M:%S")}"""

        info_label = tk.Label(info_frame, text=info_text, font=("Courier", 10), justify="left", anchor="nw")
        info_label.pack()

        # Image section (right side)
        if camera.image is not None:
            image_frame = tk.Frame(content_frame)
            image_frame.pack(side="right")

            # Convert image for display
            display_image = self.prepare_image_for_display(camera.image, max_size=(300, 200))

            if display_image:
                photo = ImageTk.PhotoImage(display_image)
                image_label = tk.Label(image_frame, image=photo)  # type: ignore
                image_label.image = photo  # type: ignore # Keep a reference
                image_label.pack()
        else:
            # No image available
            no_image_label = tk.Label(content_frame, text="❌ No image captured", font=("Arial", 10), fg="red")
            no_image_label.pack(side="right")

    def prepare_image_for_display(self, cv_image: np.ndarray, max_size: tuple[int, int]) -> Image.Image | None:
        """Convert OpenCV image to PIL Image for tkinter display."""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_image)

            # Resize to fit max_size while maintaining aspect ratio
            pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)

            return pil_image
        except Exception as e:
            print(f"Warning: Failed to prepare image for display: {e}")
            return None

    def show(self):
        """Display the window."""
        # Center the window
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

        self.root.mainloop()


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
            window = CameraDisplayWindow(cameras)
            window.show()
        except Exception as e:
            print(f"Warning: Failed to show GUI: {e}")
    elif not GUI_AVAILABLE:
        print("\n⚠️  GUI libraries not available. Install tkinter and PIL for visual display.")
    elif args.no_gui:
        print("\n📝 GUI disabled by user.")

    print("\n✅ Camera scan complete!")


if __name__ == "__main__":
    main()
