#!/usr/bin/env python3
"""
Simple webcam capture class for loading frames and streaming video.
"""

import threading
import time

import cv2
import numpy as np


class WebcamCapture:
    """
    A simple webcam capture class that can connect to a webcam and capture frames.

    Features:
    - Connect to webcam by index (usually 0 for default camera)
    - Load single frames
    - Stream video with background thread
    - Configure resolution and FPS
    """

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize the webcam capture.

        Args:
            camera_index: Camera device index (0 for default camera)
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps

        self.cap: cv2.VideoCapture | None = None
        self.is_streaming = False
        self.stream_thread: threading.Thread | None = None
        self.latest_frame: np.ndarray | None = None
        self.frame_lock = threading.Lock()

    def connect(self) -> bool:
        """
        Connect to the webcam.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_index}")
                return False

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            print(f"Camera connected: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
            return True

        except Exception as e:
            print(f"Error connecting to camera: {e}")
            return False

    def is_connected(self) -> bool:
        """
        Check if camera is connected.

        Returns:
            bool: True if connected, False otherwise
        """
        return self.cap is not None and self.cap.isOpened()

    def load_frame(self) -> np.ndarray | None:
        """
        Capture and return a single frame from the webcam.

        Returns:
            np.ndarray: Frame as BGR image array, or None if failed
        """
        if not self.is_connected():
            print("Error: Camera not connected")
            return None

        ret, frame = self.cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            return None

        return frame

    def start_streaming(self) -> bool:
        """
        Start streaming video in a background thread.

        Returns:
            bool: True if streaming started successfully, False otherwise
        """
        if not self.is_connected():
            print("Error: Camera not connected")
            return False

        if self.is_streaming:
            print("Already streaming")
            return True

        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()
        print("Video streaming started")
        return True

    def stop_streaming(self):
        """Stop video streaming."""
        if self.is_streaming:
            self.is_streaming = False
            if self.stream_thread and self.stream_thread.is_alive():
                self.stream_thread.join(timeout=2.0)
            print("Video streaming stopped")

    def get_latest_frame(self) -> np.ndarray | None:
        """
        Get the latest frame from the streaming thread.

        Returns:
            np.ndarray: Latest frame as BGR image array, or None if no frame available
        """
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def _stream_loop(self):
        """Internal streaming loop that runs in background thread."""
        while self.is_streaming and self.is_connected():
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
            else:
                print("Warning: Failed to capture frame in streaming loop")
                time.sleep(0.01)  # Small delay to prevent busy loop

    def display_stream(self, window_name: str = "Webcam Stream") -> None:
        """
        Display the video stream in a window. Press 'q' to quit.

        Args:
            window_name: Name of the display window
        """
        if not self.start_streaming():
            return

        print(f"Displaying stream in window '{window_name}'. Press 'q' to quit.")

        try:
            while True:
                frame = self.get_latest_frame()
                if frame is not None:
                    cv2.imshow(window_name, frame)

                # Check for 'q' key press to quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cv2.destroyWindow(window_name)
            self.stop_streaming()

    def save_frame(self, filename: str) -> bool:
        """
        Save current frame to file.

        Args:
            filename: Path to save the image file

        Returns:
            bool: True if saved successfully, False otherwise
        """
        frame = self.load_frame()
        if frame is not None:
            success = cv2.imwrite(filename, frame)
            if success:
                print(f"Frame saved to {filename}")
            return success
        return False

    def get_frame_info(self) -> tuple[int, int, float]:
        """
        Get current frame dimensions and FPS.

        Returns:
            Tuple[int, int, float]: (width, height, fps)
        """
        if not self.is_connected():
            return (0, 0, 0.0)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        return (width, height, fps)

    def disconnect(self):
        """Disconnect from the webcam and clean up resources."""
        self.stop_streaming()

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        cv2.destroyAllWindows()
        print("Camera disconnected")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


def main():
    """Example usage of the WebcamCapture class."""
    print("WebcamCapture Example")
    print("====================")

    # Example 1: Basic frame capture
    print("\n1. Basic frame capture:")
    camera = WebcamCapture(camera_index=0, width=640, height=480, fps=30)

    if camera.connect():
        # Load and save a single frame
        frame = camera.load_frame()
        if frame is not None:
            print(f"Captured frame shape: {frame.shape}")
            camera.save_frame("captured_frame.jpg")

        # Get camera info
        width, height, fps = camera.get_frame_info()
        print(f"Camera info: {width}x{height} @ {fps:.1f} FPS")

        camera.disconnect()

    # Example 2: Using context manager for streaming
    print("\n2. Video streaming with context manager:")
    print("Press 'q' in the video window to quit")

    with WebcamCapture(camera_index=0) as cam:
        if cam.is_connected():
            cam.display_stream("My Webcam")


if __name__ == "__main__":
    main()
