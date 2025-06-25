#!/usr/bin/env python3
"""
Live Chess Analysis Streaming Tool

Stream live chess analysis from webcam with real-time visualization.

USAGE:
======
# Basic streaming
python scripts/stream_chess_analysis.py --camera 0

# With analysis options
python scripts/stream_chess_analysis.py --camera 0 --conf 0.6 --computer-playing-as white

# With recording
python scripts/stream_chess_analysis.py --camera 0 --record --output-dir recordings/

# Custom settings
python scripts/stream_chess_analysis.py --camera 0 --fps 2 --resolution 1280x720 --verbose
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from src.camera import WebcamCapture
from src.chess_board_detection.chess_board_analyzer import ChessBoardAnalyzer
from src.datatypes import ChessAnalysis
from src.visualisation import create_combined_visualization_array

load_dotenv()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Live chess analysis streaming from webcam",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Camera settings
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument(
        "--resolution",
        type=str,
        default="1280x720",
        help="Camera resolution in WIDTHxHEIGHT format (default: 1280x720)",
    )
    parser.add_argument("--fps", type=int, default=4, help="Analysis frames per second (default: 4)")

    # Analysis settings (similar to analyze_chess_board.py)
    parser.add_argument(
        "--segmentation-model",
        type=str,
        default="dopaul/chess_board_segmentation",
        help="Segmentation model name or path (default: dopaul/chess_board_segmentation)",
    )
    parser.add_argument(
        "--piece-model",
        type=str,
        default="dopaul/chess_piece_detection",
        help="Piece detection model path or HuggingFace model name",
    )
    parser.add_argument(
        "--corner-method",
        type=str,
        default="approx",
        choices=["approx", "extended"],
        help="Corner extraction method (default: approx)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for piece detection (default: 0.5)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="Percentage expansion of chess board from center (0-100, default: 0)",
    )
    parser.add_argument(
        "--white-playing-from",
        type=str,
        default="b",
        choices=["b", "t", "l", "r"],
        help="Camera perspective - side where white is playing from: 'b' (bottom), 't' (top), 'l' (left), 'r' (right) (default: b)",
    )
    parser.add_argument(
        "--use-weighted-center",
        action="store_true",
        default=True,
        help="Use weighted center coordinates for piece mapping (default: True)",
    )
    parser.add_argument(
        "--use-geometric-center",
        action="store_true",
        help="Use geometric center coordinates instead of weighted center",
    )

    # Move prediction options
    parser.add_argument(
        "--computer-playing-as",
        type=str,
        choices=["white", "black"],
        help="Color the computer is playing as for move prediction ('white' or 'black')",
    )
    parser.add_argument(
        "--engine-type",
        type=str,
        choices=["stockfish", "simple"],
        default="stockfish",
        help="Chess engine to use for move prediction (default: stockfish)",
    )
    parser.add_argument(
        "--engine-depth",
        type=int,
        default=10,
        help="Maximum search depth for chess engine (default: 10)",
    )
    parser.add_argument(
        "--engine-time",
        type=float,
        default=1.0,
        help="Maximum time in seconds for engine to think (default: 1.0)",
    )
    parser.add_argument(
        "--stockfish-skill",
        type=int,
        default=20,
        help="Stockfish skill level 0-20, where 20 is strongest (default: 20)",
    )

    # Recording and output
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record the analysis session to video file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="recordings",
        help="Directory to save recordings (default: recordings)",
    )
    parser.add_argument(
        "--skip-piece-detection",
        action="store_true",
        help="Skip piece detection, only detect board and grid",
    )

    # Display and debug options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed analysis information",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without display (useful for headless recording)",
    )

    return parser.parse_args()


class ChessAnalysisStreamer:
    """Live chess analysis streaming class."""

    def __init__(self, args):
        self.args = args
        self.camera: WebcamCapture | None = None
        self.analyzer: ChessBoardAnalyzer | None = None
        self.last_analysis: ChessAnalysis | None = None
        self.video_writer: cv2.VideoWriter | None = None
        self.recording_path: Path | None = None

        # Performance tracking
        self.frame_count = 0
        self.analysis_count = 0
        self.last_analysis_time = 0
        self.start_time = time.time()

        # Display settings
        self.display_width = 1280
        self.display_height = 720

        # Parse camera resolution
        try:
            width, height = map(int, args.resolution.split("x"))
            self.camera_width = width
            self.camera_height = height
        except ValueError:
            print(f"⚠️  Invalid resolution format: {args.resolution}, using default 1280x720")
            self.camera_width = 1280
            self.camera_height = 720

    def setup_camera(self) -> bool:
        """Initialize camera connection."""
        print(f"🎥 Connecting to camera {self.args.camera}...")

        self.camera = WebcamCapture(
            camera_index=self.args.camera,
            width=self.camera_width,
            height=self.camera_height,
            fps=30,  # Camera FPS (we'll control analysis FPS separately)
        )

        if not self.camera.connect():
            print(f"❌ Failed to connect to camera {self.args.camera}")
            return False

        print(f"✅ Camera connected: {self.camera_width}x{self.camera_height}")
        return True

    def setup_analyzer(self) -> bool:
        """Initialize chess board analyzer."""
        print("🧠 Initializing chess board analyzer...")

        try:
            piece_model = None if self.args.skip_piece_detection else self.args.piece_model
            self.analyzer = ChessBoardAnalyzer(
                segmentation_model=self.args.segmentation_model,
                piece_detection_model=piece_model,
                corner_method=self.args.corner_method,
                threshold=self.args.threshold,
                white_playing_from=self.args.white_playing_from,
                computer_playing_as=self.args.computer_playing_as,
                engine_type=self.args.engine_type,
                engine_depth=self.args.engine_depth,
                engine_time_limit=self.args.engine_time,
                stockfish_skill_level=self.args.stockfish_skill,
            )

            print("✅ Chess board analyzer initialized")
            return True

        except Exception as e:
            print(f"❌ Failed to initialize analyzer: {e}")
            return False

    def setup_recording(self) -> bool:
        """Setup video recording if requested."""
        if not self.args.record:
            return True

        print("📹 Setting up recording...")

        # Create output directory
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_path = output_dir / f"chess_analysis_{timestamp}.mp4"

        # Setup video writer with more compatible codec
        # Try H264 first (more compatible), fallback to mp4v
        fourcc = cv2.VideoWriter_fourcc(*"H264")
        self.video_writer = cv2.VideoWriter(
            str(self.recording_path),
            fourcc,
            self.args.fps,  # Recording FPS matches analysis FPS
            (self.display_width, self.display_height),
        )

        # If H264 fails, try mp4v
        if not self.video_writer.isOpened():
            print("⚠️  H264 codec failed, trying mp4v...")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                str(self.recording_path), fourcc, self.args.fps, (self.display_width, self.display_height)
            )

        if not self.video_writer.isOpened():
            print(f"❌ Failed to initialize video recording to {self.recording_path}")
            return False

        print(f"✅ Recording to: {self.recording_path}")
        return True

    def analyze_frame(self, frame: np.ndarray) -> ChessAnalysis | None:
        """Analyze a single frame."""
        try:
            if self.args.verbose:
                print("🔍 Analyzing frame...")

            use_weighted_center = not self.args.use_geometric_center

            analysis = self.analyzer.analyze_board(
                input_image=frame, conf_threshold=self.args.conf, use_weighted_center=use_weighted_center
            )

            self.analysis_count += 1
            return analysis

        except Exception as e:
            if self.args.verbose:
                print(f"⚠️  Analysis failed: {e}")
            return None

    def create_fallback_visualization(self, frame: np.ndarray) -> np.ndarray:
        """Create a simple visualization when analysis fails."""
        # Convert BGR to RGB for matplotlib (matplotlib.imshow expects RGB)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            display_frame = frame

        # Create a simple 2x2 layout with the original image
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # Show original image in all 4 quadrants with different titles
        titles = ["Original Camera Feed", "No Board Detected", "No Pieces Detected", "No Chess Position"]

        for ax, title in zip(axes, titles, strict=False):
            ax.imshow(display_frame)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.axis("off")

        plt.tight_layout()

        # Convert to numpy array (cross-platform compatible)
        fig.canvas.draw()

        # Try different methods for getting the buffer (macOS vs Linux compatibility)
        try:
            # Method 1: buffer_rgba (macOS and modern matplotlib)
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            buf = buf[:, :, :3]  # Remove alpha channel
        except AttributeError:
            try:
                # Method 2: tostring_rgb (older matplotlib or Linux)
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            except AttributeError:
                # Method 3: renderer buffer (fallback)
                renderer = fig.canvas.get_renderer()
                buf = np.frombuffer(renderer.tostring_rgb(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return buf

    def create_visualization(self, analysis: ChessAnalysis) -> np.ndarray:
        """Create visualization from analysis results."""
        use_weighted_center = not self.args.use_geometric_center

        return create_combined_visualization_array(
            chess_analysis=analysis,
            skip_piece_detection=self.args.skip_piece_detection,
            use_weighted_center=use_weighted_center,
            verbose=self.args.verbose,
        )

    def display_frame(self, vis_array: np.ndarray):
        """Display the visualization frame."""
        if self.args.no_display:
            return

        # Resize to display resolution
        if vis_array.shape[:2] != (self.display_height, self.display_width):
            vis_array = cv2.resize(vis_array, (self.display_width, self.display_height))

        # Convert RGB to BGR for OpenCV display
        display_frame = cv2.cvtColor(vis_array, cv2.COLOR_RGB2BGR)

        # Add performance info overlay
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        analysis_fps = self.analysis_count / elapsed_time if elapsed_time > 0 else 0

        info_text = [
            f"Frame: {self.frame_count}",
            f"Analysis: {self.analysis_count}",
            f"Camera FPS: {fps:.1f}",
            f"Analysis FPS: {analysis_fps:.1f}",
        ]

        if self.args.record:
            info_text.append("🔴 RECORDING")

        # Draw info overlay
        y_offset = 30
        for text in info_text:
            cv2.putText(display_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 25

        # Show frame
        cv2.imshow("Live Chess Analysis", display_frame)

    def record_frame(self, vis_array: np.ndarray):
        """Record the visualization frame."""
        if not self.video_writer:
            return

        # Resize to recording resolution
        if vis_array.shape[:2] != (self.display_height, self.display_width):
            vis_array = cv2.resize(vis_array, (self.display_width, self.display_height))

        # Convert RGB to BGR for video writer (OpenCV VideoWriter expects BGR)
        record_frame = cv2.cvtColor(vis_array, cv2.COLOR_RGB2BGR)

        # Debug: On first few frames, log color stats
        if hasattr(self, "_record_frame_count"):
            self._record_frame_count += 1
        else:
            self._record_frame_count = 1

        if self._record_frame_count <= 3 and self.args.verbose:
            print(f"🎥 Recording frame {self._record_frame_count}:")
            print(f"   Input (RGB): shape={vis_array.shape}, mean={vis_array.mean():.1f}")
            print(f"   Output (BGR): shape={record_frame.shape}, mean={record_frame.mean():.1f}")

        self.video_writer.write(record_frame)

    def print_stats(self):
        """Print session statistics."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        print("\n📊 SESSION STATISTICS")
        print(f"   Duration: {elapsed_time:.1f} seconds")
        print(f"   Total frames: {self.frame_count}")
        print(f"   Analysis frames: {self.analysis_count}")
        print(f"   Average camera FPS: {self.frame_count / elapsed_time:.1f}")
        print(f"   Average analysis FPS: {self.analysis_count / elapsed_time:.1f}")

        if self.args.record and self.recording_path:
            print(f"   Recording saved: {self.recording_path}")

    def run(self):
        """Main streaming loop."""
        print("🚀 Starting live chess analysis stream...")
        print("📝 Controls:")
        print("   - Press 'q' to quit")
        print("   - Press 'r' to force re-analysis")
        print("   - Press 's' to save current analysis")

        analysis_interval = 1.0 / self.args.fps  # Time between analyses

        try:
            while True:
                current_time = time.time()

                # Capture frame
                frame = self.camera.load_frame()
                if frame is None:
                    print("⚠️  Failed to capture frame")
                    continue

                self.frame_count += 1

                # Check if it's time for analysis
                should_analyze = (
                    current_time - self.last_analysis_time >= analysis_interval or self.last_analysis is None
                )

                if should_analyze:
                    analysis = self.analyze_frame(frame)
                    if analysis is not None:
                        self.last_analysis = analysis
                    self.last_analysis_time = current_time

                # Create visualization
                if self.last_analysis is not None:
                    vis_array = self.create_visualization(self.last_analysis)
                else:
                    vis_array = self.create_fallback_visualization(frame)

                # Display and record
                self.display_frame(vis_array)
                if self.args.record:
                    self.record_frame(vis_array)

                # Handle keyboard input
                if not self.args.no_display:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("r"):
                        # Force re-analysis
                        self.last_analysis_time = 0
                    elif key == ord("s") and self.last_analysis:
                        # Save current analysis (placeholder for future feature)
                        print("💾 Save feature not implemented yet")

                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n⏹️  Interrupted by user")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        print("🧹 Cleaning up...")

        if self.camera:
            self.camera.disconnect()

        if self.video_writer:
            self.video_writer.release()

        cv2.destroyAllWindows()

        self.print_stats()


def main():
    """Main function."""
    args = parse_args()

    print("🎯 Live Chess Analysis Streamer")
    print("=" * 40)

    # Print configuration
    print(f"📹 Camera: {args.camera} ({args.resolution})")
    print(f"🎯 Analysis FPS: {args.fps}")
    print(f"🔍 Confidence threshold: {args.conf}")

    perspective_names = {"b": "bottom", "t": "top", "l": "left", "r": "right"}
    print(f"📐 White playing from: {perspective_names[args.white_playing_from]}")

    if args.computer_playing_as:
        print(f"🤖 Computer playing as: {args.computer_playing_as}")
        print(f"🧠 Engine: {args.engine_type}")

    if args.record:
        print(f"📹 Recording enabled: {args.output_dir}")

    # Initialize and run streamer
    streamer = ChessAnalysisStreamer(args)

    if not streamer.setup_camera():
        sys.exit(1)

    if not streamer.setup_analyzer():
        sys.exit(1)

    if not streamer.setup_recording():
        sys.exit(1)

    try:
        streamer.run()
    except Exception as e:
        print(f"❌ Error running streamer: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
