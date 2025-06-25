# Live Chess Analysis Streaming Tool

Stream live chess analysis from your webcam with real-time visualization of board detection, piece recognition, and move prediction.

## Features

- **Real-time Analysis**: 4 FPS chess analysis from webcam feed
- **720p Display**: High-quality 1280x720 visualization
- **Recording**: Save analysis sessions to MP4 video files
- **Move Prediction**: Real-time chess engine analysis with Stockfish or simple engine
- **Fallback Handling**: Graceful display when no board is detected
- **Performance Monitoring**: Live FPS and analysis statistics

## Quick Start

```bash
# Basic streaming
python scripts/stream_chess_analysis.py --camera 0

# With move prediction
python scripts/stream_chess_analysis.py --camera 0 --computer-playing-as white

# With recording
python scripts/stream_chess_analysis.py --camera 0 --record --output-dir recordings/
```

## Usage Examples

### Basic Analysis
```bash
# Simple board detection and piece recognition
python scripts/stream_chess_analysis.py --camera 0 --conf 0.6
```

### Advanced Analysis with Move Prediction
```bash
# Computer playing as white with Stockfish
python scripts/stream_chess_analysis.py \
    --camera 0 \
    --computer-playing-as white \
    --engine-type stockfish \
    --stockfish-skill 15

# Computer playing as black with simple engine
python scripts/stream_chess_analysis.py \
    --camera 0 \
    --computer-playing-as black \
    --engine-type simple
```

### Recording Sessions
```bash
# Record analysis session
python scripts/stream_chess_analysis.py \
    --camera 0 \
    --record \
    --output-dir recordings/ \
    --computer-playing-as white

# Headless recording (no display window)
python scripts/stream_chess_analysis.py \
    --camera 0 \
    --record \
    --no-display \
    --fps 2
```

### Camera Perspectives
```bash
# Camera positioned with white at top (180° rotated board)
python scripts/stream_chess_analysis.py --camera 0 --white-playing-from t

# Camera positioned from side (white at left)
python scripts/stream_chess_analysis.py --camera 0 --white-playing-from l
```

### Performance Tuning
```bash
# Lower FPS for slower systems
python scripts/stream_chess_analysis.py --camera 0 --fps 2

# Skip piece detection for faster processing
python scripts/stream_chess_analysis.py --camera 0 --skip-piece-detection

# Verbose output for debugging
python scripts/stream_chess_analysis.py --camera 0 --verbose
```

## Controls

When running the stream, you can use these keyboard controls:

- **'q'** - Quit the application
- **'r'** - Force re-analysis of current frame
- **'s'** - Save current analysis (future feature)

## Command Line Options

### Camera Settings
- `--camera, -c` - Camera device index (default: 0)
- `--resolution` - Camera resolution like "1280x720" (default: 1280x720)
- `--fps` - Analysis frames per second (default: 4)

### Analysis Settings
- `--conf` - Confidence threshold for piece detection (default: 0.5)
- `--white-playing-from` - Camera perspective: 'b' (bottom), 't' (top), 'l' (left), 'r' (right)
- `--threshold` - Board expansion percentage (0-100, default: 0)
- `--skip-piece-detection` - Only detect board, skip piece detection

### Move Prediction
- `--computer-playing-as` - Color for computer analysis ('white' or 'black')
- `--engine-type` - Chess engine: 'stockfish' or 'simple' (default: stockfish)
- `--stockfish-skill` - Stockfish skill level 0-20 (default: 20)
- `--engine-depth` - Maximum search depth (default: 10)
- `--engine-time` - Engine thinking time in seconds (default: 1.0)

### Recording & Output
- `--record` - Enable video recording
- `--output-dir` - Recording directory (default: recordings)
- `--no-display` - Run without display window (for headless recording)

### Debug Options
- `--verbose, -v` - Show detailed analysis information
- `--use-geometric-center` - Use geometric center instead of weighted center

## Output

### Display Window
The live display shows a 2x2 grid with:
1. **Board Corners and Chess Grid** - Detected board outline and square grid
2. **Chess Piece Bounding Boxes** - Detected pieces with confidence scores
3. **Chess Piece Centers** - Piece center points mapped to squares
4. **Chess Position Diagram** - Traditional chess board view with pieces

### Performance Overlay
The display includes real-time statistics:
- Frame count and analysis count
- Camera FPS and analysis FPS
- Recording status indicator

### Recording Files
When recording is enabled, videos are saved as:
```
recordings/chess_analysis_YYYYMMDD_HHMMSS.mp4
```

## Technical Details

### Performance
- **Analysis FPS**: Configurable (default: 4 FPS)
- **Display Resolution**: 1280x720
- **Camera FPS**: 30 FPS (analysis sub-sampled)
- **Analysis Latency**: ~250ms per analysis cycle

### Fallback Behavior
When no chess board is detected:
- Shows original camera feed in all 4 quadrants
- Maintains last successful analysis for subsequent frames
- Graceful error handling without crashing

### Dependencies
- OpenCV for camera capture and display
- Matplotlib for visualization generation
- ChessBoardAnalyzer for analysis
- Stockfish for chess engine (optional)

## Troubleshooting

### Camera Issues
```bash
# List available cameras
python -c "import cv2; [print(f'Camera {i}: {cv2.VideoCapture(i).isOpened()}') for i in range(10)]"

# Test specific camera
python scripts/stream_chess_analysis.py --camera 1 --verbose
```

### Performance Issues
```bash
# Reduce analysis frequency
python scripts/stream_chess_analysis.py --camera 0 --fps 2

# Skip piece detection
python scripts/stream_chess_analysis.py --camera 0 --skip-piece-detection

# Use simple engine instead of Stockfish
python scripts/stream_chess_analysis.py --camera 0 --engine-type simple
```

### Analysis Issues
```bash
# Increase confidence threshold
python scripts/stream_chess_analysis.py --camera 0 --conf 0.7

# Try different board expansion
python scripts/stream_chess_analysis.py --camera 0 --threshold 10

# Enable verbose logging
python scripts/stream_chess_analysis.py --camera 0 --verbose
```

## Integration

This tool reuses the same analysis and visualization code as `analyze_chess_board.py`:
- `ChessBoardAnalyzer` for analysis
- `create_combined_visualization_array()` for visualization
- Same CLI argument structure for consistency

The streaming approach provides a live, interactive version of the batch analysis workflow.
