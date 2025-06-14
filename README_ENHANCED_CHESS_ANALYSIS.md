# Enhanced Chess Board Analysis

This document describes the enhanced chess board analysis system that integrates YOLO-based piece detection with HuggingFace Hub model loading.

## 🚀 Key Features

### Enhanced Piece Detection
- **YOLO-based Detection**: Uses state-of-the-art YOLO models for accurate chess piece recognition
- **HuggingFace Hub Integration**: Load models directly from HuggingFace Hub
- **Automatic Square Mapping**: Intelligent mapping of detected pieces to chess board squares
- **Chess Position Generation**: Automatic creation of chess.Board objects from detected pieces

### Model Loading Options
- **Local Models**: Load models from local file paths
- **HuggingFace Models**: Load models from HuggingFace Hub (e.g., `dopaul/chess_piece_detector`)
- **Fallback Support**: Graceful handling when models are not available

### Comprehensive Analysis
- **Board Detection**: Accurate chess board corner detection using segmentation
- **Piece Classification**: Recognition of all 12 chess piece types (6 pieces × 2 colors)
- **Position Validation**: Integration with python-chess for position validation
- **Visualization**: Rich visualization of detected boards, pieces, and analysis results

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For HuggingFace Hub support
pip install huggingface_hub
```

## 🔧 Usage

### Basic Usage

```python
from src.chess_board_detection.chess_board_analyzer import ChessBoardAnalyzer

# Initialize with HuggingFace models
analyzer = ChessBoardAnalyzer(
    segmentation_model="dopaul/chess_board_segmentation",
    piece_detection_model="dopaul/chess_piece_detector"
)

# Analyze a chess board image
result = analyzer.analyze_board("path/to/chess_image.jpg")

# Access results
position = result['chess_position']  # chess.Board object
pieces = result['detected_pieces']   # List of detected pieces
corners = result['board_corners']    # Board corner coordinates
```

### CLI Usage

```bash
# Basic analysis with HuggingFace models
python examples/analyze_chess_board.py --image chess.jpg

# With custom confidence threshold
python examples/analyze_chess_board.py --image chess.jpg --conf 0.6

# Use specific models
python examples/analyze_chess_board.py \
    --image chess.jpg \
    --piece-model dopaul/chess_piece_detector \
    --segmentation-model dopaul/chess_board_segmentation

# Verbose output with detailed results
python examples/analyze_chess_board.py --image chess.jpg --verbose

# Get coordinates for specific squares
python examples/analyze_chess_board.py --image chess.jpg --squares e4,d4,a1,h8
```

### Direct Model Usage

```python
from src.chess_piece_detection.model import ChessModel

# Load model from HuggingFace Hub
model = ChessModel("dopaul/chess_piece_detector")

# Detect pieces and map to squares
piece_mappings, detection_image = model.detect_pieces_in_squares(
    image="chess.jpg",
    square_coordinates=square_coords,
    conf_threshold=0.5
)

# Create chess position
chess_position = model.create_chess_position(piece_mappings)
```

## 🎯 Model Architecture

### ChessModel Class
- **Base**: Inherits from `BaseYOLOModel`
- **Detection**: YOLO-based object detection
- **Classification**: 12-class chess piece classification
- **Mapping**: Intelligent piece-to-square mapping

### Supported Piece Classes
```python
class_dict = {
    0: "black-bishop",   1: "black-king",     2: "black-knight",
    3: "black-pawn",     4: "black-queen",    5: "black-rook",
    6: "white-bishop",   7: "white-king",     8: "white-knight",
    9: "white-pawn",     10: "white-queen",   11: "white-rook",
}
```

## 🤗 HuggingFace Integration

### Model Loading
The system automatically detects HuggingFace model names and downloads them:

```python
# This will download from HuggingFace Hub
model = ChessModel("dopaul/chess_piece_detector")

# This will load from local path
model = ChessModel("models/local_model.pt")
```

### Caching
Models are cached locally in `models/huggingface_cache/` for faster subsequent loads.

## 📊 Output Format

### Analysis Result Structure
```python
{
    'chess_position': chess.Board,           # Chess position object
    'detected_pieces': [(square, class)],    # List of detected pieces
    'position_string': str,                  # FEN string representation
    'board_corners': {                       # Board corner coordinates
        'top_left': (x, y),
        'top_right': (x, y),
        'bottom_right': (x, y),
        'bottom_left': (x, y)
    },
    'square_coordinates': {                  # Pixel coordinates for each square
        1: [(center), (corners)...],
        2: [(center), (corners)...],
        # ... squares 1-64
    },
    'processing_info': {                     # Processing metadata
        'original_dimensions': (width, height),
        'piece_detection_model': str,
        'confidence_threshold': float,
        # ...
    },
    'visualizations': {                      # Visualization images
        'original_image': np.ndarray,
        'detection_image': np.ndarray,
        'board_with_grid': np.ndarray,
        # ...
    }
}
```

## 🎨 Visualization Features

### Available Visualizations
- **Board Detection**: Corner detection and board extraction
- **Piece Detection**: Bounding boxes and piece classifications
- **Grid Overlay**: Chess board grid with square numbers
- **Complete Analysis**: Combined visualization of all results

### Customization Options
```python
# Create custom visualization
vis_image = analyzer.visualize_result(
    result,
    show_pieces=True,      # Show piece detections
    show_grid=True,        # Show chess board grid
    show_corners=True      # Show board corners
)
```

## 🔍 Example Workflows

### 1. Complete Analysis Pipeline
```python
# Initialize analyzer
analyzer = ChessBoardAnalyzer(
    piece_detection_model="dopaul/chess_piece_detector"
)

# Analyze image
result = analyzer.analyze_board("chess_position.jpg")

# Get chess position
if result['chess_position']:
    board = result['chess_position']
    print(f"FEN: {board.fen()}")
    print(f"Legal moves: {len(list(board.legal_moves))}")
```

### 2. Piece Detection Only
```python
# Load piece detection model
model = ChessModel("dopaul/chess_piece_detector")

# Detect pieces (requires pre-computed square coordinates)
pieces, detection_image = model.detect_pieces_in_squares(
    image="chess.jpg",
    square_coordinates=squares,
    conf_threshold=0.5
)

# Get piece names
for square_num, piece_class in pieces:
    piece_name = model.get_piece_name(piece_class)
    print(f"Square {square_num}: {piece_name}")
```

### 3. Position Analysis
```python
# Analyze and get detailed position information
result = analyzer.analyze_board("chess.jpg")

# Get piece at specific square
piece = analyzer.get_piece_at_square(result, "e4")
print(f"Piece at e4: {piece}")

# Get square coordinates
coords = analyzer.get_square_coordinates(result, "e4")
print(f"e4 center: {coords['center']}")
```

## 🛠️ Advanced Configuration

### Model Parameters
```python
analyzer = ChessBoardAnalyzer(
    segmentation_model="dopaul/chess_board_segmentation",
    piece_detection_model="dopaul/chess_piece_detector",
    corner_method="approx"  # or "extended"
)
```

### Detection Parameters
```python
result = analyzer.analyze_board(
    input_image="chess.jpg",
    conf_threshold=0.5,      # Confidence threshold
    target_size=512          # Processing image size
)
```

## 🐛 Troubleshooting

### Common Issues

1. **HuggingFace Model Loading**
   ```bash
   # Ensure huggingface_hub is installed
   pip install huggingface_hub
   ```

2. **Model Not Found**
   - Check if model exists on HuggingFace Hub
   - Verify model name spelling
   - Check internet connection for downloads

3. **Detection Issues**
   - Adjust confidence threshold
   - Ensure good image quality
   - Check if chess board is clearly visible

## 📈 Performance Considerations

### Model Loading
- **First Load**: Downloads from HuggingFace Hub (may take time)
- **Subsequent Loads**: Uses cached models (faster)
- **Local Models**: Fastest loading option

### Inference Speed
- **Image Size**: Larger images take longer to process
- **Confidence Threshold**: Lower thresholds increase processing time
- **Device**: GPU acceleration recommended for best performance

## 🔮 Future Enhancements

### Planned Features
- **Move Prediction**: Integration with chess engines for move suggestions
- **Live Analysis**: Real-time analysis from camera feeds
- **Multi-board Support**: Analysis of multiple boards in single image
- **Enhanced Visualization**: 3D board representations

### Model Improvements
- **Accuracy**: Continuous improvement of piece detection accuracy
- **Speed**: Optimization for faster inference
- **Robustness**: Better handling of difficult lighting conditions

## 📝 Contributing

When contributing to the enhanced chess analysis system:

1. **Model Training**: Use consistent class mappings
2. **Testing**: Test with diverse chess positions
3. **Documentation**: Update this README for new features
4. **Compatibility**: Maintain backward compatibility where possible

## 🎯 Conclusion

The enhanced chess board analysis system provides a comprehensive solution for:
- Accurate chess piece detection
- Seamless model loading from HuggingFace Hub
- Complete chess position analysis
- Rich visualization capabilities

This system is designed to be both powerful for advanced users and accessible for beginners, with clear APIs and comprehensive documentation. 