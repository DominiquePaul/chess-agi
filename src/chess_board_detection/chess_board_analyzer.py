#!/usr/bin/env python3
"""
Comprehensive Chess Board Analyzer

This module combines segmentation-based corner detection with classical image processing
to provide complete chess board analysis including piece detection and position reading.

Usage:
    # Initialize analyzer with default model
    analyzer = ChessBoardAnalyzer()
    
    # Analyze chess board from image path
    result = analyzer.analyze_board("path/to/chess_image.jpg")
    
    # Analyze chess board from numpy array
    result = analyzer.analyze_board(image_array)
    
    # Use custom model
    analyzer = ChessBoardAnalyzer(segmentation_model="custom/model")
    
    # Get chess position
    position = result['chess_position']
    pieces = result['detected_pieces']
    corners = result['board_corners']
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional
import tempfile
import os

# Import segmentation model
from .yolo.segmentation.segmentation_model import ChessBoardSegmentationModel

# Import classical method utilities
from .classical_method import (
    detect_chess_pieces, 
    map_pieces_to_squares,
    create_chess_board_representation,
    apply_perspective_transformation,
    draw_squares,
    get_square_coordinates,
    visualize_results_on_original,
)


class ChessBoardAnalyzer:
    """
    Comprehensive chess board analyzer that combines segmentation-based corner detection
    with classical image processing for full chess position analysis.
    
    This class provides a unified interface for:
    - Detecting chess board corners using segmentation
    - Extracting chess board using perspective transformation
    - Detecting and classifying chess pieces
    - Reading chess position in standard notation
    - Preparing data structures for future move prediction
    """
    
    def __init__(self, 
                 segmentation_model: str = "dopaul/chess_board_segmentation",
                 piece_detection_model: str = "models/yolo_chess_piece_detector/training_v2/weights/best.pt",
                 corner_method: str = "approx"):
        """
        Initialize the chess board analyzer.
        
        Args:
            segmentation_model: HuggingFace model name or local path for board segmentation
            piece_detection_model: Path to YOLO model for piece detection
            corner_method: Method for corner extraction ('approx', 'extended', etc.)
        """
        self.segmentation_model_name = segmentation_model
        self.piece_detection_model_path = piece_detection_model
        self.corner_method = corner_method
        
        # Initialize segmentation model
        print(f"🚀 Loading segmentation model: {segmentation_model}")
        try:
            # For now, we'll use the local model path approach
            # TODO: Add HuggingFace model loading support
            if segmentation_model == "dopaul/chess_board_segmentation":
                # Use default local model path
                model_path = Path("models/chess_board_segmentation/polygon_segmentation_training/weights/best.pt")
                if not model_path.exists():
                    raise FileNotFoundError(f"Default model not found at {model_path}. Please ensure the model is downloaded.")
                self.segmentation_model = ChessBoardSegmentationModel(model_path=model_path)
            else:
                # Assume it's a local path
                self.segmentation_model = ChessBoardSegmentationModel(model_path=Path(segmentation_model))
            print("✅ Segmentation model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load segmentation model: {e}")
            raise
        
        # Store piece detection model path for later use
        if piece_detection_model is not None:
            self.piece_model_path = Path(piece_detection_model)
            if not self.piece_model_path.exists():
                print(f"⚠️  Warning: Piece detection model not found at {piece_detection_model}")
                print("    Piece detection will be skipped if model is not available")
        else:
            self.piece_model_path = None
            print("ℹ️  Piece detection disabled (no model specified)")
    
    def analyze_board(self, 
                     input_image: Union[str, Path, np.ndarray],
                     conf_threshold: float = 0.5,
                     target_size: int = 512) -> Dict:
        """
        Analyze a chess board image to extract corners, pieces, and position.
        
        Args:
            input_image: Image path (string/Path) or numpy array
            conf_threshold: Confidence threshold for piece detection
            target_size: Target size for image processing
            
        Returns:
            Dictionary containing:
            - chess_position: chess.Board object with detected position
            - detected_pieces: List of detected pieces with coordinates
            - board_corners: Corner coordinates in original image
            - square_coordinates: Mapping of chess squares to pixel coordinates
            - processing_info: Additional processing information
            - visualizations: Various visualization images
        """
        
        print(f"🔍 Starting chess board analysis...")
        
        # Step 1: Prepare image and get original dimensions
        original_image, image_path_str, temp_files = self._prepare_image(input_image)
        original_height, original_width = original_image.shape[:2]
        print(f"📏 Original image dimensions: {original_width}x{original_height}")
        
        try:
            # Step 2: Detect chess board using segmentation
            print("🎯 Detecting chess board with segmentation...")
            board_corners, segmentation_info = self._detect_board_with_segmentation(image_path_str)
            
            if board_corners is None:
                raise ValueError("Failed to detect chess board corners")
            
            print(f"✅ Board corners detected: {len(board_corners)} corners")
            
            # Step 3: Extract chess board using perspective transformation
            print("🔄 Applying perspective transformation...")
            transform_results = self._extract_chess_board(original_image, board_corners, target_size)
            
            # Step 4: Generate square coordinates
            print("📍 Generating square coordinates...")
            square_coordinates = self._generate_square_coordinates(
                transform_results['transform_matrix'],
                original_width, original_height
            )
            
            # Step 5: Detect chess pieces (if model is available)
            pieces_info = {}
            if self.piece_model_path is not None and self.piece_model_path.exists():
                print("♟️  Detecting chess pieces...")
                pieces_info = self._detect_pieces(image_path_str, square_coordinates, conf_threshold)
            else:
                if self.piece_model_path is None:
                    print("⚠️  Skipping piece detection - disabled")
                else:
                    print("⚠️  Skipping piece detection - model not available")
                pieces_info = {
                    'detections': None,
                    'game_list': [],
                    'chess_board': None,
                    'position_string': '',
                    'detection_image': None
                }
            
            # Step 6: Create comprehensive result
            result = {
                # Core chess analysis results
                'chess_position': pieces_info.get('chess_board'),
                'detected_pieces': pieces_info.get('game_list', []),
                'position_string': pieces_info.get('position_string', ''),
                
                # Geometric information
                'board_corners': {
                    'top_left': board_corners[0] if len(board_corners) > 0 else None,
                    'top_right': board_corners[1] if len(board_corners) > 1 else None,
                    'bottom_right': board_corners[2] if len(board_corners) > 2 else None,
                    'bottom_left': board_corners[3] if len(board_corners) > 3 else None,
                },
                'square_coordinates': square_coordinates,
                
                # Processing information
                'processing_info': {
                    'original_dimensions': (original_width, original_height),
                    'segmentation_method': self.corner_method,
                    'corner_extraction_info': segmentation_info,
                    'transform_matrix': transform_results['transform_matrix'],
                    'piece_detection_available': self.piece_model_path is not None and self.piece_model_path.exists(),
                    'confidence_threshold': conf_threshold
                },
                
                # Visualizations
                'visualizations': {
                    'original_image': original_image,
                    'warped_board': transform_results['warped_image'],
                    'board_with_grid': transform_results['board_with_squares'],
                    'piece_detections': pieces_info.get('detection_image'),
                    'segmentation_info': segmentation_info
                },
                
                # Future extensions placeholder
                'move_analysis': {
                    'next_move': None,  # Placeholder for future implementation
                    'move_coordinates': None,  # Placeholder for pixel coordinates of moves
                    'legal_moves': None,  # Placeholder for legal moves
                }
            }
            
            print("✅ Chess board analysis completed successfully!")
            if pieces_info.get('game_list'):
                print(f"♟️  Found {len(pieces_info['game_list'])} chess pieces")
            
            return result
            
        finally:
            # Clean up temporary files
            self._cleanup_temp_files(temp_files)
    
    def _prepare_image(self, input_image: Union[str, Path, np.ndarray]) -> Tuple[np.ndarray, str, List[str]]:
        """
        Prepare image for processing, handling both file paths and numpy arrays.
        
        Returns:
            Tuple of (original_image, image_path_string, temp_files_to_cleanup)
        """
        temp_files = []
        
        if isinstance(input_image, (str, Path)):
            # Input is a file path
            image_path_str = str(input_image)
            original_image = cv2.imread(image_path_str)
            if original_image is None:
                raise ValueError(f"Could not load image from path: {image_path_str}")
                
        elif isinstance(input_image, np.ndarray):
            # Input is a numpy array
            original_image = input_image.copy()
            if len(original_image.shape) == 3 and original_image.shape[2] == 3:
                # Assume RGB, convert to BGR for OpenCV compatibility
                original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
            
            # Save to temporary file for processing
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            cv2.imwrite(temp_file.name, original_image)
            image_path_str = temp_file.name
            temp_files.append(temp_file.name)
            temp_file.close()
        else:
            raise ValueError("Input must be either a file path (string/Path) or numpy array")
        
        return original_image, image_path_str, temp_files
    
    def _detect_board_with_segmentation(self, image_path: str) -> Tuple[Optional[List], Dict]:
        """
        Detect chess board corners using the segmentation model.
        
        Returns:
            Tuple of (corner_coordinates, debug_info)
        """
        try:
            # Get polygon coordinates from segmentation
            polygon_info, is_valid = self.segmentation_model.get_polygon_coordinates(
                image_path, conf=0.25, iou=0.7, max_segments=1
            )
            
            if not is_valid or not polygon_info:
                return None, {'error': 'No valid chess board detected by segmentation'}
            
            # Extract corners using the specified method
            corners, debug_info = self.segmentation_model.extract_corners_from_segmentation(
                image_path, polygon_info, method=self.corner_method, debug=True
            )
            
            if not corners or len(corners) != 4:
                return None, {'error': f'Corner extraction failed: got {len(corners) if corners else 0} corners'}
            
            # Convert corner format to simple coordinate list
            corner_coords = []
            for corner in corners:
                corner_coords.append((corner['x'], corner['y']))
            
            return corner_coords, {
                'method': self.corner_method,
                'polygon_info': polygon_info,
                'debug_info': debug_info,
                'corners_raw': corners
            }
            
        except Exception as e:
            return None, {'error': f'Segmentation failed: {str(e)}'}
    
    def _extract_chess_board(self, original_image: np.ndarray, 
                           board_corners: List[Tuple], 
                           target_size: int) -> Dict:
        """
        Extract chess board using perspective transformation.
        
        Args:
            original_image: Original image array
            board_corners: List of (x, y) corner coordinates
            target_size: Target size for warped image
            
        Returns:
            Dictionary with transformation results
        """
        # Convert corners to the format expected by perspective transformation
        # Assuming corners are in order: [top_left, top_right, bottom_right, bottom_left]
        if len(board_corners) == 4:
            top_left, top_right, bottom_right, bottom_left = board_corners
        else:
            raise ValueError(f"Expected 4 corners, got {len(board_corners)}")
        
        # Apply perspective transformation
        transform_matrix, warped_image = apply_perspective_transformation(
            original_image, top_left, top_right, bottom_left, bottom_right
        )
        
        # Draw grid on warped image
        board_with_squares = draw_squares(warped_image)
        
        return {
            'transform_matrix': transform_matrix,
            'warped_image': warped_image,
            'board_with_squares': board_with_squares,
            'corners_used': {
                'top_left': top_left,
                'top_right': top_right,
                'bottom_right': bottom_right,
                'bottom_left': bottom_left
            }
        }
    
    def _generate_square_coordinates(self, transform_matrix: np.ndarray,
                                   original_width: int, original_height: int) -> Dict:
        """
        Generate coordinates for all chess squares in the original image.
        
        Returns:
            Dictionary mapping square numbers to coordinates
        """
        # Calculate inverse transformation matrix
        M_inv = cv2.invert(transform_matrix)[1]
        
        # Define warped image parameters
        rows, cols = 8, 8
        width, height = 1200, 1200
        square_width = width // cols
        square_height = height // rows
        
        # Generate square coordinates in warped space
        squares_data_warped = []
        
        # Iterate from bottom to top (chess notation order) and left to right
        for i in range(rows - 1, -1, -1):
            for j in range(cols):
                # Calculate square corners
                top_left_sq = (j * square_width, i * square_height)
                top_right_sq = ((j + 1) * square_width, i * square_height)
                bottom_left_sq = (j * square_width, (i + 1) * square_height)
                bottom_right_sq = ((j + 1) * square_width, (i + 1) * square_height)
                
                # Calculate center
                x_center = (top_left_sq[0] + bottom_right_sq[0]) // 2
                y_center = (top_left_sq[1] + bottom_right_sq[1]) // 2
                
                squares_data_warped.append([
                    (x_center, y_center),
                    bottom_right_sq,
                    top_right_sq,
                    top_left_sq,
                    bottom_left_sq
                ])
        
        # Transform coordinates back to original image space
        squares_data_warped_np = np.array(squares_data_warped, dtype=np.float32).reshape(-1, 1, 2)
        squares_data_original_np = cv2.perspectiveTransform(squares_data_warped_np, M_inv)
        squares_data_original = squares_data_original_np.reshape(-1, 5, 2)
        
        # Convert to dictionary format
        return get_square_coordinates(squares_data_original)
    
    def _detect_pieces(self, image_path: str, square_coordinates: Dict, 
                      conf_threshold: float) -> Dict:
        """
        Detect chess pieces using YOLO model.
        
        Returns:
            Dictionary with piece detection results
        """
        try:
            # Detect pieces
            chess_pieces_preds, detection_image = detect_chess_pieces(
                image_path, str(self.piece_model_path), conf=conf_threshold
            )
            
            # Map pieces to squares
            game_list = map_pieces_to_squares(chess_pieces_preds, square_coordinates)
            
            # Create chess board representation
            chess_board, chess_board_image, position_string = create_chess_board_representation(game_list)
            
            return {
                'detections': chess_pieces_preds,
                'game_list': game_list,
                'chess_board': chess_board,
                'position_string': position_string,
                'detection_image': detection_image,
                'chess_board_image': chess_board_image
            }
            
        except Exception as e:
            print(f"❌ Piece detection failed: {e}")
            return {
                'detections': None,
                'game_list': [],
                'chess_board': None,
                'position_string': '',
                'detection_image': None,
                'error': str(e)
            }
    
    def _cleanup_temp_files(self, temp_files: List[str]):
        """Clean up temporary files."""
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def visualize_result(self, result: Dict, 
                        show_pieces: bool = True, 
                        show_grid: bool = True, 
                        show_corners: bool = True) -> np.ndarray:
        """
        Create a visualization of the analysis results.
        
        Args:
            result: Result dictionary from analyze_board()
            show_pieces: Whether to show piece detections
            show_grid: Whether to show chess board grid
            show_corners: Whether to show corner points
            
        Returns:
            Visualization image as numpy array
        """
        # Prepare data for visualization
        original_image = result['visualizations']['original_image']
        
        # Create a mock results structure compatible with the classical method visualizer
        classical_results = {
            'original_image': original_image,
            'corners_original': result['board_corners'],
            'squares_original': result['square_coordinates'],
            'piece_detections': result['processing_info'].get('piece_detections'),
            'scale_factor': 1.0,  # No scaling since we're working with original image
            'offset': (0, 0)
        }
        
        return visualize_results_on_original(
            classical_results, 
            show_pieces=show_pieces, 
            show_grid=show_grid, 
            show_corners=show_corners
        )
    
    def get_piece_at_square(self, result: Dict, square_name: str) -> Optional[str]:
        """
        Get the piece at a specific chess square.
        
        Args:
            result: Result dictionary from analyze_board()
            square_name: Chess square name (e.g., 'e4', 'a1')
            
        Returns:
            Piece name or None if no piece at that square
        """
        chess_board = result.get('chess_position')
        if chess_board is None:
            return None
        
        try:
            import chess
            square = chess.parse_square(square_name)
            piece = chess_board.piece_at(square)
            return str(piece) if piece else None
        except:
            return None
    
    def get_square_coordinates(self, result: Dict, square_name: str) -> Optional[Dict]:
        """
        Get pixel coordinates for a specific chess square.
        
        Args:
            result: Result dictionary from analyze_board()
            square_name: Chess square name (e.g., 'e4', 'a1')
            
        Returns:
            Dictionary with center, corners coordinates or None
        """
        # Convert chess notation to square number (1-64)
        try:
            file = ord(square_name[0].lower()) - ord('a')  # a=0, b=1, ..., h=7
            rank = int(square_name[1]) - 1  # 1=0, 2=1, ..., 8=7
            
            # Convert to square number (1-64, starting from bottom-left)
            square_num = (7 - rank) * 8 + file + 1
            
            square_coords = result['square_coordinates'].get(square_num)
            if square_coords:
                return {
                    'center': square_coords[0],
                    'bottom_right': square_coords[1],
                    'top_right': square_coords[2],
                    'top_left': square_coords[3],
                    'bottom_left': square_coords[4]
                }
        except:
            pass
        
        return None


# Convenience function for quick analysis
def analyze_chess_board_image(image_path: Union[str, Path], 
                            segmentation_model: str = "dopaul/chess_board_segmentation",
                            **kwargs) -> Dict:
    """
    Convenience function to analyze a chess board image with default settings.
    
    Args:
        image_path: Path to chess board image
        segmentation_model: HuggingFace model name or local path
        **kwargs: Additional arguments passed to analyze_board()
        
    Returns:
        Analysis result dictionary
    """
    analyzer = ChessBoardAnalyzer(segmentation_model=segmentation_model)
    return analyzer.analyze_board(image_path, **kwargs)


if __name__ == "__main__":
    # Example usage
    print("Chess Board Analyzer - Example Usage")
    
    # Test with default model
    try:
        analyzer = ChessBoardAnalyzer()
        
        # Test image path
        test_image = Path("data/eval_images/chess_4.jpeg")
        if test_image.exists():
            print(f"\n🔍 Analyzing test image: {test_image}")
            result = analyzer.analyze_board(test_image)
            
            print(f"✅ Analysis completed!")
            print(f"📍 Board corners detected: {bool(result['board_corners'])}")
            print(f"♟️  Pieces detected: {len(result['detected_pieces'])}")
            
            # Show some example square coordinates
            if result['square_coordinates']:
                print(f"📍 Square 'e4' coordinates: {analyzer.get_square_coordinates(result, 'e4')}")
                print(f"♟️  Piece at 'e4': {analyzer.get_piece_at_square(result, 'e4')}")
        else:
            print(f"⚠️  Test image not found: {test_image}")
            
    except Exception as e:
        print(f"❌ Example failed: {e}") 