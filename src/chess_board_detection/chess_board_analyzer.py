#!/usr/bin/env python3
"""
Comprehensive Chess Board Analyzer

This module combines segmentation-based corner detection with YOLO-based piece detection
to provide complete chess board analysis including piece detection and position reading.

Usage:
    # Initialize analyzer with default model
    analyzer = ChessBoardAnalyzer()
    
    # Analyze chess board from image path
    result = analyzer.analyze_board("path/to/chess_image.jpg")
    
    # Analyze chess board from numpy array
    result = analyzer.analyze_board(image_array)
    
    # Use custom model or HuggingFace model
    analyzer = ChessBoardAnalyzer(piece_detection_model="dopaul/chess_piece_detector")
    
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
import matplotlib.pyplot as plt
import matplotlib.patheffects

from src.chess_board_detection.yolo.segmentation.segmentation_model import ChessBoardSegmentationModel
from src.chess_piece_detection.model import ChessModel
from src.chess_board_detection.classical_method import (
    apply_perspective_transformation,
    draw_squares,
    get_square_coordinates,
    visualize_results_on_original,
)


class ChessBoardAnalyzer:
    """
    Comprehensive chess board analyzer that combines segmentation-based corner detection
    with YOLO-based piece detection for full chess position analysis.
    
    This class provides a unified interface for:
    - Detecting chess board corners using segmentation
    - Extracting chess board using perspective transformation
    - Detecting and classifying chess pieces
    - Reading chess position in standard notation
    - Preparing data structures for future move prediction
    """
    
    def __init__(self, 
                 segmentation_model: str = "dopaul/chess_board_segmentation",
                 piece_detection_model: str = "dopaul/chess_piece_detection",
                 corner_method: str = "approx"):
        """
        Initialize the chess board analyzer.
        
        Args:
            segmentation_model: HuggingFace model name or local path for board segmentation
            piece_detection_model: HuggingFace model name or local path for piece detection
            corner_method: Method for corner extraction ('approx', 'extended', etc.)
        """
        self.segmentation_model_name = segmentation_model
        self.piece_detection_model_name = piece_detection_model
        self.corner_method = corner_method
        
        # Initialize segmentation model
        print(f"🚀 Loading segmentation model: {segmentation_model}")
        try:
            # Use the enhanced ChessBoardSegmentationModel with HuggingFace support
            self.segmentation_model = ChessBoardSegmentationModel(model_path=segmentation_model)
            print("✅ Segmentation model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load segmentation model: {e}")
            raise
        
        # Initialize piece detection model
        if piece_detection_model is not None:
            print(f"🚀 Loading piece detection model: {piece_detection_model}")
            try:
                self.piece_model = ChessModel(piece_detection_model)
                print("✅ Piece detection model loaded successfully!")
            except Exception as e:
                print(f"⚠️  Warning: Failed to load piece detection model: {e}")
                print("    Piece detection will be skipped")
                self.piece_model = None
        else:
            self.piece_model = None
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
            if self.piece_model is not None:
                print("♟️  Detecting chess pieces...")
                pieces_info = self._detect_pieces(image_path_str, square_coordinates, conf_threshold)
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
                    'piece_detection_available': self.piece_model is not None,
                    'piece_detection_model': self.piece_detection_model_name,
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
            # Use the enhanced ChessModel to detect pieces and map to squares
            piece_mappings, detection_image = self.piece_model.detect_pieces_in_squares(
                image_path, square_coordinates, conf_threshold
            )
            
            # Create chess board representation
            chess_board = self.piece_model.create_chess_position(piece_mappings)
            
            # Create position string for compatibility
            position_string = ""
            if chess_board:
                try:
                    position_string = chess_board.fen()
                except:
                    position_string = "Invalid position"
            
            return {
                'detections': None,  # Raw YOLO results not exposed in new API
                'game_list': piece_mappings,
                'chess_board': chess_board,
                'position_string': position_string,
                'detection_image': detection_image,
                'chess_board_image': None  # Could be added if needed
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
    
    def create_corners_and_grid_visualization(self, result: Dict) -> np.ndarray:
        """
        Create visualization showing board corners and chess grid with square labels.
        
        Args:
            result: Result dictionary from analyze_board()
            
        Returns:
            Visualization image as numpy array
        """
        # Get original image and create a copy
        original_image = result['visualizations']['original_image'].copy()
        vis_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Draw corner points
        corners = result['board_corners']
        corner_points = [corners['top_left'], corners['top_right'], 
                        corners['bottom_left'], corners['bottom_right']]
        corner_labels = ['TL', 'TR', 'BL', 'BR']
        
        for i, (point, label) in enumerate(zip(corner_points, corner_labels)):
            if point is not None:
                x, y = int(point[0]), int(point[1])
                # Draw large corner point
                cv2.circle(vis_image, (x, y), 20, (255, 0, 0), -1)
                # Add corner label
                cv2.putText(vis_image, label, (x-15, y-25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw chess board grid with square labels
        squares = result['square_coordinates']
        
        # Draw grid lines and square labels
        for square_num, coords in squares.items():
            if len(coords) >= 5:
                # coords format: [center, bottom_right, top_right, top_left, bottom_left]
                center, bottom_right, top_right, top_left, bottom_left = coords
                
                # Draw square outline
                points = np.array([
                    [int(top_left[0]), int(top_left[1])],
                    [int(top_right[0]), int(top_right[1])],
                    [int(bottom_right[0]), int(bottom_right[1])],
                    [int(bottom_left[0]), int(bottom_left[1])]
                ], np.int32)
                
                cv2.polylines(vis_image, [points], True, (0, 255, 0), 2)
                
                # Convert square number to chess notation (a1, b2, etc.)
                if 1 <= square_num <= 64:
                    file = chr(ord('a') + ((square_num - 1) % 8))  # a-h
                    rank = str(8 - ((square_num - 1) // 8))        # 1-8
                    square_label = f"{file}{rank}"
                    
                    center_x, center_y = int(center[0]), int(center[1])
                    cv2.putText(vis_image, square_label, (center_x-15, center_y+5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return vis_image
    
    def create_piece_detections_visualization(self, result: Dict) -> np.ndarray:
        """
        Create visualization showing only piece detections with confidence scores.
        
        Args:
            result: Result dictionary from analyze_board()
            
        Returns:
            Visualization image as numpy array
        """
        # Get piece detection image if available
        detection_image = result['visualizations'].get('piece_detections')
        if detection_image is not None:
            # Use the existing detection image but enhance it
            vis_image = detection_image.copy()
        else:
            # Fallback to original image
            original_image = result['visualizations']['original_image'].copy()
            vis_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Add piece information with color coding
        detected_pieces = result['detected_pieces']
        squares = result['square_coordinates']
        
        # Define colors for piece types
        white_piece_color = (100, 149, 237)  # CornflowerBlue
        black_piece_color = (220, 20, 60)    # Crimson
        
        for square_num, piece_class_id in detected_pieces:
            square_coords = squares.get(square_num)
            if square_coords and len(square_coords) >= 5:
                center = square_coords[0]
                
                # Get piece name
                if hasattr(self.piece_model, 'get_piece_name'):
                    piece_name = self.piece_model.get_piece_name(piece_class_id)
                else:
                    piece_name = f"Class_{piece_class_id}"
                
                # Determine color based on piece
                if 'white' in piece_name.lower():
                    color = white_piece_color
                else:
                    color = black_piece_color
                
                # Draw piece information
                center_x, center_y = int(center[0]), int(center[1])
                
                # Draw a small circle to mark the piece location
                cv2.circle(vis_image, (center_x, center_y), 8, color, -1)
                
                # Add piece name (smaller text to avoid crowding)
                short_name = piece_name.replace('-', ' ').title()
                cv2.putText(vis_image, short_name, (center_x-25, center_y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return vis_image
    
    def create_chess_diagram_png(self, result: Dict, output_path: Path) -> np.ndarray:
        """
        Create PNG chess diagram with classical piece symbols.
        
        Args:
            result: Result dictionary from analyze_board()
            output_path: Path where to save the PNG file
            
        Returns:
            Diagram image as numpy array
        """
        chess_position = result.get('chess_position')
        
        # Create the chess diagram using matplotlib
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Unicode chess symbols
        piece_symbols = {
            'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',  # White
            'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'   # Black
        }
        
        # Board colors
        light_color = '#f0d9b5'
        dark_color = '#b58863'
        
        # Draw the board squares
        for rank in range(8):
            for file in range(8):
                # Determine square color
                is_light = (rank + file) % 2 == 0
                color = light_color if is_light else dark_color
                
                # Draw square
                square = plt.Rectangle((file, 7-rank), 1, 1, facecolor=color, edgecolor='#8b4513', linewidth=1)
                ax.add_patch(square)
                
                # Add piece if chess position is available
                if chess_position is not None:
                    try:
                        import chess
                        chess_square = chess.square(file, rank)
                        piece = chess_position.piece_at(chess_square)
                        
                        if piece:
                            piece_symbol = piece_symbols.get(str(piece), str(piece))
                            # Position text in center of square
                            text_x = file + 0.5
                            text_y = 7 - rank + 0.5
                            
                            # Choose text color based on piece color and square color
                            if piece.color:  # White piece
                                text_color = 'white' if not is_light else '#333333'
                                edge_color = '#000000' if not is_light else '#666666'
                            else:  # Black piece
                                text_color = '#000000'
                                edge_color = '#ffffff' if is_light else '#cccccc'
                            
                            # Add piece symbol with outline for better visibility
                            ax.text(text_x, text_y, piece_symbol, fontsize=28, ha='center', va='center',
                                   color=text_color, weight='bold',
                                   path_effects=[plt.matplotlib.patheffects.Stroke(linewidth=1.5, foreground=edge_color),
                                               plt.matplotlib.patheffects.Normal()])
                    except Exception as e:
                        # Skip if chess library issues
                        pass
        
        # Add file labels (a-h) at bottom
        for file in range(8):
            file_letter = chr(ord('a') + file)
            ax.text(file + 0.5, -0.2, file_letter, fontsize=14, ha='center', va='center', 
                   color='#8b4513', weight='bold')
        
        # Add rank labels (1-8) on left side
        for rank in range(8):
            rank_number = rank + 1
            ax.text(-0.2, rank + 0.5, str(rank_number), fontsize=14, ha='center', va='center',
                   color='#8b4513', weight='bold')
        
        # Set up the plot
        ax.set_xlim(-0.5, 8.5)
        ax.set_ylim(-0.5, 8.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title
        if chess_position is not None:
            title = f"Chess Position ({len(result['detected_pieces'])} pieces detected)"
            if chess_position.turn:
                title += " - White to move"
            else:
                title += " - Black to move"
        else:
            title = "Chess Board (No pieces detected)"
        
        ax.set_title(title, fontsize=16, weight='bold', pad=20)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        
        # Convert to numpy array for return
        fig.canvas.draw()
        try:
            # Try newer matplotlib method first (>= 3.8)
            buffer = fig.canvas.buffer_rgba()
            img_array = np.asarray(buffer)
            # Convert RGBA to RGB
            if img_array.shape[-1] == 4:
                img_array = img_array[:, :, :3]
        except AttributeError:
            try:
                # Try tobytes method (matplotlib >= 3.0)
                img_array = np.frombuffer(fig.canvas.tobytes(), dtype=np.uint8)
                img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            except AttributeError:
                # Fallback for older matplotlib versions
                img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return img_array
    



# Convenience function for quick analysis
def analyze_chess_board_image(image_path: Union[str, Path], 
                            segmentation_model: str = "dopaul/chess_board_segmentation",
                            piece_detection_model: str = "dopaul/chess_piece_detection",
                            **kwargs) -> Dict:
    """
    Convenience function to analyze a chess board image with default settings.
    
    Args:
        image_path: Path to chess board image
        segmentation_model: HuggingFace model name or local path for board segmentation
        piece_detection_model: HuggingFace model name or local path for piece detection
        **kwargs: Additional arguments passed to analyze_board()
        
    Returns:
        Analysis result dictionary
    """
    analyzer = ChessBoardAnalyzer(
        segmentation_model=segmentation_model,
        piece_detection_model=piece_detection_model
    )
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