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

from pathlib import Path
from typing import Any

import chess
import cv2
import numpy as np

from src.chess_board_detection.classical_method import (
    apply_perspective_transformation,
    get_square_coordinates,
)
from src.chess_board_detection.yolo.segmentation.segmentation_model import (
    ChessBoardSegmentationModel,
)
from src.chess_piece_detection.model import ChessModel
from src.datatypes import (
    ChessAnalysis,
    ChessBoard,
    ChessBoardSquare,
    ChessBoardVisualisations,
    ChessPieceBoundingBox,
    Metadata,
    MoveAnalysis,
    Point,
)
from src.visualisation import (
    create_corners_and_grid_visualization,
    create_piece_bounding_boxes_visualization,
    create_piece_centers_visualization,
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

    def __init__(
        self,
        segmentation_model: str = "dopaul/chess_board_segmentation",
        piece_detection_model: str | None = "dopaul/chess_piece_detection",
        corner_method: str = "approx",
        create_visualizations: bool = True,
        threshold: int = 0,
        white_playing_from: str = "b",
        computer_playing_as: str | None = None,
        engine_type: str = "stockfish",
        engine_depth: int = 10,
        engine_time_limit: float = 1.0,
        stockfish_skill_level: int = 20,
    ):
        """
        Initialize the chess board analyzer.

        Args:
            segmentation_model: HuggingFace model name or local path for board segmentation
            piece_detection_model: HuggingFace model name or local path for piece detection
            corner_method: Method for corner extraction ('approx', 'extended', etc.)
            create_visualizations: Whether to create visualization images
            threshold: Percentage expansion of chess board from center (0-100, default: 0)
            white_playing_from: Side where white is playing from - "b" (bottom), "t" (top), "l" (left), "r" (right) (default: "b")
            computer_playing_as: Color the computer is playing as - "white", "black", or None to disable move prediction (default: None)
            engine_type: Chess engine to use - "stockfish" or "simple" (default: "stockfish")
            engine_depth: Maximum search depth for engine (default: 10)
            engine_time_limit: Maximum time in seconds for engine to think (default: 1.0)
            stockfish_skill_level: Stockfish skill level 0-20, where 20 is strongest (default: 20)
        """
        self.segmentation_model_name = segmentation_model
        self.piece_detection_model_name = piece_detection_model
        self.corner_method = corner_method
        self.create_visualizations = create_visualizations
        self.threshold = threshold
        self.white_playing_from = white_playing_from
        self.computer_playing_as = computer_playing_as
        self.engine_type = engine_type
        self.engine_depth = engine_depth
        self.engine_time_limit = engine_time_limit
        self.stockfish_skill_level = stockfish_skill_level

        # Validate white_playing_from parameter
        if white_playing_from not in ["b", "t", "l", "r"]:
            raise ValueError(
                f"Invalid white_playing_from: {white_playing_from}. Must be one of: 'b' (bottom), 't' (top), 'l' (left), 'r' (right)"
            )

        # Validate computer_playing_as parameter
        if computer_playing_as is not None and computer_playing_as not in ["white", "black"]:
            raise ValueError(
                f"Invalid computer_playing_as: {computer_playing_as}. Must be one of: 'white', 'black', or None"
            )

        # Validate engine parameters
        if engine_type not in ["stockfish", "simple"]:
            raise ValueError(f"Invalid engine_type: {engine_type}. Must be 'stockfish' or 'simple'")

        if not 0 <= stockfish_skill_level <= 20:
            raise ValueError(f"Invalid stockfish_skill_level: {stockfish_skill_level}. Must be between 0-20")

        # Configuration is now printed by the main script, not here
        # to avoid duplicate logging

        # Initialize chess engine
        self.chess_engine = None
        if computer_playing_as and engine_type == "stockfish":
            self._initialize_stockfish()

        # Initialize segmentation model
        try:
            # Use the enhanced ChessBoardSegmentationModel with HuggingFace support
            self.segmentation_model = ChessBoardSegmentationModel(model_path=segmentation_model)
        except Exception as e:
            print(f"❌ Failed to load segmentation model: {e}")
            raise

        # Initialize piece detection model
        if piece_detection_model is not None:
            try:
                self.piece_model = ChessModel(model_path=piece_detection_model)
            except Exception as e:
                print(f"⚠️  Warning: Failed to load piece detection model: {e} - Piece detection will be skipped")
                self.piece_model = None
        else:
            self.piece_model = None
            print("ℹ️  Piece detection disabled (no model specified)")

    def analyze_board(
        self,
        input_image: str | Path | np.ndarray,
        conf_threshold: float = 0.5,
        use_weighted_center: bool = True,
        computer_playing_as: str | None = None,
    ) -> ChessAnalysis:
        """
        Analyze a chess board image to extract corners, pieces, and position.

        Args:
            input_image: Image path (string/Path) or numpy array
            conf_threshold: Confidence threshold for piece detection
            use_weighted_center: If True, use weighted_center (default), if False use center
            computer_playing_as: Override computer color for move prediction ("white", "black", or None).
                               If None, uses the default from initialization.

        Returns:
            ChessAnalysis object containing all analysis results
        """
        # Step 1: Prepare image and get original dimensions
        original_image = self._load_image(input_image)
        original_height, original_width = original_image.shape[:2]

        # Step 2: Detect chess board using segmentation
        print("🎯 Detecting chess board with segmentation...")
        board_corners, debug_info, polygon_info = self._detect_board_with_segmentation(original_image)

        if board_corners is None:
            raise ValueError("Failed to detect chess board corners")
        print(f"✅ Board corners detected: {len(board_corners)} corners")

        # Step 3: Extract chess board using perspective transformation
        print("🔄 Applying perspective transformation...")
        if len(board_corners) == 4:
            top_left, top_right, bottom_right, bottom_left = board_corners
        else:
            raise ValueError(f"Expected 4 corners, got {len(board_corners)}")

        # Apply perspective transformation
        transform_matrix, warped_image = apply_perspective_transformation(
            original_image, top_left, top_right, bottom_left, bottom_right, threshold=self.threshold
        )

        # Step 4: Generate square coordinates
        print("📍 Generating square coordinates...")
        chess_squares = self._generate_square_coordinates(transform_matrix)

        # Step 5: Detect chess pieces (if model is available)
        if self.piece_model is not None:
            print("♟️  Detecting chess pieces...")
            # Use the enhanced ChessModel to detect pieces and map to squares
            chess_pieces = self.piece_model.detect_pieces_in_squares(original_image, conf_threshold)

            # Map detections to squares
            chess_pieces, chess_squares = self._map_detections_to_squares(
                chess_pieces, chess_squares, use_weighted_center
            )

            print(f"🎯 Detected {len(chess_pieces)} chess pieces")
        else:
            print("⚠️  Skipping piece detection - model not available")
            chess_pieces = []

        # Transform chess squares to standard notation if needed
        if self.white_playing_from != "b":
            print(f"🔄 Transforming squares from {self.white_playing_from} perspective to standard notation")
            transformed_squares = {}

            for camera_square_num, square_data in chess_squares.items():
                # Transform the square number to standard notation
                standard_square_num = self._transform_square_number_to_standard(camera_square_num)
                transformed_squares[standard_square_num] = square_data

            chess_squares = transformed_squares

        # Create chess board (squares are now in standard notation)
        chess_board = ChessBoard(
            board_corners=board_corners,
            chess_squares=chess_squares,
            chess_pieces=chess_pieces,
            white_playing_from=self.white_playing_from,
        )

        # Step 6: Determine which color the computer should play as
        effective_computer_color = computer_playing_as if computer_playing_as is not None else self.computer_playing_as

        # Step 7: Predict the best move if requested
        move_analysis = self._predict_best_move(chess_board, effective_computer_color)

        # Step 8: Create comprehensive result
        chess_analysis = ChessAnalysis(
            metadata=Metadata(
                original_image=original_image,
                original_dimensions=(original_width, original_height),
                segmentation_model=self.segmentation_model_name,
                segmentation_method=self.corner_method,
                piece_detection_model=self.piece_detection_model_name or None,
                piece_detection_available=self.piece_model is not None,
                confidence_threshold=conf_threshold,
                white_playing_from=self.white_playing_from,
            ),
            chess_board=chess_board,
            visualisations=None,
            move_analysis=move_analysis,
        )
        print("✅ Chess board analysis completed successfully!")

        if self.create_visualizations:
            # Create visualizations without logging (handled by main script)
            corners_and_grid_vis = create_corners_and_grid_visualization(chess_analysis, verbose=False)
            piece_boxes_vis = create_piece_bounding_boxes_visualization(chess_analysis, verbose=False)
            piece_centers_vis = create_piece_centers_visualization(chess_analysis, use_weighted_center, verbose=False)

            chess_analysis.visualisations = ChessBoardVisualisations(
                original_image=chess_analysis.metadata.original_image,
                warped_board=corners_and_grid_vis,  # Using corners visualization as warped board
                board_with_grid=corners_and_grid_vis,
                piece_boxes=piece_boxes_vis,
                piece_centers=piece_centers_vis,
            )

        return chess_analysis

    def _load_image(self, input_image: str | Path | np.ndarray) -> np.ndarray:
        """
        Load and validate input image.

        Args:
            input_image: Either a file path (string/Path) or numpy array

        Returns:
            numpy.ndarray: Loaded image in BGR format for OpenCV

        Raises:
            ValueError: If input_image is not a valid file path or numpy array, or if image cannot be loaded

        Notes:
            - For file paths: loads image using cv2.imread
            - For numpy arrays: assumes RGB format and converts to BGR for OpenCV compatibility
            - Creates a copy of the input array to avoid modifying the original
            - temp_files (list[str]): list of temporary file paths that need cleanup

        Raises:
            ValueError: If input_image is not a valid file path or numpy array, or if image cannot be loaded
        """

        if isinstance(input_image, str | Path):
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
        else:
            raise ValueError("Input must be either a file path (string/Path) or numpy array")
        return original_image

    def _detect_board_with_segmentation(self, image: np.ndarray) -> tuple[list[Point], dict[str, Any], dict[str, Any]]:
        """
        Detect chess board corners using the segmentation model.

        Returns:
            Tuple of (corner_coordinates, debug_info, polygon_info)
        """
        try:
            # Get polygon coordinates from segmentation
            polygon_info, is_valid = self.segmentation_model.get_polygon_coordinates(
                image, conf=0.25, iou=0.7, max_segments=1
            )

            if not is_valid or not polygon_info:
                raise ValueError("No valid chess board detected by segmentation")

            # Extract corners using the specified method
            corners, debug_info = self.segmentation_model.extract_corners_from_segmentation(
                image, polygon_info, method=self.corner_method, debug=True
            )

            if not corners or len(corners) != 4:
                raise ValueError(f"Corner extraction failed: got {len(corners) if corners else 0} corners")

            # Convert corner format to simple coordinate list
            corner_coords = [Point(corner["x"], corner["y"]) for corner in corners]

            # Ensure debug_info and polygon_info are dictionaries
            if debug_info is None:
                debug_info = {}
            if polygon_info is None:
                polygon_info = {}

            return corner_coords, debug_info, polygon_info
        except Exception as e:
            raise ValueError(f"Segmentation failed: {str(e)}") from e

    def _generate_square_coordinates(self, transform_matrix: np.ndarray) -> dict[int, ChessBoardSquare]:
        """
        Generate coordinates for all chess squares in the original image.

        Always generates squares in standard order (1-64 from bottom-left to top-right)
        regardless of camera perspective. Perspective transformation is handled during FEN conversion.

        Returns:
            dictionary mapping square numbers to coordinates
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

        print(f"🔄 Generating squares (standard order) for perspective: {self.white_playing_from}")

        # Always generate squares in standard order (bottom-left to top-right)
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

                squares_data_warped.append(
                    [
                        (x_center, y_center),
                        bottom_right_sq,
                        top_right_sq,
                        top_left_sq,
                        bottom_left_sq,
                    ]
                )

        # Transform coordinates back to original image space
        squares_data_warped_np = np.array(squares_data_warped, dtype=np.float32).reshape(-1, 1, 2)
        squares_data_original_np = cv2.perspectiveTransform(squares_data_warped_np, M_inv)
        squares_data_original = squares_data_original_np.reshape(-1, 5, 2)

        # Convert to dictionary format
        return get_square_coordinates(squares_data_original)

    def _map_detections_to_squares(
        self,
        chess_pieces: list[ChessPieceBoundingBox],
        chess_squares: dict[int, ChessBoardSquare],
        use_weighted_center: bool = True,
    ) -> tuple[list[ChessPieceBoundingBox], dict[int, ChessBoardSquare]]:
        """Map YOLO detections to chess board squares, handling duplicates by keeping highest confidence.

        Kings are mapped first to ensure exactly one white king and one black king on the board.
        All other king detections are discarded completely.

        Args:
            chess_pieces: List of detected chess piece bounding boxes
            chess_squares: Dictionary mapping square numbers to square coordinates
            use_weighted_center: If True, use weighted_center (default), if False use center
        """
        square_to_best_piece = {}  # Track best piece for each square
        occupied_squares = set()  # Track squares that are already occupied

        # Determine which coordinate method to use
        coordinate_method = "weighted_center" if use_weighted_center else "center"
        print(f"📍 Using {coordinate_method} coordinates for piece mapping")

        # Step 1: Separate kings from other pieces
        white_kings = [p for p in chess_pieces if "white" in p.piece_name.lower() and "king" in p.piece_name.lower()]
        black_kings = [p for p in chess_pieces if "black" in p.piece_name.lower() and "king" in p.piece_name.lower()]
        other_pieces = [p for p in chess_pieces if "king" not in p.piece_name.lower()]

        print(f"👑 Found {len(white_kings)} white kings, {len(black_kings)} black kings")

        # Step 2: Select best kings and discard all others
        selected_kings = []

        if len(white_kings) > 1:
            best_white_king = max(white_kings, key=lambda k: k.confidence)
            print(
                f"⚠️  Warning: {len(white_kings)} white kings detected, selecting highest confidence ({best_white_king.confidence:.3f})"
            )
            selected_kings.append(best_white_king)
            print(f"🗑️  Discarding {len(white_kings) - 1} other white king detections")
        elif len(white_kings) == 1:
            selected_kings.append(white_kings[0])
            print(f"✅ Found 1 white king (confidence: {white_kings[0].confidence:.3f})")
        else:
            print("⚠️  Warning: No white king detected")

        if len(black_kings) > 1:
            best_black_king = max(black_kings, key=lambda k: k.confidence)
            print(
                f"⚠️  Warning: {len(black_kings)} black kings detected, selecting highest confidence ({best_black_king.confidence:.3f})"
            )
            selected_kings.append(best_black_king)
            print(f"🗑️  Discarding {len(black_kings) - 1} other black king detections")
        elif len(black_kings) == 1:
            selected_kings.append(black_kings[0])
            print(f"✅ Found 1 black king (confidence: {black_kings[0].confidence:.3f})")
        else:
            print("⚠️  Warning: No black king detected")

        # Step 3: Update chess_pieces list to only include selected kings + other pieces
        # This ensures discarded kings don't interfere with anything
        updated_chess_pieces = selected_kings + other_pieces
        print(
            f"📋 Updated piece list: {len(selected_kings)} selected kings + {len(other_pieces)} other pieces = {len(updated_chess_pieces)} total"
        )

        # Step 4: Map selected kings first (they get priority)
        for king in selected_kings:
            # Get the appropriate coordinate based on the parameter
            king_coord = king.weighted_center if use_weighted_center else king.center

            # Find which square this king belongs to
            for square_num, cs in chess_squares.items():
                if square_num in occupied_squares:
                    continue  # Skip already occupied squares

                min_x = min(cs.tl.x, cs.tr.x, cs.br.x, cs.bl.x)
                max_x = max(cs.tl.x, cs.tr.x, cs.br.x, cs.bl.x)
                min_y = min(cs.tl.y, cs.tr.y, cs.br.y, cs.bl.y)
                max_y = max(cs.tl.y, cs.tr.y, cs.br.y, cs.bl.y)

                if min_x <= king_coord.x <= max_x and min_y <= king_coord.y <= max_y:
                    king.assigned_square = square_num
                    square_to_best_piece[square_num] = king
                    occupied_squares.add(square_num)
                    print(f"👑 Mapped {king.piece_name} to square {square_num} using {coordinate_method}")
                    break

        # Step 5: Map remaining pieces to remaining squares
        for chess_piece in other_pieces:
            # Get the appropriate coordinate based on the parameter
            piece_coord = chess_piece.weighted_center if use_weighted_center else chess_piece.center

            # Find which square this center belongs to
            for square_num, cs in chess_squares.items():
                min_x = min(cs.tl.x, cs.tr.x, cs.br.x, cs.bl.x)
                max_x = max(cs.tl.x, cs.tr.x, cs.br.x, cs.bl.x)
                min_y = min(cs.tl.y, cs.tr.y, cs.br.y, cs.bl.y)
                max_y = max(cs.tl.y, cs.tr.y, cs.br.y, cs.bl.y)

                if min_x <= piece_coord.x <= max_x and min_y <= piece_coord.y <= max_y:
                    chess_piece.assigned_square = square_num

                    # Check if this square already has a piece detected
                    if (
                        square_num not in square_to_best_piece
                        or chess_piece.confidence > square_to_best_piece[square_num].confidence
                    ):
                        square_to_best_piece[square_num] = chess_piece
                    break

        # Step 6: Update chess squares with the best piece for each square
        for square_num, best_piece in square_to_best_piece.items():
            if square_num in chess_squares:
                chess_squares[square_num].piece_class_id = best_piece.piece_class
                chess_squares[square_num].piece_name = best_piece.piece_name

        # Step 7: Validation - Check final king count
        final_white_kings = sum(
            1
            for s in chess_squares.values()
            if s.piece_name and "white" in s.piece_name.lower() and "king" in s.piece_name.lower()
        )
        final_black_kings = sum(
            1
            for s in chess_squares.values()
            if s.piece_name and "black" in s.piece_name.lower() and "king" in s.piece_name.lower()
        )

        if final_white_kings == 1 and final_black_kings == 1:
            print("✅ Kings mapped successfully: 1 white king, 1 black king")
        else:
            print(f"⚠️  King mapping issue: {final_white_kings} white kings, {final_black_kings} black kings on board")

        # Return the updated chess_pieces list (with discarded kings removed)
        return updated_chess_pieces, chess_squares

    def _initialize_stockfish(self):
        """Initialize Stockfish chess engine with configured settings."""
        try:
            from stockfish import Stockfish

            # Initialize Stockfish with the correct path
            self.chess_engine = Stockfish(path="/opt/homebrew/bin/stockfish")

            # Configure Stockfish settings
            self.chess_engine.set_depth(self.engine_depth)

            # Set skill level (0-20, where 20 is strongest)
            self.chess_engine.set_skill_level(self.stockfish_skill_level)

            print("✅ Stockfish engine initialized successfully")

        except Exception as e:
            print(f"⚠️  Warning: Failed to initialize Stockfish: {e}")
            print("   Falling back to simple evaluation")
            self.chess_engine = None
            self.engine_type = "simple"

    def _get_stockfish_move(self, board: chess.Board) -> tuple[chess.Move | None, float | None]:
        """
        Get the best move from Stockfish engine.

        Args:
            board: Current chess board position

        Returns:
            Tuple of (best_move, evaluation_score)
        """
        try:
            if self.chess_engine is None:
                return None, None

            # Set the position in Stockfish
            self.chess_engine.set_fen_position(board.fen())

            # Get the best move from Stockfish
            best_move_str = self.chess_engine.get_best_move_time(int(self.engine_time_limit * 1000))

            if best_move_str:
                # Convert string to chess.Move
                best_move = chess.Move.from_uci(best_move_str)

                # Get evaluation score
                evaluation = self.chess_engine.get_evaluation()
                if evaluation and evaluation.get("type") == "cp":
                    # Centipawn evaluation (divide by 100 to get pawn units)
                    score = evaluation.get("value", 0) / 100.0
                elif evaluation and evaluation.get("type") == "mate":
                    # Mate score
                    mate_moves = evaluation.get("value", 0)
                    score = 10000 if mate_moves > 0 else -10000
                else:
                    score = 0.0

                print(f"🏆 Stockfish found: {best_move} (eval: {score:.2f})")
                return best_move, score
            else:
                print("⚠️  Stockfish could not find a move")
                return None, None

        except Exception as e:
            print(f"❌ Error getting Stockfish move: {e}")
            return None, None

    def _get_simple_move(
        self, board: chess.Board, legal_moves: list, computer_is_white: bool
    ) -> tuple[chess.Move | None, float | None]:
        """
        Get the best move using simple evaluation (fallback method).

        Args:
            board: Current chess board position
            legal_moves: List of legal moves
            computer_is_white: True if computer is playing white

        Returns:
            Tuple of (best_move, evaluation_score)
        """
        print("🔍 Using simple evaluation...")

        best_move = None
        best_score = float("-inf") if computer_is_white else float("inf")

        for move in legal_moves:
            # Make the move temporarily
            board.push(move)

            # Evaluate the position
            score = self._evaluate_position(board, computer_is_white)

            # Check if this is the best move so far
            if computer_is_white and score > best_score:
                best_score = score
                best_move = move
            elif not computer_is_white and score < best_score:
                best_score = score
                best_move = move

            # Undo the move
            board.pop()

        return best_move, best_score

    def _predict_best_move(self, chess_board: ChessBoard, computer_playing_as: str | None) -> MoveAnalysis:
        """
        Predict the best move for the specified color using a simple evaluation algorithm.

        Args:
            chess_board: ChessBoard object containing the current position
            computer_playing_as: Color the computer is playing as ("white", "black", or None)

        Returns:
            MoveAnalysis object with predicted move and analysis
        """
        # If no computer color specified or no valid board position, return empty analysis
        if computer_playing_as is None or chess_board.board_position is None:
            return MoveAnalysis(
                next_move=None,
                move_coordinates=None,
                legal_moves=None,
                evaluation_score=None,
                computer_playing_as=computer_playing_as,
            )

        try:
            board = chess_board.board_position
            legal_moves = list(board.legal_moves)

            if not legal_moves:
                print("⚠️  No legal moves available")
                return MoveAnalysis(
                    next_move=None,
                    move_coordinates=None,
                    legal_moves=[],
                    evaluation_score=None,
                    computer_playing_as=computer_playing_as,
                )

            print(f"🤔 Analyzing {len(legal_moves)} legal moves for {computer_playing_as}...")

            # Determine if it's the computer's turn
            computer_is_white = computer_playing_as == "white"
            is_computer_turn = board.turn == computer_is_white

            if not is_computer_turn:
                print(f"ℹ️  It's not {computer_playing_as}'s turn to move")
                return MoveAnalysis(
                    next_move=None,
                    move_coordinates=None,
                    legal_moves=legal_moves,
                    evaluation_score=None,
                    computer_playing_as=computer_playing_as,
                )

            # Use Stockfish if available, otherwise fall back to simple evaluation
            if self.engine_type == "stockfish" and self.chess_engine is not None:
                best_move, best_score = self._get_stockfish_move(board)
            else:
                best_move, best_score = self._get_simple_move(board, legal_moves, computer_is_white)

            if best_move:
                print(f"🎯 Best move found: {best_move} (score: {best_score:.2f})")

                # Get move coordinates
                move_coordinates = self._get_move_coordinates(best_move, chess_board.chess_squares)

                return MoveAnalysis(
                    next_move=best_move,
                    move_coordinates=move_coordinates,
                    legal_moves=legal_moves,
                    evaluation_score=best_score,
                    computer_playing_as=computer_playing_as,
                )
            else:
                print("⚠️  Failed to find best move")
                return MoveAnalysis(
                    next_move=None,
                    move_coordinates=None,
                    legal_moves=legal_moves,
                    evaluation_score=None,
                    computer_playing_as=computer_playing_as,
                )

        except Exception as e:
            print(f"❌ Error during move prediction: {e}")
            return MoveAnalysis(
                next_move=None,
                move_coordinates=None,
                legal_moves=None,
                evaluation_score=None,
                computer_playing_as=computer_playing_as,
            )

    def _evaluate_position(self, board: chess.Board, computer_is_white: bool) -> float:
        """
        Simple position evaluation function.

        Args:
            board: Current board position
            computer_is_white: True if computer is playing white

        Returns:
            Position evaluation score (positive = good for white, negative = good for black)
        """
        if board.is_checkmate():
            # Checkmate is very good/bad depending on who's mated
            return 10000 if board.turn != computer_is_white else -10000

        if board.is_stalemate() or board.is_insufficient_material():
            return 0  # Draw

        # Simple material evaluation
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0,  # King value doesn't matter for material count
        }

        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value

        # Add small bonus for piece activity (number of legal moves)
        mobility_bonus = len(list(board.legal_moves)) * 0.1
        if board.turn == chess.WHITE:
            score += mobility_bonus
        else:
            score -= mobility_bonus

        return score

    def _get_move_coordinates(
        self, move: chess.Move, chess_squares: dict[int, ChessBoardSquare]
    ) -> dict[str, Point] | None:
        """
        Get pixel coordinates for a chess move.

        Args:
            move: Chess move object
            chess_squares: Dictionary mapping square numbers to coordinates

        Returns:
            Dictionary with 'from' and 'to' Point coordinates, or None if not found
        """
        try:
            # Convert chess squares to square numbers (1-64 format used in chess_squares)
            from_square_chess = move.from_square
            to_square_chess = move.to_square

            # Convert chess library squares to our square numbering (1-64)
            # chess library uses 0-63, we use 1-64
            # chess library: a1=0, h1=7, a8=56, h8=63
            # our format: a1=57, h1=64, a8=1, h8=8 (bottom-left to top-right)

            from_file = chess.square_file(from_square_chess)  # 0-7 for a-h
            from_rank = chess.square_rank(from_square_chess)  # 0-7 for 1-8
            to_file = chess.square_file(to_square_chess)
            to_rank = chess.square_rank(to_square_chess)

            # Convert to our square numbering (1-64, bottom-left to top-right)
            from_square_num = (7 - from_rank) * 8 + from_file + 1
            to_square_num = (7 - to_rank) * 8 + to_file + 1

            from_square_data = chess_squares.get(from_square_num)
            to_square_data = chess_squares.get(to_square_num)

            if from_square_data and to_square_data:
                return {
                    "from": from_square_data.center,
                    "to": to_square_data.center,
                }
            else:
                print(f"⚠️  Could not find coordinates for move {move}")
                return None

        except Exception as e:
            print(f"❌ Error getting move coordinates: {e}")
            return None

    def get_piece_at_square(self, result: dict, square_name: str) -> str | None:
        """
        Get the piece at a specific chess square.

        Args:
            result: Result dictionary from analyze_board()
            square_name: Chess square name (e.g., 'e4', 'a1')

        Returns:
            Piece name or None if no piece at that square
        """
        chess_board = result.get("chess_position")
        if chess_board is None:
            return None

        try:
            square = chess.parse_square(square_name)
            piece = chess_board.piece_at(square)
            return str(piece) if piece else None
        except (ValueError, TypeError):
            return None

    def get_square_coordinates(self, result: dict, square_name: str) -> dict | None:
        """
        Get pixel coordinates for a specific chess square.

        Args:
            result: Result dictionary from analyze_board()
            square_name: Chess square name (e.g., 'e4', 'a1')

        Returns:
            dictionary with center, corners coordinates or None
        """
        # Convert chess notation to square number (1-64)
        try:
            file = ord(square_name[0].lower()) - ord("a")  # a=0, b=1, ..., h=7
            rank = int(square_name[1]) - 1  # 1=0, 2=1, ..., 8=7

            # Convert to square number (1-64, starting from bottom-left)
            square_num = (7 - rank) * 8 + file + 1

            square_coords = result["square_coordinates"].get(square_num)
            if square_coords:
                return {
                    "center": square_coords[0],
                    "bottom_right": square_coords[1],
                    "top_right": square_coords[2],
                    "top_left": square_coords[3],
                    "bottom_left": square_coords[4],
                }
        except (ValueError, TypeError, IndexError, KeyError):
            pass

        return None

    # Perspective transformation utility methods
    def _transform_square_number_to_standard(self, camera_square_num: int) -> int:
        """
        Transform a square number from camera perspective to standard chess notation (white's perspective).

        Args:
            camera_square_num: Square number from camera perspective (1-64)

        Returns:
            Square number in standard chess notation
        """
        if not (1 <= camera_square_num <= 64):
            raise ValueError(f"Square number must be between 1 and 64, got {camera_square_num}")

        # Convert to 0-based indexing for easier math
        camera_idx = camera_square_num - 1
        camera_row = camera_idx // 8
        camera_col = camera_idx % 8

        # Transform based on white's position
        if self.white_playing_from == "b":  # Bottom (standard perspective) - no transformation needed
            standard_row = camera_row
            standard_col = camera_col
        elif self.white_playing_from == "t":  # Top (rotated 180°)
            standard_row = 7 - camera_row
            standard_col = 7 - camera_col
        elif self.white_playing_from == "l":  # Left (rotated 90° clockwise)
            standard_row = camera_col
            standard_col = 7 - camera_row
        elif self.white_playing_from == "r":  # Right (rotated 90° counter-clockwise)
            standard_row = 7 - camera_col
            standard_col = camera_row
        else:
            raise ValueError(f"Invalid white_playing_from: {self.white_playing_from}")

        # Convert back to 1-based square number
        return standard_row * 8 + standard_col + 1

    def _transform_square_number_to_camera(self, standard_square_num: int) -> int:
        """
        Transform a square number from standard chess notation to camera perspective.

        Args:
            standard_square_num: Square number in standard chess notation (1-64)

        Returns:
            Square number in camera perspective
        """
        if not (1 <= standard_square_num <= 64):
            raise ValueError(f"Square number must be between 1 and 64, got {standard_square_num}")

        # Convert to 0-based indexing for easier math
        standard_idx = standard_square_num - 1
        standard_row = standard_idx // 8
        standard_col = standard_idx % 8

        # Transform based on white's position
        if self.white_playing_from == "b":  # Bottom (standard perspective) - no transformation needed
            camera_row = standard_row
            camera_col = standard_col
        elif self.white_playing_from == "t":  # Top (rotated 180°)
            camera_row = 7 - standard_row
            camera_col = 7 - standard_col
        elif self.white_playing_from == "l":  # Left (rotated 90° counter-clockwise from standard)
            camera_row = 7 - standard_col
            camera_col = standard_row
        elif self.white_playing_from == "r":  # Right (rotated 90° clockwise from standard)
            camera_row = standard_col
            camera_col = 7 - standard_row
        else:
            raise ValueError(f"Invalid white_playing_from: {self.white_playing_from}")

        # Convert back to 1-based square number
        return camera_row * 8 + camera_col + 1
