from dataclasses import dataclass

import chess
import numpy as np


@dataclass
class Metadata:
    original_image: np.ndarray
    original_dimensions: tuple[float, float]
    segmentation_model: str
    segmentation_method: str
    piece_detection_model: str | None
    piece_detection_available: bool
    confidence_threshold: float
    white_playing_from: str


@dataclass
class Point:
    x: float
    y: float

    def __array__(self, dtype=None):
        """Convert Point to numpy array for use with np.array()"""
        return np.array([self.x, self.y], dtype=dtype)


@dataclass
class ChessPieceBoundingBox:
    piece_class: int
    piece_name: str
    confidence: float
    x_left: float
    y_top: float
    x_right: float
    y_bottom: float
    assigned_square: int | None

    @property
    def center(self) -> Point:
        return Point((self.x_left + self.x_right) / 2, (self.y_top + self.y_bottom) / 2)

    @property
    def weighted_center(self) -> Point:
        return Point((self.x_left + self.x_right) / 2, (1 * self.y_top + 3 * self.y_bottom) / 4)


@dataclass
class ChessBoardSquare:
    center: Point
    br: Point
    tr: Point
    tl: Point
    bl: Point
    piece_class_id: int | None
    piece_name: str | None

    @property
    def coords(self) -> list[Point]:
        return [self.tl, self.tr, self.br, self.bl]


@dataclass
class ChessBoard:
    board_corners: list[Point] | None
    chess_squares: dict[int, ChessBoardSquare]
    chess_pieces: list[ChessPieceBoundingBox]
    white_playing_from: str = "b"
    position_string: str = "Invalid position"
    board_position: chess.Board | None = None

    def __post_init__(self):
        """Create chess board position and FEN string after initialization"""
        # Create empty board
        try:
            import chess

            board = chess.Board(None)

            # Piece mapping from class names to chess piece types
            piece_type_mapping = {
                "white-pawn": (chess.PAWN, chess.WHITE),
                "black-pawn": (chess.PAWN, chess.BLACK),
                "white-knight": (chess.KNIGHT, chess.WHITE),
                "black-knight": (chess.KNIGHT, chess.BLACK),
                "white-bishop": (chess.BISHOP, chess.WHITE),
                "black-bishop": (chess.BISHOP, chess.BLACK),
                "white-rook": (chess.ROOK, chess.WHITE),
                "black-rook": (chess.ROOK, chess.BLACK),
                "white-queen": (chess.QUEEN, chess.WHITE),
                "black-queen": (chess.QUEEN, chess.BLACK),
                "white-king": (chess.KING, chess.WHITE),
                "black-king": (chess.KING, chess.BLACK),
            }

            # Place pieces on board - assumes squares are already in standard chess notation
            # Perspective transformation should be handled by ChessBoardAnalyzer before creating ChessBoard
            for square_num, cs in self.chess_squares.items():
                piece_name = cs.piece_name

                if piece_name in piece_type_mapping:
                    piece_type, color = piece_type_mapping[piece_name]

                    # Convert square number to chess square (assumes standard notation)
                    # Square numbers are 1-64, starting from bottom-left (a1=1, h8=64)
                    square_idx = square_num - 1
                    file = square_idx % 8  # 0-7 for a-h
                    rank = 7 - (square_idx // 8)  # 0-7 for ranks 1-8 (inverted)

                    chess_square = chess.square(file, rank)
                    board.set_piece_at(chess_square, chess.Piece(piece_type, color))

            self.board_position = board
            self.position_string = board.fen()

        except (ValueError, TypeError, ImportError):
            self.board_position = None
            self.position_string = "Invalid position"


@dataclass
class ChessBoardVisualisations:
    original_image: np.ndarray
    warped_board: np.ndarray
    board_with_grid: np.ndarray
    piece_boxes: np.ndarray
    piece_centers: np.ndarray


@dataclass
class MoveAnalysis:
    next_move: None
    move_coordinates: None
    legal_moves: None


@dataclass
class ChessAnalysis:
    metadata: Metadata
    chess_board: ChessBoard
    visualisations: ChessBoardVisualisations | None
    move_analysis: MoveAnalysis
