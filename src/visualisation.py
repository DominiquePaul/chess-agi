from pathlib import Path

import cv2
import matplotlib.patches
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np

from src.datatypes import ChessAnalysis


def create_corners_and_grid_visualization(chess_analysis: ChessAnalysis) -> np.ndarray:
    """
    Create visualization showing board corners and chess grid with square labels.

    Args:
        chess_analysis: ChessAnalysis dataclass containing all analysis results

    Returns:
        Visualization image as numpy array
    """
    # Get original image and create a copy
    original_image = chess_analysis.metadata.original_image.copy()
    vis_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Extract corners from chess board squares
    # Get corner coordinates from the chess squares (corners would be at squares 1, 8, 57, 64)

    board_corners = chess_analysis.chess_board.board_corners

    if board_corners:
        # Use board corners in order: top_left, top_right, bottom_right, bottom_left
        corner_points = [
            (board_corners[0].x, board_corners[0].y),
            (board_corners[1].x, board_corners[1].y),
            (board_corners[2].x, board_corners[2].y),
            (board_corners[3].x, board_corners[3].y),
        ]
        corner_labels = ["TL", "TR", "BL", "BR"]

        for point, label in zip(corner_points, corner_labels, strict=False):
            if point is not None:
                x, y = int(point[0]), int(point[1])
                # Draw corner point at 40% size (8 instead of 20)
                cv2.circle(vis_image, (x, y), 8, (255, 0, 0), -1)
                # Add corner label
                cv2.putText(
                    vis_image,
                    label,
                    (x - 15, y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

    # Draw chess board grid with square labels
    for square_num, square in chess_analysis.chess_board.chess_squares.items():
        # Draw square outline
        points = np.array(
            [
                [int(square.tl.x), int(square.tl.y)],
                [int(square.tr.x), int(square.tr.y)],
                [int(square.br.x), int(square.br.y)],
                [int(square.bl.x), int(square.bl.y)],
            ],
            np.int32,
        )

        cv2.polylines(vis_image, [points], True, (0, 255, 0), 2)

        # Convert square number to chess notation (a1, b2, etc.)
        if 1 <= square_num <= 64:
            file = chr(ord("a") + ((square_num - 1) % 8))  # a-h
            rank = str(8 - ((square_num - 1) // 8))  # 1-8
            square_label = f"{file}{rank}"

            center_x, center_y = int(square.center.x), int(square.center.y)
            cv2.putText(
                vis_image,
                square_label,
                (center_x - 15, center_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

    return vis_image


def create_piece_bounding_boxes_visualization(chess_analysis: ChessAnalysis) -> np.ndarray:
    """
    Create visualization showing ONLY bounding boxes for detected pieces on the original image.

    Shows:
    - Colored bounding boxes around detected pieces
    - Piece names and confidence scores
    - NO center dots

    Args:
        chess_analysis: ChessAnalysis dataclass containing all analysis results

    Returns:
        Visualization image as numpy array
    """
    # Use the original image from metadata
    original_image = chess_analysis.metadata.original_image.copy()
    vis_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Get detected chess pieces with bounding boxes
    chess_pieces = chess_analysis.chess_board.chess_pieces

    # Define colors for piece types
    white_piece_color = (100, 149, 237)  # CornflowerBlue
    black_piece_color = (220, 20, 60)  # Crimson

    print(f"📦 Drawing {len(chess_pieces)} bounding boxes...")

    for piece in chess_pieces:
        # Determine color based on piece name
        if "white" in piece.piece_name.lower():
            color = white_piece_color
        else:
            color = black_piece_color

        # Draw bounding box
        x_left = int(piece.x_left)
        y_top = int(piece.y_top)
        x_right = int(piece.x_right)
        y_bottom = int(piece.y_bottom)

        # Draw rectangle (bounding box)
        cv2.rectangle(vis_image, (x_left, y_top), (x_right, y_bottom), color, 2)

        # Add piece name and confidence score
        short_name = piece.piece_name.replace("-", " ").title()
        confidence_text = f"{short_name} ({piece.confidence:.2f})"

        # Position text above the bounding box
        text_x = x_left
        text_y = y_top - 10 if y_top > 20 else y_bottom + 20

        # Add text with background for better visibility
        (text_width, text_height), _ = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw background rectangle for text
        cv2.rectangle(vis_image, (text_x, text_y - text_height - 5), (text_x + text_width, text_y + 5), color, -1)

        # Draw text
        cv2.putText(
            vis_image,
            confidence_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),  # White text
            1,
        )

    return vis_image


def create_piece_centers_visualization(chess_analysis: ChessAnalysis, use_weighted_center: bool = True) -> np.ndarray:
    """
    Create visualization showing center dots for detected pieces and chess board grid on the original image.

    Shows:
    - Chess board grid with square labels
    - Small colored dots at piece centers (weighted center by default)
    - Piece names near the dots
    - NO bounding boxes or corner markers

    Args:
        chess_analysis: ChessAnalysis dataclass containing all analysis results
        use_weighted_center: If True, use weighted_center (default), if False use center

    Returns:
        Visualization image as numpy array
    """
    # Use the original image from metadata
    original_image = chess_analysis.metadata.original_image.copy()
    vis_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Determine which coordinate method to use
    coordinate_method = "weighted_center" if use_weighted_center else "center"
    print(f"🎯 Drawing piece centers using {coordinate_method} coordinates")

    # Draw chess board grid with square labels (from corners visualization)
    for square_num, square in chess_analysis.chess_board.chess_squares.items():
        # Draw square outline
        points = np.array(
            [
                [int(square.tl.x), int(square.tl.y)],
                [int(square.tr.x), int(square.tr.y)],
                [int(square.br.x), int(square.br.y)],
                [int(square.bl.x), int(square.bl.y)],
            ],
            np.int32,
        )

        cv2.polylines(vis_image, [points], True, (0, 255, 0), 2)

        # Convert square number to chess notation (a1, b2, etc.)
        if 1 <= square_num <= 64:
            file = chr(ord("a") + ((square_num - 1) % 8))  # a-h
            rank = str(8 - ((square_num - 1) // 8))  # 1-8
            square_label = f"{file}{rank}"

            center_x, center_y = int(square.center.x), int(square.center.y)
            cv2.putText(
                vis_image,
                square_label,
                (center_x - 15, center_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

    # Get detected chess pieces
    chess_pieces = chess_analysis.chess_board.chess_pieces

    # Define colors for piece types
    white_piece_color = (100, 149, 237)  # CornflowerBlue
    black_piece_color = (220, 20, 60)  # Crimson

    print(f"🎯 Drawing {len(chess_pieces)} center dots...")

    for piece in chess_pieces:
        # Determine color based on piece name
        if "white" in piece.piece_name.lower():
            color = white_piece_color
        else:
            color = black_piece_color

        # Get the appropriate coordinate based on the parameter
        piece_coord = piece.weighted_center if use_weighted_center else piece.center

        # Draw center dot
        center_x = int(piece_coord.x)
        center_y = int(piece_coord.y)
        cv2.circle(vis_image, (center_x, center_y), 6, color, -1)

        # Add white outline to make dot more visible
        cv2.circle(vis_image, (center_x, center_y), 6, (255, 255, 255), 2)

        # Add piece name near the dot
        short_name = piece.piece_name.replace("-", " ").title()
        confidence_text = f"{short_name} ({piece.confidence:.2f})"

        # Position text near the dot
        text_x = center_x + 10
        text_y = center_y - 10

        # Add text with background for better visibility
        (text_width, text_height), _ = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

        # Draw background rectangle for text
        cv2.rectangle(vis_image, (text_x, text_y - text_height - 3), (text_x + text_width, text_y + 3), color, -1)

        # Draw text
        cv2.putText(
            vis_image,
            confidence_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),  # White text
            1,
        )

    return vis_image


def create_chess_diagram_png(chess_analysis: ChessAnalysis, output_path: Path) -> np.ndarray:
    """
    Create PNG chess diagram with classical piece symbols.

    Args:
        chess_analysis: ChessAnalysis dataclass containing all analysis results
        output_path: Path where to save the PNG file

    Returns:
        Diagram image as numpy array
    """
    chess_position = chess_analysis.chess_board.board_position

    # Create the chess diagram using matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))

    # Unicode chess symbols
    piece_symbols = {
        "K": "♔",
        "Q": "♕",
        "R": "♖",
        "B": "♗",
        "N": "♘",
        "P": "♙",  # White
        "k": "♚",
        "q": "♛",
        "r": "♜",
        "b": "♝",
        "n": "♞",
        "p": "♟",  # Black
    }

    # Board colors
    light_color = "#f0d9b5"
    dark_color = "#b58863"

    # Draw the board squares
    for rank in range(8):
        for file in range(8):
            # Determine square color
            is_light = (rank + file) % 2 == 0
            color = light_color if is_light else dark_color

            # Draw square
            square = matplotlib.patches.Rectangle(
                (file, 7 - rank),
                1,
                1,
                facecolor=color,
                edgecolor="#8b4513",
                linewidth=1,
            )
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
                            text_color = "white" if not is_light else "#333333"
                            edge_color = "#000000" if not is_light else "#666666"
                        else:  # Black piece
                            text_color = "#000000"
                            edge_color = "#ffffff" if is_light else "#cccccc"

                        # Add piece symbol with outline for better visibility
                        ax.text(
                            text_x,
                            text_y,
                            piece_symbol,
                            fontsize=28,
                            ha="center",
                            va="center",
                            color=text_color,
                            weight="bold",
                            path_effects=[
                                matplotlib.patheffects.Stroke(linewidth=1.5, foreground=edge_color),
                                matplotlib.patheffects.Normal(),
                            ],
                        )
                except Exception:
                    # Skip if chess library issues
                    pass

    # Add file labels (a-h) at bottom
    for file in range(8):
        file_letter = chr(ord("a") + file)
        ax.text(
            file + 0.5,
            -0.2,
            file_letter,
            fontsize=14,
            ha="center",
            va="center",
            color="#8b4513",
            weight="bold",
        )

    # Add rank labels (1-8) on left side
    for rank in range(8):
        rank_number = rank + 1
        ax.text(
            -0.2,
            rank + 0.5,
            str(rank_number),
            fontsize=14,
            ha="center",
            va="center",
            color="#8b4513",
            weight="bold",
        )

    # Set up the plot
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 8.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Add title
    if chess_position is not None:
        piece_count = len([s for s in chess_analysis.chess_board.chess_squares.values() if s.piece_name is not None])
        title = f"Chess Position ({piece_count} pieces detected)"
        if chess_position.turn:
            title += " - White to move"
        else:
            title += " - Black to move"
    else:
        title = "Chess Board (No pieces detected)"

    ax.set_title(title, fontsize=16, weight="bold", pad=20)

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")

    # Read the saved image as numpy array for return
    img_array = cv2.imread(str(output_path))
    if img_array is not None:
        # Convert BGR to RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    else:
        # Fallback: create a simple placeholder
        img_array = np.zeros((800, 800, 3), dtype=np.uint8)

    plt.close(fig)

    return img_array
