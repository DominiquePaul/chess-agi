import tempfile
from pathlib import Path

import cv2
import matplotlib.patches
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np

from src.datatypes import ChessAnalysis


def draw_move_arrow(vis_image: np.ndarray, chess_analysis: ChessAnalysis, verbose: bool = False) -> np.ndarray:
    """
    Draw an arrow showing the predicted best move on a visualization image.

    Args:
        vis_image: The visualization image to draw on
        chess_analysis: ChessAnalysis containing move prediction data
        verbose: Whether to print debug messages

    Returns:
        Image with move arrow drawn (modifies original)
    """
    move_analysis = chess_analysis.move_analysis

    # Only draw if we have a predicted move with coordinates
    if (
        move_analysis
        and move_analysis.next_move
        and move_analysis.move_coordinates
        and move_analysis.computer_playing_as
    ):
        try:
            from_coord = move_analysis.move_coordinates["from"]
            to_coord = move_analysis.move_coordinates["to"]

            # Convert to integer coordinates
            start_point = (int(from_coord.x), int(from_coord.y))
            end_point = (int(to_coord.x), int(to_coord.y))

            # Choose arrow color based on computer color
            if move_analysis.computer_playing_as == "white":
                arrow_color = (128, 0, 128)  # Purple for white
                outline_color = (255, 255, 255)  # White outline
            else:
                arrow_color = (128, 0, 128)  # Purple for black
                outline_color = (255, 255, 255)  # White outline

            # Draw thick arrow with outline for visibility
            # First draw a thicker outline
            cv2.arrowedLine(vis_image, start_point, end_point, outline_color, 8, tipLength=0.3)
            # Then draw the main arrow
            cv2.arrowedLine(vis_image, start_point, end_point, arrow_color, 5, tipLength=0.3)

            # Add move label
            move_text = f"Best: {move_analysis.next_move}"
            if move_analysis.evaluation_score is not None:
                move_text += f" ({move_analysis.evaluation_score:.2f})"

            # Position label near the start of the arrow
            label_x = start_point[0] - 20
            label_y = start_point[1] - 20

            # Ensure label is within image bounds
            if label_x < 0:
                label_x = start_point[0] + 20
            if label_y < 20:
                label_y = start_point[1] + 40

            # Get text size for background
            (text_width, text_height), _ = cv2.getTextSize(move_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            # Draw background rectangle for text
            bg_color = outline_color
            cv2.rectangle(
                vis_image,
                (label_x - 5, label_y - text_height - 5),
                (label_x + text_width + 5, label_y + 5),
                bg_color,
                -1,
            )

            # Draw text
            cv2.putText(vis_image, move_text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, arrow_color, 2)

            if verbose:
                print(f"🏹 Added move arrow: {move_analysis.next_move} ({move_analysis.computer_playing_as})")

        except Exception as e:
            if verbose:
                print(f"⚠️  Failed to draw move arrow: {e}")

    return vis_image


def create_corners_and_grid_visualization(chess_analysis: ChessAnalysis, verbose: bool = False) -> np.ndarray:
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
        corner_labels = ["TL", "TR", "BR", "BL"]

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

        # Display square number
        if 1 <= square_num <= 64:
            square_label = str(square_num)

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

    # Add move arrow if available
    vis_image = draw_move_arrow(vis_image, chess_analysis, verbose)

    return vis_image


def create_piece_bounding_boxes_visualization(chess_analysis: ChessAnalysis, verbose: bool = False) -> np.ndarray:
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

    if verbose:
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

    # Add move arrow if available
    vis_image = draw_move_arrow(vis_image, chess_analysis, verbose)

    return vis_image


def create_piece_centers_visualization(
    chess_analysis: ChessAnalysis, use_weighted_center: bool = True, verbose: bool = False
) -> np.ndarray:
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
    if verbose:
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

        # Display square number
        if 1 <= square_num <= 64:
            square_label = str(square_num)

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

    if verbose:
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


def draw_move_arrow_matplotlib(ax, chess_analysis: ChessAnalysis, verbose: bool = False) -> None:
    """
    Draw an arrow showing the predicted best move on a matplotlib axes.

    Args:
        ax: Matplotlib axes to draw on
        chess_analysis: ChessAnalysis containing move prediction data
    """
    move_analysis = chess_analysis.move_analysis

    # Only draw if we have a predicted move
    if move_analysis and move_analysis.next_move and move_analysis.computer_playing_as:
        try:
            import chess

            # Get the move squares
            from_square = move_analysis.next_move.from_square
            to_square = move_analysis.next_move.to_square

            # Convert to board coordinates (0-7 for files, 0-7 for ranks)
            from_file = chess.square_file(from_square)
            from_rank = chess.square_rank(from_square)
            to_file = chess.square_file(to_square)
            to_rank = chess.square_rank(to_square)

            # Convert to matplotlib coordinates (y-axis flipped for board display)
            from_x = from_file + 0.5
            from_y = 7 - from_rank + 0.5
            to_x = to_file + 0.5
            to_y = 7 - to_rank + 0.5

            # Choose arrow color based on computer color
            if move_analysis.computer_playing_as == "white":
                arrow_color = "purple"
                edge_color = "white"
            else:
                arrow_color = "purple"
                edge_color = "white"

            # Draw arrow
            arrow = matplotlib.patches.FancyArrowPatch(
                (from_x, from_y),
                (to_x, to_y),
                arrowstyle="->",
                mutation_scale=25,
                linewidth=4,
                facecolor=arrow_color,
                edgecolor=edge_color,
                alpha=0.8,
                zorder=10,
            )
            ax.add_patch(arrow)

            # Add move label
            move_text = f"Best: {move_analysis.next_move}"
            if move_analysis.evaluation_score is not None:
                move_text += f" ({move_analysis.evaluation_score:.2f})"

            # Position label near the start of the arrow
            label_x = from_x - 0.3
            label_y = from_y - 0.5

            # Ensure label is within bounds
            if label_x < 0:
                label_x = from_x + 0.3
            if label_y < 0:
                label_y = from_y + 0.5

            # Add text with background
            ax.text(
                label_x,
                label_y,
                move_text,
                fontsize=10,
                weight="bold",
                color=arrow_color,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": edge_color, "alpha": 0.7},
                zorder=11,
            )

            if verbose:
                print(
                    f"🏹 Added move arrow to chess diagram: {move_analysis.next_move} ({move_analysis.computer_playing_as})"
                )

        except Exception as e:
            if verbose:
                print(f"⚠️  Failed to draw move arrow on diagram: {e}")


def create_chess_diagram_png(chess_analysis: ChessAnalysis, output_path: Path, verbose: bool = False) -> np.ndarray:
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

    # Draw move arrows
    draw_move_arrow_matplotlib(ax, chess_analysis, verbose)

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


def create_combined_visualization(
    chess_analysis: ChessAnalysis,
    output_dir: Path,
    image_name: str,
    skip_piece_detection: bool = False,
    use_weighted_center: bool = True,
    verbose: bool = False,
) -> None:
    """
    Create a combined visualization with all plots as subplots.

    Args:
        chess_analysis: ChessAnalysis dataclass containing all analysis results
        output_dir: Directory to save the combined visualization
        image_name: Base name for the output file
        skip_piece_detection: Whether piece detection was skipped
        use_weighted_center: Whether to use weighted center coordinates for piece mapping
    """
    try:
        # Prepare the subplot data
        plots_data = []
        titles = []

        # 1. Corners and grid visualization
        corners_vis = create_corners_and_grid_visualization(chess_analysis, verbose)
        plots_data.append(corners_vis)
        titles.append("Board Corners and Chess Grid")

        # 2. Piece bounding boxes visualization (if pieces detected and not skipped)
        if not skip_piece_detection and chess_analysis.chess_board.chess_pieces:
            pieces_boxes_vis = create_piece_bounding_boxes_visualization(chess_analysis, verbose)
            plots_data.append(pieces_boxes_vis)
            titles.append("Chess Piece Bounding Boxes")

            # 3. Piece centers visualization
            coordinate_method = "weighted_center" if use_weighted_center else "geometric_center"
            pieces_centers_vis = create_piece_centers_visualization(chess_analysis, use_weighted_center, verbose)
            plots_data.append(pieces_centers_vis)
            titles.append(f"Chess Piece Centers ({coordinate_method.replace('_', ' ').title()})")

        # 4. Chess diagram (if chess position exists)
        chess_position = chess_analysis.chess_board.board_position
        if chess_position:
            # Use the existing create_chess_diagram_png function
            try:
                # Create a temporary file for the chess diagram
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    temp_path = Path(temp_file.name)

                # Generate the chess diagram using the existing function
                chess_diagram_array = create_chess_diagram_png(chess_analysis, temp_path, verbose)

                # Clean up the temporary file
                temp_path.unlink(missing_ok=True)

                plots_data.append(chess_diagram_array)
                titles.append("Chess Position Diagram")

            except Exception as e:
                print(f"      ⚠️  Could not include chess diagram in combined plot: {e}")

        # Determine subplot layout
        n_plots = len(plots_data)
        if n_plots == 1:
            rows, cols = 1, 1
        elif n_plots == 2:
            rows, cols = 1, 2
        elif n_plots == 3:
            rows, cols = 2, 2  # 2x2 with one empty
        else:  # n_plots == 4
            rows, cols = 2, 2

        # Create the combined plot
        fig, axes = plt.subplots(rows, cols, figsize=(16, 12))

        # Handle case where axes is not a 2D array
        if n_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        else:
            axes = axes.flatten()

        # Plot each visualization
        for i, (plot_data, title) in enumerate(zip(plots_data, titles, strict=False)):
            axes[i].imshow(plot_data)
            axes[i].set_title(title, fontsize=12, fontweight="bold")
            axes[i].axis("off")

        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].axis("off")

        # Adjust layout and save
        plt.tight_layout()
        combined_path = output_dir / f"{image_name}_combined_analysis.png"
        plt.savefig(combined_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"      💾 Saved: {combined_path}")

    except Exception as e:
        print(f"      ⚠️  Failed to create combined visualization: {e}")
