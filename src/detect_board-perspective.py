## Extracting Chess Squares with Perspective Transformation ( image --> fen format)
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import math
import os
import chess
import chess.svg

from src.chess_piece_detection.model import ChessModel
from src.utils import is_notebook

# Configure matplotlib for inline plotting in Jupyter notebooks
if is_notebook():
    print("Notebook environment detected - configuring matplotlib for inline plotting")
    try:
        # Try to get the IPython instance and set matplotlib inline
        from IPython.core.getipython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            ipython.run_line_magic('matplotlib', 'inline')
            print("✓ Matplotlib inline mode enabled")
    except Exception as e:
        print(f"Note: Could not enable matplotlib inline mode: {e}")
        # Fallback to setting matplotlib backend directly
        import matplotlib
        matplotlib.use('inline')
        print("✓ Matplotlib inline backend set directly")
else:
    print("Non-notebook environment detected")

# Configuration constants
show_images = False  # Disable intermediate visualizations
class_dict = {
    0: "black-bishop",
    1: "black-king",
    2: "black-knight",
    3: "black-pawn",
    4: "black-queen",
    5: "black-rook",
    6: "white-bishop",
    7: "white-king",
    8: "white-knight",
    9: "white-pawn",
    10: "white-queen",
    11: "white-rook",
}

# Path of Image that you want to convert
image_path = Path(os.environ["DATA_FOLDER_PATH"]) / "eval_images/chess_4.jpeg"



# Check if the image exists
if not os.path.exists(image_path):
    print(f"Warning: Image file {image_path} does not exist.")
    print(
        "Please add a test image to the data/test_images directory or update the image_path variable."
    )
    # Use a placeholder path for demonstration
    print("Using a placeholder for now to avoid errors...")
    # You can either exit or continue with just the code structure
    # exit(1)


def detect_edges(path: str, hough_threshold: int = 200, min_line_length: int = 200, max_line_gap: int = 800, dilation_kernel_size: int=1):
    image = cv2.imread(path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # rgb_image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    ## Processing Image  -->  OTSU Threshold , Canny edge detection , dilate , HoughLinesP

    # OTSU threshold
    ret, otsu_binary = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Canny edge detection
    canny_image = cv2.Canny(otsu_binary, 1, 255)

    # Dilation
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilation_image = cv2.dilate(canny_image, kernel, iterations=1)

    if show_images:
        plt.figure(figsize=(10, 8))
        plt.imshow(dilation_image, cmap='gray')
        plt.title('Dilation Image')
        plt.axis('off')
        plt.show()

    # Hough Lines
    lines = cv2.HoughLinesP(
        dilation_image, 1, np.pi / 180, threshold=hough_threshold, minLineLength=min_line_length, maxLineGap=max_line_gap
    )
    print(len(lines) if lines is not None else 0)
    # Create an image that contains only black pixels
    black_image = np.zeros_like(dilation_image)

    # Draw only lines that are output of HoughLinesP function to the "black_image"
    if lines is not None and len(lines) > 0:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # draw only lines to the "black_image"
            cv2.line(black_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Dilation
    kernel = np.ones((3, 3), np.uint8)
    black_image = cv2.dilate(black_image, kernel, iterations=1)

    return image, canny_image, black_image, lines


"""
 Find Contours , sort contours points(4 point), and display valid squares on new fully black image
 By saying "valid squares" , I mean geometrically. With some threshold value , 4 length of a square must be close to each other 
  4 point --> bottomright , topright , topleft , bottomleft
"""


def detect_squares(image, black_image, min_area_threshold: float = 0.001, max_area_threshold: float = 0.03125):
    # find contours
    board_contours, hierarchy = cv2.findContours(
        black_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # blank image for displaying all contours
    all_contours_image = np.zeros_like(black_image)

    # Copy blank image for displaying all squares
    squares_image = np.copy(image)

    # blank image for displaying valid contours (squares)
    valid_squares_image = np.zeros_like(black_image)

    size_img = image.shape[0] * image.shape[1]

    # loop through contours and filter them by deciding if they are potential squares
    for contour in board_contours:
        if min_area_threshold * size_img < cv2.contourArea(contour) < max_area_threshold * size_img:
            # Approximate the contour to a simpler shape
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # if polygon has 4 vertices
            if len(approx) == 4:
                # 4 points of polygon
                pts = [pt[0].tolist() for pt in approx]

                # create same pattern for points , bottomright(1) , topright(2) , topleft(3) , bottomleft(4)
                index_sorted = sorted(pts, key=lambda x: x[0], reverse=True)

                #  Y values
                if index_sorted[0][1] < index_sorted[1][1]:
                    cur = index_sorted[0]
                    index_sorted[0] = index_sorted[1]
                    index_sorted[1] = cur

                if index_sorted[2][1] > index_sorted[3][1]:
                    cur = index_sorted[2]
                    index_sorted[2] = index_sorted[3]
                    index_sorted[3] = cur

                # bottomright(1) , topright(2) , topleft(3) , bottomleft(4)
                pt1 = index_sorted[0]
                pt2 = index_sorted[1]
                pt3 = index_sorted[2]
                pt4 = index_sorted[3]

                # find rectangle that fits 4 point
                x, y, w, h = cv2.boundingRect(contour)
                # find center of rectangle
                center_x = (x + (x + w)) / 2
                center_y = (y + (y + h)) / 2

                # calculate length of 4 side of rectangle
                l1 = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
                l2 = math.sqrt((pt2[0] - pt3[0]) ** 2 + (pt2[1] - pt3[1]) ** 2)
                l3 = math.sqrt((pt3[0] - pt4[0]) ** 2 + (pt3[1] - pt4[1]) ** 2)
                l4 = math.sqrt((pt1[0] - pt4[0]) ** 2 + (pt1[1] - pt4[1]) ** 2)

                # Create a list of lengths
                lengths = [l1, l2, l3, l4]

                # Get the maximum and minimum lengths
                max_length = max(lengths)
                min_length = min(lengths)

                # Check if this length values are suitable for a square , this threshold value plays crucial role for squares ,
                if (
                    (max_length - min_length) <= 35
                ):  # 20 for smaller boards  , 50 for bigger , 35 works most of the time
                    valid_square = True
                else:
                    valid_square = False

                if valid_square:
                    # Draw the lines between the points
                    cv2.line(squares_image, pt1, pt2, (255, 255, 0), 7)
                    cv2.line(squares_image, pt2, pt3, (255, 255, 0), 7)
                    cv2.line(squares_image, pt3, pt4, (255, 255, 0), 7)
                    cv2.line(squares_image, pt1, pt4, (255, 255, 0), 7)

                    # Draw only valid squares to "valid_squares_image"
                    cv2.line(valid_squares_image, pt1, pt2, (255, 255, 0), 7)
                    cv2.line(valid_squares_image, pt2, pt3, (255, 255, 0), 7)
                    cv2.line(valid_squares_image, pt3, pt4, (255, 255, 0), 7)
                    cv2.line(valid_squares_image, pt1, pt4, (255, 255, 0), 7)

                # Draw only valid squares to "valid_squares_image"
                cv2.line(all_contours_image, pt1, pt2, (255, 255, 0), 7)
                cv2.line(all_contours_image, pt2, pt3, (255, 255, 0), 7)
                cv2.line(all_contours_image, pt3, pt4, (255, 255, 0), 7)
                cv2.line(all_contours_image, pt1, pt4, (255, 255, 0), 7)

    #### Dilation to the image that contains only valid squares (gemoetrically valid)
    # Apply dilation to the valid_squares_image
    kernel = np.ones((1, 1), np.uint8)
    dilated_valid_squares_image = cv2.dilate(valid_squares_image, kernel, iterations=1)

    #### Find biggest contour of image

    # Find contours of dilated_valid_squares_image
    contours, _ = cv2.findContours(
        dilated_valid_squares_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # take biggest contour
    largest_contour = max(contours, key=cv2.contourArea)

    return largest_contour, all_contours_image, dilated_valid_squares_image


def find_extreme_points(largest_contour):
    #### Find 4 extreme point of chess board
    # Initialize variables to store extreme points
    top_left = None
    top_right = None
    bottom_left = None
    bottom_right = None

    # Loop through the contour to find extreme points
    for point in largest_contour[:, 0]:
        x, y = point

        if top_left is None or (x + y < top_left[0] + top_left[1]):
            top_left = (x, y)

        if top_right is None or (x - y > top_right[0] - top_right[1]):
            top_right = (x, y)

        if bottom_left is None or (x - y < bottom_left[0] - bottom_left[1]):
            bottom_left = (x, y)

        if bottom_right is None or (x + y > bottom_right[0] + bottom_right[1]):
            bottom_right = (x, y)
    return top_left, top_right, bottom_left, bottom_right


def apply_perspective_transformation(
    image, top_left, top_right, bottom_left, bottom_right
):
    # read image and convert it to different color spaces
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define the four source points (replace with actual coordinates)
    extreme_points_list = np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32)

    threshold = 0  # Extra space on all sides

    width, height = 1200, 1200

    # Define the destination points (shifted by 'threshold' on all sides)
    dst_pts = np.array(
        [
            [threshold, threshold],
            [width + threshold, threshold],
            [threshold, height + threshold],
            [width + threshold, height + threshold],
        ], dtype=np.float32
    )

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(extreme_points_list, dst_pts)

    # Apply the transformation with extra width and height
    warped_image = cv2.warpPerspective(
        rgb_image, M, (width + 2 * threshold, height + 2 * threshold)
    )

    return M, warped_image


def draw_squares(warped_image):
    #### Divide board to 64 square
    # Assuming area_warped is already defined
    # Define number of squares (8x8 for chessboard)
    warped_image_copy = warped_image.copy()
    rows, cols = 8, 8

    # Calculate the width and height of each square
    width, height = 1200, 1200
    square_width = width // cols
    square_height = height // rows

    # Draw the squares on the warped image
    for i in range(rows):
        for j in range(cols):
            # Calculate top-left and bottom-right corners of each square
            top_left = (j * square_width, i * square_height)
            bottom_right = ((j + 1) * square_width, (i + 1) * square_height)

            # Draw a rectangle for each square
            cv2.rectangle(
                warped_image_copy, top_left, bottom_right, (0, 255, 0), 4
            )  # Green color, thickness 2
    return warped_image_copy


def get_unwarped_grid_coordinates(squares_data_original, rows=8, cols=8):
    """
    Returns the coordinates for drawing the grid on the unwarped image.
    """
    # Get the actual board corners from the correct squares
    bottom_left_square = squares_data_original[0]  # First square (bottom left)
    bottom_right_square = squares_data_original[7]  # Last square of first row
    top_left_square = squares_data_original[56]  # First square of last row
    top_right_square = squares_data_original[63]  # Last square (top right)

    # Store the four extreme points of the board
    extreme_points_list = np.array(
        [
            top_left_square[3],  # Top left of top-left square
            top_right_square[2],  # Top right of top-right square
            bottom_left_square[4],  # Bottom left of bottom-left square
            bottom_right_square[1],  # Bottom right of bottom-right square
        ], dtype=np.float32
    )

    return extreme_points_list, squares_data_original


def draw_unwarped_grid(
    image, extreme_points_list, squares_data_original, rows=8, cols=8
):
    """
    Draws the grid and corner markers on the unwarped image.
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw the grid lines on the original image
    for idx, square in enumerate(squares_data_original):
        # Calculate row and column from index
        i = idx // cols  # Row number
        j = idx % cols  # Column number

        # Convert coordinates to integers for drawing
        x_center, y_center = tuple(map(int, square[0]))
        bottom_right = tuple(map(int, square[1]))
        top_right = tuple(map(int, square[2]))
        top_left = tuple(map(int, square[3]))
        bottom_left = tuple(map(int, square[4]))

        # Draw the grid lines
        # Only draw top and left lines for each square to avoid duplicate lines
        cv2.line(rgb_image, top_left, top_right, (0, 255, 0), 6)  # Top line
        cv2.line(rgb_image, top_left, bottom_left, (0, 255, 0), 6)  # Left line

        # Draw right and bottom lines only for the last column and first row respectively
        if j == cols - 1:
            cv2.line(rgb_image, top_right, bottom_right, (0, 255, 0), 8)  # Right line
        if i == 0:
            cv2.line(
                rgb_image, bottom_left, bottom_right, (0, 255, 0), 8
            )  # Bottom line

    # Draw white circles at the extreme points to mark the corners
    for point in extreme_points_list:
        x, y = int(point[0]), int(point[1])
        cv2.circle(rgb_image, (x, y), 25, (255, 255, 255), -1)

    return rgb_image


def get_square_coordinates(squares_data_original):
    """
    Convert square coordinates to a dictionary format.

    Args:
        squares_data_original: List of square coordinates in format [center, bottom_right, top_right, top_left, bottom_left]

    Returns:
        dict: Dictionary with cell numbers as keys and coordinate lists as values including center point
    """
    # Create temporary list to store coordinates
    coordinates_list = []

    # Process each square's coordinates
    for coordinate in squares_data_original:
        center, bottom_right, top_right, top_left, bottom_left = coordinate
        coordinates_list.append(
            [
                float(center[0]),
                float(center[1]),  # center x, y
                float(bottom_right[0]),
                float(bottom_right[1]),  # x1, y1
                float(top_right[0]),
                float(top_right[1]),  # x2, y2
                float(top_left[0]),
                float(top_left[1]),  # x3, y3
                float(bottom_left[0]),
                float(bottom_left[1]),  # x4, y4
            ]
        )

    # Convert to dictionary format
    coord_dict = {}
    for cell, row in enumerate(coordinates_list, 1):
        coord_dict[cell] = [
            [row[0], row[1]],  # center
            [row[2], row[3]],  # bottom right
            [row[4], row[5]],  # top right
            [row[6], row[7]],  # top left
            [row[8], row[9]],  # bottom left
        ]

    return coord_dict


def process_square_coordinates_and_draw(image_path_str, squares_data_original):
    """
    Process square coordinates and draw polygons on the image.
    
    Returns:
        tuple: (image_with_polygons, square_coordinates_data)
    """
    square_coordinates_data = get_square_coordinates(squares_data_original)
    image = cv2.imread(image_path_str)

    # Loop through the square_coordinates_data dictionary and draw polygons
    for cell_id, coords in square_coordinates_data.items():
        # Extract coordinates from the dictionary
        # Format: [bottom_right, top_right, top_left, bottom_left]
        pts = np.array(coords, np.int32)
        pts = pts.reshape((-1, 1, 2))

        # Draw circle at the center of each square
        square_idx = cell_id - 1  # Convert to 0-indexed for squares_data_original
        center_point = (
            int(squares_data_original[square_idx][0][0]),
            int(squares_data_original[square_idx][0][1]),
        )
        cv2.circle(image, center_point, 3, (0, 255, 0), 3)

        # Draw polygon around each square
        cv2.polylines(
            image, [pts], True, (255, 255, 255), thickness=8
        )  # White polygon with thickness 8

    return image, square_coordinates_data


def detect_chess_pieces(image_path_str, model_path, conf=0.5):
    """
    Load YOLO model and detect chess pieces in the image.
    
    Returns:
        tuple: (chess_pieces_predictions, image_with_detections)
    """
    # YOLOv8 model
    model = ChessModel(Path(model_path))
    # Lower confidence threshold to detect more pieces
    chess_pieces_preds = model.predict(image_path_str, conf=conf)
    
    # Print debugging information
    if chess_pieces_preds.boxes is not None:
        num_detections = len(chess_pieces_preds.boxes.xyxy)
        print(f"Number of detections in boxes: {num_detections}")
        print(f"Boxes shape: {chess_pieces_preds.boxes.xyxy.shape}")
    else:
        num_detections = 0
        print(f"Number of detections: {num_detections} (no boxes found)")
    
    if chess_pieces_preds.boxes is not None and chess_pieces_preds.boxes.conf is not None:
        confidences = chess_pieces_preds.boxes.conf
        # Convert tensor to numpy array properly (handle MPS/CUDA tensors)
        try:
            confidences = confidences.cpu().numpy()
        except AttributeError:
            # Already a numpy array
            if not isinstance(confidences, np.ndarray):
                confidences = np.array(confidences)
        print(f"Detection confidences: {confidences}")
        print(f"Min confidence: {confidences.min():.3f}, Max confidence: {confidences.max():.3f}")
    
    # Create visualization with bounding boxes
    im_array = chess_pieces_preds.plot(line_width=1, font_size=6)  # plot a BGR numpy array of predictions
    
    # Alternative: Create custom visualization if the default one isn't working well
    if num_detections > 1:
        print(f"✓ Model detected {num_detections} chess pieces - visualization should show all bounding boxes")
        # Create a custom visualization with enhanced bounding boxes
        im_array = create_enhanced_detection_visualization(image_path_str, chess_pieces_preds)
    else:
        print(f"⚠ Only {num_detections} detection found - this might indicate an issue")
    
    return chess_pieces_preds, im_array


def create_enhanced_detection_visualization(image_path_str, prediction):
    """
    Create enhanced visualization with clearly visible bounding boxes and labels.
    """
    import cv2
    import numpy as np
    
    # Read the original image
    image = cv2.imread(image_path_str)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if prediction.boxes is not None and len(prediction.boxes.xyxy) > 0:
        boxes = prediction.boxes.xyxy
        confidences = prediction.boxes.conf if prediction.boxes.conf is not None else None
        classes = prediction.boxes.cls if prediction.boxes.cls is not None else None
        
        # Convert tensors to numpy arrays properly (handle MPS/CUDA tensors)
        try:
            boxes = boxes.cpu().numpy()
        except AttributeError:
            if not isinstance(boxes, np.ndarray):
                boxes = np.array(boxes)
                
        if confidences is not None:
            try:
                confidences = confidences.cpu().numpy()
            except AttributeError:
                if not isinstance(confidences, np.ndarray):
                    confidences = np.array(confidences)
                    
        if classes is not None:
            try:
                classes = classes.cpu().numpy()
            except AttributeError:
                if not isinstance(classes, np.ndarray):
                    classes = np.array(classes)
        
        # Draw each bounding box
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            
            # Choose color based on class
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                     (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
                     (255, 192, 203), (0, 128, 0), (128, 128, 0), (128, 0, 0)]
            
            class_idx = int(classes[i]) if classes is not None and i < len(classes) else 0
            color = colors[class_idx % len(colors)]
            
            # Draw thicker bounding box
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 4)
            
            # Add label with confidence
            if confidences is not None and i < len(confidences):
                conf = confidences[i]
                class_name = class_dict.get(class_idx, f"Class_{class_idx}")
                label = f"{class_name}: {conf:.2f}"
                
                # Draw label background
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(image_rgb, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                cv2.putText(image_rgb, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        print(f"Enhanced visualization created with {len(boxes)} bounding boxes")
    
    return image_rgb


def map_pieces_to_squares(chess_pieces_preds, coord_dict):
    """
    Map detected chess pieces to chess squares.
    
    Returns:
        list: List of [cell_value, piece_class_id] pairs
    """
    # list for cell number and piece id (class value)
    game_list = []

    # chess_pieces_preds is a single result object, not a list
    if chess_pieces_preds.boxes is not None and chess_pieces_preds.boxes.xyxy is not None:
        for id, box in enumerate(chess_pieces_preds.boxes.xyxy):  # box with xyxy format, (N, 4)
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            # find middle of bounding boxes for x and y
            bb_center_x_coordinate = int((x1 + x2) / 2)
            bb_center_y_coordinate = int((y1 + y2) / 2)

            for cell_value, coordinates in coord_dict.items():
                x_values = [point[0] for point in coordinates]
                y_values = [point[1] for point in coordinates]

                if (min(x_values) <= bb_center_x_coordinate <= max(x_values)) and (
                    min(y_values) <= bb_center_y_coordinate <= max(y_values)
                ):
                    if chess_pieces_preds.boxes.cls is not None:
                        pred_cls_id = int(chess_pieces_preds.boxes.cls[id])

                        print(f" cell :  {cell_value} --> {pred_cls_id} ")
                        # add cell values and piece cell_value(class value
                        game_list.append([cell_value, pred_cls_id])
                        break

    return game_list


def parse_coordinates(input_str):
    """
    Parse the input string to extract the positions of the chess pieces.
    """
    rows = input_str.strip().split("\n")
    chess_pieces = []
    for row in rows:  # Reversing rows to invert ranks
        pieces = row.strip().split()
        chess_pieces.extend(pieces)
    return chess_pieces


def create_chess_board_representation(game_list):
    """
    Convert game list to chess board representation and create visualizations.
    
    Returns:
        tuple: (chess_board, chess_board_image, chess_str)
    """
    # show game , if cell value exist in game_list , then print piece in that cell , otherwise print space
    chess_str = ""
    for i in range(1, 65):
        for slist in game_list:
            if slist[0] == i:
                print(class_dict[slist[1]], end=" ")
                chess_str += f" {class_dict[slist[1]]} "
                break
        else:
            print("space", end=" ")
            chess_str += " space "

        if i % 8 == 0:
            print("\n")
            chess_str += "\n"

    chess_pieces = parse_coordinates(chess_str)

    # Create a blank chess board
    board = chess.Board(None)

    piece_mapping = {
        'white-pawn': chess.PAWN,
        'black-pawn': chess.PAWN,
        'white-knight': chess.KNIGHT,
        'black-knight': chess.KNIGHT,
        'white-bishop': chess.BISHOP,
        'black-bishop': chess.BISHOP,
        'white-rook': chess.ROOK,
        'black-rook': chess.ROOK,
        'white-queen': chess.QUEEN,
        'black-queen': chess.QUEEN,
        'white-king': chess.KING,
        'black-king': chess.KING,
        'space': None
    }

    # Place pieces on the board
    for rank in range(8):
        for file in range(8):
            piece = chess_pieces[rank * 8 + file]
            if piece != 'space':
                color = chess.WHITE if piece.startswith('white') else chess.BLACK
                piece_type = piece_mapping[piece]
                board.set_piece_at(chess.square(file, 7-rank), chess.Piece(piece_type, color))  # Fix rank inversion

    # Create chess board visualization using matplotlib
    chess_board_image = save_chess_board_as_image(board, return_image=True)
    
    return board, chess_board_image, chess_str


def save_chess_board_as_image(board, output_path=None, return_image=False):
    """
    Create a visual representation of the chess board and save as image.
    """
    try:
        # Create a matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create board colors
        colors = ['#F0D9B5', '#B58863']  # Light and dark squares
        
        # Draw the board
        for rank in range(8):
            for file in range(8):
                color = colors[(rank + file) % 2]
                square = patches.Rectangle((file, rank), 1, 1, facecolor=color, edgecolor='black', linewidth=1)
                ax.add_patch(square)
                
                # Get piece at this square
                piece = board.piece_at(chess.square(file, 7-rank))  # Flip rank for display
                if piece:
                    # Simple text representation of pieces
                    piece_symbols = {
                        'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',  # White pieces
                        'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'   # Black pieces
                    }
                    symbol = piece_symbols.get(str(piece), str(piece))
                    ax.text(file + 0.5, rank + 0.5, symbol, ha='center', va='center', 
                           fontsize=24, color='black' if piece.color else 'white')
        
        # Set up the board display
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        ax.set_xticks(range(9))
        ax.set_yticks(range(9))
        ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', ''])
        ax.set_yticklabels(['1', '2', '3', '4', '5', '6', '7', '8', ''])
        ax.set_title('Detected Chess Position', fontsize=16, fontweight='bold')
        
        if return_image:
            # Return the figure as an image array
            fig.canvas.draw()
            # Convert figure to numpy array using a more compatible method
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            from PIL import Image
            img = Image.open(buf)
            img_array = np.array(img)
            buf.close()
            plt.close(fig)
            return img_array
        
        if output_path:
            # Save the figure
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)  # Close to free memory
            print(f"Chess board saved as {output_path}")
            return True
        
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"Error saving chess board: {e}")
        return False


def create_comprehensive_visualization(all_results, display=False):
    """
    Create comprehensive visualization with all intermediate steps.
    
    Args:
        image_path_str: Path to the input image
        all_results: Dictionary containing all intermediate results
        display: Whether to display the figure immediately
    """
    # Always create visualization when called
    print("Creating comprehensive visualization...")
    
    # Create a single figure with all subplots
    fig = plt.figure(figsize=(24, 18))

    # Row 1: Basic processing steps
    plt.subplot(4, 4, 1)
    plt.imshow(all_results['original_image'])
    plt.title("1_Original Image")
    plt.axis("off")

    plt.subplot(4, 4, 2)
    plt.imshow(all_results['canny_image'])
    plt.title("2_Canny Edge Detection")
    plt.axis("off")

    plt.subplot(4, 4, 3)
    plt.imshow(all_results['black_image'])
    plt.title("3_Hough Lines")
    plt.axis("off")

    plt.subplot(4, 4, 4)
    plt.imshow(all_results['all_contours_image'])
    plt.title("4_All Contours")
    plt.axis("off")

    # Row 2: Square detection and contour processing
    plt.subplot(4, 4, 5)
    plt.imshow(all_results['dilated_valid_squares_image'])
    plt.title("5_Dilated Valid Squares")
    plt.axis("off")

    plt.subplot(4, 4, 6)
    plt.imshow(all_results['biggest_area_image'])
    plt.title("6_Largest Contour")
    plt.axis("off")

    plt.subplot(4, 4, 7)
    plt.imshow(all_results['extreme_points_image'])
    plt.title("7_Extreme Points")
    plt.axis("off")

    plt.subplot(4, 4, 8)
    plt.imshow(all_results['warped_image'])
    plt.title("8_Warped Image")
    plt.axis("off")

    # Row 3: Grid processing and piece detection
    plt.subplot(4, 4, 9)
    plt.imshow(all_results['warped_image_with_squares'])
    plt.title("9_Warped Image with Squares")
    plt.axis("off")

    plt.subplot(4, 4, 10)
    plt.imshow(all_results['unwarped_grid_image'])
    plt.title("10_Unwarped Grid")
    plt.axis("off")

    plt.subplot(4, 4, 11)
    plt.imshow(all_results['image_with_detections'])
    plt.title("11_Piece Detections")
    plt.axis("off")

    plt.subplot(4, 4, 12)
    if all_results['chess_board_image'] is not None:
        plt.imshow(all_results['chess_board_image'])
        plt.title("12_Final Chess Board")
        plt.axis("off")

    plt.tight_layout()
    
    if display:
        plt.show()
        return None
    else:
        # In Jupyter notebooks, close the figure to prevent automatic display
        # but return it first so it can still be saved
        fig_copy = fig
        plt.close(fig)  # This prevents automatic display in Jupyter
        return fig_copy


def save_results(all_results, image_path_str, output_dir="artifacts"):
    """
    Save all results including individual images and comprehensive visualization.
    
    Args:
        all_results: Dictionary containing all intermediate results
        image_path_str: Path to input image
        output_dir: Output directory for saving results
    """
    # Create output directory
    timestamp = Path(image_path_str).stem
    result_dir = Path(output_dir) / f"chess_analysis_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual images
    image_files = {}
    # Define the order for numbering based on the analysis subplot order
    image_order = [
        'original_image', 'canny_image', 'black_image', 'all_contours_image',
        'dilated_valid_squares_image', 'biggest_area_image', 'extreme_points_image', 'warped_image',
        'warped_image_with_squares', 'unwarped_grid_image', 'image_with_detections', 'chess_board_image',
        'image_with_polygons'  # Include any remaining images not in main analysis
    ]
    
    for index, name in enumerate(image_order, 1):
        if name in all_results and all_results[name] is not None and name != 'chess_board':
            image = all_results[name]
            filename = f"{index}_{name}.png"
            filepath = result_dir / filename
            
            # Handle different image formats
            if len(image.shape) == 3:  # Color image
                if image.shape[2] == 3:  # RGB
                    plt.imsave(filepath, image)
                else:  # BGR, convert to RGB
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    plt.imsave(filepath, rgb_image)
            else:  # Grayscale
                plt.imsave(filepath, image, cmap='gray')
                
            image_files[name] = filepath
            print(f"Saved {name} to {filepath}")
    
    # Create and save comprehensive visualization
    fig = create_comprehensive_visualization(all_results, display=False)
    if fig is not None:
        analysis_path = result_dir / "analysis.png"
        fig.savefig(analysis_path, dpi=300, bbox_inches='tight')
        print(f"Saved analysis to {analysis_path}")
        # Don't close the figure immediately so it displays inline
    
    # Save SVG board if available
    if all_results.get('chess_board') is not None:
        try:
            svgboard = chess.svg.board(all_results['chess_board'])
            svg_path = result_dir / "chess_board.svg"
            with open(svg_path, "w") as f:
                f.write(svgboard)
            print(f"Saved SVG board to {svg_path}")
        except Exception as e:
            print(f"Error saving SVG board: {e}")
    
    print(f"\nAll results saved to: {result_dir}")
    return result_dir


# Main execution
def main():
    """Main function to orchestrate the chess board analysis."""

    # Step 1: Basic image processing
    image, canny_image, black_image, lines = detect_edges(str(image_path))
    largest_contour, all_contours_image, dilated_valid_squares_image = detect_squares(
        image, black_image
    )
    top_left, top_right, bottom_left, bottom_right = find_extreme_points(largest_contour)
    M, warped_image = apply_perspective_transformation(
        image, top_left, top_right, bottom_left, bottom_right
    )
    warped_image_with_squares = draw_squares(warped_image)

    # Step 2: Generate square coordinates
    # Calculate inverse perspective transformation matrix to map warped coordinates back to original image
    M_inv = cv2.invert(M)[1]  # Get the inverse of the perspective matrix

    # Define chessboard dimensions
    rows, cols = 8, 8  # Standard 8x8 chessboard
    width, height = 1200, 1200  # Target dimensions for warped image

    # Calculate dimensions of each square in the warped image
    square_width = width // cols
    square_height = height // rows

    # Initialize list to store coordinates of each square in the warped image
    # Coordinates stored in order: [center, bottom_right, top_right, top_left, bottom_left]
    squares_data_warped = []

    # Generate coordinates for each square in the warped image
    # Iterate from bottom to top (chess notation order) and left to right
    for i in range(rows - 1, -1, -1):  # Start from bottom row and move up
        for j in range(cols):  # Left to right order
            # Calculate the four corners of the current square
            top_left_sq = (j * square_width, i * square_height)
            top_right_sq = ((j + 1) * square_width, i * square_height)
            bottom_left_sq = (j * square_width, (i + 1) * square_height)
            bottom_right_sq = ((j + 1) * square_width, (i + 1) * square_height)

            # Calculate the center point of the square
            x_center = (top_left_sq[0] + bottom_right_sq[0]) // 2
            y_center = (top_left_sq[1] + bottom_right_sq[1]) // 2

            # Store all points in the required order
            squares_data_warped.append(
                [
                    (x_center, y_center),  # Center point
                    bottom_right_sq,  # Bottom right corner
                    top_right_sq,  # Top right corner
                    top_left_sq,  # Top left corner
                    bottom_left_sq,  # Bottom left corner
                ]
            )

    # Convert list to numpy array and reshape for perspective transformation
    # Shape: (num_squares, 1, 2) for each point
    squares_data_warped_np = np.array(squares_data_warped, dtype=np.float32).reshape(
        -1, 1, 2
    )

    # Transform all points from warped image back to original image coordinates
    squares_data_original_np = cv2.perspectiveTransform(squares_data_warped_np, M_inv)

    # Reshape back to list format: (num_squares, 5 points, x/y coordinates)
    squares_data_original = squares_data_original_np.reshape(-1, 5, 2)

    # Get coordinates and draw grid
    extreme_points_list, squares_data_original = get_unwarped_grid_coordinates(
        squares_data_original
    )
    unwarped_grid_image = draw_unwarped_grid(
        image, extreme_points_list, squares_data_original
    )

    # Step 3: Process square coordinates and draw polygons
    image_with_polygons, square_coordinates_data = process_square_coordinates_and_draw(
        str(image_path), squares_data_original
    )

    # Step 4: Detect chess pieces
    chess_pieces_preds, image_with_detections = detect_chess_pieces(
        str(image_path), "models/yolo_chess_piece_detector/training_v2/weights/best.pt"
    )

    # Step 5: Map pieces to squares
    game_list = map_pieces_to_squares(chess_pieces_preds, square_coordinates_data)

    # Step 6: Create chess board representation
    chess_board, chess_board_image, chess_str = create_chess_board_representation(game_list)

    # Step 7: Create additional visualization images
    # Create biggest contour image
    biggest_area_image = np.zeros_like(dilated_valid_squares_image)
    cv2.drawContours(biggest_area_image, [largest_contour], -1, (255, 255, 255), 5)

    # Create extreme points image
    extreme_points_image = np.zeros_like(dilated_valid_squares_image, dtype=np.uint8)
    cv2.drawContours(
        extreme_points_image, [largest_contour], -1, (255, 255, 255), thickness=2
    )

    # Mark the extreme points
    extreme_points = [top_left, top_right, bottom_left, bottom_right]
    for point in extreme_points:
        if point is not None:
            cv2.circle(
                extreme_points_image,
                (int(point[0]), int(point[1])),
                15,
                (255, 255, 255),
                -1,
            )

    # Step 8: Collect all results
    all_results = {
        'original_image': image,
        'canny_image': canny_image,
        'black_image': black_image,
        'all_contours_image': all_contours_image,
        'dilated_valid_squares_image': dilated_valid_squares_image,
        'biggest_area_image': biggest_area_image,
        'extreme_points_image': extreme_points_image,
        'warped_image': warped_image,
        'warped_image_with_squares': warped_image_with_squares,
        'unwarped_grid_image': unwarped_grid_image,
        'image_with_polygons': image_with_polygons,
        'image_with_detections': image_with_detections,
        'chess_board_image': chess_board_image,
        'chess_board': chess_board  # Keep the chess.Board object for SVG generation
    }

    # Step 9: Display comprehensive visualization and final chess board
    print("Creating comprehensive visualization...")
    fig = create_comprehensive_visualization(all_results, display=True)

    print("Displaying final chess board result...")
    if chess_board_image is not None:
        plt.figure(figsize=(8, 8))
        plt.imshow(chess_board_image)
        plt.title('Final Detected Chess Position', fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # Step 10: Save all results
    save_results(all_results, str(image_path))

    return all_results


if __name__ == "__main__":
    results = main()
