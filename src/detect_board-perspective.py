## Extracting Chess Squares with Perspective Transformation ( image --> fen format)

import cv2
import numpy as np
import matplotlib.pyplot as plt 
from ultralytics import YOLO
import math
import os
# Optional imports - uncomment if available in your environment
# from svglib.svglib import svg2rlg
# from reportlab.graphics import renderPM
# import chess
# import chess.svg

show_images = True


# Path of Image that you want to convert
image_path = r"data/test_images/chess_3.jpeg"

# Check if the image exists
if not os.path.exists(image_path):
    print(f"Warning: Image file {image_path} does not exist.")
    print("Please add a test image to the data/test_images directory or update the image_path variable.")
    # Use a placeholder path for demonstration
    print("Using a placeholder for now to avoid errors...")
    # You can either exit or continue with just the code structure
    # exit(1)


def detect_edges(path: str):
    image = cv2.imread(path)
    gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # rgb_image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    ## Processing Image  -->  OTSU Threshold , Canny edge detection , dilate , HoughLinesP 

    # OTSU threshold
    ret, otsu_binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Canny edge detection
    canny_image = cv2.Canny(otsu_binary, 20, 255)

    # Dilation
    kernel = np.ones((2,2), np.uint8)
    dilation_image = cv2.dilate(canny_image, kernel, iterations=1)

    if show_images:
        plt.imshow(dilation_image)


    # Hough Lines
    lines = cv2.HoughLinesP(dilation_image, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=400)
    print(len(lines))
    # Create an image that contains only black pixels
    black_image = np.zeros_like(dilation_image)

    # Draw only lines that are output of HoughLinesP function to the "black_image"
    if lines is not None:
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

def detect_squares(image, black_image):

    # find contours
    board_contours, hierarchy = cv2.findContours(black_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # blank image for displaying all contours
    all_contours_image= np.zeros_like(black_image)

    # Copy blank image for displaying all squares 
    squares_image = np.copy(image) 

    # blank image for displaying valid contours (squares)
    valid_squares_image = np.zeros_like(black_image)
    
    size_img = image.shape[0] * image.shape[1]

    # loop through contours and filter them by deciding if they are potential squares
    for contour in board_contours:
        if 0.001*size_img < cv2.contourArea(contour) < size_img / 32:

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
                if index_sorted[0][1]< index_sorted[1][1]:
                    cur=index_sorted[0]
                    index_sorted[0] =  index_sorted[1]
                    index_sorted[1] = cur

                if index_sorted[2][1]> index_sorted[3][1]:
                    cur=index_sorted[2]
                    index_sorted[2] =  index_sorted[3]
                    index_sorted[3] = cur

                # bottomright(1) , topright(2) , topleft(3) , bottomleft(4)
                pt1=index_sorted[0]
                pt2=index_sorted[1]
                pt3=index_sorted[2]
                pt4=index_sorted[3]

                # find rectangle that fits 4 point 
                x, y, w, h = cv2.boundingRect(contour)
                # find center of rectangle 
                center_x=(x+(x+w))/2
                center_y=(y+(y+h))/2

                

                # calculate length of 4 side of rectangle
                l1 = math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                l2 = math.sqrt((pt2[0] - pt3[0])**2 + (pt2[1] - pt3[1])**2)
                l3 = math.sqrt((pt3[0] - pt4[0])**2 + (pt3[1] - pt4[1])**2)
                l4 = math.sqrt((pt1[0] - pt4[0])**2 + (pt1[1] - pt4[1])**2)
    
    
                # Create a list of lengths
                lengths = [l1, l2, l3, l4]
                
                # Get the maximum and minimum lengths
                max_length = max(lengths)
                min_length = min(lengths)

                # Check if this length values are suitable for a square , this threshold value plays crucial role for squares ,  
                if (max_length - min_length) <= 35: # 20 for smaller boards  , 50 for bigger , 35 works most of the time 
                    valid_square=True
                else:
                    valid_square=False
    
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
    contours, _ = cv2.findContours(dilated_valid_squares_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
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


def apply_perspective_transformation(image, top_left, top_right, bottom_left, bottom_right):
    # read image and convert it to different color spaces 
    rgb_image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    # Define the four source points (replace with actual coordinates)
    extreme_points_list = np.float32([top_left, top_right, bottom_left, bottom_right])

    threshold = 0  # Extra space on all sides

    width, height = 1200 , 1200 

    # Define the destination points (shifted by 'threshold' on all sides)
    dst_pts = np.float32([
        [threshold, threshold], 
        [width + threshold, threshold], 
        [threshold, height + threshold], 
        [width + threshold, height + threshold]
    ])

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(extreme_points_list, dst_pts)

    # Apply the transformation with extra width and height
    warped_image = cv2.warpPerspective(rgb_image, M, (width + 2 * threshold, height + 2 * threshold))

    return M, warped_image


def draw_squares(warped_image):
    #### Divide board to 64 square
    # Assuming area_warped is already defined
    # Define number of squares (8x8 for chessboard)
    warped_image_copy = warped_image.copy()
    rows, cols = 8, 8

    # Calculate the width and height of each square
    width, height = 1200 , 1200 
    square_width = width // cols
    square_height = height // rows

    # Draw the squares on the warped image
    for i in range(rows):
        for j in range(cols):
            # Calculate top-left and bottom-right corners of each square
            top_left = (j * square_width, i * square_height)
            bottom_right = ((j + 1) * square_width, (i + 1) * square_height)
            
            # Draw a rectangle for each square
            cv2.rectangle(warped_image_copy, top_left, bottom_right, (0, 255, 0), 4)  # Green color, thickness 2
    return warped_image_copy

def get_unwarped_grid_coordinates(squares_data_original, rows=8, cols=8):
    """
    Returns the coordinates for drawing the grid on the unwarped image.
    """
    # Get the actual board corners from the correct squares
    bottom_left_square = squares_data_original[0]  # First square (bottom left)
    bottom_right_square = squares_data_original[7]  # Last square of first row
    top_left_square = squares_data_original[56]    # First square of last row
    top_right_square = squares_data_original[63]   # Last square (top right)

    # Store the four extreme points of the board
    extreme_points_list = np.float32([
        top_left_square[3],      # Top left of top-left square
        top_right_square[2],     # Top right of top-right square
        bottom_left_square[4],   # Bottom left of bottom-left square
        bottom_right_square[1]   # Bottom right of bottom-right square
    ])
    
    return extreme_points_list, squares_data_original

def draw_unwarped_grid(image, extreme_points_list, squares_data_original, rows=8, cols=8):
    """
    Draws the grid and corner markers on the unwarped image.
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw the grid lines on the original image
    for idx, square in enumerate(squares_data_original):
        # Calculate row and column from index
        i = idx // cols  # Row number
        j = idx % cols   # Column number
        
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
            cv2.line(rgb_image, bottom_left, bottom_right, (0, 255, 0), 8)  # Bottom line

    # Draw white circles at the extreme points to mark the corners
    for point in extreme_points_list:
        x, y = int(point[0]), int(point[1])
        cv2.circle(rgb_image, (x, y), 25, (255, 255, 255), -1)
    
    return rgb_image

image, canny_image, black_image, lines = detect_edges(image_path)
largest_contour, all_contours_image, dilated_valid_squares_image = detect_squares(image, black_image)
top_left, top_right, bottom_left, bottom_right = find_extreme_points(largest_contour)
M, warped_image = apply_perspective_transformation(image, top_left, top_right, bottom_left, bottom_right)
warped_image_with_squares = draw_squares(warped_image)

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
        top_left = (j * square_width, i * square_height)
        top_right = ((j + 1) * square_width, i * square_height)
        bottom_left = (j * square_width, (i + 1) * square_height)
        bottom_right = ((j + 1) * square_width, (i + 1) * square_height)

        # Calculate the center point of the square
        x_center = (top_left[0] + bottom_right[0]) // 2
        y_center = (top_left[1] + bottom_right[1]) // 2

        # Store all points in the required order
        squares_data_warped.append([
            (x_center, y_center),  # Center point
            bottom_right,          # Bottom right corner
            top_right,            # Top right corner
            top_left,             # Top left corner
            bottom_left           # Bottom left corner
        ])

# Convert list to numpy array and reshape for perspective transformation
# Shape: (num_squares, 1, 2) for each point
squares_data_warped_np = np.array(squares_data_warped, dtype=np.float32).reshape(-1, 1, 2)

# Transform all points from warped image back to original image coordinates
squares_data_original_np = cv2.perspectiveTransform(squares_data_warped_np, M_inv)

# Reshape back to list format: (num_squares, 5 points, x/y coordinates)
squares_data_original = squares_data_original_np.reshape(-1, 5, 2)

# Get coordinates and draw grid
extreme_points_list, squares_data_original = get_unwarped_grid_coordinates(squares_data_original)
unwarped_grid_image = draw_unwarped_grid(image, extreme_points_list, squares_data_original)

if show_images:
    # Create a single figure with all subplots
    plt.figure(figsize=(20, 15))
    
    # First row of subplots
    plt.subplot(431)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(432)
    plt.imshow(canny_image)
    plt.title('Canny Edge Detection')
    plt.axis('off')
    
    plt.subplot(433)
    plt.imshow(black_image)
    plt.title('Hough Lines')
    plt.axis('off')
    
    # Second row of subplots
    plt.subplot(434)
    plt.imshow(all_contours_image)
    plt.title('All Contours')
    plt.axis('off')
    
    plt.subplot(435)
    plt.imshow(dilated_valid_squares_image)
    plt.title('Dilated Valid Squares')
    plt.axis('off')
    
    # Create and add biggest contour to the image
    biggest_area_image = np.zeros_like(dilated_valid_squares_image)
    cv2.drawContours(biggest_area_image, [largest_contour], -1, (255,255,255), 5)
    plt.subplot(436)
    plt.imshow(biggest_area_image)
    plt.title('Largest Contour')
    plt.axis('off')
    
    # Create and add extreme points to the image
    extreme_points_image = np.zeros_like(dilated_valid_squares_image, dtype=np.uint8)
    cv2.drawContours(extreme_points_image, [largest_contour], -1, (255, 255, 255), thickness=2)
    
    # Mark the extreme points
    if top_left is not None:
        cv2.circle(extreme_points_image, (int(top_left[0]), int(top_left[1])), 15, (255, 255, 255), -1)
    if top_right is not None:
        cv2.circle(extreme_points_image, (int(top_right[0]), int(top_right[1])), 15, (255, 255, 255), -1)
    if bottom_left is not None:
        cv2.circle(extreme_points_image, (int(bottom_left[0]), int(bottom_left[1])), 15, (255, 255, 255), -1)
    if bottom_right is not None:
        cv2.circle(extreme_points_image, (int(bottom_right[0]), int(bottom_right[1])), 15, (255, 255, 255), -1)
    
    plt.subplot(437)
    plt.imshow(extreme_points_image)
    plt.title('Extreme Points')
    plt.axis('off')
    
    # Add warped image as 8th subplot
    plt.subplot(438)
    plt.imshow(warped_image)
    plt.title('Warped Image')
    plt.axis('off')
    
    # Add warped image with squares as 9th subplot
    plt.subplot(439)
    plt.imshow(warped_image_with_squares)
    plt.title('Warped Image with Squares')
    plt.axis('off')
    
    # Add unwarped grid image as 10th subplot
    plt.subplot(4,3,10)
    plt.imshow(unwarped_grid_image)
    plt.title('Unwarped Grid')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()



def get_square_coordinates(squares_data_original):
    """
    Convert square coordinates to a dictionary format.
    
    Args:
        squares_data_original: List of square coordinates in format [center, bottom_right, top_right, top_left, bottom_left]
        
    Returns:
        dict: Dictionary with cell numbers as keys and coordinate lists as values
    """
    # Create temporary list to store coordinates
    coordinates_list = []
    
    # Process each square's coordinates
    for coordinate in squares_data_original:
        center, bottom_right, top_right, top_left, bottom_left = coordinate
        coordinates_list.append([
            bottom_right[0], bottom_right[1],  # x1, y1
            top_right[0], top_right[1],        # x2, y2
            top_left[0], top_left[1],          # x3, y3
            bottom_left[0], bottom_left[1]     # x4, y4
        ])
    
    # Convert to dictionary format
    coord_dict = {}
    for cell, row in enumerate(coordinates_list, 1):
        coord_dict[cell] = [
            [row[0], row[1]],  # bottom right
            [row[2], row[3]],  # top right
            [row[4], row[5]],  # top left
            [row[6], row[7]]   # bottom left
        ]
    
    return coord_dict


square_coordinates_data = get_square_coordinates(squares_data_original)

image = cv2.imread(image_path) 

# Loop through the square_coordinates_data dictionary and draw polygons
for cell_id, coords in square_coordinates_data.items():
    # Extract coordinates from the dictionary
    # Format: [bottom_right, top_right, top_left, bottom_left]
    pts = np.array(coords, np.int32)
    pts = pts.reshape((-1,1,2))
    
    # Draw circle at the center of each square
    square_idx = cell_id - 1  # Convert to 0-indexed for squares_data_original
    center_point = (int(squares_data_original[square_idx][0][0]), int(squares_data_original[square_idx][0][1]))
    cv2.circle(image, center_point, 3, (0,255,0), 3)
    
    # Draw polygon around each square
    cv2.polylines(image, [pts], True, (255,255,255), thickness=8)  # White polygon with thickness 8

# Use square_coordinates_data directly instead of reading from CSV
# The dictionary structure is already in the format we need
coord_dict = square_coordinates_data.copy()



# class values , these values are decided before training
names: ['black-bishop', 'black-king', 'black-knight', 'black-pawn', 'black-queen', 'black-rook', 'white-bishop', 'white-king', 'white-knight', 'white-pawn', 'white-queen', 'white-rook'] # type: ignore
class_dict={0:'black-bishop',1:'black-king',2:'black-knight',3:'black-pawn',4: 'black-queen',5: 'black-rook',
            6:'white-bishop',7:'white-king',8: 'white-knight',9: 'white-pawn',10: 'white-queen',11:'white-rook'}

print("\n\n") 

# YOLOv8  model  
model = YOLO("chess-model-yolov8m.pt") 

# make prediction
results = model(image_path) # path to test image
im_array = results[0].plot(); # plot a BGR numpy array of predictions

print("\n\n") 

# list for cell number and piece id (class value)
game_list=[]

for result in results:  # results is model's prediction     
    for id,box in enumerate(result.boxes.xyxy) : # box with xyxy format, (N, 4)
            
            x1,y1,x2,y2=int(box[0]),int(box[1]),int(box[2]),int(box[3]) # take coordinates 

            # find middle of bounding boxes for x and y 
            x_mid=int((x1+x2)/2) 
            # add padding to y values
            y_mid=int((y1+y2)/2)+25

            for cell_value, coordinates in coord_dict.items():
                x_values = [point[0] for point in coordinates]
                y_values = [point[1] for point in coordinates]
                 
                if (min(x_values) <= x_mid <= max(x_values)) and (min(y_values) <= y_mid <= max(y_values)):
                    a=int(result.boxes.cls[id])

                    print(f" cell :  {cell_value} --> {a} ")
                    # add cell values and piece cell_value(class value
                    game_list.append([cell_value,a]) 
                    break

print("\n\n\n")        

# show game , if cell value exist in game_list , then print piece in that cell , otherwise print space 
chess_str=""
for i in range(1, 65):
    
    for slist in game_list:
        if slist[0] == i:
            print(class_dict[slist[1]], end=" ")
            chess_str+=f" {class_dict[slist[1]]} "
            break
    else:
        print("space", end=" ")
        chess_str+=" space "

    if i % 8 == 0:
        print("\n")
        chess_str+="\n"
 

def parse_coordinates(input_str):
    """
    Parse the input string to extract the positions of the chess pieces.
    """
    rows = input_str.strip().split('\n')
    chess_pieces = []
    for row in rows:  # Reversing rows to invert ranks
        pieces = row.strip().split()
        chess_pieces.extend(pieces)
    return chess_pieces

 
input_str=chess_str

chess_pieces = parse_coordinates(input_str)

# The following chess-related code requires python-chess module
# If you have it installed, you can uncomment this section
'''
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

# Create SVG representation of the board
svgboard = chess.svg.board(board)
with open("extracted-data/2Dboard.svg", "w") as f:
    f.write(svgboard)
'''

# Comment out the SVG to PNG conversion since it relies on the svglib module
'''
# Function to convert SVG to PNG
def convert_svg_to_png(svg_file_path, png_file_path):
    # Read the SVG file and convert it to a ReportLab Drawing
    drawing = svg2rlg(svg_file_path)
    # Render the drawing to a PNG file
    renderPM.drawToFile(drawing, png_file_path, fmt='jpeg')
    print(f"Converted {svg_file_path} to {png_file_path}")

# Example usage
svg_file = 'extracted-data/2Dboard.svg'
png_file = 'extracted-data/Extracted-Board.jpeg'
convert_svg_to_png(svg_file, png_file)
'''

original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(14, 10))  # Increase the figure size

plt.subplot(121)
plt.title(f"{image_path}")
plt.imshow(original_image)

plt.subplot(122)
plt.title("Extracted Squares")
plt.imshow(image)

# Save the figure as a PNG file
output_path = 'output_figure.png'
plt.savefig(output_path)

plt.show()  