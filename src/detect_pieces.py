import kagglehub
import torch
import matplotlib
matplotlib.use("TkAgg")  # Set the backend to TkAgg
import glob
import os
import warnings

import cv2
import matplotlib.pyplot as plt
import ultralytics
from IPython.display import Video
from ultralytics import YOLO

# Enable inline plotting for Jupyter notebooks
try:
    from IPython.core.getipython import get_ipython
    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic('matplotlib', 'inline')
except:
    pass  # Not running in a notebook

warnings.filterwarnings("ignore")

# Check if MPS is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Download latest version
# chess_piece_datase t_path = kagglehub.dataset_download(
#     "imtkaggleteam/chess-pieces-detection-image-dataset"
# )
# print("Path to dataset files:", chess_piece_dataset_path)

chess_piece_dataset_path = "data/chess_pieces_roboflow"
my_data = "data/mychess"


ultralytics.checks()
# Ultralytics 8.3.140 🚀 Python-3.11.6 torch-2.7.0 CPU (Apple M3 Pro)
# Setup complete ✅ (12 CPUs, 36.0 GB RAM, 867.7/926.4 GB disk)


# if not os.path.exists("runs/detect/train"):
if False:
    # Initialize model with MPS device
    model = YOLO("models/yolov8s.pt")
    model.to(device)
    yaml_path = os.path.abspath(os.path.join(chess_piece_dataset_path, "data.yaml"))
    print(f"Using dataset config: {yaml_path}")
    train_path = os.path.join(chess_piece_dataset_path, "train", "images")
    valid_path = os.path.join(chess_piece_dataset_path, "valid", "images")
    results = model.train(
        data=yaml_path,
        epochs=15,
        batch=32,
        lr0=0.0001,
        lrf=0.1,
        imgsz=640,
        plots=True,
        device=device,
    )
    # Save the trained model
    # model.save("models/chess_piece_detector.pt")
    # print("Model saved as 'models/chess_piece_detector.pt'")
else:
    model = YOLO("runs/detect/train9/weights/best.pt")


# Load and display validation images
data_dir = os.path.join(chess_piece_dataset_path, "valid", "images")
img_dir = os.path.join(data_dir, "*g")
files = glob.glob(img_dir)
imgs = []
for image in files:
    img = cv2.imread(image)
    if img is not None:  # Check if image was loaded successfully
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        imgs.append(img)
        # plt.figure(figsize=(10, 10), dpi=200)
        # plt.imshow(img)
        # plt.axis("off")  # Hide axes
        # plt.show()
    else:
        print(f"Failed to load image: {image}")


chess_img_path = "data/test_images/chess_2.jpeg"

def eval_image(model, img_path):
    # Load and display the chess board image
    # Get prediction results
    results = model.predict(img_path, conf=0.1, save=True)[0]

    # Print all possible class names (targets) that the model can detect
    print("Possible detection targets:")
    print('", "'.join(results.names.values()))

    # Create figure and display original image
    plt.figure(figsize=(10, 10), dpi=200)
    plt.imshow(results.orig_img)

    # Draw bounding boxes and labels
    for box in results.boxes:
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        # Get class name and confidence
        cls = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        label = f"{results.names[cls]} {conf:.2f}"
        
        # Draw rectangle
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                            fill=False, color='red', linewidth=2)
        plt.gca().add_patch(rect)
        
        # Add label
        plt.text(x1, y1-5, label, color='red', 
                bbox=dict(facecolor='white', alpha=0.7))

    plt.axis("off")  # Hide axes
    plt.show()



model9 = YOLO("runs/detect/train9/weights/best.pt")
model9 = model9.to(device)

model95 = YOLO("runs/detect/train95/weights/best.pt")
eval_image(model9, chess_img_path)
eval_image(model95, chess_img_path)

# ### Video
# example_video_path = "/kaggle/input/chess-pieces-detection-image-dataset/Chess_pieces/Chess_video_example.mp4"

# video_output = model.predict(source=example_video_path, conf=0.6, save=True)

# # !ffmpeg -y -loglevel panic -i /kaggle/working/runs/detect/train2/Chess_video_example.avi Chess_video_example.mp4

# Video("Chess_video_example.mp4", embed=True, width=960)

# Continue training on my data
yaml_path = os.path.join(my_data, "data.yaml")
results = model9.train(
    data=yaml_path,
    epochs=10,
    batch=32,
    lr0=0.0001,
    lrf=0.1,
    imgsz=640,
    plots=True,
    device=device,
)

eval_image(model9, chess_img_path)