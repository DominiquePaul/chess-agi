"""
The Roboflow has labels that has an additional class for "bishop" that is not present in the images. To align the labels with the standard YOLOv8 format, this script will remove the "bishop" class and shift the class IDs of the remaining classes down by 1.

Original:

  0: bishop
  1: black-bishop
  2: black-king
  3: black-knight
  4: black-pawn
  5: black-queen
  6: black-rook
  7: white-bishop
  8: white-king
  9: white-knight
  10: white-pawn
  11: white-queen
  12: white-rook


New:

  0: black-bishop
  1: black-king
  ...
  11: white-rook


We also update the paths in the data.yaml
"""

import os
import kagglehub
import shutil
import yaml
from pathlib import Path
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

DATA_FOLDER_PATH = Path(os.environ["DATA_FOLDER_PATH"])
DATASET_NAME = Path("chess_pieces_roboflow")


def process_label_file(file_path: Path):
    """Process a single label file to update class IDs"""
    modified = False
    new_lines = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            new_lines.append(line)
            continue

        class_id = int(parts[0])

        # Skip the generic "bishop" class
        if class_id == 0:
            modified = True
            continue

        # Shift class IDs down by 1
        new_class_id = class_id - 1
        new_line = f"{new_class_id} {' '.join(parts[1:])}\n"
        new_lines.append(new_line)
        modified = True

    if modified:
        with open(file_path, "w") as f:
            f.writelines(new_lines)
        return True
    return False


def clean_roboflow_data(data_folder_path: Path, folder_name: Path):
    data_path = data_folder_path / folder_name
    label_dirs = [
        data_path / "train" / "labels",
        data_path / "test" / "labels",
        data_path / "valid" / "labels",
    ]

    # Process all label files
    files_processed = 0
    files_modified = 0

    for label_dir in label_dirs:
        if not label_dir.exists():
            print(f"Warning: Directory {label_dir} does not exist")
            continue

        for label_file in label_dir.glob("*.txt"):
            files_processed += 1
            if process_label_file(label_file):
                files_modified += 1

    print(f"Processed {files_processed} files, modified {files_modified} files")


def download_and_move_roboflow_data(data_folder_path: Path, folder_name: Path):
    download_path = kagglehub.dataset_download(
        "imtkaggleteam/chess-pieces-detection-image-dataset"
    )
    download_path = Path(download_path)
    new_path = os.path.join(data_folder_path, folder_name)
    os.makedirs(new_path, exist_ok=True)

    # Move entire folder contents
    shutil.copytree(
        download_path / "Chess Pieces.yolov8-obb", new_path, dirs_exist_ok=True
    )


def update_data_yaml(data_folder_path: Path, folder_name: Path):
    """Update the data.yaml file to remove the bishop class and update paths"""
    data_path = data_folder_path / folder_name / "data.yaml"

    # Original class names with the extra bishop class
    original_names = {
        0: "bishop",
        1: "black-bishop",
        2: "black-king",
        3: "black-knight",
        4: "black-pawn",
        5: "black-queen",
        6: "black-rook",
        7: "white-bishop",
        8: "white-king",
        9: "white-knight",
        10: "white-pawn",
        11: "white-queen",
        12: "white-rook",
    }

    # Create updated class names (removing bishop and shifting down)
    updated_names = {}
    for i in range(1, len(original_names)):
        updated_names[i - 1] = original_names[i]

    # Create updated data.yaml content with standard relative paths
    data_yaml = {
        "train": "train/images",
        "val": "valid/images", 
        "test": "test/images",
        "names": updated_names,
        "nc": len(updated_names)
    }

    # Write to the original data.yaml file in the dataset directory
    with open(data_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    print(f"✅ Updated {data_path}")


if __name__ == "__main__":
    download_and_move_roboflow_data(DATA_FOLDER_PATH, DATASET_NAME)
    clean_roboflow_data(DATA_FOLDER_PATH, DATASET_NAME)
    update_data_yaml(DATA_FOLDER_PATH, DATASET_NAME)
