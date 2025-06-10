# Example: Loading models from Hugging Face Hub
import os
from pathlib import Path
from src.chess_piece_detection.model import ChessModel

single_eval_img_path = Path(os.environ["DATA_FOLDER_PATH"]) / "eval_images/chess_2.jpeg"

# Load a specific model version
model_from_hf = ChessModel.from_huggingface(f"dopaul/chess-piece-detector-v1")

# List available model files in a repository
available_files = ChessModel.list_huggingface_files(f"dopaul/chess-piece-detector-v1")

# Load a specific model file
model_custom = ChessModel.from_huggingface(f"dopaul/chess-piece-detector-v1", filename="last.pt")

# Evaluate the downloaded model
model_from_hf.evaluate(single_eval_img_path)
