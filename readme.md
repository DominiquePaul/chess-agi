# Training a SO-100 robot to play chess

## Roadmap

- [ ] Chess piece detection works reliably
- [ ] Board detection works in new setting
- [ ] Board detection + piece detection works together
- [ ] Board position can be fed into chess engine and fetch next move
- [ ] User can collect data with dynamically marked points on board.
- [ ] Collected dataset of 250 samples
- [ ] Train ACT policy on dataset
- [ ] Control-loop for playing a game
- [ ] Needs to detect whether other player made a move
- [ ] Add optional voice over comment with the voice of Borat


# Setup

```
pip install uv
uv sync
```


## Download chess piece data

### From Hugging Face (recommended)

The easiest way to get the chess piece detection datasets is to download them directly from Hugging Face:

```bash
# Download merged dataset (recommended for training)
python src/data_prep/download_from_hf.py --dataset merged

# Download all available datasets  
python src/data_prep/download_from_hf.py --dataset all

# Download specific datasets
python src/data_prep/download_from_hf.py --dataset dominique roboflow

# List available datasets
python src/data_prep/download_from_hf.py --list

# Download to custom directory
python src/data_prep/download_from_hf.py --dataset merged --output-dir ./my_data
```

Or use the Hugging Face datasets directly in your code:

```python
from datasets import load_dataset

# Load individual datasets
dominique_dataset = load_dataset("dopaul/chess-pieces-dominique")
roboflow_dataset = load_dataset("dopaul/chess-pieces-roboflow") 

# Load merged dataset (recommended for training)
merged_dataset = load_dataset("dopaul/chess-pieces-merged")

# Access different splits
train_data = merged_dataset["train"]
valid_data = merged_dataset["valid"]
test_data = merged_dataset["test"]
```

### Recreate datasets from source

If you want to recreate the datasets from scratch (for development or to upload your own version), follow these steps:

#### Prerequisites

1. **Kaggle API**: Create an API key at [www.kaggle.com/settings](https://www.kaggle.com/settings) and place the `kaggle.json` file in `~/.kaggle/kaggle.json`

2. **Roboflow CLI**: Install if not already available:
   ```bash
   uv add roboflow
   ```

3. **Environment setup**: Make sure your `.env` file contains:
   ```bash
   DATA_FOLDER_PATH=/path/to/your/data/folder
   HF_USERNAME=your-huggingface-username  # Set this if you want to upload
   ```

4. **Hugging Face authentication** (if uploading):
   ```bash
   huggingface-cli login
   ```

#### Step-by-step recreation

```bash
# Complete pipeline: download, process, and upload all datasets
python src/data_prep/prepare_and_upload_datasets.py

# Alternative options:
python src/data_prep/prepare_and_upload_datasets.py --dry-run           # See what would be done
python src/data_prep/prepare_and_upload_datasets.py --no-individual    # Only upload merged dataset
python src/data_prep/prepare_and_upload_datasets.py --skip-download     # Only upload existing datasets
```

**What the script does:**

- **Step 1**: Downloads the chess pieces dataset from Kaggle, processes it to remove the extra "bishop" class, and standardizes the labels for YOLOv8 format
- **Step 2**: Downloads the complementary dataset from Roboflow with additional chess piece images  
- **Step 3**: Uploads both datasets individually and creates a merged version on Hugging Face for easy access

The script automatically checks prerequisites and provides clear status updates. You can inspect `src/data_prep/prepare_and_upload_datasets.py` to see exactly what it does - it imports functions from the existing processing scripts to keep the code clean and readable.

## Important links:

- **Hugging Face Profile**: https://huggingface.co/dopaul
- **Chess Datasets**:
  - [Merged Dataset (Recommended)](https://huggingface.co/datasets/dopaul/chess-pieces-merged) - Combined dataset for training
  - [Dominique Dataset](https://huggingface.co/datasets/dopaul/chess-pieces-dominique) - Individual dataset from Roboflow
  - [Roboflow Dataset](https://huggingface.co/datasets/dopaul/chess-pieces-roboflow) - Processed dataset from Kaggle
- **Original Data Sources**:
  - [Kaggle Chess Pieces Dataset](https://www.kaggle.com/datasets/imtkaggleteam/chess-pieces-detection-image-dataset)
  - [Roboflow Chess Detection](https://universe.roboflow.com/gustoguardian/chess-piece-detection-bltvi/dataset/6)

### Notes

- I used Roboflow to label my data. You can download data you labelled there like this: