# Training a SO-100 robot to play chess


## Download chess piece data

### From Huggingface (recommended)

### Recreate dataset from source

First, create an API key at www.kaggle.com/settings and place the kaggle.json file in `~/.kaggle/kaggle.json` to use the API. This is required to run `src/data_prep/roboflow_chess.py`.

```bash
python src/data_prep/roboflow_chess.py
roboflow download -f yolov8 -l data/chess_pieces_dominique gustoguardian/chess-piece-detection-bltvi/4
```

## Important links:

- My Huggingface profile: https://huggingface.co/dopaul
- HF dataset for all chess pieces.

### Notes

- I used Roboflow to label my data. You can download data you labelled there like this: