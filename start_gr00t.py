import os
import json
import requests

API_ENDPOINT = "http://0.0.0.0"
DATASET_NAME = "dopaul/simple_pawn_move_v4"
BATCH_SIZE = 80
# BATCH_SIZE = 128
N_EPOCHS = 50

r = requests.post(
  f"{API_ENDPOINT}/training/start",
  json={
    "model_type": "gr00t",
    "dataset_name": DATASET_NAME,
    "model_name": f"Gr00t_simple_pawn_move_v4_{N_EPOCHS}",
    "wandb_api_key": os.environ["WANDB_API_KEY"],
    "training_params": {
      "train_test_split": 1.0,
      "batch_size": BATCH_SIZE,
      "epochs": N_EPOCHS,
      "learning_rate": 0.0001
    }
  }
)

print(r.status_code)
print(json.dumps(r.json(), indent=2))
