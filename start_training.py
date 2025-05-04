import os
import requests

API_ENDPOINT = "http://0.0.0.0"
DATASET_NAME = "dopaul/simple_pawn_move_v3"
N_STEPS = 500
BATCH_SIZE = 64

r = requests.post(
  f"{API_ENDPOINT}/training/start",
  json={
    "model_type": "gr00t",
    "dataset_name": DATASET_NAME,
    "model_name": f"Gr00t_simple_pawn_move_v3_{N_STEPS}",
    "wandb_api_key": os.environ["WANDB_API_KEY"],
    "training_params": {
      "batch_size": BATCH_SIZE,
      "steps": N_STEPS,
      "train_test_split": 0.8,
      "epochs": 5,
      "learning_rate": 0.0001
    }
  }
)

print(r.status_code)
print(r.json())
