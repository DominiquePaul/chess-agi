import os
import json
import requests

API_ENDPOINT = "http://0.0.0.0"
DATASET_NAME = "dopaul/simple_pawn_move_v5"
BATCH_SIZE = 80
# BATCH_SIZE = 128
# N_STEPS = 100_000
N_STEPS = 10_000

r = requests.post(
  f"{API_ENDPOINT}/training/start",
  json={
    "model_type": "ACT",
    "dataset_name": DATASET_NAME,
    "model_name": f"ACT_simple_pawn_move_v5_{N_STEPS}",
    "wandb_api_key": os.environ["WANDB_API_KEY"],
    "training_params": {
      "batch_size": BATCH_SIZE,
      "steps": N_STEPS,
    }
  }
)

print(r.status_code)
print(json.dumps(r.json(), indent=2))
