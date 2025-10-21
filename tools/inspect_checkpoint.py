
import torch
import sys

# Добавляем путь к rl-trading-binance, чтобы можно было импортировать config
sys.path.append(r'C:\Python\Prosperous_Bot\third_party\rl-trading-binance')

from config import MasterConfig

def inspect_checkpoint(ckpt_path):
    """Loads a checkpoint and prints its keys."""
    try:
        device = torch.device('cpu')
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        if not isinstance(checkpoint, dict):
            print(f"Checkpoint is not a dictionary, but a {type(checkpoint)}")
            return

        print("Keys found in checkpoint:")
        for key in checkpoint.keys():
            print(f"- {key}")

    except Exception as e:
        print(f"Failed to load or inspect checkpoint: {e}")

if __name__ == "__main__":
    # Путь к чекпоинту из configs/alpha.py
    path = r"C:\Python\Prosperous_Bot\third_party\FMProducer\fmproducer_1_eval\saved_models\session_1\best.pth"
    inspect_checkpoint(path)

