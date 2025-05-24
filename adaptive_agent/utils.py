
import os
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

def ensure_directory(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    logger.info("Directory %s ensured", directory)

def save_to_csv(dataframe, path):
    ensure_directory(Path(path).parent)
    dataframe.to_csv(path, index=False)
    logger.info("Saved CSV to %s", path)

def load_symbols(config_path="config_signal.json") -> List[str]:
    with open(config_path, "r") as f:
        config = json.load(f)
    return config.get("symbols", [])

def save_bot_state(state: Dict, path="bot_state.json"):
    with open(path, "w") as f:
        json.dump(state, f, indent=2)

def load_bot_state(path="bot_state.json") -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)

def check_signal_expired(signal_time_str: str, timeout_minutes=30) -> bool:
    signal_time = datetime.fromisoformat(signal_time_str)
    return datetime.utcnow() - signal_time > timedelta(minutes=timeout_minutes)

def save_last_signals_to_config(config_path: str, last_signals: dict):
    with open(config_path, "r") as f:
        cfg = json.load(f)
    cfg["last_signals"] = last_signals
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
