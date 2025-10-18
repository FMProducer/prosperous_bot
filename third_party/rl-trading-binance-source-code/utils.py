# utils.py
import datetime as dt
import importlib.util
import logging
import os
import random
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from config import MasterConfig

logger = logging.getLogger(__name__)


GroupedSignals = Dict[dt.datetime, List[Tuple[str, np.ndarray]]]


def set_random_seed(seed: int = 25) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def load_npz_dataset(
    file_path: str,
    name_dataset: str,
    plot_dir: str,
    debug_max_size: Optional[int] = None,
    plot_examples: int = 0,
    plot_channel_idx: int = 4,
    pre_signal_len: int = 90,
) -> List[Tuple[Any, np.ndarray]]:
    logger.info(f"Loading dataset '{name_dataset}' from {file_path}")
    if debug_max_size:
        logger.debug(f"DEBUG mode: limit dataset size to {debug_max_size}")

    experiences: List[Tuple[Any, np.ndarray]] = []
    try:
        with np.load(file_path, allow_pickle=True) as data:
            if "_keys_map_" in data:
                keys_map = data["_keys_map_"].item()
                for idx, (str_key, orig_key) in enumerate(
                    tqdm(keys_map.items(), desc=f"Loading {name_dataset}", leave=False)
                ):
                    if str_key in data:
                        experiences.append((orig_key, data[str_key]))
                    else:
                        logger.warning(f"Missing array for key {str_key}")
                    if debug_max_size and idx + 1 >= debug_max_size:
                        break
            else:
                for key in (k for k in data.files if not k.startswith("_")):
                    experiences.append((key, data[key]))
        count = len(experiences)
        logger.info(f"Loaded {count} sequences from {file_path}")

        dates = []
        for orig_key, _ in experiences:
            if isinstance(orig_key, tuple) and len(orig_key) == 2:
                _, dt = orig_key
                dates.append(dt)
        if dates:
            dates_sorted = sorted(dates)
            logger.info(f"Dataset '{name_dataset}' period: from {dates_sorted[0].date()} to {dates_sorted[-1].date()}")

        if plot_examples and experiences:
            sns.set_style("whitegrid")
            os.makedirs(plot_dir, exist_ok=True)
            import random

            sampled = random.sample(experiences, min(plot_examples, count))
            for i, (orig_key, seq) in enumerate(sampled, 1):
                ticker = orig_key[0] if isinstance(orig_key, tuple) else str(orig_key)
                dt = orig_key[1] if isinstance(orig_key, tuple) else None
                prices = seq[:, plot_channel_idx]

                plt.figure(figsize=(10, 5))
                plt.plot(prices, color="green", linewidth=2, label="Price")

                plt.axvline(x=pre_signal_len - 1, color="magenta", linestyle="--", lw=1.5, label="Session Start")
                title_dt = dt.strftime("%Y-%m-%d %H:%M") if dt is not None else ""
                plt.title(f"{ticker} {title_dt}  {name_dataset}", fontsize=14)
                plt.xlabel("Time (minutes)")
                plt.ylabel("Price")
                plt.legend()
                plt.tight_layout()
                fname = f"{name_dataset}_example_{i}_{ticker}_{title_dt}.png"
                out = os.path.join(plot_dir, fname)
                plt.savefig(out, dpi=300)
                plt.close()
                logger.info(f"Saved example plot: {fname}")

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}", exc_info=True)

    return experiences


def select_and_arrange_channels(
    raw_seq: np.ndarray, file_channels: List[str], use_channels: List[str]
) -> Optional[np.ndarray]:
    if raw_seq.shape[1] != len(file_channels):
        logger.error("Channel count mismatch in select_and_arrange_channels")
        return None
    df = pd.DataFrame(raw_seq, columns=file_channels)
    missing = [ch for ch in use_channels if ch not in df.columns]
    if missing:
        logger.error(f"Missing channels: {missing}")
        return None
    return df[use_channels].to_numpy(dtype=np.float32)


def calculate_normalization_stats(
    sequences: List[np.ndarray],
    use_channels: List[str],
    price_channels: List[str],
    volume_channels: List[str],
    other_channels: List[str],
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {"means": {}, "stds": {}}
    if not sequences:
        logger.warning("Empty training set for normalization stats")
        return stats

    data_accum: Dict[str, List[float]] = {ch: [] for ch in use_channels}
    for seq in tqdm(sequences, desc="Calculating normalization stats ...", leave=False):
        for idx, ch in enumerate(use_channels):
            arr = seq[:, idx].astype(np.float64)
            if ch in price_channels:
                changes = arr[1:] / (arr[:-1] + 1e-9)
                vals = np.log(np.maximum(changes, 1e-9))
            elif ch in volume_channels:
                vals = np.log(arr + 1.0)
            elif ch in other_channels:
                vals = arr
            else:
                continue
            finite = vals[np.isfinite(vals)]
            data_accum[ch].extend(finite.tolist())

    for ch, values in data_accum.items():
        if not values:
            logger.warning(f"No data for stats on channel {ch}, defaulting to mean=0, std=1")
            stats["means"][ch], stats["stds"][ch] = 0.0, 1.0
        else:
            arr = np.array(values, dtype=np.float32)
            m, s = float(arr.mean()), float(arr.std())
            if s < 1e-7:
                logger.debug(f"Std too small for {ch}, setting to 1.0")
                s = 1.0
            stats["means"][ch], stats["stds"][ch] = m, s
    logger.info("Normalization statistics computed")
    return stats


def apply_normalization(
    window: np.ndarray,
    stats: Dict[str, Dict[str, float]],
    use_channels: List[str],
    price_channels: List[str],
    volume_channels: List[str],
    other_channels: List[str],
    agent_history_len: int,
    input_history_len: int,
) -> Optional[np.ndarray]:
    seq_len, channels = window.shape
    if seq_len != agent_history_len or channels != len(use_channels):
        logger.error("Window shape mismatch in apply_normalization")
        return None

    out = np.zeros((input_history_len, channels), dtype=np.float32)

    for i, ch in enumerate(use_channels):
        arr = window[:, i].astype(np.float64)
        mean, std = stats["means"].get(ch, 0.0), stats["stds"].get(ch, 1.0)
        if ch in price_channels:
            rel = arr[1:] / (arr[:-1] + 1e-9)
            logs = np.log(np.maximum(rel, 1e-9))
            normed = (logs - mean) / std
        elif ch in volume_channels:
            logv = np.log(arr + 1.0)
            normed = (logv - mean) / std
            normed = normed[-input_history_len:]
        elif ch in other_channels:
            normed = (arr - mean) / std
            normed = normed[-input_history_len:]
        else:
            normed = arr[-input_history_len:]
        out[:, i] = np.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)

    return out


def load_config(path: str) -> MasterConfig:
    cfg_path = Path(path)
    spec = importlib.util.spec_from_file_location("experiment_cfg", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not getattr(module.cfg.paths, "config_name", None):
        module.cfg.paths.config_name = cfg_path.stem

    return module.cfg


def setup_logging(session_name: str, cfg: MasterConfig, log_dir_override: Optional[str] = None) -> None:
    """
    Sets up logging. By default, writes to a shared log file for the entire Optuna session.
    If the `cfg.logging.per_trial_logs` flag is enabled, creates a separate log file per trial.

    Args:
        session_name (str): name of the session or experiment.
        cfg (MasterConfig): main configuration object.
        log_dir_override (str, optional): override directory path if provided.
    """
    if cfg.logging.per_trial_logs and log_dir_override:
        log_dir = os.path.join(log_dir_override, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{session_name}.log")
    else:
        log_dir = cfg.paths.log_dir
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{session_name}.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        # handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        handlers=[logging.FileHandler(log_file)],
    )
    logging.info(f"Logging to: {log_file}")


def calculate_price_change(first_price: float, second_price: float) -> float:
    return (second_price - first_price) / first_price


def compute_metrics(sequences: List[np.ndarray], predictions: np.ndarray, cfg: MasterConfig) -> Tuple[float, float]:
    """
    Computes average PnL and win rate based on predicted trade directions (0=short, 1=long).

    Args:
        sequences (List[np.ndarray]): List of sessions (raw prices).
        predictions (np.ndarray): Model predictions (0 or 1).
        cfg (MasterConfig): Configuration.

    Returns:
        Tuple[float, float]: mean_pnl, win_rate
    """
    total_pnls = []
    wins = 0
    close_idx = cfg.data.data_channels.index("close")

    for session, direction in zip(sequences, predictions):
        if session.shape[0] <= cfg.seq.pre_signal_len + cfg.seq.agent_session_len:
            continue

        start_price = session[cfg.seq.pre_signal_len - 1, close_idx]
        end_price = session[cfg.seq.pre_signal_len + cfg.seq.agent_session_len - 1, close_idx]

        if start_price <= 0 or end_price <= 0:
            continue

        position_size = cfg.market.initial_balance / start_price

        if direction == 1:  # LONG
            entry = start_price * (1 + cfg.market.slippage)
            exit = end_price * (1 - cfg.market.slippage)
            fee = (entry + exit) * position_size * cfg.market.transaction_fee
            pnl = (exit - entry) * position_size - fee

        else:  # SHORT
            entry = start_price * (1 - cfg.market.slippage)
            exit = end_price * (1 + cfg.market.slippage)
            fee = (entry + exit) * position_size * cfg.market.transaction_fee
            pnl = (entry - exit) * position_size - fee

        total_pnls.append(pnl)
        if pnl > 0:
            wins += 1

    if not total_pnls:
        return 0.0, 0.0

    return np.mean(total_pnls), wins / len(total_pnls)


def create_signal_groups(npz_dataset: List[Tuple[Tuple[str, dt.datetime], np.ndarray]]) -> GroupedSignals:
    grouped: GroupedSignals = defaultdict(list)

    for (ticker_name, signal_dt), data_array in npz_dataset:
        grouped[signal_dt].append((ticker_name, data_array))

    logger.info(f"From a dataset of {len(npz_dataset)} signals, {len(grouped.keys())} groups were created.")

    return dict(sorted(grouped.items()))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def millify(n, precision=1):
    millnames = ['', 'K', 'M', 'B', 'T', 'P']
    n = float(n)
    millidx = max(0, min(len(millnames) - 1, int(math.floor(math.log10(abs(n))) / 3))) if n != 0 else 0
    return f"{n / 10 ** (3 * millidx):.{precision}f}{millnames[millidx]}"

