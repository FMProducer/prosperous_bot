# utils.py 201125
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


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Строгая детерминированность PyTorch
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Совместимость со старыми версиями torch
            pass
    else:
        torch.backends.cudnn.benchmark = True
    logger.info(f"Random seed set to {seed}, deterministic: {deterministic}")


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


def load_npz_dataset_keys(file_path: str) -> List[Tuple[str, dt.datetime]]:
    """
    Loads only the keys (metadata) from an NPZ dataset without loading the large arrays.
    Returns a list of (symbol, datetime) tuples.
    """
    logger.info(f"Loading dataset keys from {file_path}")
    keys: List[Tuple[str, dt.datetime]] = []
    try:
        with np.load(file_path, allow_pickle=True) as data:
            if "_keys_map_" in data:
                keys_map = data["_keys_map_"].item()
                keys = list(keys_map.values())
            else:
                # Fallback for older format without a keys map
                keys = [k for k in data.files if not k.startswith("_")]
        logger.info(f"Loaded {len(keys)} keys.")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"Error loading keys from {file_path}: {e}", exc_info=True)

    # Ensure keys are sorted by datetime for deterministic order
    return sorted(keys, key=lambda x: (x[1], x[0]))


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
    price_idx: List[int],
    vol_idx: List[int],
    other_idx: List[int],
    datachannels: List[str],
) -> Dict[str, Dict[str, float]]:
    all_idx = price_idx + vol_idx + other_idx
    if not sequences:
        logger.warning("Empty training set for normalization stats")
        return {}

    # Защитный ассерт: Senior-подход к отладке продакшн-кода
    sample_seq = sequences[0]
    max_requested_idx = max(all_idx)
    if max_requested_idx >= sample_seq.shape[1]:
        raise IndexError(
            f"Config/Index mismatch: Requested index {max_requested_idx} "
            f"but data has only {sample_seq.shape[1]} channels."
        )

    stats: Dict[str, Dict[str, float]] = {"means": {}, "stds": {}}
    data_accum: Dict[str, List[float]] = {ch: [] for ch in datachannels}

    for seq in tqdm(sequences, desc="Calculating normalization stats ...", leave=False):
        for idx, ch in enumerate(datachannels):
            arr = seq[:, idx].astype(np.float64)
            if idx in price_idx:
                changes = arr[1:] / (arr[:-1] + 1e-9)
                vals = np.log(np.maximum(changes, 1e-9))
            elif idx in vol_idx:
                vals = np.log(arr + 1.0)
            elif idx in other_idx:
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
    pricechannels: List[str],
    volumechannels: List[str],
    otherchannels: List[str],
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
        if ch in pricechannels:
            rel = arr[1:] / (arr[:-1] + 1e-9)
            logs = np.log(np.maximum(rel, 1e-9))
            normed_logs = (logs - mean) / std
            # Pad with 0.0 for the first undefined value and match length
            padded_normed = np.concatenate((np.array([0.0], dtype=np.float32), normed_logs.astype(np.float32)))
            normed = padded_normed[-input_history_len:]
        elif ch in volumechannels:
            logv = np.log(arr + 1.0)
            normed = (logv - mean) / std
            normed = normed[-input_history_len:]
        elif ch in otherchannels:
            normed = (arr - mean) / std
            normed = normed[-input_history_len:]
        else:
            normed = arr[-input_history_len:]
        out[:, i] = np.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)

    return out


def apply_normalization_to_sequence(
    seq: np.ndarray,
    stats: Dict[str, Dict[str, float]],
    cfg: MasterConfig,
) -> Optional[np.ndarray]:
    """
    Нормализует *всю* последовательность (L, C) в соответствии со статистиками,
    рассчитанными `calculate_normalization_stats`.

    Для price-каналов применяется log-return между соседними точками (длина L-1),
    затем z-score; первый элемент заполняется 0.0, чтобы сохранить длину L.
    Для volume-каналов применяется log(x + 1), затем z-score.
    Для other-каналов применяется z-score к исходным значениям.
    """
    if seq is None:
        return None
    if not isinstance(seq, np.ndarray):
        seq = np.asarray(seq)
    if seq.ndim != 2:
        logger.error(f"apply_normalization_to_sequence ожидает 2D массив (L, C), получено: {seq.shape}")
        return None

    if not isinstance(stats, dict) or "means" not in stats or "stds" not in stats:
        logger.error("Некорректная структура stats: ожидается {'means': {...}, 'stds': {...}}")
        return None

    use_channels = cfg.data.datachannels
    pricechannels = cfg.data.pricechannels
    volumechannels = cfg.data.volumechannels
    otherchannels = cfg.data.otherchannels

    L, C = seq.shape
    if C != len(use_channels):
        logger.error(
            f"apply_normalization_to_sequence: mismatch каналов: seq.shape[1]={C}, "
            f"len(cfg.data.datachannels)={len(use_channels)}"
        )
        return None

    out = np.zeros_like(seq, dtype=np.float32)
    eps = 1e-9

    for i, ch in enumerate(use_channels):
        arr = seq[:, i].astype(np.float64)
        mean = float(stats["means"].get(ch, 0.0))
        std = float(stats["stds"].get(ch, 1.0))
        if not math.isfinite(std) or std < 1e-7:
            std = 1.0

        if ch in pricechannels:
            rel = arr[1:] / (arr[:-1] + eps)
            logs = np.log(np.maximum(rel, eps))
            norm = (logs - mean) / std
            out[0, i] = 0.0
            out[1:, i] = norm.astype(np.float32, copy=False)
        elif ch in volumechannels:
            vals = np.log(arr + 1.0)
            out[:, i] = ((vals - mean) / std).astype(np.float32, copy=False)
        elif ch in otherchannels:
            out[:, i] = ((arr - mean) / std).astype(np.float32, copy=False)
        else:
            out[:, i] = arr.astype(np.float32, copy=False)

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def load_config(path: str, return_module: bool = False) -> MasterConfig | Tuple[MasterConfig, Any]:
    cfg_path = Path(path)
    spec = importlib.util.spec_from_file_location("experiment_cfg", path)
    if spec is None:
        raise FileNotFoundError(f"Config file not found or invalid path: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if getattr(module.cfg.paths, "config_name", None) is None:
        module.cfg.paths.config_name = cfg_path.stem
    
    if return_module:
        return module.cfg, module
    else:
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
    else:
        log_dir = cfg.paths.log_dir
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{session_name}.log")
    abs_log_file = os.path.abspath(log_file)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(abs_log_file, encoding="utf-8"), logging.StreamHandler()],
    )
    logging.info(f"Logging to: {abs_log_file}")


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
    close_idx = cfg.data.datachannels.index("close")

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

def create_walk_forward_folds(
    merged_sequences: List[Tuple[Any, np.ndarray]],
    train_months: int,
    test_months: int,
    step_months: int
) -> List[Tuple[List[Any], List[Any]]]:
    # Sort by datetime (assuming key is (ticker, dt))
    # Filter out items that don't match the expected key format
    valid_seqs = [s for s in merged_sequences if isinstance(s[0], tuple) and len(s[0]) == 2]
    sorted_seqs = sorted(valid_seqs, key=lambda x: x[0][1])

    if not sorted_seqs:
        return []

    start_date = sorted_seqs[0][0][1]
    end_date = sorted_seqs[-1][0][1]

    folds = []
    current_start = start_date

    while True:
        train_end = current_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)

        if test_end > end_date:
            break

        train_fold = [s for s in sorted_seqs if current_start <= s[0][1] < train_end]
        test_fold = [s for s in sorted_seqs if train_end <= s[0][1] < test_end]

        if train_fold and test_fold:
            folds.append((train_fold, test_fold))

        current_start += pd.DateOffset(months=step_months)

    return folds

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def millify(n, precision=1):
    millnames = ['', 'K', 'M', 'B', 'T', 'P']
    n = float(n)
    millidx = max(0, min(len(millnames) - 1, int(math.floor(math.log10(abs(n))) / 3))) if n != 0 else 0
    return f"{n / 10 ** (3 * millidx):.{precision}f}{millnames[millidx]}"

# ------------------------------ Volatility Spike Detector ------------------------------
def _abs_change_pct(series: pd.Series) -> float:
    """Абсолютное изменение цены между первым и последним значением, %."""
    if series.empty:
        return 0.0
    first, last = float(series.iloc[0]), float(series.iloc[-1])
    if first <= 0:
        return 0.0
    return abs((last - first) / first) * 100.0

def _avg_abs_minute_ret(series: pd.Series) -> float:
    """Средний |минутный доход| в процентах (как прокси стабильности/волатильности)."""
    if series.size < 2:
        return 0.0
    rets = series.pct_change().abs().dropna()
    return float(rets.mean() * 100.0)

def find_spike_windows(
    df: pd.DataFrame,
    *,
    context_minutes: int = 90,
    window_minutes: int = 10,
    abs_change_threshold_pct: float = 5.0,
    contrast_min: float = 5.0,
    cooldown_minutes: int = 60,
    use_lookahead: bool = False,
) -> List[Tuple[dt.datetime, dt.datetime, dt.datetime, dt.datetime, float]]:
    """
    По минутным данным df (index=UTC, колонки содержат 'close') возвращает список окон:
    (ctx_start, ctx_end, session_start, session_end, abs_change_pct).
    * use_lookahead=True  — как в бэктесте: спайк оценивается на [t, t+window].
    * use_lookahead=False — реал-режим: спайк оценивается на [t-window, t] (без заглядывания вперёд).
    """
    if df.empty or "close" not in df.columns:
        return []
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex (UTC).")
    s = df["close"].astype(float).copy()
    # Безопасная монотонная итерация по времени
    s = s.sort_index()
    out: List[Tuple[dt.datetime, dt.datetime, dt.datetime, dt.datetime, float]] = []
    # Границы перебора t: это конец контекста; окно спайка зависит от lookahead
    t0 = s.index.min() + pd.Timedelta(minutes=context_minutes)
    t1 = s.index.max() - pd.Timedelta(minutes=window_minutes if use_lookahead else 0)
    t = t0
    while t <= t1:
        ctx_start = t - pd.Timedelta(minutes=context_minutes)
        ctx_end = t
        if use_lookahead:
            win_start = t
            win_end = t + pd.Timedelta(minutes=window_minutes)
        else:
            # Реал-режим: оцениваем всплеск на [t-window, t], но торговать начинаем с момента t
            win_start = t - pd.Timedelta(minutes=window_minutes)
            win_end = t
        ctx_slice = s.loc[ctx_start:ctx_end]
        win_slice = s.loc[win_start:win_end]
        # Требуем почти полную заполненность окна (минутные бары, включительно по краям)
        if len(ctx_slice) < context_minutes or len(win_slice) < window_minutes:
            t += pd.Timedelta(minutes=1)
            continue
        abs_chg = _abs_change_pct(win_slice)
        pre_avg_abs = _avg_abs_minute_ret(ctx_slice)
        contrast = abs_chg / max(pre_avg_abs, 1e-9)
        if abs_chg >= abs_change_threshold_pct and contrast >= contrast_min:
            # Начало торговой сессии:
            #  - look-ahead=True  -> стартуем с начала окна (t)
            #  - look-ahead=False -> стартуем с конца окна (t), чтобы не заглядывать в будущее
            session_start = win_start if use_lookahead else win_end
            # Предзаполним session_end длиной оценочного окна; фактическая длительность может быть переопределена конфигом
            session_end = session_start + pd.Timedelta(minutes=window_minutes)
            out.append((ctx_start.to_pydatetime(), ctx_end.to_pydatetime(),
                        session_start.to_pydatetime(), session_end.to_pydatetime(), abs_chg))
            # Кулдаун: пропускаем окна вблизи
            t += pd.Timedelta(minutes=cooldown_minutes)
        else:
            t += pd.Timedelta(minutes=1)
    return out

def preprocess_sequences(
    sequences: List[np.ndarray],
    stats: Dict[str, Dict[str, float]],
    datachannels: List[str],
    pricechannels: List[str],
    volumechannels: List[str],
    otherchannels: List[str]
) -> List[np.ndarray]:
    """Pre-normalize all sequences to avoid runtime overhead."""
    normalized = []
    for seq in tqdm(sequences, desc="Normalizing sequences"):
        norm_seq = apply_normalization(
            seq, stats, datachannels,
            pricechannels, volumechannels, otherchannels,
            agent_history_len=seq.shape[0],
            input_history_len=seq.shape[0]
        )
        normalized.append(norm_seq)
    return normalized

def create_validation_episodes(
    val_sequences: List[np.ndarray],
    val_keys: List[Tuple[str, dt.datetime]],
    num_episodes: int = 750,
    num_symbols: int = 256,
    min_episodes_per_symbol: int = 1,
    max_episodes_per_symbol: int = 10,
    seed: int = 404
) -> Tuple[List[np.ndarray], List[Tuple[str, dt.datetime]]]:
    """
    Performs stratified sampling on the validation set to ensure symbol diversity.
    """
    if not val_sequences:
        return [], []

    # 1. Group episodes by symbol
    episodes_by_symbol = defaultdict(list)
    for i, key in enumerate(val_keys):
        # key can be a string or a tuple. Handle both cases.
        if isinstance(key, tuple):
            symbol = key[0]
        else:
            symbol = key.split('_')[0]
        episodes_by_symbol[symbol].append(i)

    # 2. First pass: sample from each symbol group
    selected_indices = []
    for symbol, indices in episodes_by_symbol.items():
        # Determine number of samples for this symbol
        n_samples = min(len(indices), max_episodes_per_symbol)
        n_samples = max(n_samples, min_episodes_per_symbol)
        
        # Ensure we don't sample more than available
        n_samples = min(n_samples, len(indices))

        random.seed(seed)
        selected_indices.extend(random.sample(indices, n_samples))

    # 3. Second pass: if we have more than num_episodes, downsample
    if len(selected_indices) > num_episodes:
        random.seed(seed)
        final_indices = random.sample(selected_indices, num_episodes)
    else:
        final_indices = selected_indices

    # 4. Shuffle the final list of indices
    random.seed(seed)
    random.shuffle(final_indices)

    # 5. Create the final lists of sequences and keys
    final_sequences = [val_sequences[i] for i in final_indices]
    final_keys = [val_keys[i] for i in final_indices]

    # Logging
    final_symbols = {val_keys[i][0] if isinstance(val_keys[i], tuple) else val_keys[i].split('_')[0] for i in final_indices}
    logging.info(f"Stratified sampling complete. "
                 f"Initial episodes: {len(val_sequences)}, "
                 f"Sampled episodes: {len(final_sequences)}, "
                 f"Symbol coverage: {len(final_symbols)}/{len(episodes_by_symbol)}")

    return final_sequences, final_keys

def load_and_prep_data_from_source(sequences, keys, split_name, norm_stats):
    # This is a helper to adapt the existing load_and_prep_data logic for in-memory data
    prepped_sequences = []
    valid_keys = []
    for i, seq in enumerate(sequences):
        key = keys[i]
        try:
            asset_name = key[0] if isinstance(key, tuple) else key.split('_')[0]
        except IndexError:
            continue

        asset_stats = norm_stats.get(asset_name)
        if not asset_stats:
            continue

        means = np.array(asset_stats['means'])
        stds = np.array(asset_stats['stds'])

        seq_float = seq.astype(np.float32)
        if seq_float.shape[1] != len(means):
            continue

        seq_norm = (seq_float - means) / stds
        seq_norm = seq_norm.T
        seq_norm = np.expand_dims(seq_norm, -1)
        prepped_sequences.append(seq_norm)
        valid_keys.append(key)

    return prepped_sequences, valid_keys