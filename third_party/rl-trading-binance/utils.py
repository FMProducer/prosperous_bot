# utils.py
import datetime as dt
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text
from typing import Any, Dict, List, Optional, Tuple
import logging
import gc
import importlib.util
import os
import random
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
            normed_logs = (logs - mean) / std
            # Pad with 0.0 for the first undefined value and match length
            padded_normed = np.concatenate((np.array([0.0], dtype=np.float32), normed_logs.astype(np.float32)))
            normed = padded_normed[-input_history_len:]
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


def load_config(path: str, return_module: bool = False) -> MasterConfig | Tuple[MasterConfig, Any]:
    cfg_path = Path(path)
    spec = importlib.util.spec_from_file_location("experiment_cfg", path)
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
    progress_bar: bool = True,  # Optional tqdm
) -> List[Tuple[dt.datetime, dt.datetime, dt.datetime, dt.datetime, float]]:
    """
    По минутным данным df (index=UTC, колонки содержат 'close') возвращает список окон:
    (ctx_start, ctx_end, session_start, session_end, abs_change_pct).
    * use_lookahead=True — как в бэктесте: спайк оценивается на [t, t+window].
    * use_lookahead=False — реал-режим: спайк оценивается на [t-window, t] (без заглядывания вперёд).
    """
    import pandas as pd
    print(f"Pandas version in use: {pd.__version__}")
    if df.empty or "close" not in df.columns:
        return []
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex (UTC).")

    # Precompute для vectorization (fast rolling на full series)
    s = df["close"].astype(float).copy()
    s = s.sort_index()
    n = len(s)
    if n < context_minutes + window_minutes:
        return []

    # Rolling minute rets для contrast (window=1 min, but for pre-avg)
    min_rets = s.pct_change().abs() * 100.0  # % abs change per min
    pre_avg_abs_rolling = min_rets.rolling(window=context_minutes, min_periods=context_minutes).mean()

    out: List[Tuple[dt.datetime, dt.datetime, dt.datetime, dt.datetime, float]] = []

    # Границы перебора t: это конец контекста; окно спайка зависит от lookahead
    t0 = s.index.min() + pd.Timedelta(minutes=context_minutes)
    t1 = s.index.max() - pd.Timedelta(minutes=window_minutes if use_lookahead else 0)
    
    total_steps = int((t1 - t0).total_seconds() / 60) + 1
    pbar = tqdm(range(total_steps), desc="Detecting spikes", disable=not progress_bar, leave=False) if progress_bar else range(total_steps)

    last_spike_end = t0 - pd.Timedelta(minutes=cooldown_minutes + 1)

    for step in pbar:
        t = t0 + pd.Timedelta(minutes=step)
        if t > t1:
            break

        if t < last_spike_end + pd.Timedelta(minutes=cooldown_minutes):
            continue

        if use_lookahead:
            win_start = t
            win_end = t + pd.Timedelta(minutes=window_minutes)
        else:
            win_start = t - pd.Timedelta(minutes=window_minutes)
            win_end = t
        
        win_slice = s.loc[win_start:win_end]
        if len(win_slice) < window_minutes:
            continue

        t_idx = s.index.get_indexer([t], method='nearest')[0]
        pre_avg_abs = float(pre_avg_abs_rolling.iloc[t_idx]) if not pd.isna(pre_avg_abs_rolling.iloc[t_idx]) else 0.0
        
        abs_chg = _abs_change_pct(win_slice)
        contrast = abs_chg / max(pre_avg_abs, 1e-9)

        if abs_chg >= abs_change_threshold_pct and contrast >= contrast_min:
            ctx_start = t - pd.Timedelta(minutes=context_minutes)
            ctx_end = t
            session_start = win_start if use_lookahead else win_end
            session_end = session_start + pd.Timedelta(minutes=window_minutes)
            out.append((ctx_start.to_pydatetime(), ctx_end.to_pydatetime(),
                        session_start.to_pydatetime(), session_end.to_pydatetime(), abs_chg))
            last_spike_end = win_end
        
        if progress_bar and isinstance(pbar, tqdm):
            pbar.set_postfix({"t": t.strftime("%H:%M"), "spikes": len(out)})

    return out

def preprocess_sequences(
    sequences: List[np.ndarray],
    stats: Dict[str, Dict[str, float]],
    data_channels: List[str],
    price_channels: List[str],
    volume_channels: List[str],
    other_channels: List[str]
) -> List[np.ndarray]:
    """Pre-normalize all sequences to avoid runtime overhead."""
    normalized = []
    for seq in tqdm(sequences, desc="Normalizing sequences"):
        norm_seq = apply_normalization(
            seq, stats, data_channels,
            price_channels, volume_channels, other_channels,
            agent_history_len=seq.shape[0],
            input_history_len=seq.shape[0]
        )
        normalized.append(norm_seq)
    return normalized

def get_engine(dsn: str):
    """
    Создаёт и возвращает SQLAlchemy engine для подключения к Postgres БД.
    Кэширует engine для повторных вызовов (global _engine_cache).
    """
    global _engine_cache
    if '_engine_cache' not in globals():
        _engine_cache = {}
    if dsn not in _engine_cache:
        if not dsn:
            raise ValueError("DSN not provided for database connection")
        _engine_cache[dsn] = create_engine(dsn, pool_pre_ping=True)  # pool_pre_ping для стабильности
        logger.info(f"Created new SQLAlchemy engine for DSN: {dsn.split('@')[1] if '@' in dsn else dsn}")
    return _engine_cache[dsn]

def load_sequences_from_db(
    cfg: MasterConfig,
    split: str = "train",
    debug_max_size: Optional[int] = None
) -> List[Tuple[Tuple[str, datetime], np.ndarray]]:
    """
    Загружает сырые минутные бары из Postgres БД (таблица klines_1m),
    детектирует спайки с find_spike_windows и генерирует sequences.
    Возвращает формат как load_npz_dataset: List[( (symbol, ctx_start_dt), array )],
    где array shape (full_seq_len, len(data_channels)).
    Адаптировано под схему: open_time_ms → timestamp, open_price → open, base_volume → volume.
    Фильтр: is_closed=true (завершённые свечи).
    Batching: Если symbols=None, batch по max_symbols=10 (load/process/del per batch).
    Query top symbols if all (ORDER BY total_volume DESC LIMIT).
    """
    if not cfg.db.dsn:
        raise ValueError(f"DB DSN not set in config for split '{split}'. Set cfg.db.dsn.")
    
    # Маппинг периодов по split
    periods = {
        "train": (cfg.db.train_period_start, cfg.db.train_period_end),
        "val": (cfg.db.val_period_start, cfg.db.val_period_end),
        "test": (cfg.db.test_period_start, cfg.db.test_period_end),
    }
    if split not in periods:
        raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'.")
    period_start, period_end = periods[split]
    
    engine = get_engine(cfg.db.dsn)
    
    # Get symbols list if None (top N volatile for spikes; adjust N)
    all_symbols = cfg.db.symbols
    if all_symbols is None:
        # Raw query для топ symbols по base_volume sum (volatile markets)
        top_query = text("""
        SELECT symbol FROM (
          SELECT symbol, SUM(base_volume) as total_vol 
          FROM klines_1m 
          WHERE is_closed=true AND (open_time_ms / 1000) BETWEEN :start_sec AND :end_sec
          GROUP BY symbol ORDER BY total_vol DESC LIMIT 10  -- Top 10; increase to 50 for more
        ) t
        """)
        period_start_sec = int(pd.to_datetime(period_start).timestamp())
        period_end_sec = int(pd.to_datetime(period_end).timestamp())
        symbols_df = pd.read_sql(top_query, engine, params={"start_sec": period_start_sec, "end_sec": period_end_sec})
        all_symbols = symbols_df['symbol'].tolist()
        logger.info(f"Auto-selected top {len(all_symbols)} symbols: {all_symbols}")

    # Фильтр символов, если задан
    symbols_filter = ""
    if all_symbols:
        symbols_str = "', '".join(all_symbols)
        symbols_filter = f"AND symbol IN ('{symbols_str}')"

    # --- Динамическая генерация SQL-запроса ---
    DB_COLUMN_MAP = {
        "open": "open_price", "high": "high_price", "low": "low_price", "close": "close_price",
        "volume": "base_volume", "num_trades": "trade_count", "quote_volume": "quote_asset_volume",
        "taker_buy_base_volume": "taker_buy_base_asset_volume",
        "taker_buy_quote_volume": "taker_buy_quote_asset_volume",
    }
    
    select_expressions = []
    for channel in cfg.data.data_channels:
        db_col = DB_COLUMN_MAP.get(channel)
        if db_col:
            select_expressions.append(f"{db_col} AS {channel}")
        else:
            logger.warning(f"Channel '{channel}' from config is not mapped to a DB column and will be skipped.")

    always_required_cols = {
        "timestamp_unix_sec": "open_time_ms / 1000",
        "symbol": "symbol",
    }
    final_select_cols = ", ".join(
        [f"{v} AS {k}" for k, v in always_required_cols.items()] + select_expressions
    )
    
    query = f"""
    SELECT 
        {final_select_cols}
    FROM klines_1m
    WHERE is_closed = true
      AND (open_time_ms / 1000)::bigint BETWEEN 
          EXTRACT(EPOCH FROM '{period_start}'::timestamptz)::bigint 
          AND EXTRACT(EPOCH FROM '{period_end}'::timestamptz)::bigint
      {symbols_filter}
    ORDER BY symbol, open_time_ms
    """
    
    logger.info(f"Executing DB query for {split}: {period_start} to {period_end} ({symbols_filter or 'all symbols'})")
    
    # Batching: Process symbols по batch_size=10 (del sym_df to free RAM)
    batch_size = 10  # Adjust: 5 for low RAM, 20 for 32GB+
    sequences = []
    total_symbols = len(all_symbols)
    for i in range(0, total_symbols, batch_size):
        batch_symbols = all_symbols[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}: symbols {batch_symbols}")
        
        in_clause = ','.join("'" + s + "'" for s in batch_symbols)
        batch_filter = f"AND symbol IN ({in_clause})"
        batch_query = query.replace(symbols_filter, batch_filter)  # Reuse query template
        batch_df = pd.read_sql(batch_query, engine, parse_dates=False)
        
        # Конверт timestamp_unix_sec в datetime UTC
        batch_df['timestamp'] = pd.to_datetime(batch_df['timestamp_unix_sec'], unit='s', utc=True)
        batch_df = batch_df.drop('timestamp_unix_sec', axis=1)
        batch_df = batch_df.sort_values(['symbol', 'timestamp']).set_index('timestamp')
        
        if batch_df.empty:
            continue
        
        logger.info(f"Loaded {len(batch_df)} raw bars from DB for batch")
        
        batch_sequences = []
        for symbol, sym_df in batch_df.groupby('symbol'):
            if len(sym_df) < cfg.seq.full_seq_len:
                logger.warning(f"Skipping symbol {symbol}: only {len(sym_df)} bars < full_seq_len {cfg.seq.full_seq_len}")
                continue
            
            spikes = find_spike_windows(
                sym_df,
                context_minutes=cfg.seq.pre_signal_len,
                window_minutes=cfg.seq.agent_session_len,
                abs_change_threshold_pct=cfg.detector.abs_change_pct,
                contrast_min=cfg.detector.contrast_min,
                cooldown_minutes=cfg.detector.cooldown_minutes,
                use_lookahead=True
            )
            
            for ctx_start, ctx_end, sess_start, sess_end, _ in spikes:
                full_start = ctx_start - pd.Timedelta(minutes=cfg.seq.pre_signal_len)
                full_slice = sym_df.loc[full_start:sess_end]
                
                if len(full_slice) == cfg.seq.full_seq_len:
                    slice_channels = full_slice[cfg.data.data_channels].values.astype(np.float32)
                    dt_key = (symbol, ctx_start.to_pydatetime())
                    batch_sequences.append((dt_key, slice_channels))
                    
                    if debug_max_size and len(sequences) + len(batch_sequences) >= debug_max_size:
                        break
                else:
                    logger.debug(f"Skipping spike for {symbol}: slice len {len(full_slice)} != {cfg.seq.full_seq_len}")
            
            if debug_max_size and len(sequences) + len(batch_sequences) >= debug_max_size:
                break
        
        sequences.extend(batch_sequences)
        del batch_df
        gc.collect()
        logger.info(f"Batch done: +{len(batch_sequences)} sequences (total {len(sequences)})")

        if debug_max_size and len(sequences) >= debug_max_size:
            break
    
    logger.info(f"Generated {len(sequences)} sequences from DB spikes for {split}")
    if not sequences:
        logger.warning(f"No sequences generated for {split}. Check spike params or data volume.")
    
    return sequences