import datetime as dt
import importlib.util
import logging
import sys
import os
import json

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

from config import MasterConfig
from config import cfg as default_cfg
from test_agent import init_agent
from trading_environment import TradingEnvironment
from utils import load_npz_dataset, setup_logging


def load_session_from_npz(
    file_path: str, target_symbol: str, target_dt: dt.datetime, pre_signal_len: int, cfg: MasterConfig
) -> np.ndarray | None:
    """Загружает один конкретный сеанс из файла NPZ."""
    logging.info(f"Searching for {target_symbol} @ {target_dt} in {file_path}")
    dataset = load_npz_dataset(
        file_path,
        "NPZ",
        plot_dir=cfg.paths.plot_dir,
        pre_signal_len=pre_signal_len,
        debug_max_size=None,
        plot_examples=0,  # We don't need plots for frame check
    )
    for (symbol, signal_dt), arr in dataset:
        if symbol == target_symbol and signal_dt == target_dt:
            logging.info("Found matching session in NPZ file.")
            return arr
    logging.warning("Session not found in NPZ file.")
    return None


def load_session_from_db(
    cfg: MasterConfig, cfg_mod: object, target_symbol: str, target_dt: dt.datetime
) -> np.ndarray | None:
    """Загружает один конкретный сеанс из базы данных."""
    logging.info(f"Loading session for {target_symbol} @ {target_dt} from database.")
    dsn = cfg.db.dsn
    seq_start_dt = target_dt - dt.timedelta(minutes=cfg.seq.pre_signal_len)
    seq_end_dt = target_dt + dt.timedelta(minutes=cfg.seq.post_signal_len)

    start_ts = int(seq_start_dt.timestamp() * 1000)
    end_ts = int(seq_end_dt.timestamp() * 1000)

    try:
        conn = psycopg2.connect(dsn)
        cur = conn.cursor(cursor_factory=RealDictCursor)

        query = (
            "SELECT ts, open, high, low, close, volume, volume_weighted_average, num_trades "
            "FROM v_klines_1m_npz WHERE symbol = %s AND ts >= %s AND ts < %s ORDER BY ts ASC;"
        )
        cur.execute(query, (target_symbol, start_ts, end_ts))
        rows = cur.fetchall()

        if not rows:
            logging.warning("No data found in DB for the given time range.")
            return None

        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("ts")

        if len(df) != cfg.seq.full_seq_len:
            logging.warning(f"Data from DB has incorrect length: {len(df)}, expected {cfg.seq.full_seq_len}")
            return None

        # Убедимся, что порядок каналов соответствует ожиданиям
        seq_arr = df[cfg.data.expected_channels].to_numpy(dtype=np.float32)
        logging.info("Successfully loaded session from DB.")
        return seq_arr

    except Exception as e:
        logging.error(f"Failed to load data from DB: {e}")
        return None
    finally:
        if "conn" in locals() and conn:
            conn.close()


def get_q_values(agent, session_data, stats, cfg):
    """Инициализирует среду и получает Q-значения для первого шага."""
    env = TradingEnvironment(
        sequences=[session_data],
        stats=stats,
        render_mode=None,
        full_seq_len=cfg.seq.full_seq_len,
        num_features=cfg.seq.num_features,
        num_actions=cfg.market.num_actions,
        flat_state_size=cfg.seq.flat_state_size,
        initial_balance=cfg.market.initial_balance,
        pre_signal_len=cfg.seq.pre_signal_len,
        data_channels=cfg.data.data_channels,
        slippage=0,
        transaction_fee=0,
        agent_session_len=cfg.seq.agent_session_len,
        agent_history_len=cfg.seq.agent_history_len,
        input_history_len=cfg.seq.input_history_len,
        price_channels=cfg.data.price_channels,
        volume_channels=cfg.data.volume_channels,
        other_channels=cfg.data.other_channels,
        action_history_len=cfg.seq.action_history_len,
        inaction_penalty_ratio=0,
        backtest_mode=True,
    )
    obs, _ = env.reset(options={"forced_index": 0})
    q_vals = agent.select_action(state=obs, training=False, return_qvals=True)
    return q_vals


def run_frame_check(cfg: MasterConfig, cfg_mod: object):
    """Выполняет сравнение тензоров признаков из NPZ и БД."""
    setup_logging("frame_check_session", cfg)
    logging.info("--- Starting Frame-Check Test ---")

    # Целевой сигнал для проверки (одна из катастрофических сделок)
    TARGET_SYMBOL = "OMUSDT"
    TARGET_DATETIME = dt.datetime(2025, 4, 13, 18, 37, 0, tzinfo=dt.timezone.utc)

    # 1. Загрузка статистик и агента
    stats_path = cfg.paths.norm_stats_path
    if not os.path.exists(stats_path):
        logging.error(f"Normalization stats not found at {stats_path}")
        return
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    logging.info(f"Loaded normalization stats from {stats_path}")

    model_base = cfg.paths.extra_model_dir or cfg.paths.model_dir
    model_folder = os.path.join(model_base, sorted(os.listdir(model_base))[-1])
    model_path = os.path.join(model_folder, "best.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_folder, "final.pth")
    
    agent = init_agent(model_path, cfg, None) # No cache for this test
    logging.info(f"Loaded agent from {model_path}")


    # 2. Загрузка из NPZ
    npz_path = cfg.paths.backtest_data_path
    tensor_npz = load_session_from_npz(npz_path, TARGET_SYMBOL, TARGET_DATETIME, cfg.seq.pre_signal_len, cfg)

    # 3. Загрузка из БД
    tensor_db = load_session_from_db(cfg, cfg_mod, TARGET_SYMBOL, TARGET_DATETIME)

    # 4. Сравнение тензоров (Frame-check)
    if tensor_npz is None or tensor_db is None:
        logging.error("Could not perform comparison. One of the tensors is missing.")
        return

    logging.info("\n--- Comparison Results ---")
    logging.info(f"Tensor from NPZ shape: {tensor_npz.shape}, dtype: {tensor_npz.dtype}")
    logging.info(f"Tensor from DB  shape: {tensor_db.shape}, dtype: {tensor_db.dtype}")

    if tensor_npz.shape != tensor_db.shape:
        logging.error("SHAPES MISMATCH!")
        return

    are_close = np.allclose(tensor_npz, tensor_db, atol=1e-6)
    mae = np.mean(np.abs(tensor_npz - tensor_db))

    if are_close:
        logging.info(f"✅ SUCCESS: Tensors are almost identical (MAE: {mae:.8f}).")
    else:
        logging.error(f"❌ FAILURE: Tensors are different (MAE: {mae:.8f}).")
        diff_indices = np.where(np.abs(tensor_npz - tensor_db) > 1e-6)
        logging.error(f"First difference at index: {diff_indices[0][0]}, {diff_indices[1][0]}")
        logging.error(f"Value NPZ: {tensor_npz[diff_indices[0][0], diff_indices[1][0]]}")
        logging.error(f"Value DB:  {tensor_db[diff_indices[0][0], diff_indices[1][0]]}")
        return # Stop if frames are different

    # 5. Сравнение решений (Decision-diff)
    logging.info("\n--- Decision-diff Test ---")
    q_values_npz = get_q_values(agent, tensor_npz, stats, cfg)
    q_values_db = get_q_values(agent, tensor_db, stats, cfg)

    logging.info(f"Q-values from NPZ data: {q_values_npz}")
    logging.info(f"Q-values from DB data:  {q_values_db}")

    q_are_close = np.allclose(q_values_npz, q_values_db, atol=1e-6)
    q_mae = np.mean(np.abs(q_values_npz - q_values_db))

    if q_are_close:
        logging.info(f"✅ SUCCESS: Q-values are identical (MAE: {q_mae:.8f}).")
    else:
        logging.error(f"❌ FAILURE: Q-values are different (MAE: {q_mae:.8f}).")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/alpha.py"

    if config_path:
        spec = importlib.util.spec_from_file_location("experiment_cfg", config_path)
        cfg_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_mod)
        cfg = cfg_mod.cfg
    else:
        cfg_mod = None
        cfg = default_cfg

    run_frame_check(cfg, cfg_mod)