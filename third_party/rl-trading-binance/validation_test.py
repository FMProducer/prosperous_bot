import logging
import os
import sys
import json
import argparse
from pathlib import Path
import datetime as dt
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Необходимо добавить путь к rl-trading-binance в sys.path, чтобы работали импорты
# Это делается относительно расположения самого скрипта validation_test.py
script_dir = Path(__file__).parent.resolve()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from agent import D3QN_PER_Agent
from config import MasterConfig
from trading_environment import TradingEnvironment
from utils import load_config, set_random_seed

# Глобальная настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_prep_data(npz_path: str, split_name: str, norm_stats: dict, expected_channels: int) -> list:
    """
    Загружает NPZ, применяет z-нормализацию и изменяет форму данных для модели.
    """
    if not npz_path or not os.path.exists(npz_path):
        logging.error("%s data file not found or path not specified: %s", split_name, npz_path)
        sys.exit(1)

    if not norm_stats:
        logging.error("Normalization stats are required for %s but not provided.", split_name)
        sys.exit(1)
    
    d = np.load(npz_path, allow_pickle=True)
    data_keys = [k for k in d.files if not k.startswith('_')]
    sequences = []
    logging.info("Loading %d sequences from %s...", len(data_keys), split_name)
    
    means = np.array(norm_stats['mean'])
    stds = np.array(norm_stats['std'])
    
    if len(means) != expected_channels:
        logging.warning(f"Config/stats mismatch: norm_stats has {len(means)} channels, but config expects {expected_channels}. Using {len(means)} from stats.")
    
    for key in tqdm(data_keys, desc=f"Normalizing {split_name}"):
        seq = d[key].astype(np.float32)
        if seq.shape[1] != len(means):
            # Пропускаем последовательности, которые не соответствуют статистике нормализации
            logging.warning(f"Skipping sequence {key} with shape {seq.shape} as it doesn't match norm_stats channels ({len(means)}).")
            continue
        
        seq = (seq - means) / stds
        # Reshape для CNN: (L, C) -> (C, L, 1)
        seq = seq.T
        seq = np.expand_dims(seq, -1)
        sequences.append(seq)
    
    d.close()
    if sequences:
        logging.info("Prepared %d sequences, shape: %s", len(sequences), sequences[0].shape)
    return sequences


def run_validation(entry_cfg: MasterConfig):
    """
    Основная функция для запуска валидации модели.
    Использует config_train.json как источник истины для параметров.
    """
    # 1. Определение путей из входного конфига
    model_path_str = entry_cfg.paths.model_path
    
    if not model_path_str or not os.path.exists(model_path_str):
        logging.error(f"Model file not found at path specified in config: {model_path_str}")
        sys.exit(1)
    
    model_path = Path(model_path_str)
    output_dir = model_path.parent
    
    # 2. Загрузка ИСТИННОЙ конфигурации из `config_train.json`
    config_train_path = output_dir / "config_train.json"
    if not config_train_path.exists():
        logging.error(f"CRITICAL: `config_train.json` not found in model directory: {config_train_path}")
        sys.exit(1)
        
    logging.info(f"Loading ground truth config from: {config_train_path}")
    with open(config_train_path, 'r') as f:
        true_config_data = json.load(f)
    
    # Создаем объект Pydantic из истинного конфига
    cfg = MasterConfig.model_validate(true_config_data)
    logging.info(f"Successfully loaded and validated config from training.")
    
    # 3. Обновляем пути в истинном конфиге актуальными значениями
    cfg.paths.model_path = entry_cfg.paths.model_path
    cfg.paths.norm_stats_path = entry_cfg.paths.norm_stats_path
    cfg.paths.val_data_path = entry_cfg.paths.val_data_path
    
    set_random_seed(cfg.random_seed)

    # 4. Загрузка данных и статистики по ИСТИННОМУ конфигу
    stats_path = Path(cfg.paths.norm_stats_path)
    if not stats_path.exists():
        logging.error(f"Norm stats file not found at path: {stats_path}")
        sys.exit(1)
        
    with open(stats_path, 'r') as f:
        norm_stats = json.load(f)
    
    # Определяем количество каналов из конфига, с которым велась тренировка
    expected_channels = len(cfg.data.data_channels)
    logging.info(f"Expecting {expected_channels} data channels based on training config.")

    val_seqs = load_and_prep_data(cfg.paths.val_data_path, "Validation", norm_stats, expected_channels)
    if not val_seqs:
        logging.error("Validation data could not be loaded or is empty. Exiting.")
        sys.exit(1)

    # 5. Инициализация окружения и агента по ИСТИННОМУ конфигу
    num_features = val_seqs[0].shape[0]
    if num_features != expected_channels:
        logging.error(f"FATAL: Mismatch between data channels in prepped data ({num_features}) and training config ({expected_channels}).")
        sys.exit(1)

    input_history_len = cfg.seq.agent_history_len
    num_actions = cfg.market.num_actions
    action_history_len = cfg.seq.action_history_len
    
    flat_features = input_history_len * num_features 
    extras = 4
    history_vector_size = num_actions * action_history_len if action_history_len > 0 else 0
    flat_state_size = flat_features + extras + history_vector_size

    env_kwargs = {
        "sequences": val_seqs,
        "stats": norm_stats,
        "render_mode": None,
        "full_seq_len": cfg.seq.full_seq_len,
        "num_features": num_features,
        "num_actions": num_actions,
        "flat_state_size": flat_state_size,
        "initial_balance": cfg.market.initial_balance,
        "pre_signal_len": cfg.seq.pre_signal_len,
        "data_channels": cfg.data.data_channels,
        "slippage": cfg.market.slippage,
        "transaction_fee": cfg.market.transaction_fee,
        "agent_session_len": cfg.seq.agent_session_len,
        "agent_history_len": cfg.seq.agent_history_len,
        "input_history_len": input_history_len,
        "price_channels": cfg.data.price_channels,
        "volume_channels": cfg.data.volume_channels,
        "other_channels": cfg.data.other_channels,
        "action_history_len": action_history_len,
        "inaction_penalty_ratio": cfg.market.inaction_penalty_ratio,
        "backtest_mode": True,
        "use_risk_management": getattr(cfg.backtest, "use_risk_management", True),
        "bankruptcy_threshold": cfg.market.bankruptcy_threshold,
        "bankruptcy_penalty": cfg.market.bankruptcy_penalty,
        "max_drawdown_threshold": cfg.market.max_drawdown_threshold,
        "max_drawdown_penalty": cfg.market.max_drawdown_penalty,
        "max_drawdown_penalty_type": cfg.market.max_drawdown_penalty_type,
    }
    env = TradingEnvironment(**env_kwargs)
    if hasattr(cfg.backtest, "exec_delay_bars"):
        setattr(env, "exec_delay_bars", int(cfg.backtest.exec_delay_bars))

    agent = D3QN_PER_Agent(
        state_shape=cfg.state_shape,
        action_dim=cfg.market.num_actions,
        cnn_maps=cfg.model.cnn_maps,
        cnn_kernels=cfg.model.cnn_kernels,
        cnn_strides=cfg.model.cnn_strides,
        cnn_dilations=cfg.model.cnn_dilations,
        dense_val=cfg.model.dense_val,
        dense_adv=cfg.model.dense_adv,
        additional_feats=cfg.model.additional_feats,
        dropout_model=cfg.model.dropout_p,
        device=cfg.device.device,
        learning_rate=0, gamma=0, batch_size=1, buffer_size=1, target_update_freq=1,
        train_start=1, per_alpha=0, per_beta_start=0, per_beta_frames=1,
        eps_start=0, eps_end=0, eps_frames=1, epsilon=0, max_gradient_norm=0,
        perf_cfg=cfg.perf,
    )

    logging.info(f"Loading model weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location=cfg.device.device)
    
    if "policy_state" in checkpoint:
        state_dict = checkpoint["policy_state"]
        logging.info("Found 'policy_state' key in checkpoint.")
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        logging.info("Found 'state_dict' key in checkpoint, falling back.")
    else:
        state_dict = checkpoint
        logging.info("No 'policy_state' or 'state_dict' key found, assuming raw state_dict.")
        
    agent.policy_net.load_state_dict(state_dict)
    agent.policy_net.eval()
    
    if hasattr(agent, "mc_enable"):
        agent.mc_enable = False
        logging.info("MC-Dropout has been explicitly disabled for validation.")

    logging.info("Agent loaded in evaluation mode.")

    # 6. Запуск цикла валидации
    trades_log = []
    equity_log = []
    
    total_trades = 0
    total_correct = 0
    trade_pnls: list[float] = []
    exit_counts: Dict[str, int] = {}
    tsl_hits = 0
    
    num_episodes = len(val_seqs)
    stub_dt = dt.datetime(2020, 1, 1, 0, 0)

    for i in tqdm(range(num_episodes), desc="Running Validation Episodes"):
        obs, info = env.reset(seed=None, options={"forced_index": i})
        
        episode_start_dt = stub_dt + dt.timedelta(minutes=i * env.agent_session_len)
        equity_log.append({"timestamp": episode_start_dt, "equity": info['portfolio_value']})
        
        done = False
        while not done:
            action = agent.select_action(obs, training=False)
            obs, reward, done, _, info = env.backtest_step(
                action=action,
                signal_dt=episode_start_dt,
                ticker="VALIDATION",
                trailing_stop=getattr(cfg.backtest, "trailing_stop", None),
                trailing_stop_min=getattr(cfg.backtest, "trailing_stop_min", None),
                fee_buffer_mult=getattr(cfg.backtest, "fee_buffer_mult", None),
                delta_p_hysteresis=getattr(cfg.backtest, "delta_p_hysteresis", None),
            )
            
            step_dt = episode_start_dt + dt.timedelta(minutes=env.step_idx)
            equity_log.append({"timestamp": step_dt, "equity": env._get_info()['portfolio_value']})

            if info.get("position_closed", False):
                pnl = float(info.get("trade_realized_pnl", 0.0) or 0.0)
                trade_pnls.append(pnl)
                total_trades += 1
                if info.get("correct_prediction", False):
                    total_correct += 1
                
                reason = info.get("exit_reason", "")
                if reason:
                    exit_counts[reason] = exit_counts.get(reason, 0) + 1
                if info.get("tsl_triggered", False):
                    tsl_hits += 1
                
                trade_info = {
                    'entry_time': info.get('trade_dt'),
                    'exit_time': step_dt,
                    'direction': info.get('direction'),
                    'amount': info.get('trade_amount'),
                    'pnl_net': pnl,
                    'commission': info.get('trade_commission'),
                    'exit_reason': reason,
                }
                trades_log.append(trade_info)

    # 7. Расчет и вывод метрик
    initial_balance = float(cfg.market.initial_balance)
    mean_reward = (sum(trade_pnls) / initial_balance) / max(1, num_episodes) if trade_pnls else 0.0
    mean_pnl = (sum(trade_pnls) / max(1, total_trades)) if total_trades else 0.0
    wr_ratio = total_correct / max(1, total_trades) if total_trades > 0 else 0.0
    
    pos_sum = sum(p for p in trade_pnls if p > 0)
    neg_sum = sum(p for p in trade_pnls if p < 0)
    profit_factor = (pos_sum / abs(neg_sum)) if neg_sum < 0 else float("inf")

    if trade_pnls:
        equity_curve = np.cumsum(trade_pnls) + initial_balance
        peak = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - peak) / peak
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    else:
        max_dd = 0.0

    returns = np.asarray(trade_pnls, dtype=np.float64) / max(1e-9, initial_balance)
    if returns.size > 1:
        mean_r = float(returns.mean())
        std_r = float(returns.std(ddof=1))
        downside = np.minimum(0.0, returns)
        downside_std = float(np.sqrt(np.mean(downside * downside)))
        sharpe = (mean_r / std_r) if std_r > 1e-12 else 0.0
        sortino = (mean_r / downside_std) if downside_std > 1e-12 else (float("inf") if mean_r > 0 else 0.0)
    else:
        sharpe, sortino = 0.0, 0.0

    print("\n--- Validation Results ---")
    print(f"Mean Reward:         {mean_reward:.6f}")
    print(f"Mean PnL per Trade:  {mean_pnl:+.2f}")
    print(f"Win Rate:            {wr_ratio:.2%}")
    print(f"Profit Factor:       {profit_factor:.4f}")
    print(f"Max Drawdown:        {max_dd:.2%}")
    print(f"Total Trades:        {total_trades}")
    print(f"Sharpe Ratio:        {sharpe:.3f}")
    print(f"Sortino Ratio:       {sortino:.3f}")
    if exit_counts:
        print(f"Exit Reasons:        {sorted(exit_counts.items(), key=lambda x: -x[1])}")
    if total_trades:
        print(f"TSL Hits:            {tsl_hits} ({100.0 * tsl_hits / total_trades:.2f}%)")
    print("------------------------\n")

    # 8. Сохранение CSV
    trades_df = pd.DataFrame(trades_log)
    equity_df = pd.DataFrame(equity_log)

    trades_csv_path = output_dir / "paper_trades.csv"
    equity_csv_path = output_dir / "paper_equity.csv"

    trades_df.to_csv(trades_csv_path, index=False)
    equity_df.to_csv(equity_csv_path, index=False)

    logging.info(f"Saved trades log to: {trades_csv_path}")
    logging.info(f"Saved equity curve to: {equity_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run validation on a trained RL model.")
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the Python configuration file for the model.",
    )
    args = parser.parse_args()

    # Загружаем конфиг из .py файла только для получения путей
    entry_cfg, _ = load_config(args.config_path, return_module=True)
    if not entry_cfg:
        logging.error(f"Could not load entry config from {args.config_path}")
        sys.exit(1)
        
    run_validation(entry_cfg)