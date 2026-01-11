# validate_model.py
import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict
import numpy as np
import torch
import datetime as dt

# Добавляем текущую директорию в путь, чтобы импортировать модули проекта
sys.path.append(os.getcwd())

from agent import D3QN_PER_Agent
from config import MasterConfig
from trading_environment import TradingEnvironment
from utils import load_npz_dataset, create_validation_episodes

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_prep_data(npz_path: str, split_name: str, norm_stats: dict) -> tuple[list, list]:
    """
    Загружает NPZ, применяет Z-нормализацию для каждого актива отдельно, решейпит в (C, L, 1).
    """
    if not npz_path or not os.path.exists(npz_path):
        logger.warning(f"{split_name} data file not found or path not specified: {npz_path}")
        return [], []

    if not norm_stats:
        raise ValueError(f"norm_stats не предоставлен для {split_name}, но он обязателен.")

    d = np.load(npz_path, allow_pickle=True)
    data_keys = [k for k in d.files if not k.startswith('_')]
    sequences = []
    valid_keys = []
    logger.info(f"Загрузка {len(data_keys)} последовательностей из {split_name}...")
    
    for key in data_keys:
        try:
            asset_name = key.split('_')[0]
        except IndexError:
            logger.warning(f"Пропуск ключа с некорректным форматом: {key}")
            continue

        asset_specific_stats = norm_stats.get(asset_name)
        if asset_specific_stats is None:
            logger.warning(f"Пропуск ключа '{key}', т.к. статистики для актива '{asset_name}' не найдены.")
            continue

        means = np.array(asset_specific_stats.get('mean', asset_specific_stats.get('means')))
        stds = np.array(asset_specific_stats.get('std', asset_specific_stats.get('stds')))

        seq = d[key].astype(np.float32)
        if seq.shape[1] != len(means):
            logger.error(f"Ошибка размерности для ключа {key}: ожидалось {len(means)} каналов, получено {seq.shape[1]}")
            continue

        # Z-norm по каждому каналу
        seq = (seq - means) / (stds + 1e-8)
        # Reshape для CNN: (L, C) -> (C, L, 1)
        seq = seq.T
        seq = np.expand_dims(seq, -1)
        sequences.append(seq)
        valid_keys.append(key)
    
    d.close()
    if sequences:
        logger.info(f"Подготовлено {len(sequences)} последовательностей, форма: {sequences[0].shape}")
    return sequences, valid_keys


def evaluate_agent(
    env: TradingEnvironment,
    agent: D3QN_PER_Agent,
    split_label: str,
    cfg: MasterConfig,
    keys: list = None,
) -> Dict[str, Any]:
    """
    Greedy-оценка (без ε-эксплорации и MC-Dropout) в backtest-режиме:
    считает MeanReward/MeanPnL/WinRate/PF/MaxDD, логирует распределение exit_reason и TSL-срабатывания,
    возвращает словарь с ключами вроде 'Validation_win_rate', 'Test_profit_factor' и т.д.
    """
    agent.policy_net.eval()
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0

    total_reward = 0.0
    total_trades = 0
    total_correct = 0
    trade_pnls: list[float] = []
    ep_pnls: list[float] = []
    exit_counts: Dict[str, int] = {}
    tsl_hits = 0
    bankruptcy_episodes = 0
    
    all_trades_info = []
    total_commission = 0.0
    long_trades = 0
    short_trades = 0
    holding_times = []
    total_bars_processed = 0
    start_time = time.time()

    num_eval_episodes = len(env.sequences)

    for i in range(num_eval_episodes):
        obs, _ = env.reset(options={"forced_index": i})
        done = False
        ep_reward = 0.0
        is_bankrupt = False

        signal_dt_for_step = dt.datetime(2000, 1, 1, 0, 0)
        ticker_name = "UNKNOWN"
        if keys and i < len(keys):
            try:
                key_parts = keys[i].split('_')
                ticker_name = key_parts[0]
                if len(key_parts) > 1:
                    start_dt_str = key_parts[1]
                    signal_dt_for_step = dt.datetime.fromisoformat(start_dt_str)
            except (IndexError, AttributeError, ValueError):
                logging.warning(f"Could not parse ticker/date from key: {keys[i]}")
        
        while not done:
            action = agent.select_action(obs, training=False)
            obs, reward, done, _, info = env.backtest_step(
                action=action,
                signal_dt=signal_dt_for_step,
                ticker=ticker_name,
                stop_loss=None,
                take_profit=None,
                trailing_stop=getattr(cfg.backtest, "trailing_stop", None),
                fee_buffer_mult=getattr(cfg.backtest, "fee_buffer_mult", None),
            )
            ep_reward += float(reward or 0.0)
            total_bars_processed += 1
            if info.get("bankruptcy", False):
                is_bankrupt = True
            if info.get("position_closed", False):
                pnl = float(info.get("trade_realized_pnl", 0.0) or 0.0)
                total_trades += 1
                trade_pnls.append(pnl)
                if info.get("correct_prediction", False):
                    total_correct += 1
                reason = (info.get("exit_reason") or "")
                if reason:
                    exit_counts[reason] = exit_counts.get(reason, 0) + 1
                if info.get("tsl_triggered", False) or ("TSL" in reason):
                    tsl_hits += 1

                all_trades_info.append(info)
                total_commission += info.get('trade_commission', 0.0)
                direction = info.get('direction', '')
                if direction == 'LONG':
                    long_trades += 1
                elif direction == 'SHORT':
                    short_trades += 1

                if 'holding_duration_bars' in info:
                    holding_times.append(info['holding_duration_bars'])

        if is_bankrupt:
            bankruptcy_episodes += 1
        total_reward += ep_reward
        ep_pnls.append(sum(trade_pnls))

    total_duration = time.time() - start_time
    win_count = total_correct
    loss_count = total_trades - total_correct
    wr_ratio = total_correct / max(1, total_trades) if total_trades > 0 else 0.0

    net_pnl = sum(trade_pnls)
    initial_balance = float(getattr(cfg.market, "initial_balance", 10000.0))

    if trade_pnls:
        equity = float(initial_balance)
        peak = float(initial_balance)
        max_dd = 0.0
        for pnl in trade_pnls:
            equity += pnl
            if equity > peak:
                peak = equity
            if peak > 0.0:
                dd = (equity - peak) / peak
                if dd < max_dd:
                    max_dd = dd
    else:
        max_dd = 0.0

    returns = np.asarray(trade_pnls, dtype=np.float64) / max(1e-9, initial_balance)
    if returns.size > 1:
        mean_r = float(returns.mean())
        std_r  = float(returns.std(ddof=1))
        downside = np.minimum(0.0, returns)
        downside_std = float(np.sqrt(np.mean(downside * downside)))
        sharpe   = (mean_r / std_r) if std_r > 1e-12 else 0.0
        sortino  = (mean_r / downside_std) if downside_std > 1e-12 else (float("inf") if mean_r > 0.0 else 0.0)
    else:
        sharpe, sortino = 0.0, 0.0

    pos_sum = sum(p for p in trade_pnls if p > 0)
    neg_sum = sum(p for p in trade_pnls if p < 0)
    profit_factor = (pos_sum / abs(neg_sum)) if neg_sum < 0 else float("inf")

    L = split_label
    metrics: Dict[str, Any] = {
        f"{L}_sortino": float(np.clip(sortino, -10.0, 10.0)),
        f"{L}_sharpe": float(np.clip(sharpe, -10.0, 10.0)),
        f"{L}_net_pnl": float(net_pnl),
        f"{L}_win_rate": float(wr_ratio),
        f"{L}_trades": int(total_trades),
        f"{L}_profit_factor": float(profit_factor),
        f"{L}_max_drawdown": float(max_dd),
        # Add other metrics as needed
    }

    logger.info(f"[{L}] Validation Complete. Trades: {metrics[f'{L}_trades']}, Net PnL: {metrics[f'{L}_net_pnl']:.2f}, Sortino: {metrics[f'{L}_sortino']:.3f}")

    return metrics


def validate(config_path, checkpoint_path, output_path, episode_num):
    # 1. Загрузка конфигурации
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    cfg = MasterConfig.model_validate(config_dict)

    val_data_path = cfg.paths.val_data_path
    norm_stats_path = cfg.paths.norm_stats_path

    # 2. Загрузка данных
    if not norm_stats_path or not os.path.exists(norm_stats_path):
        logger.error(f"Norm stats not found: {norm_stats_path}")
        sys.exit(1)

    with open(norm_stats_path, 'r') as f:
        norm_stats = json.load(f)

    device = torch.device("cpu")

    # 3. Подготовка данных и среды
    val_seqs, val_keys = load_and_prep_data(val_data_path, "Validation", norm_stats)
    if not val_seqs:
        logger.error("No validation data loaded. Exiting.")
        sys.exit(1)

    val_seqs, val_keys = create_validation_episodes(
        val_sequences=val_seqs,
        val_keys=val_keys,
        num_episodes=cfg.trainlog.num_val_ep,
        seed=cfg.random_seed
    )

    env_kwargs = {
        "sequences": val_seqs,
        "keys": val_keys,
        "stats": norm_stats,
        "full_seq_len": cfg.seq.full_seq_len,
        "num_features": val_seqs[0].shape[0],
        "num_actions": cfg.market.num_actions,
        "initial_balance": cfg.market.initial_balance,
        "pre_signal_len": cfg.seq.pre_signal_len,
        "datachannels": cfg.data.datachannels,
        "position_fraction": cfg.market.position_fraction,
        "slippage": cfg.market.slippage,
        "transaction_fee": cfg.market.transaction_fee,
        "agent_session_len": cfg.seq.agent_session_len,
        "agent_history_len": cfg.seq.agent_history_len,
        "input_history_len": cfg.seq.input_history_len,
        "pricechannels": cfg.data.pricechannels,
        "volumechannels": cfg.data.volumechannels,
        "otherchannels": cfg.data.otherchannels,
        "action_history_len": cfg.seq.action_history_len,
        "backtest_mode": True
    }
    val_env = TradingEnvironment(**env_kwargs)

    # 4. Инициализация Агента
    agent = D3QN_PER_Agent(
        state_shape=tuple(cfg.seq.state_shape),
        action_dim=cfg.market.num_actions,
        cnn_maps=cfg.model.cnn_maps,
        cnn_kernels=cfg.model.cnn_kernels,
        cnn_strides=cfg.model.cnn_strides,
        dense_val=cfg.model.dense_val,
        dense_adv=cfg.model.dense_adv,
        additional_feats=cfg.model.additional_feats,
        dropout_model=cfg.model.dropout_p,
        device=device,
        gamma=cfg.rl.gamma,
        learning_rate=cfg.rl.lr,
        batch_size=cfg.rl.batch_size,
        buffer_size=cfg.per.buffer_size,
        target_update_freq=cfg.rl.target_update_freq,
        train_start=cfg.rl.train_start,
        per_alpha=cfg.per.per_alpha,
        per_beta_start=cfg.per.per_beta_start,
        per_beta_frames=cfg.per.per_beta_frames,
        eps_start=0.0,
        eps_end=0.0,
        eps_frames=1,
        epsilon=0.0,
        max_gradient_norm=cfg.rl.max_gradient_norm,
        cnn_dilations=cfg.model.cnn_dilations
    )

    # 5. Загрузка чекпойнта
    try:
        agent.load_model(checkpoint_path, strict=False) # Use strict=False for flexibility
        logger.info(f"Loaded model from {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        sys.exit(1)

    # 6. Валидация
    metrics = evaluate_agent(val_env, agent, "Validation", cfg, keys=val_keys)

    # 7. Сохранение результатов
    with open(output_path, 'w') as f:
        json.dump({
            "episode": episode_num,
            "metrics": metrics,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }, f, indent=2)

    logger.info(f"Validation results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a trained model.")
    parser.add_argument("--config", required=True, help="Path to config_train.json")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pth")
    parser.add_argument("--out", required=True, help="Path to output metrics.json")
    parser.add_argument("--episode", required=True, type=int, help="Current episode number for logging")

    args = parser.parse_args()
    validate(args.config, args.checkpoint, args.out, args.episode)