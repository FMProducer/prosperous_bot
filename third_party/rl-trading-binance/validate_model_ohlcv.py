# validate_model_ohlcv.py
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
from numpy.lib.stride_tricks import sliding_window_view

# Добавляем текущую директорию в путь
sys.path.append(os.getcwd())

from agent import D3QN_PER_Agent
from config import MasterConfig
from trading_environment import TradingEnvironment
from utils import create_validation_episodes, load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def load_and_prep_data(npz_path: str, split_name: str, cfg: MasterConfig = None) -> tuple[list, list]:
    """
    Загружает NPZ, применяет Z-нормализацию для каждого актива отдельно, решейпит в (C, L, 1).
    """
    if not npz_path or not os.path.exists(npz_path):
        logger.warning(f"{split_name} data file not found or path not specified: {npz_path}")
        return [], []

    d = np.load(npz_path, allow_pickle=True)
    data_keys = [k for k in d.files if not k.startswith('_')]
    sequences = []
    valid_keys = []
    logger.info(f"Загрузка {len(data_keys)} последовательностей из {split_name}...")

    for key in data_keys:
        seq = d[key].astype(np.float32)
        # Просто загружаем сырые данные, нормализация будет в среде
        sequences.append(seq)
        valid_keys.append(key)

    d.close()
    if sequences:
        logger.info(f"Подготовлено {len(sequences)} сырых последовательностей, форма: {sequences[0].shape}")
    return sequences, valid_keys


class RollingZScoreTradingEnvironment(TradingEnvironment):
    """
    Среда, которая применяет Rolling Z-score нормализацию на лету.
    Она ожидает на вход СЫРЫЕ, ненормализованные данные.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Убедимся, что window_size доступен для _get_observation
        self.window_size = kwargs.get('agent_history_len', 90)

    def _get_observation(self) -> np.ndarray:
        # --- FIX: Prevent out-of-bounds access ---
        max_len = len(self.current_seq)
        end = min(self.pre_signal_len + self.step_idx, max_len)
        start = end - self.agent_history_len
        if start < 0: start = 0

        # 1. Получаем окно СЫРЫХ данных
        raw_window = self.current_seq[start:end]

        if len(raw_window) < self.agent_history_len:
            pad_len = self.agent_history_len - len(raw_window)
            padding = np.zeros((pad_len, self.num_features), dtype=np.float32)
            # Для сырых данных лучше пандировать последним известным значением, но нули проще
            raw_window = np.concatenate((padding, raw_window), axis=0)

        # 2. Применяем Rolling Z-Score к этому окну
        # Так как окно уже имеет нужную длину, z-score будет по этому окну
        means = np.mean(raw_window, axis=0)
        stds = np.std(raw_window, axis=0)
        stds = np.where(stds == 0, 1e-6, stds)
        normalized_window = (raw_window - means) / stds

        # 3. Формируем состояние (state) как в оригинальной среде
        unrealized = 0.0
        if self.position != 0:
            # Важно: self.entry_price здесь - это реальная цена, так как среда работает с сырыми данными
            current_price = self.current_seq[end - 1, self.close_idx]
            delta = (current_price - self.entry_price) * self.position
            # Денормализация не нужна, но для консистентности делим на цену входа
            unrealized = delta / (self.entry_price + 1e-8)


        time_elapsed = float(self.step_idx) / self.agent_session_len
        time_remaining = float(self.agent_session_len - self.step_idx) / self.agent_session_len
        extras = np.array(
            [
                float(self.position),
                unrealized,
                time_elapsed,
                time_remaining,
            ],
            dtype=np.float32,
        )

        # Reshape для CNN: (L, C) -> (C, L, 1)
        normalized_window = normalized_window.T
        normalized_window = np.expand_dims(normalized_window, -1)

        # Этот код для плоского вектора (MLP) нужно адаптировать или удалить, если используется CNN
        # Для простоты предполагаем, что модель ожидает CNN-формат, как в оригинальном load_and_prep_data
        # Если нужен плоский вектор, нужно будет раскомментировать и адаптировать код ниже

        # hist_len = normalized_window.size
        # self._obs_buffer[:hist_len] = normalized_window.ravel()
        # self._obs_buffer[hist_len:hist_len+4] = extras
        # ... (код для action history)
        # return self._obs_buffer

        return normalized_window.astype(np.float32)

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
    agent.qnetwork_local.eval()
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
                trailing_stop_min=getattr(cfg.backtest, "trailing_stop_min", None),
                fee_buffer_mult=getattr(cfg.backtest, "fee_buffer_mult", None),
                delta_p_hysteresis=getattr(cfg.backtest, "delta_p_hysteresis", None),
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
    initial_balance = float(getattr(cfg.market, "initial_balance", 10000.0))

    if all_trades_info:
        gross_pnl = sum(t.get('trade_realized_pnl', 0.0) + t.get('trade_commission', 0.0) for t in all_trades_info)
        net_pnl = sum(t.get('trade_realized_pnl', 0.0) for t in all_trades_info)
        avg_pnl_per_trade = net_pnl / len(all_trades_info) if all_trades_info else 0.0
        
        trade_pnls_all = [t.get('trade_realized_pnl', 0.0) for t in all_trades_info]
        best_trade = max(trade_pnls_all) if trade_pnls_all else 0.0
        worst_trade = min(trade_pnls_all) if trade_pnls_all else 0.0
        
        avg_holding_time = np.mean(holding_times) if holding_times else 0.0
        max_holding_time = max(holding_times) if holding_times else 0.0
        min_holding_time = min(holding_times) if holding_times else 0.0
        
        bars_per_day = 1440
        trading_time_days = total_bars_processed / bars_per_day if bars_per_day > 0 else 0.0
        
        if trade_pnls:
            avg_win_size = np.mean([p for p in trade_pnls if p > 0]) if any(p > 0 for p in trade_pnls) else 0.0
            avg_loss_size = np.mean([p for p in trade_pnls if p < 0]) if any(p < 0 for p in trade_pnls) else 0.0
            win_loss_ratio = abs(avg_win_size / avg_loss_size) if avg_loss_size < -1e-6 else float('inf')
            expectancy = (wr_ratio * avg_win_size) - ((1 - wr_ratio) * abs(avg_loss_size))
        else:
            avg_win_size = 0.0
            avg_loss_size = 0.0
            win_loss_ratio = 0.0
            expectancy = 0.0
            
        commission_pct = (total_commission / abs(gross_pnl)) * 100 if abs(gross_pnl) > 1e-6 else 0.0
        roi_percent = (net_pnl / initial_balance) * 100 if initial_balance > 0 else 0.0
        roi_annualized = roi_percent * (365.0 / trading_time_days) if trading_time_days > 0 else 0.0
    else:
        gross_pnl = net_pnl = avg_pnl_per_trade = 0.0
        best_trade = worst_trade = 0.0
        avg_holding_time = max_holding_time = min_holding_time = 0.0
        trading_time_days = 0.0
        avg_win_size = avg_loss_size = win_loss_ratio = expectancy = 0.0
        commission_pct = roi_percent = roi_annualized = 0.0

    pnl_per_day = net_pnl / trading_time_days if trading_time_days > 0 else 0.0

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
    
    logger.info(
        f"[{L}] Trades: {total_trades} (Long: {long_trades}, Short: {short_trades}, "
        f"Win: {win_count}, Loss: {loss_count}) | WinRate: {wr_ratio*100:.2f}% | PF: {profit_factor:.4f}"
    )
    logger.info(
        f"[{L}] Gross PnL: {gross_pnl:.2f} | Net PnL: {net_pnl:.2f} | "
        f"Commission: {total_commission:.2f} | Avg/Trade: {avg_pnl_per_trade:.2f}"
    )
    logger.info(
        f"[{L}] Best Trade: {best_trade:+.2f} | Worst Trade: {worst_trade:+.2f} | "
        f"MaxDD: {abs(max_dd)*100:.2f}% | Sharpe: {sharpe:.3f} | Sortino: {sortino:.3f}"
    )
    logger.info(
        f"[{L}] Avg Hold: {avg_holding_time:.2f} bars | "
        f"Min Hold: {min_holding_time} bars | Max Hold: {max_holding_time} bars"
    )
    logger.info(f"[{L}] Duration: {total_duration:.2f}s | Bars: {total_bars_processed} | "
                 f"Trading Days: {trading_time_days:.1f}")
    logger.info(f"[{L}] PnL/Day: {pnl_per_day:.2f} USDT | "
                 f"ROI: {roi_percent:.2f}% | Annualized ROI: {roi_annualized:.1f}%")
    logger.info(f"[{L}] Commission: {commission_pct:.1f}% of gross | "
                 f"Avg Win: {avg_win_size:.2f} | Avg Loss: {avg_loss_size:.2f} | "
                 f"W/L Ratio: {win_loss_ratio:.2f}")
    logger.info(f"[{L}] Expectancy/Trade: {expectancy:.2f} USDT")
    
    if exit_counts:
        logger.info(f"[{L}] Exit reasons: {dict(sorted(exit_counts.items(), key=lambda x:(-x[1], x[0])))}")
    if total_trades:
        logger.info(f"[{L}] TSL hits: {tsl_hits} ({100.0*tsl_hits/max(1,total_trades):.2f}%)")

    metrics: Dict[str, Any] = {
        f"{L}_sortino": float(np.clip(sortino, -10.0, 10.0)),
        f"{L}_sharpe": float(np.clip(sharpe, -10.0, 10.0)),
        f"{L}_net_pnl": float(net_pnl),
        f"{L}_win_rate": float(wr_ratio),
        f"{L}_trades": int(total_trades),
        f"{L}_profit_factor": float(profit_factor),
        f"{L}_max_drawdown": float(max_dd),
        f"{L}_gross_pnl": float(gross_pnl),
        f"{L}_total_commission": float(total_commission),
        f"{L}_avg_pnl_per_trade": float(avg_pnl_per_trade),
        f"{L}_pnl_per_day": float(pnl_per_day),
        f"{L}_best_trade": float(best_trade),
        f"{L}_worst_trade": float(worst_trade),
        f"{L}_long_trades": int(long_trades),
        f"{L}_short_trades": int(short_trades),
        f"{L}_win_trades": int(win_count),
        f"{L}_loss_trades": int(loss_count),
        f"{L}_avg_holding_time": float(avg_holding_time),
        f"{L}_max_holding_time": float(max_holding_time),
        f"{L}_min_holding_time": float(min_holding_time),
        f"{L}_total_duration_seconds": float(total_duration),
        f"{L}_bars_processed": int(total_bars_processed),
        f"{L}_trading_time_days": float(trading_time_days),
        f"{L}_roi_percent": float(roi_percent),
        f"{L}_roi_annualized": float(roi_annualized),
        f"{L}_commission_percent": float(commission_pct),
        f"{L}_avg_win_size": float(avg_win_size),
        f"{L}_avg_loss_size": float(avg_loss_size),
        f"{L}_win_loss_ratio": float(win_loss_ratio),
        f"{L}_expectancy": float(expectancy),
        f"{L}_tsl_hits": int(tsl_hits),
        f"{L}_exit_reasons": {k: int(v) for k, v in exit_counts.items()},
    }

    return metrics


def validate(config_path, checkpoint_path, out_dir, episode_num, args):
    # 1. Загрузка конфигурации
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    cfg_mod = None
    if config_path.endswith('.py'):
        result = load_config(config_path, return_module=True)
        if isinstance(result, tuple):
            cfg, cfg_mod = result
        else:
            cfg = result
        logger.info(f"Loaded configuration from python file: {config_path}")
    else:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        cfg = MasterConfig.model_validate(config_dict)
        logger.info(f"Loaded configuration from JSON file: {config_path}")
    
    os.makedirs(out_dir, exist_ok=True)
     
    val_data_path = cfg.paths.val_data_path

    # 2. Загружаем СЫРЫЕ данные
    val_seqs, val_keys = load_and_prep_data(val_data_path, "Validation", cfg=cfg)
    if not val_seqs:
        logger.error("No validation data loaded. Exiting.")
        sys.exit(1)

    val_seqs, val_keys = create_validation_episodes(
        val_sequences=val_seqs,
        val_keys=val_keys,
        num_episodes=cfg.trainlog.num_val_ep,
        seed=cfg.random_seed
    )

    # 3. Создаем фиктивные статистики, чтобы среда работала с сырыми ценами
    # Это заставляет среду думать, что данные уже "нормализованы" (mean=0, std=1)
    # real_price = norm_price * std + mean  => real_price = raw_price * 1 + 0
    num_features = val_seqs[0].shape[1]
    dummy_stats = {
        key.split('_')[0]: {
            'mean': np.zeros(num_features),
            'std': np.ones(num_features)
        } for key in val_keys
    }

    device = torch.device("cpu")

    env_filter = getattr(cfg.market, "filter_direction", None)
    env_allowed = getattr(cfg.market, "allowed_directions", None)
    max_trades = getattr(cfg.market, "max_trades_per_episode", 100)

    # Используем нашу кастомную среду
    env_kwargs = {
        "sequences": val_seqs,
        "keys": val_keys,
        "stats": dummy_stats,
        "full_seq_len": val_seqs[0].shape[0], # cfg.seq.full_seq_len,
        "num_features": num_features, # val_seqs[0].shape[0],
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
        "backtest_mode": True,
        "render_mode": None,
        "flat_state_size": cfg.seq.flat_state_size,
        "inaction_penalty_ratio": cfg.market.inaction_penalty_ratio,
        "filter_direction": env_filter,
        "allowed_directions": env_allowed,
        "use_risk_management": getattr(cfg.backtest, "use_risk_management", True),
        "max_trades_per_episode": max_trades,
    }
    val_env = RollingZScoreTradingEnvironment(**env_kwargs)

    # 4. Инициализация Агента
    # Важно: state_shape должен соответствовать выходу _get_observation
    # (Channels, Length, 1) -> (num_features, agent_history_len, 1)
    agent_state_shape = (num_features, cfg.seq.agent_history_len, 1)

    agent = D3QN_PER_Agent(
        state_shape=agent_state_shape, #tuple(cfg.seq.state_shape),
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
        agent.load_model(checkpoint_path, strict=False)
        logger.info(f"Loaded model from {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        sys.exit(1)

    # 6. Валидация
    metrics = evaluate_agent(val_env, agent, "Validation", cfg, keys=val_keys)
     
    # 7. Формирование имени JSON с метриками
    sortino = metrics.get("Validation_sortino", 0.0)
    sharpe = metrics.get("Validation_sharpe", 0.0)
    trades = int(metrics.get("Validation_trades", 0))
    
    json_filename = f"checkpoint_ep{episode_num:05d}_sortino{sortino:.3f}_sharpe{sharpe:.3f}_trades{trades}.json"
    output_path = os.path.join(out_dir, json_filename)
    
    # 8. Сохранение результатов
    with open(output_path, 'w') as f:
        json.dump({
            "episode": episode_num,
            "metrics": metrics,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }, f, indent=2)
     
    logger.info(f"Validation results saved to {output_path}")
    
    print(f"RESULT_JSON: {output_path}")
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a trained model.")
    parser.add_argument("--config", required=True, help="Path to config_train.json")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pth")
    parser.add_argument("--out-dir", required=True, help="Directory to save output metrics JSON")
    parser.add_argument("--episode", required=True, type=int, help="Current episode number for logging")
    parser.add_argument("--agent-mode", type=str, help="Override agent mode (LONG_ONLY, SHORT_ONLY)")

    args = parser.parse_args()
    validate(args.config, args.checkpoint, args.out_dir, args.episode, args)
