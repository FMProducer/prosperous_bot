# train.py
import logging
import os
import sys
import time
from collections import deque
from typing import Any, Dict
import hashlib, tarfile
import datetime as dt

# CuBLAS: детерминизм требует рабочего пространства; задаём до импорта torch
if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import platform
import json
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm, trange

from agent import D3QN_PER_Agent
from config import MasterConfig
from config import cfg as default_cfg
from subproc_vec_env import SubprocVecEnv
from vec_env import DummyVecEnv # noqa: F401
from trading_environment import TradingEnvironment
from utils import (
    calculate_normalization_stats,
    load_config,
    load_npz_dataset,
    select_and_arrange_channels,
    set_random_seed,
    setup_logging,
) # noqa: F401

def _make_train_env_fns(env_kwargs, n: int):
    # фабрика копий среды для векторизации
    return [lambda ek=env_kwargs: TradingEnvironment(**ek) for _ in range(n)]


def _rollout_vectorized_episode(train_env: DummyVecEnv, agent: D3QN_PER_Agent):
    """
    Один "батч-эпизод" на N средах:
    - параллельно идём до завершения каждой под-среды (autoreset внутри VecEnv),
    - накапливаем опыт и возвращаем средний суммарный reward за эпизоды.
    """
    obs_batch, _ = train_env.reset(seed=None, options=None)
    done_mask = np.zeros(train_env.num_envs, dtype=bool)
    ep_reward = np.zeros(train_env.num_envs, dtype=float)
    step_iters = 0
    win_rates = []
    while not done_mask.all():
        prev_done = done_mask.copy()
        actions = [agent.select_action(obs_batch[i], training=True) for i in range(train_env.num_envs)]
        next_obs_b, rewards, dones, trunc, infos = train_env.step(actions)
        # в DQN/пер меры используем done (без разгадки truncated), как и было в одиночной логике
        for i in range(train_env.num_envs):
            # Корректный next_state при done: брать финальное наблюдение из info
            if bool(dones[i]) and isinstance(infos[i], dict):
                next_state = infos[i].get("terminal_observation",
                               infos[i].get("final_observation", next_obs_b[i]))
            else:
                next_state = next_obs_b[i]
            agent.store_experience(obs_batch[i], actions[i], float(rewards[i]), next_state, bool(dones[i]))
            if bool(dones[i]) and isinstance(infos[i], dict):
                wr = infos[i].get("episode_win_rate", None)
                if wr is not None:
                    win_rates.append(float(wr))
        # Накапливать награды только для тех подсред, которые ещё не были завершены до этого шага
        for i in range(train_env.num_envs):
            if not prev_done[i]:
                ep_reward[i] += float(rewards[i])
        obs_batch = next_obs_b
        done_mask |= dones  # эпизод для каждой под-среды
        step_iters += 1
        # В каждом "батч-шаге" получаем по одному переходу на среду
        for _ in range(train_env.num_envs):
            agent.increment_step()
        # (Опционально) вызывать шаг обучения на каждом батч-шаге, как в одиночной ветке:
        # loss = agent.learn()
        # if loss is not None:
        #     ep_losses.append(loss)
    avg_reward = float(ep_reward.mean())
    avg_win_rate = float(np.mean(win_rates)) if win_rates else 0.0
    transitions_count = int(step_iters * train_env.num_envs)
    return avg_reward, avg_win_rate, transitions_count


def plot_training_progress(history: dict, save_dir: str, window_size: int) -> None:
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    episodes = history.get("episodes", [])
    rewards = history.get("rewards", [])
    mean_rewards = history.get("mean_rewards_N", [])
    losses = history.get("losses", [])
    mean_losses = history.get("mean_losses_N", [])
    epsilons = history.get("epsilons", [])
    win_rates = history.get("win_rates", [])
    mean_win_rates = history.get("mean_win_rates_N", [])

    if episodes and rewards:
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            x=episodes,
            y=rewards,
            label="Reward per Episode",
            color="tab:blue",
            alpha=0.3,
            linewidth=1.5,
        )

        if mean_rewards:
            sns.lineplot(
                x=episodes,
                y=mean_rewards,
                label=f"Moving Avg window={window_size}",
                color="tab:blue",
                linestyle="-",
                linewidth=2.5,
            )
        plt.title("Training Rewards over Episodes", fontsize=16, fontweight="bold")
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Reward", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12, loc="upper left")
        plt.tight_layout()
        save_path = os.path.join(save_dir, "training_rewards.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        logging.info(f"Saved reward plot: {save_path}")
    else:
        logging.warning("No 'episodes' or 'rewards' data available to plot the reward graph.")

    if episodes and losses:
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            x=episodes,
            y=losses,
            label="Loss per Episode",
            color="tab:red",
            alpha=0.3,
            linewidth=1.5,
        )
        if mean_losses:
            sns.lineplot(
                x=episodes,
                y=mean_losses,
                label=f"Moving Avg window={window_size}",
                color="tab:red",
                linestyle="--",
                linewidth=2.5,
            )
        plt.title("Training Loss over Episodes", fontsize=16, fontweight="bold")
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12, loc="upper right")
        plt.tight_layout()
        save_path = os.path.join(save_dir, "training_losses.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        logging.info(f"Saved loss plot: {save_path}")
    else:
        logging.warning("No 'episodes' or 'losses' data available to plot the loss graph.")

    if episodes and win_rates:
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            x=episodes,
            y=[wr * 100 for wr in win_rates],
            label="Win Rate (%) per Episode",
            color="tab:orange",
            alpha=0.3,
            linewidth=1.5,
        )
        if mean_win_rates:
            sns.lineplot(
                x=episodes,
                y=[wr * 100 for wr in mean_win_rates],
                label=f"Moving Avg window={window_size}",
                color="tab:orange",
                linestyle="--",
                linewidth=2.5,
            )
        plt.title("Training Win Rate (%) over Episodes", fontsize=16, fontweight="bold")
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Win Rate (%)", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12, loc="upper right")
        plt.ylim(-5, 105)
        plt.tight_layout()
        save_path = os.path.join(save_dir, "training_win_rate.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        logging.info(f"Saved Win Rate plot: {save_path}")
    else:
        logging.warning("No 'episodes' or 'Win Rate' data available to generate the Win Rate plot.")

    if episodes and epsilons:
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            x=episodes,
            y=epsilons,
            label="Epsilon (ε)",
            color="tab:green",
            linewidth=2.0,
        )
        plt.title("Epsilon Decay over Episodes", fontsize=16, fontweight="bold")
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Epsilon (ε)", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12, loc="upper right")
        plt.tight_layout()
        save_path = os.path.join(save_dir, "epsilon_decay.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        logging.info(f"Saved epsilon-greedy plot: {save_path}")
    else:
        logging.info("No 'epsilons' data available – skipping epsilon plot.")

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _dump_requirements_lock(dst_path: str) -> None:
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        with open(dst_path, "w", encoding="utf-8") as f:
            f.write(out)
        logging.info(f"Saved pip freeze to: {dst_path}")
    except Exception as e:
        logging.exception(f"Failed to dump requirements: {e}")

def _dump_torch_env(dst_path: str) -> None:
    lines = []
    try:
        lines.append(f"python={platform.python_version()}")
        lines.append(f"platform={platform.platform()}")
        lines.append(f"torch={torch.__version__}")
        lines.append(f"cuda={getattr(torch.version, 'cuda', None)}")
        try:
            import torch.backends.cudnn as cudnn
            lines.append(f"cudnn={getattr(cudnn, 'version', lambda: None)()})")
        except Exception:
            lines.append("cudnn=None")
        if torch.cuda.is_available():
            dev = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            lines.append(f"gpu={dev}")
            lines.append(f"gpu_cc={cc}")
        with open(dst_path, "w", encoding="utf-8") as f:
            f.write("\n".join(str(x) for x in lines) + "\n")
        logging.info(f"Saved torch env to: {dst_path}")
    except Exception as e:
        logging.exception(f"Failed to dump torch env: {e}")

def _dump_env_flags(cfg: MasterConfig, dst_path: str) -> None:
    payload = {
        "random_seed": cfg.random_seed,
        "determinism": {
            "cudnn_benchmark": cfg.perf.cudnn_benchmark,
        },
        "amp": {
            "use_amp": cfg.perf.use_amp,
            "amp_dtype": cfg.perf.amp_dtype,
        },
    }
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logging.info(f"Saved env flags to: {dst_path}")

def _build_data_manifest(cfg: MasterConfig) -> Dict[str, Any]:
    items = []
    for key, p in {
        "train_data_path": cfg.paths.train_data_path,
        "val_data_path": cfg.paths.val_data_path,
        "test_data_path": cfg.paths.test_data_path,
    }.items():
        if p and os.path.exists(p):
            try:
                items.append({
                    "name": key,
                    "path": p,
                    "sha256": _sha256(p),
                    "bytes": os.path.getsize(p),
                })
            except Exception as e:
                logging.warning(f"Manifest: cannot hash {p}: {e}")
    return {"datasets": items}

def _git_head_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""

def _numpy_json_default(obj):
    """
    Custom JSON serializer for numpy types.
    """
    if isinstance(obj, (np.integer, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def evaluate_agent(
    env: TradingEnvironment,
    agent: D3QN_PER_Agent,
    episodes: int,
    split_label: str,
    episode_num: int | None,
    seed: int | None,
    cfg: MasterConfig,
) -> Dict[str, Any]:
    """
    Greedy-оценка (без ε-эксплорации и MC-Dropout) в backtest-режиме:
    считает MeanReward/MeanPnL/WinRate/PF/MaxDD, логирует распределение exit_reason и TSL-срабатывания,
    возвращает словарь с ключами вроде 'Validation_win_rate', 'Test_profit_factor' и т.д.
    """
    # ── Жёстко выключаем стохастику выбора действий
    try:
        agent.policy_net.eval()
    except Exception:
        pass
    old_eps = getattr(agent, "epsilon", None)
    old_mc  = getattr(agent, "mc_enable", None)
    if hasattr(agent, "epsilon"): agent.epsilon = 0.0
    if hasattr(agent, "mc_enable"): agent.mc_enable = False

    # накопители
    total_reward = 0.0
    total_trades = 0
    total_correct = 0
    trade_pnls: list[float] = []
    ep_pnls:   list[float] = []
    ep_rews:   list[float] = []
    ep_wrs:    list[float] = []
    exit_counts: Dict[str,int] = {}
    tsl_hits = 0

    stub_dt = dt.datetime(2000, 1, 1, 0, 0)
    stub_tk = "VAL"

    for _ in range(int(episodes)):
        obs, _ = env.reset(seed=None)
        done = False
        ep_reward = 0.0
        ep_trades = 0
        ep_wins   = 0
        ep_trade_pnls: list[float] = []
        while not done:
            action = agent.select_action(obs, training=False)
            obs, reward, done, _, info = env.backtest_step(
                action=action,
                signal_dt=stub_dt,
                ticker=stub_tk,
                stop_loss=None,
                take_profit=None,
                trailing_stop=getattr(cfg.backtest, "trailing_stop", None),
                trailing_stop_min=getattr(cfg.backtest, "trailing_stop_min", None),
                fee_buffer_mult=getattr(cfg.backtest, "fee_buffer_mult", None),
                delta_p_hysteresis=getattr(cfg.backtest, "delta_p_hysteresis", None),
            )
            ep_reward += float(reward or 0.0)
            if info.get("position_closed", False):
                pnl = float(info.get("trade_realized_pnl", 0.0) or 0.0)
                ep_trades += 1
                trade_pnls.append(pnl)
                ep_trade_pnls.append(pnl)
                if info.get("correct_prediction", False):
                    ep_wins += 1
                reason = (info.get("exit_reason") or "")
                if reason:
                    exit_counts[reason] = exit_counts.get(reason, 0) + 1
                if info.get("tsl_triggered", False) or ("TSL" in reason):
                    tsl_hits += 1
        # завершение эпизода
        total_reward += ep_reward
        total_trades += ep_trades
        total_correct += ep_wins
        ep_pnls.append(sum(ep_trade_pnls))
        ep_rews.append(ep_reward)
        ep_wrs.append( ep_wins / max(1, ep_trades) if ep_trades else 0.0 )

    mean_reward = total_reward / max(1, episodes)
    mean_pnl = (sum(trade_pnls) / max(1, total_trades)) if total_trades else 0.0
    wr_ratio = total_correct / max(1, total_trades)
    pos_sum = sum(p for p in trade_pnls if p > 0)
    neg_sum = sum(p for p in trade_pnls if p < 0)
    profit_factor = (pos_sum / abs(neg_sum)) if neg_sum < 0 else float("inf")

    # --- Sharpe / Sortino ---
    # Sharpe: стандартно по σ общих доходностей.
    # Sortino: downside semideviation (MAR=0): sqrt(mean(min(0, r)^2)).
    try:
        initial_balance = float(getattr(cfg.market, "initial_balance", 10_000.0))
    except Exception:
        initial_balance = 10_000.0

    # MaxDD по кумулятивной equity
    eq = 0.0; peak = 0.0; max_dd = 0.0
    for p in trade_pnls:
        eq += p
        peak = max(peak, eq)
        drawdown_value = peak - eq # Это положительное число
        max_dd = max(max_dd, drawdown_value / max(1e-9, initial_balance))

    returns = np.asarray(trade_pnls, dtype=np.float64) / max(1e-9, initial_balance)
    if returns.size > 0:
        mean_r = float(returns.mean())
        std_r  = float(returns.std(ddof=1)) if returns.size > 1 else float(returns.std(ddof=0))
        # Downside semideviation (MAR=0) — без ddof, как в определении Sortino
        downside = np.minimum(0.0, returns)
        downside = float(np.sqrt(np.mean(downside * downside)))
        sharpe   = (mean_r / std_r)      if std_r      > 1e-12 else 0.0
        sortino  = (mean_r / downside)   if downside   > 1e-12 else (float("inf") if mean_r > 0.0 else 0.0)
    else:
        sharpe, sortino = 0.0, 0.0

    # лог-сводка
    logging.info(
        "[%s] MeanReward=%.6f  MeanPnL=%+.2f  WinRate=%.2f%%  PF=%.4f  MaxDD=%+.2f  Trades=%d  Sharpe=%.3f  Sortino=%.3f",
        split_label, mean_reward, mean_pnl, wr_ratio*100.0, profit_factor, -max_dd, total_trades, sharpe, sortino
    )
    if exit_counts:
        logging.info("[%s] Exit reasons: %s", split_label,
                     {k:int(v) for k,v in sorted(exit_counts.items(), key=lambda x:(-x[1], x[0]))})
    if total_trades:
        logging.info("[%s] TSL hits: %d (%.2f%%)", split_label, tsl_hits, 100.0*tsl_hits/max(1,total_trades))

    # вернуть исходные режимы агента
    if old_eps is not None: agent.epsilon = old_eps
    if old_mc  is not None: agent.mc_enable = old_mc

    # сформировать словарь под выбор метрики в тренере
    L = split_label  # "Validation" | "Test"
    out: Dict[str,Any] = {
        f"{L}_mean_reward": float(mean_reward),
        f"{L}_mean_pnl":    float(mean_pnl),
        f"{L}_win_rate":    float(wr_ratio),            # 0..1 — удобно для отбора
        f"{L}_win_rate_percent": float(wr_ratio*100.0),
        f"{L}_profit_factor": float(profit_factor),
        # FIX: Возвращаем просадку как отрицательное число, как и принято в индустрии.
        f"{L}_max_drawdown": -float(max_dd),
        f"{L}_trades": int(total_trades),
        f"{L}_tsl_hits": int(tsl_hits),
        f"{L}_exit_reasons": {k:int(v) for k,v in exit_counts.items()},
        f"{L}_sharpe":  float(np.clip(sharpe,   -10.0, 10.0)),
        f"{L}_sortino": float(np.clip(sortino,  -10.0, 10.0)),
    }
    if L == "Test":
        out.update({
            "Test_all_pnls": ep_pnls,
            "Test_all_reward": ep_rews,
            "Test_all_win_rate": ep_wrs,
        })
    return out


def process_data(raw_list, name_dataset, cfg: MasterConfig):
    seqs = []
    for _, arr in tqdm(raw_list, desc=f"Selecting and arrange channels for {name_dataset}", leave=False):
        sel = select_and_arrange_channels(arr, cfg.data.expected_channels, cfg.data.data_channels)
        if sel is not None:
            seqs.append(sel)
    return seqs


def main(cfg: MasterConfig = None):
    # Загружаем конфиг и модуль, чтобы иметь доступ ко всем переменным, включая bundle_cfg
    if cfg is not None:
        cfg_mod = None # Модуль конфига недоступен, если cfg передан напрямую
    elif len(sys.argv) > 1:
        cfg, cfg_mod = load_config(sys.argv[1], return_module=True)
    else:
        cfg, cfg_mod = default_cfg, None

    # --- MC-dropout: ищем внешний объект `mc_dropout_cfg` или создаём пустышку ---
    mc_cfg = getattr(cfg_mod, "mc_dropout_cfg", type("obj", (), {})())

    timestamp = time.strftime("date_%Y%m%d_time_%H%M%S")
    session_name = f"{cfg.project_name}_{timestamp}"
    setup_logging(session_name, cfg)
    set_random_seed(cfg.random_seed)
    # Получаем bundle_cfg из модуля или из cfg для обратной совместимости
    bundle_cfg = getattr(cfg_mod, "bundle_cfg", getattr(cfg, "bundle", object()))
    if cfg.device.device.type == "cuda":
        torch.backends.cudnn.benchmark = cfg.perf.cudnn_benchmark
    # Детерминизм по умолчанию ВКЛЮЧЕН; отключить: RL_DETERMINISTIC=0
    det = True
    env_flag = os.environ.get("RL_DETERMINISTIC")
    if env_flag is not None:
        det = env_flag not in ("0", "false", "False", "no", "No")
    # Разрешаем переопределение из конфига, если поле существует (обратная совместимость)
    det = bool(getattr(cfg, "deterministic", det)) if hasattr(cfg, "deterministic") else det
    det = bool(getattr(getattr(cfg, "perf", object()), "deterministic", det))
    if det:
        try:
            import torch.backends.cudnn as cudnn
            cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=True)
            logging.info(
                "Deterministic mode: ON "
                "(cudnn.deterministic=True, torch.use_deterministic_algorithms; "
                f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')})"
            )
        except Exception as e:
            logging.warning(f"Deterministic mode setup failed: {e}")

    # Fallback: если model_dir/plot_dir не определены, строим их из base_output_dir
    base_out = getattr(cfg.paths, "base_output_dir", None)
    if not hasattr(cfg.paths, "model_dir") or cfg.paths.model_dir in (None, ""):
        cfg.paths.model_dir = os.path.join(base_out or "output", cfg.paths.config_name, "saved_models")
    if not hasattr(cfg.paths, "plot_dir") or cfg.paths.plot_dir in (None, ""):
        cfg.paths.plot_dir = os.path.join(base_out or "output", cfg.paths.config_name, "plots")
    models_dir = os.path.join(cfg.paths.model_dir, session_name)
    plots_dir = os.path.join(cfg.paths.plot_dir, session_name)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # --- Save the full training configuration (immutable copy) ---
    config_save_path = os.path.join(models_dir, "config_train.json")
    with open(config_save_path, "w") as f:
        # Use model_dump and default=str to handle non-serializable types like torch.device
        json.dump(cfg.model_dump(), f, indent=2, default=str)
    logging.info(f"Full training configuration saved to: {config_save_path}")

    raw_train = load_npz_dataset(
        file_path=cfg.paths.train_data_path,
        name_dataset="Train",
        plot_dir=cfg.paths.plot_dir,
        debug_max_size=cfg.debug.debug_max_size_data,
        plot_examples=cfg.data.plot_examples,
        plot_channel_idx=cfg.data.plot_channel_idx,
        pre_signal_len=cfg.seq.pre_signal_len,
    )

    raw_val = (
        load_npz_dataset(
            file_path=cfg.paths.val_data_path,
            name_dataset="Val",
            plot_dir=cfg.paths.plot_dir,
            debug_max_size=cfg.debug.debug_max_size_data,
            plot_examples=cfg.data.plot_examples,
            plot_channel_idx=cfg.data.plot_channel_idx,
            pre_signal_len=cfg.seq.pre_signal_len,
        )
        if cfg.trainlog.validate_model
        else []
    )

    raw_test = load_npz_dataset(
        file_path=cfg.paths.test_data_path,
        name_dataset="Test",
        plot_dir=cfg.paths.plot_dir,
        debug_max_size=cfg.debug.debug_max_size_data,
        plot_examples=cfg.data.plot_examples,
        plot_channel_idx=cfg.data.plot_channel_idx,
        pre_signal_len=cfg.seq.pre_signal_len,
    )
    raw_test = []

    train_seqs = process_data(raw_train, "Train", cfg)
    val_seqs = process_data(raw_val, "Val", cfg)
    test_seqs = process_data(raw_test, "Test", cfg)

    if not train_seqs:
        logging.error("No training data – aborting.")
        return

    logging.info(f"Data sizes: train={len(train_seqs)}, val={len(val_seqs)}, test={len(test_seqs)}")

    train_stats = calculate_normalization_stats(
        train_seqs,
        cfg.data.data_channels,
        cfg.data.price_channels,
        cfg.data.volume_channels,
        cfg.data.other_channels,
    )

    # --- Save normalization stats for this training run ---
    stats_save_path = os.path.join(models_dir, "norm_stats.json")
    with open(stats_save_path, "w") as f:
        json.dump(train_stats, f, indent=4)
    logging.info(f"Normalization stats saved to: {stats_save_path}")


    env_kwargs = {
        "sequences": train_seqs,
        "stats": train_stats,
        "render_mode": cfg.render_mode,
        "full_seq_len": cfg.seq.full_seq_len,
        "num_features": cfg.seq.num_features,
        "num_actions": cfg.market.num_actions,
        "flat_state_size": cfg.seq.flat_state_size,
        "initial_balance": cfg.market.initial_balance,
        "pre_signal_len": cfg.seq.pre_signal_len,
        "data_channels": cfg.data.data_channels,
        "slippage": cfg.market.slippage,
        "transaction_fee": cfg.market.transaction_fee,
        "agent_session_len": cfg.seq.agent_session_len,
        "agent_history_len": cfg.seq.agent_history_len,
        "input_history_len": cfg.seq.input_history_len,
        "price_channels": cfg.data.price_channels,
        "volume_channels": cfg.data.volume_channels,
        "other_channels": cfg.data.other_channels,
        "action_history_len": cfg.seq.action_history_len,
        "inaction_penalty_ratio": cfg.market.inaction_penalty_ratio,
    }
    # --- TRAIN ENV: single vs vectorized ---
    if cfg.vec.num_envs > 1:
        env_fns = _make_train_env_fns(env_kwargs, cfg.vec.num_envs)
        if cfg.vec.backend == "subproc":
            train_env = SubprocVecEnv(env_fns, start_method=cfg.vec.start_method)
            logging.info(f"Vectorized train env: SubprocVecEnv x{cfg.vec.num_envs} (start_method='{cfg.vec.start_method}')")
        elif cfg.vec.backend == "dummy":
            train_env = DummyVecEnv(env_fns)
            logging.info(f"Vectorized train env: DummyVecEnv x{cfg.vec.num_envs}")
        else:
            raise ValueError(f"Unknown vec.backend: '{cfg.vec.backend}'")
    else:
        train_env = TradingEnvironment(**env_kwargs)
    # Валидация: backtest-режим + TSL/exec-delay
    val_env = None
    if val_seqs:
        val_kwargs = dict(env_kwargs)
        val_kwargs["sequences"] = val_seqs
        val_kwargs["backtest_mode"] = True
        val_kwargs["use_risk_management"] = getattr(cfg.backtest, "use_risk_management", True)
        val_kwargs["transaction_fee"] = getattr(cfg.market, "transaction_fee", 0.0)
        val_env = TradingEnvironment(**val_kwargs)
        if hasattr(cfg.backtest, "exec_delay_bars"):
            setattr(val_env, "exec_delay_bars", int(cfg.backtest.exec_delay_bars))

    agent = D3QN_PER_Agent(
        state_shape=(cfg.seq.num_features, cfg.seq.input_history_len, 1),
        action_dim=cfg.market.num_actions,
        cnn_maps=cfg.model.cnn_maps,
        cnn_kernels=cfg.model.cnn_kernels,
        cnn_strides=cfg.model.cnn_strides,
        cnn_dilations=getattr(cfg.model, 'cnn_dilations', None),
        dense_val=cfg.model.dense_val,
        dense_adv=cfg.model.dense_adv,
        additional_feats=cfg.model.additional_feats,
        dropout_model=cfg.model.dropout_p,
        device=cfg.device.device,
        gamma=cfg.rl.gamma,
        learning_rate=cfg.rl.learning_rate,
        batch_size=cfg.rl.batch_size,
        buffer_size=cfg.per.buffer_size,
        target_update_freq=cfg.rl.target_update_freq,
        train_start=cfg.rl.train_start,
        per_alpha=cfg.per.per_alpha,
        per_beta_start=cfg.per.per_beta_start,
        per_beta_frames=cfg.per.per_beta_frames,
        eps_start=cfg.eps.eps_start,
        eps_end=cfg.eps.eps_end,
        eps_frames=cfg.eps.eps_decay_frames,
        epsilon=cfg.per.per_eps,
        max_gradient_norm=cfg.rl.max_gradient_norm,
        backtest_cache_path=None,
        perf_cfg=cfg.perf,
        # ── НОВОЕ: MC-dropout в обучении (читаем из нескольких источников)
        **(lambda mc: dict(
            mc_enable=getattr(mc, "enable", False),
            mc_n_action_samples=getattr(mc, "n_action_samples", 1),
            mc_action_agg=getattr(mc, "action_agg", "mean"),
            mc_lcb_k=getattr(mc, "lcb_k", 0.0),
            mc_use_for_target=getattr(mc, "use_for_target", False),
            mc_n_target_samples=getattr(mc, "n_target_samples", 1),
            mc_target_agg=getattr(mc, "target_agg", "mean_max"),
            mc_uncertainty_guided_explore=getattr(mc, "uncertainty_guided_explore", False),
            mc_uncertainty_beta=getattr(mc, "uncertainty_beta", 0.0),
        ))(
            # приоритет: cfg.rl.mc_dropout → cfg.mc_dropout → cfg_mod.mc_dropout_cfg → пустой объект
            getattr(getattr(cfg, "rl", object()), "mc_dropout", None)
            or getattr(cfg, "mc_dropout", None)
            or (getattr(cfg_mod, "mc_dropout_cfg", None) if 'cfg_mod' in locals() else None)
            or object()
        ),
    )

    episode_rewards_deque = deque(maxlen=cfg.trainlog.plot_moving_avg_window)
    episode_losses_deque = deque(maxlen=cfg.trainlog.plot_moving_avg_window)
    episode_win_rate_deque = deque(maxlen=cfg.trainlog.plot_moving_avg_window)

    history = {
        "episodes": [],
        "rewards": [],
        "mean_rewards_N": [],
        "losses": [],
        "mean_losses_N": [],
        "epsilons": [],
        "win_rates": [],
        "mean_win_rates_N": [],
    }

    # Selection settings (backward-compatible defaults)
    val_direction = str(getattr(getattr(cfg, "trainlog", object()), "val_selection_direction", "max")).lower()
    val_min_delta = float(getattr(getattr(cfg, "trainlog", object()), "val_min_delta", 0.0))
    if val_direction not in ("max", "min"):
        logging.warning("Unsupported val_selection_direction=%r -> fallback to 'max'", val_direction)
        val_direction = "max"

    # Поддержим мульти-объективный режим: значение лучшей метрики храним как None|float|tuple
    best_val_metric = None
    last_val_metrics: Dict[str, Any] = {}
    best_validation: Dict[str, Any] = {}
    # --- Early Stopping ---
    # Ищем в конфиге, если нет — используем разумные значения по умолчанию
    early_stopping_patience = int(getattr(getattr(cfg, "trainlog", object()), "early_stopping_patience", 20))
    # Счётчик валидаций без улучшения
    no_improvement_count = 0

    best_episode: int | None = None
    train_steps = 0

    train_env.reset(seed=cfg.global_env_seed)
    counter = trange(1, cfg.trainlog.episodes + 1, desc="Training in episodes", leave=False)
    for ep in counter:
        if hasattr(train_env, "num_envs"):  # VecEnv путь
            ep_reward, ep_win_rate, transitions = _rollout_vectorized_episode(train_env, agent)
            # Увеличиваем число градиентных шагов пропорционально собранным переходам
            steps_to_train = max(1, transitions // cfg.rl.batch_size)
            ep_losses = []
            for _ in range(steps_to_train):
                loss = agent.learn()
                if loss is not None:
                    ep_losses.append(loss)
            train_steps += transitions
            info = {"episode_win_rate": ep_win_rate}
        else:
            obs, _ = train_env.reset(seed=None, options=None)
            ep_reward = 0.0
            ep_losses = []
            done = False
            while not done:
                action = agent.select_action(obs, training=True)
                next_obs, reward, done, _, info = train_env.step(action)
                agent.store_experience(obs, action, reward, next_obs, done)
                loss = agent.learn()
                if loss is not None:
                    ep_losses.append(loss)
                obs = next_obs
                agent.increment_step()
                train_steps += 1
                ep_reward += reward

        history["episodes"].append(ep)
        history["rewards"].append(ep_reward)

        avg_loss = np.mean(ep_losses) if ep_losses else 0.0
        history["losses"].append(avg_loss)

        episode_rewards_deque.append(ep_reward)
        mean_reward_N = float(np.mean(episode_rewards_deque))
        history["mean_rewards_N"].append(mean_reward_N)

        episode_losses_deque.append(avg_loss)
        mean_loss_N = float(np.mean(episode_losses_deque)) if episode_losses_deque else 0.0
        history["mean_losses_N"].append(mean_loss_N)

        eps_current = agent.eps_end + (agent.eps_start - agent.eps_end) * np.exp(-train_steps / agent.eps_frames)
        history["epsilons"].append(eps_current)

        episode_win_rate_deque.append(info.get("episode_win_rate", 0.0))
        history["win_rates"].append(info.get("episode_win_rate", 0.0))
        mean_win_rate_N = float(np.mean(episode_win_rate_deque)) if episode_win_rate_deque else 0.0
        history["mean_win_rates_N"].append(mean_win_rate_N)

        counter.desc = f"Training loss={avg_loss:.7f}, reward={ep_reward:.5f}"

        if val_env and ep % cfg.trainlog.val_freq == 0:
            metrics = evaluate_agent(
                val_env,
                agent,
                min(len(val_seqs), cfg.trainlog.num_val_ep),
                "Validation",
                ep,
                cfg.global_env_seed,
                cfg,
            )
            # Поддержка single- и multi-objective отбора лучшей модели.
            # Пример: cfg.trainlog.val_selection_metrics = [
            #   "Validation_sharpe", "Validation_sortino",
            #   "Validation_profit_factor", "Validation_win_rate"
            # ]
            sel_keys = cfg.trainlog.val_selection_metrics

            # Нормализация направления сравнения: для метрик из этого множества "меньше — лучше"
            lower_is_better = {
                "Validation_loss" # FIX: max_drawdown теперь отрицательный, поэтому для него "больше - лучше".
            }

            def _fetch_metric(name: str) -> float:
                v = metrics.get(name, None)
                if v is None:
                    logging.warning(f"[Validation] metric '{name}' is missing in metrics dict — using fallback -inf")
                    return float("-inf")
                try:
                    v = float(v)
                except Exception:
                    logging.warning(f"[Validation] metric '{name}' has non-numeric value '{v}' — fallback -inf")
                    return float("-inf")
                return -v if name in lower_is_better else v

            # Лексикографический приоритет по порядку в списке:
            # сначала ключ[0], затем ключ[1], ...
            if isinstance(sel_keys, (list, tuple)):
                val_metric = tuple(_fetch_metric(k) for k in sel_keys)
            else:
                val_metric = _fetch_metric(str(sel_keys))
            last_val_metrics = metrics

            # ── ВАЛИДАЦИОННЫЙ ГЕЙТ: пороги берём ТОЛЬКО из конфигурации
            # Первично ищем в cfg.trainlog.validation_gate (если у nested-конфига разрешены extra-поля),
            # иначе — fallback на верхний уровень cfg.validation_gate (MasterConfig.extra='allow').
            gate = getattr(getattr(cfg, "trainlog", object()), "validation_gate", None)
            if gate is None:
                gate = getattr(cfg, "validation_gate", None)
            def _passes_gate(m: Dict[str, Any], g: Dict[str, Any] | None) -> bool:
                if not g:
                    return True  # гейт выключен, если не задан в конфиге
                def _f(name: str, default: float | None = None) -> float:
                    v = m.get(name, default)
                    try:
                        return float(v)
                    except Exception:
                        return float("-inf")
                cur_sharpe  = _f("Validation_sharpe")
                cur_sortino = _f("Validation_sortino")
                cur_pf_raw  = m.get("Validation_profit_factor", None)
                # PF может быть float("inf")
                try:
                    cur_pf = float(cur_pf_raw)
                except Exception:
                    cur_pf = float("-inf")
                cur_dd      = _f("Validation_max_drawdown")  # уже отрицательный (−DD)
                cur_wr      = _f("Validation_win_rate")      # 0..1
                cur_trades  = int(m.get("Validation_trades", 0) or 0)

                # Параметры из конфига
                min_sharpe     = g.get("min_sharpe", None)
                min_sortino    = g.get("min_sortino", None)
                min_pf         = g.get("min_profit_factor", None)
                max_dd_at_most = g.get("max_drawdown_at_most", None)
                min_wr         = g.get("min_win_rate", None)
                min_trades     = g.get("min_trades", None)
                deny_zero_dd   = bool(g.get("deny_zero_drawdown", False))
                deny_inf_pf    = bool(g.get("deny_inf_pf", False))

                # Проверки
                ok = True
                if (min_sharpe  is not None) and not (cur_sharpe  >= float(min_sharpe)):         ok = False
                if (min_sortino is not None) and not (cur_sortino >= float(min_sortino)):        ok = False
                if deny_inf_pf and (isinstance(cur_pf_raw, str) and cur_pf_raw.lower() == "inf"): ok = False
                if deny_inf_pf and (cur_pf == float("inf")):                                      ok = False # noqa: E272
                if (min_pf      is not None) and not (cur_pf      >= float(min_pf)):             ok = False # noqa: E272
                # ИСПРАВЛЕНО: `cur_dd` должен быть БОЛЬШЕ или РАВЕН порогу (т.к. -0.01 > -0.05).
                if (max_dd_at_most is not None) and not (cur_dd   >= float(max_dd_at_most)):     ok = False # noqa: E272
                # НОВОЕ: Запрещаем модели с нулевой просадкой, если флаг установлен.
                if deny_zero_dd and cur_dd == 0.0:                                                ok = False
                if (min_wr      is not None) and not (cur_wr      >= float(min_wr)):             ok = False
                if (min_trades  is not None) and not (cur_trades  >= int(min_trades)):           ok = False
                if not ok:
                    try:
                        logging.info(
                            "[Validation] Gate FAILED: "
                            "Sharpe=%.3f(>=%s), Sortino=%.3f(>=%s), PF=%s(>=%s, deny_inf=%s), "
                            "MaxDD=%.4f(>=%s, deny_0=%s), WR=%.3f(>=%s), Trades=%d(>=%s)",
                            cur_sharpe,  min_sharpe,
                            cur_sortino, min_sortino, # noqa: E272
                            ("inf" if np.isinf(cur_pf) else f"{cur_pf:.4f}"), min_pf, str(deny_inf_pf),
                            cur_dd, max_dd_at_most, str(deny_zero_dd),
                            cur_wr, min_wr,
                            cur_trades, str(min_trades),
                        )
                    except Exception:
                        logging.info("[Validation] Gate FAILED (see metrics.json for details)")
                return ok

            # Если гейт не пройден — просто пропускаем обновление best
            if not _passes_gate(metrics, gate):
                continue

            logging.info(
                "[Validation] текущие метрики прошли валидационный гейт"
            )

            # Корректное сравнение tuple/float/None
            def _is_better(current, best):
                if best is None:
                    return True
                return current > best

            if _is_better(val_metric, best_val_metric):
                logging.info(
                    f"[Validation] New best model. Current metric: {val_metric} > Previous best: {best_val_metric}"
                )
                best_val_metric = val_metric
                best_validation = dict(metrics)  # store full snapshot
                best_episode = int(ep)
                best_path = os.path.join(models_dir, "best.pth")
                agent.save_model(best_path)

                # Human-friendly sidecar with selection info
                try:
                    # Сериализуем tuple корректно для JSON/человеческого чтения
                    _val_serializable = (
                        list(val_metric) if isinstance(val_metric, tuple) else float(val_metric)
                    )
                    best_info = {
                        "metric_name": cfg.trainlog.val_selection_metrics,
                        "direction": val_direction,
                        "min_delta": val_min_delta,
                        "value": _val_serializable,
                        "value_primary": (val_metric[0] if isinstance(val_metric, tuple) else float(val_metric)),
                        "episode": int(ep),
                        "saved_path": "best.pth",
                        "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    with open(os.path.join(models_dir, "best_model_info.json"), "w", encoding="utf-8") as bf:
                        json.dump(best_info, bf, indent=2, default=_numpy_json_default)
                except Exception as e:
                    logging.warning("Failed to write best_model_info.json: %s", e)

                # Для логирования используем оригинальные значения, а не инвертированные
                human_readable_metrics = {k: metrics.get(k, "N/A") for k in sel_keys}
                logging.info(
                    "New BEST model found at episode %d (saved to %s)",
                    ep, best_path
                )
                logging.info(
                    " -> Selection criteria: %s",
                    cfg.trainlog.val_selection_metrics
                )
                logging.info(" -> New best values: %s", human_readable_metrics)

                # Сбрасываем счётчик, т.к. нашли улучшение
                no_improvement_count = 0
            else:
                # Улучшения не было, увеличиваем счётчик
                no_improvement_count += 1

        # --- Проверка условия досрочной остановки ---
        if val_env and ep % cfg.trainlog.val_freq == 0 and best_episode is not None:
            if no_improvement_count >= early_stopping_patience:
                logging.info(
                    f"[Early Stopping] No improvement for {no_improvement_count} validation checks "
                    f"(patience={early_stopping_patience}). Stopping training at episode {ep}."
                )
                break # Выход из основного цикла обучения

    final_path = os.path.join(models_dir, "final.pth")
    agent.save_model(final_path)
    logging.info(f"Final model saved: {final_path}")
    plot_training_progress(history, plots_dir, cfg.trainlog.plot_moving_avg_window)

    train_env.close()

    if val_env:
        val_env.close()

    # --- Aggregate and persist must-have bundle artifacts in models_dir ---
    bundle_enabled = getattr(bundle_cfg, "enable", True)
    if bundle_enabled:
        # 1) metrics.json
        # NEW: Get human-readable values for the best metric tuple.
        # This ensures that the final summary log and metrics.json contain the correct, non-inverted values.
        sel_keys = cfg.trainlog.val_selection_metrics if isinstance(cfg.trainlog.val_selection_metrics, (list, tuple)) else [cfg.trainlog.val_selection_metrics]
        best_val_metric_human = tuple(best_validation.get(k, None) for k in sel_keys) if best_validation else None

        bundle_metrics = {
            "val_selection_metric": cfg.trainlog.val_selection_metrics,
            "val_selection_direction": val_direction,
            "val_min_delta": val_min_delta,
            # FIX: Store human-readable values, not the internal inverted ones.
            "best_val_metric": best_val_metric_human,
            "best_episode": best_episode,
            "best_validation": best_validation,
            "last_validation": last_val_metrics,
            "history": {
                "episodes": history.get("episodes", []),
                "mean_rewards_N": history.get("mean_rewards_N", []),
                "mean_losses_N": history.get("mean_losses_N", []),
                "mean_win_rates_N": history.get("mean_win_rates_N", []),
            },
        }
        with open(os.path.join(models_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(bundle_metrics, f, indent=2, default=_numpy_json_default)
        # 2) requirements-lock.txt
        _dump_requirements_lock(os.path.join(models_dir, "requirements-lock.txt"))
        # 3) torch_env.txt
        _dump_torch_env(os.path.join(models_dir, "torch_env.txt"))
        # 4) env_flags.json
        _dump_env_flags(cfg, os.path.join(models_dir, "env_flags.json"))
        # 5) data_manifest.json
        data_manifest = _build_data_manifest(cfg)
        with open(os.path.join(models_dir, "data_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(data_manifest, f, indent=2)

        # 6) optional: snapshot кода
        if getattr(bundle_cfg, "include_code_snapshot", False):
            snapshot_paths = getattr(bundle_cfg, "code_snapshot_paths", [])
            snap_path = os.path.join(models_dir, "code_snapshot.tar.gz")
            try:
                with tarfile.open(snap_path, "w:gzip") as tar:
                    for p in snapshot_paths:
                        if os.path.exists(p):
                            tar.add(p, arcname=os.path.basename(p))
                logging.info(f"Code snapshot saved: {snap_path}")
            except Exception as e:
                logging.warning(f"Code snapshot failed: {e}")

        # 7) MANIFEST.json (file list + sha256)
        file_entries = []
        for fn in sorted(os.listdir(models_dir)):
            fp = os.path.join(models_dir, fn)
            if os.path.isfile(fp):
                try:
                    file_entries.append({"path": fn, "sha256": _sha256(fp), "bytes": os.path.getsize(fp)})
                except Exception as e:
                    logging.warning(f"MANIFEST: failed to hash {fn}: {e}")
        manifest = {
            "schema": "MODEL_BUNDLE_V1",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code_commit": _git_head_sha(),
            "files": file_entries,
            "links": {
                "dataset_sha256": [x["sha256"] for x in data_manifest.get("datasets", [])],
                "norm_stats": "norm_stats.json",
                "config_train": "config_train.json",
            },
        }
        with open(os.path.join(models_dir, "MANIFEST.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logging.info(f"Bundle manifest written: {os.path.join(models_dir, 'MANIFEST.json')}")

        # 8) Вшиваем META в best.pth / final.pth
        def _attach_meta_to_checkpoint(path: str, meta: dict):
            try:
                ckpt = torch.load(path, map_location="cpu")
                if isinstance(ckpt, dict) and "state_dict" in ckpt:
                    ckpt["meta"] = meta
                else:
                    ckpt = {"state_dict": ckpt, "meta": meta}
                torch.save(ckpt, path)
                logging.info(f"Attached meta to: {path}")
            except Exception as e:
                logging.warning(f"Failed to attach meta to {path}: {e}")

        norm_sha = _sha256(os.path.join(models_dir, "norm_stats.json"))
        cfg_sha  = _sha256(os.path.join(models_dir, "config_train.json"))
        ds_list  = [x["sha256"] for x in data_manifest.get("datasets", [])]
        meta = {
            "dataset_sha256_list": ds_list,
            "norm_stats_sha256": norm_sha,
            "config_train_sha256": cfg_sha,
            "code_commit": _git_head_sha(),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        best_path  = os.path.join(models_dir, "best.pth")
        final_path = os.path.join(models_dir, "final.pth")
        if os.path.exists(best_path):
            _attach_meta_to_checkpoint(best_path, meta)
        if os.path.exists(final_path):
            _attach_meta_to_checkpoint(final_path, meta)

        # ── Краткое резюме метрик в лог (для аудита без открытия файлов) — только валидация
        try:
            _metrics_path = os.path.join(models_dir, "metrics.json")
            with open(_metrics_path, "r", encoding="utf-8") as _mf:
                _m = json.load(_mf)
            _best = _m.get("best_val_metric")
            _sel  = _m.get("val_selection_metric")
            _best_tuple = tuple(_best) if isinstance(_best, list) else (_best,)
            logging.info("[SUMMARY] best_val_metric=%s  val_selection_metric=%s", _best_tuple, _sel)
        except Exception as e:
            logging.warning(f"[SUMMARY] Failed to log metrics summary: {e}")


if __name__ == "__main__":
    # cfg_arg = load_config(sys.argv[1]) if len(sys.argv) > 1 else default_cfg
    # main(cfg=cfg_arg)
    # Вызываем main без аргументов, т.к. логика загрузки перенесена внутрь
    main()