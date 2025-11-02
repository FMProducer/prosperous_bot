# train.py
import logging
import os
import sys
import time
from collections import deque
from typing import Any, Dict
import hashlib, tarfile

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
from vec_env import DummyVecEnv
from trading_environment import TradingEnvironment
from utils import (
    calculate_normalization_stats,
    load_config,
    load_npz_dataset,
    select_and_arrange_channels,
    set_random_seed,
    setup_logging,
)

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
            lines.append(f"cudnn={getattr(cudnn, 'version', lambda: None)()}")
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
    num_episodes: int,
    split_name: str,
    current_episode: int = None,
    env_seed: int = 17,
) -> Dict[str, Any]:
    rewards = []
    pnls = []
    win_rates = []

    env.reset(seed=env_seed)
    logging.info(f"--- Starting Evaluation: {split_name}, episodes={num_episodes} ---")
    for ep in tqdm(range(1, num_episodes + 1), total=num_episodes, desc=f"{split_name} in episodes", leave=False):
        logging.info(f"--- Validation episode {ep}/{num_episodes} ---")
        obs, _ = env.reset(seed=None, options=None)
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.select_action(obs, training=False)
            obs, reward, done, _, info = env.step(action)
            ep_reward += reward

        pnl = info.get("episode_realized_pnl", 0.0)
        win_rate = info.get("episode_win_rate", 0.0)

        rewards.append(ep_reward)
        pnls.append(pnl)
        win_rates.append(win_rate)

    metrics = {
        f"{split_name}_mean_reward": np.mean(rewards),
        f"{split_name}_mean_pnl": np.mean(pnls),
        f"{split_name}_win_rate": np.mean(win_rates),
        f"{split_name}_all_pnls": pnls,
    }
    # для корректной генерации графиков в plot_test_distributions
    if split_name == "Test":
        metrics["Test_all_reward"] = rewards
        metrics["Test_all_win_rate"] = win_rates

    if current_episode is not None:
        episode_info = f" Ep_{current_episode}"
    else:
        episode_info = ""
    logging.info(
        f"---{episode_info} {split_name} Results: mean_reward={metrics[f'{split_name}_mean_reward']:.5f}, "
        f"Mean PnL: {metrics[f'{split_name}_mean_pnl']:.2f}, "
        f"Win rate: {metrics[f'{split_name}_win_rate']:.2%} ---"
    )
    logging.info(f"--- Finished Evaluation: {split_name} ---")
    return metrics


def process_data(raw_list, name_dataset, cfg: MasterConfig):
    seqs = []
    for _, arr in tqdm(raw_list, desc=f"Selecting and arrange channels for {name_dataset}", leave=False):
        sel = select_and_arrange_channels(arr, cfg.data.expected_channels, cfg.data.data_channels)
        if sel is not None:
            seqs.append(sel)
    return seqs


def plot_test_distributions(test_metrics: dict, plots_dir: str) -> None:
    os.makedirs(plots_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    logging.info(f"Starting to generate test distribution plots in: {plots_dir}")

    if "Test_all_pnls" in test_metrics and test_metrics["Test_all_pnls"]:
        plt.figure(figsize=(10, 6))
        sns.histplot(
            test_metrics["Test_all_pnls"],
            kde=True,
            bins=30,
            color="tab:blue",
            edgecolor="black",
            alpha=0.7,
        )
        plt.title("Distribution of Test PnL", fontsize=16, fontweight="bold")
        plt.xlabel("PnL per Episode", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        save_path = os.path.join(plots_dir, "test_pnl_distribution.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        logging.info(f"Saved PnL distribution plot: {save_path}")
    else:
        logging.warning("Test_all_pnls is missing or empty – skipping PnL plot.")

    if "Test_all_reward" in test_metrics and test_metrics["Test_all_reward"]:
        plt.figure(figsize=(10, 6))
        sns.histplot(
            test_metrics["Test_all_reward"],
            kde=True,
            bins=30,
            color="tab:green",
            edgecolor="black",
            alpha=0.7,
        )
        plt.title("Distribution of Test Rewards", fontsize=16, fontweight="bold")
        plt.xlabel("Reward per Episode", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        save_path = os.path.join(plots_dir, "test_reward_distribution.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        logging.info(f"Reward distribution plot saved: {save_path}")
    else:
        logging.warning("Test_all_reward is missing or empty – skipping Reward plot.")

    if "Test_all_win_rate" in test_metrics and test_metrics["Test_all_win_rate"]:
        plt.figure(figsize=(10, 6))
        sns.histplot(
            [wr * 100 for wr in test_metrics["Test_all_win_rate"]],
            kde=True,
            bins=30,
            color="tab:purple",
            edgecolor="black",
            alpha=0.7,
        )
        plt.title("Distribution of Test Win Rate (%)", fontsize=16, fontweight="bold")
        plt.xlabel("Win Rate (%) per Episode", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        save_path = os.path.join(plots_dir, "test_win_rate_distribution.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        logging.info(f"Win Rate distribution plot saved: {save_path}")
    else:
        logging.warning("Test_all_win_rate is missing or empty – skipping Win Rate plot.")


def main(cfg: MasterConfig = None):
    # Загружаем конфиг и модуль, чтобы иметь доступ ко всем переменным, включая bundle_cfg
    if len(sys.argv) > 1:
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
        cfg.paths.model_dir = os.path.join(base_out or "output", "alpha", "saved_models")
    if not hasattr(cfg.paths, "plot_dir") or cfg.paths.plot_dir in (None, ""):
        cfg.paths.plot_dir = os.path.join(base_out or "output", "alpha", "plots")
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
        train_env = DummyVecEnv(_make_train_env_fns(env_kwargs, cfg.vec.num_envs))
        logging.info(f"Vectorized train env: DummyVecEnv x{cfg.vec.num_envs}")
    else:
        train_env = TradingEnvironment(**env_kwargs)
    env_kwargs["sequences"] = val_seqs
    val_env = TradingEnvironment(**env_kwargs) if val_seqs else None

    agent = D3QN_PER_Agent(
        state_shape=(cfg.seq.num_features, cfg.seq.input_history_len, 1),
        action_dim=cfg.market.num_actions,
        cnn_maps=cfg.model.cnn_maps,
        cnn_kernels=cfg.model.cnn_kernels,
        cnn_strides=cfg.model.cnn_strides,
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

    best_val_metric = float("-inf")
    last_val_metrics: Dict[str, Any] = {}
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
                val_env, agent, min(len(val_seqs), cfg.trainlog.num_val_ep), "Validation", ep, cfg.global_env_seed
            )
            val_metric = metrics[cfg.trainlog.val_selection_metrics]
            last_val_metrics = metrics
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_path = os.path.join(models_dir, "best.pth")
                agent.save_model(best_path)
                logging.info(
                    f"New Best model by {cfg.trainlog.val_selection_metrics} = {val_metric:.2f}. Episode = {ep}: {best_path}"
                )

    final_path = os.path.join(models_dir, "final.pth")
    agent.save_model(final_path)
    logging.info(f"Final model saved: {final_path}")
    plot_training_progress(history, plots_dir, cfg.trainlog.plot_moving_avg_window)

    test_metrics: Dict[str, Any] = {}
    if test_seqs:
        env_kwargs["sequences"] = test_seqs
        test_env = TradingEnvironment(**env_kwargs)
        model_name = "final.pth" if cfg.debug.use_final_model else "best.pth"
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
            model_path = final_path
        agent.load_model(model_path)
        logging.info(f"Testing model: {model_path}")

        test_metrics = evaluate_agent(
            test_env, agent, min(len(test_seqs), cfg.trainlog.num_val_ep), "Test", None, cfg.global_env_seed
        )

        plot_test_distributions(test_metrics, plots_dir)
        logging.info("All test plots generated successfully.")

        test_env.close()
    else:
        logging.warning("Test data not found – skipping final evaluation.")

    train_env.close()

    if val_env:
        val_env.close()

    # --- Aggregate and persist must-have bundle artifacts in models_dir ---
    bundle_enabled = getattr(bundle_cfg, "enable", True)
    if bundle_enabled:
        # 1) metrics.json
        bundle_metrics = {
            "val_selection_metric": cfg.trainlog.val_selection_metrics,
            "best_val_metric": best_val_metric if best_val_metric != float("-inf") else None,
            "last_validation": last_val_metrics,
            "history": {
                "episodes": history.get("episodes", []),
                "mean_rewards_N": history.get("mean_rewards_N", []),
                "mean_losses_N": history.get("mean_losses_N", []),
                "mean_win_rates_N": history.get("mean_win_rates_N", []),
            },
            "test": test_metrics,
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

        # ── Краткое резюме метрик в лог (для аудита без открытия файлов)
        try:
            _metrics_path = os.path.join(models_dir, "metrics.json")
            with open(_metrics_path, "r", encoding="utf-8") as _mf:
                _m = json.load(_mf)
            _best = _m.get("best_val_metric")
            _sel  = _m.get("val_selection_metric")
            _test = _m.get("test", {}) if isinstance(_m, dict) else {}
            # Adjust keys to match what evaluate_agent produces
            _test_win_rate = _test.get("Test_win_rate")
            _test_mean_pnl = _test.get("Test_mean_pnl")

            logging.info(
                "[SUMMARY] best_val_metric=%s  val_selection_metric=%s  "
                "test.win_rate=%s  test.mean_pnl=%s",
                f"{_best:.4f}" if isinstance(_best, float) else _best,
                _sel,
                f"{_test_win_rate:.2%}" if isinstance(_test_win_rate, float) else _test_win_rate,
                f"{_test_mean_pnl:.2f}" if isinstance(_test_mean_pnl, float) else _test_mean_pnl,
            )
        except Exception as e:
            logging.warning(f"[SUMMARY] Failed to log metrics summary: {e}")


if __name__ == "__main__":
    # cfg_arg = load_config(sys.argv[1]) if len(sys.argv) > 1 else default_cfg
    # main(cfg=cfg_arg)
    # Вызываем main без аргументов, т.к. логика загрузки перенесена внутрь
    main()
