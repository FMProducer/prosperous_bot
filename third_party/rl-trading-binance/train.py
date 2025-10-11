# train.py
import logging
import os
import sys
import time
from collections import deque
from typing import Any, Dict, Tuple

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


def _rollout_vectorized_episode(train_env: DummyVecEnv, agent: D3QN_PER_Agent) -> Tuple[float, float, int, int, int, float]:
    """
    Один "батч-эпизод" на N средах (auto-reset внутри VecEnv):
    Возвращает:
      ep_reward_sum_avg, ep_win_rate_avg, transitions, term_count, trunc_count, avg_loss
    Примечание: при done[i] VecEnv уже отдал наблюдение следующего эпизода, поэтому
    для корректного next_state берём info["terminal_observation"] / "final_observation".
    См. предупреждение SB3 о VecEnv auto-reset.  # :contentReference[oaicite:4]{index=4}
    """
    obs_batch, _ = train_env.reset(seed=None, options=None)
    done_mask = np.zeros(train_env.num_envs, dtype=bool)
    ep_reward = np.zeros(train_env.num_envs, dtype=float)
    transitions = 0
    term_count, trunc_count = 0, 0
    win_rates: list[float] = []
    step_losses: list[float] = []
    while not done_mask.all():
        prev_done = done_mask.copy()
        actions = [agent.select_action(obs_batch[i], training=True) for i in range(train_env.num_envs)]
        next_obs_b, rewards, dones, trunc, infos = train_env.step(actions)
        for i in range(train_env.num_envs):
            info_i = infos[i] if isinstance(infos[i], dict) else {}
            # финальное наблюдение может называться "terminal_observation" (Gymnasium) или "final_observation" (старые обёртки)
            next_state = info_i.get("terminal_observation",
                                    info_i.get("final_observation", next_obs_b[i]))
            # Для бустрэппинга: terminal = terminated (а не truncated).
            # В DQN next_Q зануляется только для истинно терминальных состояний.  # :contentReference[oaicite:5]{index=5}
            terminal = bool(dones[i]) and not bool(trunc[i])
            agent.store_experience(obs_batch[i], actions[i], float(rewards[i]), next_state, terminal)
            if bool(dones[i]) and isinstance(infos[i], dict):
                wr = infos[i].get("episode_win_rate", None)
                if wr is not None:
                    win_rates.append(float(wr))
            # считать "terminated" только когда done и НЕ truncated
            if bool(dones[i]) and not bool(trunc[i]):
                term_count += 1
            if bool(trunc[i]):
                trunc_count += 1
        # Накапливать награды только для тех подсред, которые ещё не были завершены до этого шага
        for i in range(train_env.num_envs):
            if not prev_done[i]:
                ep_reward[i] += float(rewards[i])
        obs_batch = next_obs_b
        done_mask |= dones
        transitions += train_env.num_envs
        # Частота обучения при VecEnv: один learn() на каждый векторный шаг (паритет с single-env по шагам)
        loss = agent.learn()
        if loss is not None:
            step_losses.append(float(loss))
    ep_reward_mean = float(np.mean(ep_reward)) if ep_reward.size else 0.0
    ep_win_rate_mean = float(np.mean(win_rates)) if win_rates else 0.0
    avg_loss = float(np.mean(step_losses)) if step_losses else 0.0
    return ep_reward_mean, ep_win_rate_mean, transitions, term_count, trunc_count, avg_loss


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
    for ep in tqdm(range(1, num_episodes + 1), total=num_episodes + 1, desc=f"{split_name} in episodes", leave=False):
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
    cfg = cfg or default_cfg
    timestamp = time.strftime("date_%Y%m%d_time_%H%M%S")
    session_name = f"{cfg.project_name}_{timestamp}"
    setup_logging(session_name, cfg)
    set_random_seed(cfg.random_seed)
    if cfg.device.device.type == "cuda":
        torch.backends.cudnn.benchmark = cfg.perf.cudnn_benchmark

    models_dir = os.path.join(cfg.paths.model_dir, session_name)
    plots_dir = os.path.join(cfg.paths.plot_dir, session_name)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

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
    train_steps = 0

    train_env.reset(seed=cfg.global_env_seed)

    # --- Паритет ε между single-env и vec-env ---
    # Масштабируем продолжительность распада ε на число сред (однократно), чтобы ε(t) совпадал
    # при одинаковом числе собранных переходов. Защита от двойного масштабирования — флаг на агенте.
    if hasattr(cfg, "vec") and getattr(cfg.vec, "scale_epsilon_by_envs", True) and hasattr(agent, "eps_frames"):
        if not getattr(agent, "_eps_scaled_by_envs", False):
            try:
                original = int(agent.eps_frames)
                scale = max(1, int(getattr(cfg.vec, "num_envs", 1)))
                agent.eps_frames = original * scale
                agent._eps_scaled_by_envs = True
                logging.info(f"[VecEnv] Scaled epsilon frames by num_envs={scale}: {original} -> {agent.eps_frames}")
            except Exception as e:
                logging.warning(f"Failed to scale epsilon frames: {e}")
 
    counter = trange(1, cfg.trainlog.episodes + 1, desc="Training in episodes", leave=False)
    for ep in counter:
        if hasattr(train_env, "num_envs"):  # VecEnv путь
            ep_reward, ep_wr, transitions, term_cnt, trunc_cnt, ep_avg_loss = _rollout_vectorized_episode(train_env, agent)
            ep_losses = [] if ep_avg_loss == 0.0 else [ep_avg_loss]
            train_steps += transitions
            info = {"episode_win_rate": ep_wr, "terminated": term_cnt, "truncated": trunc_cnt, "transitions": transitions}
        else:
            obs, _ = train_env.reset(seed=None, options=None)
            ep_reward = 0.0
            ep_losses = []
            done = False
            while not done:
                action = agent.select_action(obs, training=True)
                next_obs, reward, done, truncated, info = train_env.step(action)
                # single-env: корректно отделяем truncated от terminated
                terminal = bool(done) and not bool(truncated)
                agent.store_experience(obs, action, reward, next_obs, terminal)
                loss = agent.learn()
                if loss is not None:
                    ep_losses.append(loss)
                obs = next_obs
                train_steps += 1
                ep_reward += reward
            info = {"episode_win_rate": info.get("episode_win_rate", 0.0), "terminated": int(done), "truncated": int(truncated), "transitions": train_steps}

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

        # Логируем ФАКТИЧЕСКИЙ ε агента (если доступен), иначе падать назад на формулу
        eps_current = None
        if hasattr(agent, "epsilon"):
            try:
                eps_current = float(agent.epsilon)
            except Exception:
                eps_current = None
        if eps_current is None and hasattr(agent, "get_epsilon") and callable(agent.get_epsilon):
            try:
                eps_current = float(agent.get_epsilon())
            except Exception:
                eps_current = None
        if eps_current is None:
            eps_current = float(
                agent.eps_end + (agent.eps_start - agent.eps_end)
                * np.exp(-train_steps / max(1.0, float(agent.eps_frames)))
            )
        history["epsilons"].append(eps_current)

        episode_win_rate_deque.append(info.get("episode_win_rate", 0.0))
        history["win_rates"].append(info.get("episode_win_rate", 0.0))
        mean_win_rate_N = float(np.mean(episode_win_rate_deque)) if episode_win_rate_deque else 0.0
        history["mean_win_rates_N"].append(mean_win_rate_N)
        # Доп. диагностические счётчики при VecEnv
        if hasattr(train_env, "num_envs"):
            counter.set_postfix_str(
                f"ε={eps_current:.4f} tr={info.get('transitions',0)} T={info.get('terminated',0)} U={info.get('truncated',0)}"
            )

        counter.desc = f"Training loss={avg_loss:.7f}, reward={ep_reward:.5f}"

        if val_env and ep % cfg.trainlog.val_freq == 0:
            metrics = evaluate_agent(
                val_env, agent, min(len(val_seqs), cfg.trainlog.num_val_ep), "Validation", ep, cfg.global_env_seed
            )
            val_metric = metrics[cfg.trainlog.val_selection_metrics]
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


if __name__ == "__main__":
    main(cfg=load_config(sys.argv[1]) if len(sys.argv) > 1 else default_cfg)
