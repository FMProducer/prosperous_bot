# test_agent.py
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from agent import D3QN_PER_Agent
from config import MasterConfig
from config import cfg as default_cfg
from trading_environment import TradingEnvironment
from train import evaluate_agent
from utils import (
    calculate_normalization_stats,
    load_config,
    load_npz_dataset,
    select_and_arrange_channels,
    set_random_seed,
)

ACTION_COLORS = {
    0: "gray",
    1: "green",
    2: "blue",
    3: "red",
}
ACTION_LABELS = {
    0: "Hold/Wait",
    1: "Long",
    2: "Short",
    3: "Close",
}


def record_all_sessions(env: TradingEnvironment, agent: D3QN_PER_Agent, sequences_meta: list, cfg: MasterConfig):
    sessions = []
    total = min(cfg.trainlog.num_val_ep, len(sequences_meta))

    all_indices = list(range(len(sequences_meta)))
    np.random.shuffle(all_indices)
    for idx_ep in range(total):
        idx = all_indices[idx_ep]
        orig_key, _ = sequences_meta[idx]
        obs, info = env.reset(seed=None, options={"forced_index": idx})
        prices = []
        executed_actions = []
        prev_position = 0

        for minute_step in range(cfg.seq.pre_signal_len - 1):
            price = env.current_seq[minute_step, env.data_channels.index("close")]
            prices.append(price)

        for minute_step in range(cfg.seq.agent_session_len):
            raw_action = agent.select_action(obs, training=False)
            prev_position = env.position
            next_obs, reward, done, _, info = env.step(raw_action)
            if raw_action == 1 and prev_position == 0:
                effective_action = 1
            elif raw_action == 2 and prev_position == 0:
                effective_action = 2
            elif raw_action == 3 and prev_position != 0:
                effective_action = 3
            else:
                effective_action = 0

            price = env.current_seq[cfg.seq.pre_signal_len - 1 + minute_step, env.data_channels.index("close")]
            prices.append(price)
            executed_actions.append(effective_action)

            if env.last_step and prev_position != 0:
                effective_action = 3
                executed_actions[-1] = effective_action
            if env.last_step and prev_position == 0:
                effective_action = 0
                executed_actions[-1] = effective_action

            obs = next_obs

        if cfg.trainlog.plot_metric == "pnl":
            metric = info.get("episode_realized_pnl", 0.0)
        else:
            metric = info.get("episode_win_rate", 0.0)
        sessions.append(
            {
                "orig_key": orig_key,
                "prices": np.array(prices)[-cfg.seq.agent_session_len * 2 :],
                "actions": executed_actions,
                "metric": metric,
            }
        )

    return sessions


def plot_sessions(sessions: list, title_suffix: str, cfg: MasterConfig):
    os.makedirs(cfg.paths.plot_dir, exist_ok=True)

    for i, sess in enumerate(sessions, 1):
        prices = sess["prices"]
        actions = sess["actions"]
        ticker, dt = sess["orig_key"]
        dt_str = dt.strftime("%Y-%m-%d %H:%M")

        x = np.arange(len(prices))
        y_min, y_max = prices.min(), prices.max()
        y0 = y_min - 0.05 * (y_max - y_min)

        plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")

        plt.plot(x, prices, color="green", label="Price", linewidth=2)
        plt.axvline(x=cfg.seq.agent_session_len, color="magenta", linestyle="--", linewidth=1.5, label="Session Start")
        for t, a in enumerate(actions):
            plt.scatter(cfg.seq.agent_session_len + t, y0, s=100, c=ACTION_COLORS[a], marker="s")
        handles = [
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=col, markersize=10, label=ACTION_LABELS[a])
            for a, col in ACTION_COLORS.items()
        ]
        price_line = plt.Line2D([0], [0], color="green", linewidth=2, label="Price")
        handles.append(price_line)
        plt.legend(handles=handles, loc="upper left", fontsize=10)

        plt.title(
            f"{title_suffix} Session N_{i}: {ticker} {dt_str}  {cfg.trainlog.plot_metric.upper()}={sess['metric']:.4f}",
            fontsize=14,
        )
        plt.xlabel("Time (minutes)")
        plt.ylabel("Price")
        plt.tight_layout()

        fname = f"{title_suffix.lower()}_session_{i}.png"
        out = os.path.join(cfg.paths.plot_dir, fname)
        plt.savefig(out, dpi=300)
        plt.close()
        logging.info(f"Saved plot: {fname}")


def setup_logging(cfg: MasterConfig):
    os.makedirs(cfg.paths.log_dir, exist_ok=True)
    log_path = os.path.join(cfg.paths.log_dir, "test_agent.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    logging.info("Logging initialized for test_agent.py")


def prepare_sequences(cfg: MasterConfig) -> list:
    sequences = []
    raw_list = load_npz_dataset(
        file_path=cfg.paths.test_data_path,
        name_dataset="Test",
        plot_dir=cfg.paths.plot_dir,
        debug_max_size=cfg.debug.debug_max_size_data,
        plot_examples=cfg.data.plot_examples,
        plot_channel_idx=cfg.data.plot_channel_idx,
        pre_signal_len=cfg.seq.pre_signal_len,
    )
    for original_key, arr in tqdm(raw_list, desc="Selecting and arrange channels for Test", leave=False):
        transformed = select_and_arrange_channels(
            arr,
            cfg.data.expected_channels,
            cfg.data.data_channels,
        )
        if transformed is not None:
            sequences.append((original_key, transformed))
        else:
            logging.warning(f"Sequence {original_key} skipped - channel mismatch")
    return sequences


def init_agent(model_path: str, cfg: MasterConfig, cache_path: str = None) -> D3QN_PER_Agent:
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
        backtest_cache_path=cache_path,
    )
    agent.load_model(model_path)
    return agent


def test(cfg: MasterConfig = None):
    setup_logging(cfg)
    set_random_seed(cfg.random_seed)

    test_seqs = prepare_sequences(cfg)

    if not test_seqs:
        logging.error("No test sequences found. Exiting.")
        exit(1)

    raw_train = load_npz_dataset(
        file_path=cfg.paths.train_data_path,
        name_dataset="Train",
        plot_dir=cfg.paths.plot_dir,
        debug_max_size=cfg.debug.debug_max_size_data,
        plot_examples=0,
        plot_channel_idx=None,
        pre_signal_len=cfg.seq.pre_signal_len,
    )
    train_seqs = []
    for _, arr in tqdm(raw_train, desc="Selecting and arrange channels for Train", leave=False):
        sel = select_and_arrange_channels(arr, cfg.data.expected_channels, cfg.data.data_channels)
        if sel is not None:
            train_seqs.append(sel)
    train_stats = calculate_normalization_stats(
        train_seqs,
        cfg.data.data_channels,
        cfg.data.price_channels,
        cfg.data.volume_channels,
        cfg.data.other_channels,
    )

    env_kwargs = {
        "sequences": [seq for _, seq in test_seqs],
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

    test_env = TradingEnvironment(**env_kwargs)

    model_folder = os.path.join(cfg.paths.model_dir, sorted(os.listdir(cfg.paths.model_dir))[-1])
    model_name = "final.pth" if cfg.debug.use_final_model else "best.pth"
    model_path = os.path.join(model_folder, model_name)
    # best_path = os.path.join(model_folder, "best.pth")
    # model_path = best_path if os.path.exists(best_path) else os.path.join(model_folder, "final.pth")
    agent = init_agent(model_path, cfg)

    all_sessions = record_all_sessions(test_env, agent, test_seqs, cfg)

    all_sessions_sorted = sorted(all_sessions, key=lambda x: x["metric"])
    top_n = cfg.trainlog.plot_top_n
    worst = all_sessions_sorted[:top_n]
    best = all_sessions_sorted[-top_n:]

    plot_sessions(best, "Profitable", cfg)
    plot_sessions(worst, "Unprofitable", cfg)

    test_metrics = evaluate_agent(
        test_env, agent, min(len(test_seqs), cfg.trainlog.num_val_ep), "Test", None, cfg.global_env_seed
    )

    plot_folder = os.path.join(cfg.paths.plot_dir, sorted(os.listdir(cfg.paths.model_dir))[-1])
    plt.figure(figsize=(10, 6))
    sns.histplot(
        test_metrics[cfg.trainlog.test_selection_metrics],
        kde=True,
        bins=30,
        color="tab:blue",
        edgecolor="black",
        alpha=0.7,
    )
    plt.title(f"Distribution of {cfg.trainlog.test_selection_metrics}", fontsize=16, fontweight="bold")
    plt.xlabel(f"{cfg.trainlog.test_selection_metrics} per Episode", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    save_plot_path = os.path.join(plot_folder, f"test_agent_{cfg.trainlog.test_selection_metrics}_distribution.png")
    plt.savefig(save_plot_path, dpi=300)
    plt.close()
    logging.info(f"Distribution plot saved: test_agent_{cfg.trainlog.test_selection_metrics}_distribution.png")

    test_env.close()


if __name__ == "__main__":
    test(cfg=load_config(sys.argv[1]) if len(sys.argv) > 1 else default_cfg)
