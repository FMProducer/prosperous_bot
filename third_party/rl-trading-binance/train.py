# train.py 201125
import logging
import os
import sys
import time
from collections import deque, defaultdict
from typing import Any, Dict, List, Optional
import hashlib, tarfile
import datetime as dt
from functools import partial
from pathlib import Path
# CuBLAS: детерминизм требует рабочего пространства; задаём до импорта torch
if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import platform
import json
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    setup_logging,
    select_and_arrange_channels,
    set_random_seed,
    create_validation_episodes,
    load_config,
    load_npz_dataset,
    create_walk_forward_folds,
    load_and_prep_data_from_source,
    calculate_normalization_stats,
) # noqa: F401

class TopKCheckpointManager:
    """
    Менеджер для сохранения топ-K лучших чекпоинтов с метаданными. 
    
    Автоматически удаляет худшие чекпоинты при превышении лимита top_k.
    Сохраняет полные метрики в JSON для последующего анализа.
    """
    
    def __init__(self, save_dir: str, top_k: int = 10, metric_key: str = "Validation_sortino", mode: str = "max"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.metric_key = metric_key
        self.mode = mode
        self.checkpoints = []  # List of (metric_value, episode, filepath, metrics_dict)
        
        logging.info(f"TopKCheckpointManager initialized: save_dir={save_dir}, top_k={top_k}, metric={metric_key}, mode={mode})")
    
    def save_checkpoint(self, agent, episode: int, metrics: Dict[str, Any]) -> bool:
        """
        Сохраняет чекпоинт, если он входит в топ-K по целевой метрике. 
        
        Returns:
            bool: True если чекпоинт сохранён, False если отклонён
        """
        
        metric_value = metrics.get(self.metric_key, None)
        
        if metric_value is None:
            logging.warning(f"Metric '{self.metric_key}' not found in validation metrics. Skipping checkpoint save.")
            return False
        
        try:
            metric_value = float(metric_value)
        except (TypeError, ValueError):
            logging.warning(f"Metric '{self.metric_key}' has non-numeric value: {metric_value}. Skipping.")
            return False
        
        # Создать имя файла с ключевыми метриками
        sortino = metrics.get("Validation_sortino", 0.0)
        sharpe = metrics.get("Validation_sharpe", 0.0)
        trades = metrics.get("Validation_trades", 0)
        
        filename = (
            f"checkpoint_ep{episode:05d}_"
            f"sortino{sortino:.3f}_"
            f"sharpe{sharpe:.3f}_"
            f"trades{trades:.0f}.pth"
        )
        filepath = self.save_dir / filename
        
        # Сохранить модель
        try:
            agent.save_model(str(filepath))
        except Exception as e:
            logging.error(f"Failed to save model checkpoint: {e}")
            return False
        
        # Сохранить метаданные отдельно в JSON
        metadata_path = filepath.with_suffix('.json')
        try:
            with open(metadata_path, 'w') as f:
                json.dump({
                    'episode': episode,
                    'metrics': metrics,
                    'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }, f, indent=2, default=_numpy_json_default)
        except Exception as e:
            logging.warning(f"Failed to save checkpoint metadata: {e}")
        
        # Добавить в список и отсортировать
        self.checkpoints.append((metric_value, episode, filepath, metrics))
        self.checkpoints.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
        
        # Удалить худшие чекпоинты, если превышен лимит
        if len(self.checkpoints) > self.top_k:
            to_remove = self.checkpoints[self.top_k:]
            for _, _, fpath, _ in to_remove:
                try:
                    fpath.unlink(missing_ok=True)
                    fpath.with_suffix('.json').unlink(missing_ok=True)
                    logging.debug(f"Removed old checkpoint: {fpath.name}")
                except Exception as e:
                    logging.warning(f"Failed to remove checkpoint {fpath}: {e}")
            
            self.checkpoints = self.checkpoints[:self.top_k]
        
        logging.info(
            f"[TopK] Saved checkpoint (rank {len([c for c in self.checkpoints if c[0] >= metric_value])}/{len(self.checkpoints)}): "
            f"{filename} | {self.metric_key}={metric_value:.4f}"
        )
        
        return True
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Возвращает путь к лучшему чекпоинту"""
        return self.checkpoints[0][2] if self.checkpoints else None

def compute_norm_stats(npz_path: str, cfg: MasterConfig, norm_stats_path: str) -> dict:
    """
    Вычисляет mean/std для каждого актива (тикера) в файле NPZ.
    Берет случайную выборку `num_samples_per_asset` для каждого тикера для ускорения.
    Сохраняет результат в `norm_stats.json`.
    Returns: {'TICKER1': {'mean': [], 'std': []}, 'TICKER2': ...}
    """
    num_samples_per_asset = cfg.data.norm_num_samples_per_asset
    seed = cfg.data.norm_seed
    np.random.seed(seed)
    try:
        d = np.load(npz_path, allow_pickle=True)
    except FileNotFoundError:
        logging.error(f"Файл данных не найден: {npz_path}")
        return {}

    data_keys = [k for k in d.files if not k.startswith('_')]
    
    # Группировка ключей по тикерам
    asset_keys = defaultdict(list)
    for key in data_keys:
        try:
            asset_name = key.split('_')[0]
            asset_keys[asset_name].append(key)
        except IndexError:
            logging.warning(f"Не удалось извлечь имя актива из ключа: {key}")
            continue
            
    all_stats = {}
    logging.info(f"Найдено {len(asset_keys)} активов. Расчет статистик...")

    for asset, keys in tqdm(asset_keys.items(), desc="Computing stats per asset"):
        if len(keys) > num_samples_per_asset:
            sample_keys = np.random.choice(keys, num_samples_per_asset, replace=False)
        else:
            sample_keys = keys
        
        try:
            # Загружаем данные только для выбранных ключей этого ассета
            asset_data = np.stack([d[key].astype(np.float32) for key in sample_keys], axis=0)
            
            if asset_data.ndim == 3 and asset_data.shape[0] > 0: # (N, L, C)
                means = np.mean(asset_data, axis=(0, 1))
                stds = np.std(asset_data, axis=(0, 1)) + 1e-8
                all_stats[asset] = {'mean': means.tolist(), 'std': stds.tolist()}
            else:
                logging.warning(f"Неверная форма или пустые данные для ассета {asset}: {asset_data.shape}")
        except Exception as e:
            logging.error(f"Ошибка при обработке ассета {asset}: {e}")


    d.close()
    
    # Сохраняем в файл
    Path(norm_stats_path).write_text(json.dumps(all_stats, indent=2))
    logging.info(f"Сохранены статистики для {len(all_stats)} активов в {norm_stats_path}")
    return all_stats

def load_and_prep_data(npz_path: str, split_name: str, norm_stats: dict, allowed_assets: Optional[List[str]] = None) -> tuple[list, list]:
    """
    Загружает NPZ, применяет Z-нормализацию для каждого актива отдельно, решейпит в (C, L, 1).
    Требует предоставления `norm_stats` с данными для каждого актива.
    Фильтрует активы по списку `allowed_assets`, если он предоставлен.
    Returns: list of np.arrays (samples), list of keys.
    """
    if not npz_path or not os.path.exists(npz_path):
        logging.warning(f"{split_name} data file not found or path not specified: {npz_path}")
        return [], []

    if not norm_stats:
        raise ValueError(f"norm_stats не предоставлен для {split_name}, но он обязателен.")

    d = np.load(npz_path, allow_pickle=True)
    data_keys = [k for k in d.files if not k.startswith('_')]
    sequences = []
    valid_keys = []
    logging.info(f"Загрузка {len(data_keys)} последовательностей из {split_name}...")
    
    for key in tqdm(data_keys, desc=f"Normalizing {split_name}"):
        try:
            asset_name = key.split('_')[0]
        except IndexError:
            logging.warning(f"Пропуск ключа с некорректным форматом: {key}")
            continue
        
        # Фильтрация по списку разрешенных активов
        if allowed_assets and asset_name not in allowed_assets:
            continue

        asset_specific_stats = norm_stats.get(asset_name)
        if asset_specific_stats is None:
            if not allowed_assets or asset_name in allowed_assets:
                 logging.warning(f"Пропуск ключа '{key}', т.к. статистики для актива '{asset_name}' не найдены.")
            continue

        means = np.array(asset_specific_stats['mean'])
        stds = np.array(asset_specific_stats['std'])
        
        seq = d[key].astype(np.float32)
        if seq.shape[1] != len(means):
            logging.error(f"Ошибка размерности для ключа {key}: ожидалось {len(means)} каналов, получено {seq.shape[1]}")
            continue
            
        # Z-norm по каждому каналу
        seq = (seq - means) / stds
        # Reshape для CNN: (L, C) -> (C, L, 1)
        seq = seq.T
        seq = np.expand_dims(seq, -1)
        sequences.append(seq)
        valid_keys.append(key)
    
    d.close()
    if sequences:
        logging.info(f"Подготовлено {len(sequences)} последовательностей, форма: {sequences[0].shape}")
    return sequences, valid_keys


def make_env(env_kwargs: dict):
    """Helper function to create a TradingEnvironment, designed to be picklable."""
    return TradingEnvironment(**env_kwargs)

def _rollout_vectorized_episode(train_env: DummyVecEnv, agent: D3QN_PER_Agent, agent_session_len: int):
    """
    Один "батч-эпизод" на N средах:
    - параллельно идём до завершения каждой под-среды (autoreset внутри VecEnv),
    - накапливаем опыт и возвращаем средний суммарный reward за эпизоды.
    """
    reset_out = train_env.reset()
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs_batch, _ = reset_out   # (obs, infos)
    else:
        obs_batch = reset_out      # на случай старого API
    done_mask = np.zeros(train_env.num_envs, dtype=bool)
    ep_reward = np.zeros(train_env.num_envs, dtype=float)
    ep_reward_per_episode = []
    win_rates = []
    ep_losses = []
    last_info = {}
    transitions_count = 0
    episode_infos = []

    # FIX: создаем 4 прогресс-бара по ЭПИЗОДАМ, а не по шагам
    pbars = [
        tqdm(
            total=0,            # будем увеличивать total динамически
            desc=f"Env {i}",
            position=i + 1,     # строки под основным training-bar
            unit="ep",
            leave=False,
        )
        for i in range(train_env.num_envs)
    ]

    while not done_mask.all():
        actions = [agent.select_action(obs_batch[i], training=True) for i in range(train_env.num_envs)]
        next_obs_b, rewards, dones, trunc, infos = train_env.step(actions)

        transitions_count += train_env.num_envs  # один переход на каждую среду

        for i in range(train_env.num_envs):
            # Корректный next_state при done: брать финальное наблюдение из info
            # В векторизованном режиме всегда используем batched next_obs_b[i]
            # чтобы гарантировать одинаковую форму состояний в буфере.
            next_state = next_obs_b[i]
            if dones[i] and isinstance(infos[i], dict):
                next_state = infos[i].get("terminal_observation", infos[i].get("final_observation", next_state))

            # Накапливаем награды для каждого env отдельно
            ep_reward[i] += float(rewards[i])

            # Жёстко приводим и state, и next_state к плоскому float32-вектору.
            state_vec = np.asarray(obs_batch[i], dtype=np.float32).reshape(-1)
            next_state_vec = np.asarray(next_state, dtype=np.float32).reshape(-1)

            agent.store_experience(state_vec, actions[i], float(rewards[i]), next_state_vec, bool(dones[i])) # noqa: E501
            if bool(dones[i]) and isinstance(infos[i], dict):
                episode_infos.append(infos[i])
                ep_reward_per_episode.append(ep_reward[i])
                wr = infos[i].get("episode_win_rate", None)
                if wr is not None:
                    # считаем завершённый эпизод для этого env
                    pbars[i].total += 1
                    pbars[i].update(1)
                    pbars[i].set_postfix_str(f"R={ep_reward[i]:.3f} WR={wr:.2%}")
                    win_rates.append(float(wr))
                
                ep_reward[i] = 0.0

        last_info = infos[0] if len(infos) > 0 and isinstance(infos[0], dict) else {}
        # Шаги больше не рисуем: бары будут обновляться только при завершении эпизода.
        prev_done = done_mask.copy()

        # Накапливать награды только для тех подсред, которые ещё не были завершены до этого шага
        obs_batch = next_obs_b # noqa: F841
        done_mask |= dones  # эпизод для каждой под-среды
        # В каждом "батч-шаге" получаем по одному переходу на среду
        for _ in range(train_env.num_envs):
            agent.increment_step()

        # Один вызов обучения на batched шаг.
        loss = agent.learn()
        if loss is not None:
            ep_losses.append(loss)

    # Убедимся, что все прогресс-бары закрыты в конце
    for pbar in pbars:
        pbar.close()

    # --- Bankruptcy Rate Metric ---
    bankruptcy_count = sum(1 for info in episode_infos if info.get('bankruptcy', False))
    total_episodes = len(episode_infos)
    if total_episodes > 0:
        bankruptcy_rate = bankruptcy_count / total_episodes
        logging.info(f"Bankruptcy Rate: {bankruptcy_rate:.2%}")

    avg_reward = float(np.mean(ep_reward_per_episode)) if ep_reward_per_episode else 0.0
    avg_win_rate = float(np.mean(win_rates)) if win_rates else 0.0
    avg_loss = np.mean(ep_losses) if ep_losses else 0.0
    # Aggregate infos from all sub-environments. A simple approach is to merge them,
    # or return the info from the first completed environment. Here we just return the last one.
    return avg_reward, avg_win_rate, transitions_count, avg_loss, last_info


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
        lines.append(f"python={platform.python_version}()")
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
    elif isinstance(obj, (np.floating, float)):
        if np.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        if np.isnan(obj):
            return "nan"
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
    keys: list = None,
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
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0
    if hasattr(agent, "mc_enable"):
        agent.mc_enable = False

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
    bankruptcy_episodes = 0
    
    # Дополнительные метрики
    all_trades_info = []
    total_commission = 0.0
    long_trades = 0
    short_trades = 0
    holding_times = []
    total_bars_processed = 0
    start_time = time.time()

    num_eval_episodes = len(env.sequences)

    for i in range(int(num_eval_episodes)):
        obs, _ = env.reset(options={"forced_index": i})
        done = False
        ep_reward = 0.0
        ep_trades = 0
        ep_wins   = 0
        ep_trade_pnls: list[float] = []
        is_bankrupt = False
        
        # ИСПРАВЛЕНО: Извлекаем дату начала семпла из ключа, а не используем заглушку
        signal_dt_for_step = dt.datetime(2000, 1, 1, 0, 0) # Fallback
        ticker_name = "UNKNOWN"
        if keys and i < len(keys):
            try:
                key_parts = keys[i].split('_')
                ticker_name = key_parts[0]
                # Ожидаемый формат ключа: TICKER_STARTISO_ENDISO
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
                
                # Сбор расширенной информации о сделках
                all_trades_info.append(info)
                total_commission += info.get('trade_commission', 0.0)
                direction = info.get('direction', '')
                if direction == 'LONG':
                    long_trades += 1
                elif direction == 'SHORT':
                    short_trades += 1
                
                if 'holding_duration_bars' in info:
                    holding_times.append(info['holding_duration_bars'])

        # завершение эпизода
        if is_bankrupt:
            bankruptcy_episodes += 1
        total_reward += ep_reward
        total_trades += ep_trades
        total_correct += ep_wins
        ep_pnls.append(sum(ep_trade_pnls))
        ep_rews.append(ep_reward)
        ep_wrs.append( ep_wins / max(1, ep_trades) if ep_trades else 0.0 )

    # --- Расчет дополнительных метрик ---
    total_duration = time.time() - start_time
    win_count = total_correct
    loss_count = total_trades - total_correct
    wr_ratio = total_correct / max(1, total_trades) if total_trades > 0 else 0.0

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
        
        # Calculate trading time in days
        bars_per_day = 1440  # 24 hours * 60 minutes
        trading_time_days = total_bars_processed / bars_per_day if bars_per_day > 0 else 0.0
        
        try:
            initial_balance = float(getattr(cfg.market, "initial_balance", 10000.0))
        except Exception:
            initial_balance = 10000.0
        
        # Calculate derived metrics
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
        avg_holding_time = 0.0
        max_holding_time = 0.0
        min_holding_time = 0.0
        trading_time_days = 0.0
        avg_win_size = 0.0
        avg_loss_size = 0.0
        win_loss_ratio = 0.0
        expectancy = 0.0
        commission_pct = 0.0
        roi_percent = 0.0
        roi_annualized = 0.0
        # wr_ratio already defined above

    pnl_per_day = net_pnl / trading_time_days if trading_time_days > 0 else 0.0

    # ИСПРАВЛЕНО: MeanReward в валидации — рассчитываем из normalized PnL сделок
    # (backtest_step возвращает reward=0.0, так как reward не используется в оценке)
    try:
        initial_balance = float(getattr(cfg.market, "initial_balance", 10_000.0))
    except Exception:
        initial_balance = 10_000.0
    
    # Средний normalized reward = sum(pnl) / initial_balance / episodes
    mean_reward = (sum(trade_pnls) / initial_balance) / max(1, num_eval_episodes) if trade_pnls else 0.0
    
    mean_pnl = (sum(trade_pnls) / max(1, total_trades)) if total_trades else 0.0
    
    pos_sum = sum(p for p in trade_pnls if p > 0)
    neg_sum = sum(p for p in trade_pnls if p < 0)
    profit_factor = (pos_sum / abs(neg_sum)) if neg_sum < 0 else float("inf")
    
    # --- Sharpe / Sortino ---
    if trade_pnls:
        denorm_pnls = np.array(trade_pnls, dtype=np.float64)
        equity = float(initial_balance)
        peak = float(initial_balance)
        max_dd = 0.0
        for pnl in denorm_pnls:
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
    if returns.size > 0:
        mean_r = float(returns.mean())
        std_r  = float(returns.std(ddof=1)) if returns.size > 1 else float(returns.std(ddof=0))
        downside = np.minimum(0.0, returns)
        downside = float(np.sqrt(np.mean(downside * downside)))
        sharpe   = (mean_r / std_r)      if std_r      > 1e-12 else 0.0
        sortino  = (mean_r / downside)   if downside   > 1e-12 else (float("inf") if mean_r > 0.0 else 0.0)
    else:
        sharpe, sortino = 0.0, 0.0

    bankruptcy_rate = bankruptcy_episodes / max(1, num_eval_episodes)

    # --- Расширенный лог ---
    logging.info(
        f"[{split_label}] Trades: {total_trades} (Long: {long_trades}, Short: {short_trades}, "
        f"Win: {win_count}, Loss: {loss_count}) | WinRate: {wr_ratio*100:.2f}% | PF: {profit_factor:.4f}"
    )
    logging.info(
        f"[{split_label}] Gross PnL: {gross_pnl:.2f} | Net PnL: {net_pnl:.2f} | "
        f"Commission: {total_commission:.2f} | Avg/Trade: {avg_pnl_per_trade:.2f}"
    )
    logging.info(
        f"[{split_label}] Best Trade: {best_trade:+.2f} | Worst Trade: {worst_trade:+.2f} | "
        f"MaxDD: {abs(max_dd)*100:.2f}% | Sharpe: {sharpe:.3f} | Sortino: {sortino:.3f}"
    )
    logging.info(
        f"[{split_label}] Avg Hold: {avg_holding_time:.2f} bars | "
        f"Min Hold: {min_holding_time} bars | Max Hold: {max_holding_time} bars"
    )
    # Enhanced metrics output
    logging.info(f"[{split_label}] Duration: {total_duration:.2f}s | Bars: {total_bars_processed} | "
                 f"Trading Days: {trading_time_days:.1f}")
    logging.info(f"[{split_label}] PnL/Day: {pnl_per_day:.2f} USDT | "
                 f"ROI: {roi_percent:.2f}% | Annualized ROI: {roi_annualized:.1f}%")
    logging.info(f"[{split_label}] Commission: {commission_pct:.1f}% of gross | "
                 f"Avg Win: {avg_win_size:.2f} | Avg Loss: {avg_loss_size:.2f} | "
                 f"W/L Ratio: {win_loss_ratio:.2f}")
    logging.info(f"[{split_label}] Expectancy/Trade: {expectancy:.2f} USDT")
    
    if exit_counts:
        logging.info("[%s] Exit reasons: %s", split_label,
                     {k:int(v) for k,v in sorted(exit_counts.items(), key=lambda x:(-x[1], x[0]))})
    if total_trades:
        logging.info("[%s] TSL hits: %d (%.2f%%)", split_label, tsl_hits, 100.0*tsl_hits/max(1,total_trades))

    # вернуть исходные режимы агента
    if old_eps is not None:
        agent.epsilon = old_eps
    if old_mc  is not None:
        agent.mc_enable = old_mc

    # сформировать словарь под выбор метрики в тренере
    L = split_label
    out: Dict[str,Any] = {
        f"{L}_mean_reward": float(mean_reward),
        f"{L}_mean_pnl":    float(mean_pnl),
        f"{L}_win_rate":    float(wr_ratio),
        f"{L}_win_rate_percent": float(wr_ratio*100.0),
        f"{L}_profit_factor": float(profit_factor),
        f"{L}_max_drawdown": float(max_dd),
        f"{L}_trades": int(total_trades),
        f"{L}_tsl_hits": int(tsl_hits),
        f"{L}_exit_reasons": {k:int(v) for k,v in exit_counts.items()},
        f"{L}_sharpe":  float(np.clip(sharpe,   -10.0, 10.0)),
        f"{L}_sortino": float(np.clip(sortino,  -10.0, 10.0)),
        f"{L}_bankruptcy_rate": float(bankruptcy_rate),
        # Новые метрики
        f"{L}_gross_pnl": float(gross_pnl),
        f"{L}_net_pnl": float(net_pnl),
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
        sel = select_and_arrange_channels(arr, cfg.data.expectedchannels, cfg.data.datachannels)
        if sel is not None:
            seqs.append(sel)
    return seqs


def run_training_session(
    train_sequences: List[np.ndarray],
    train_keys: List[Any],
    val_sequences: List[np.ndarray],
    val_keys: List[Any],
    cfg: MasterConfig,
    norm_stats: Dict[str, Any],
    models_dir: str,
    plots_dir: str,
    session_name: str,
    cfg_mod: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Runs a complete training and validation session for a given dataset.
    """
    
    # --- MC-dropout: ищем внешний объект `mc_dropout_cfg` или создаём пустышку ---
    mc_cfg = getattr(cfg_mod, "mc_dropout_cfg", type("obj", (), {})())

    # Set episodes from total_timesteps if not set
    if not hasattr(cfg.trainlog, 'episodes') or cfg.trainlog.episodes is None:
        cfg.trainlog.episodes = cfg.rl.total_timesteps // cfg.rl.n_steps

    # Масштабируем скорость затухания эпсилон, если включена опция и есть несколько сред
    eps_decay_frames = cfg.eps.eps_decay_frames
    if cfg.vec.num_envs > 1 and cfg.vec.scale_epsilon_by_envs:
        # Эта логика имеет смысл в основном для `subproc` бэкенда
        eps_decay_frames *= cfg.vec.num_envs
        logging.info(f"Epsilon decay frames scaled by num_envs ({cfg.vec.num_envs}): {cfg.eps.eps_decay_frames} -> {eps_decay_frames}")

    # Подготовка настроек MC-Dropout для передачи в конструктор
    mc_settings = getattr(cfg, "mc_dropout", mc_cfg)
    def _get_mc_val(obj, key, default):
        if isinstance(obj, dict): return obj.get(key, default)
        return getattr(obj, key, default)

    agent = D3QN_PER_Agent(
        state_shape=cfg.state_shape,  # (10,150,1)
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
        learning_rate=cfg.rl.lr,
        gamma=cfg.rl.gamma,
        batch_size=cfg.rl.batch_size,
        buffer_size=cfg.per.buffer_size,
        target_update_freq=cfg.rl.target_update_freq,
        train_start=cfg.rl.train_start,
        per_alpha=cfg.per.per_alpha,
        per_beta_start=cfg.per.per_beta_start,
        per_beta_frames=cfg.per.per_beta_frames,
        eps_start=cfg.eps.eps_start,
        eps_end=cfg.eps.eps_end,
        eps_frames=eps_decay_frames,
        epsilon=cfg.per.per_eps,  # PER eps
        max_gradient_norm=cfg.rl.max_gradient_norm,
        perf_cfg=cfg.perf,
        # Передаем параметры MC-Dropout явно в конструктор
        mc_enable=_get_mc_val(mc_settings, "enable", False),
        mc_n_action_samples=_get_mc_val(mc_settings, "n_action_samples", 1),
        mc_action_agg=_get_mc_val(mc_settings, "action_agg", "mean"),
        mc_lcb_k=_get_mc_val(mc_settings, "lcb_k", 0.5),
        mc_use_for_target=_get_mc_val(mc_settings, "use_for_target", False),
        mc_n_target_samples=_get_mc_val(mc_settings, "n_target_samples", 1),
        mc_target_agg=_get_mc_val(mc_settings, "target_agg", "mean_max"),
        mc_uncertainty_guided_explore=_get_mc_val(mc_settings, "uncertainty_guided", False),
        mc_uncertainty_beta=_get_mc_val(mc_settings, "uncertainty_beta", 0.0),
    )

    # After reshape, num_features becomes the number of channels in original data
    if len(train_sequences[0].shape) == 3:
        num_features = train_sequences[0].shape[0]  # C from (C, L, 1)
    else:
        num_features = train_sequences[0].shape[1]  # C from (L, C)
    input_history_len = cfg.seq.input_history_len or cfg.seq.agent_history_len
    num_actions = cfg.market.num_actions
    action_history_len = cfg.seq.action_history_len
    
    max_trades = getattr(cfg.market, "max_trades_per_episode", 100)
    if cfg_mod is not None and hasattr(cfg_mod, "MAX_TRADES_PER_EPISODE"):
        max_trades = cfg_mod.MAX_TRADES_PER_EPISODE
        logging.info(f"Override max_trades_per_episode from config module: {max_trades}")

    flat_features = input_history_len * num_features
    extras = 4  # position, unrealized, time_elapsed, time_remaining
    history_vector_size = num_actions * action_history_len if action_history_len > 0 else 0
    flat_state_size = flat_features + extras + history_vector_size
    
    env_kwargs = {
        "sequences": train_sequences,
        "keys": train_keys,
        "stats": norm_stats,
        "render_mode": cfg.render_mode,
        "full_seq_len": cfg.seq.full_seq_len,
        "num_features": num_features,
        "num_actions": num_actions,
        "flat_state_size": flat_state_size,
        "initial_balance": cfg.market.initial_balance,
        "pre_signal_len": cfg.seq.pre_signal_len,
        "datachannels": cfg.data.datachannels,
        "position_fraction": cfg.market.position_fraction,
        "slippage": cfg.market.slippage,
        "transaction_fee": cfg.market.transaction_fee,
        "agent_session_len": cfg.seq.agent_session_len,
        "agent_history_len": cfg.seq.agent_history_len,
        "input_history_len": input_history_len,
        "pricechannels": cfg.data.pricechannels,
        "volumechannels": cfg.data.volumechannels,
        "otherchannels": cfg.data.otherchannels,
        "action_history_len": action_history_len,
        "inaction_penalty_ratio": cfg.market.inaction_penalty_ratio,
        "bankruptcy_threshold": cfg.market.bankruptcy_threshold,
        "bankruptcy_penalty": cfg.market.bankruptcy_penalty,
        "max_drawdown_threshold": cfg.market.max_drawdown_threshold,
        "max_drawdown_penalty": cfg.market.max_drawdown_penalty,
        "max_drawdown_penalty_type": cfg.market.max_drawdown_penalty_type,
        "new_equity_peak_reward": cfg.market.new_equity_peak_reward,
        "perfect_entry_reward": cfg.market.perfect_entry_reward,
        "risk_reward_ratio_threshold": cfg.market.risk_reward_ratio_threshold,
        "risk_reward_ratio_reward": cfg.market.risk_reward_ratio_reward,
        "continuous_pain_penalty_ratio": cfg.market.continuous_pain_penalty_ratio,
        "good_exit_bonus": cfg.market.good_exit_bonus,
        "fast_exit_bonus": cfg.market.fast_exit_bonus,
        "low_balance_penalty": cfg.market.low_balance_penalty,
        "bankruptcy_slippage_penalty": cfg.market.bankruptcy_slippage_penalty,
        "holding_penalty_multiplier": cfg.market.holding_penalty_multiplier,
        "greed_penalty_multiplier": cfg.market.greed_penalty_multiplier,
        "premature_exit_penalty": cfg.market.premature_exit_penalty,
        "allow_opposite_trades": getattr(cfg.market, "allow_opposite_trades", True),
        "max_trades_per_episode": max_trades,
        "close_action_index": getattr(cfg.market, "close_action_index", None),
        "filter_direction": getattr(cfg.market, "filter_direction", None),
        "allowed_directions": getattr(cfg.market, "allowed_directions", None),
    }
    num_envs = getattr(cfg.vec, "num_envs", 1)
    if num_envs > 1:
        base_seed = cfg.global_env_seed
        env_fns = [partial(make_env, env_kwargs={**env_kwargs, "seed": base_seed + i}) for i in range(num_envs)]
        train_env = DummyVecEnv(env_fns) if cfg.vec.backend == "dummy" else SubprocVecEnv(env_fns, start_method=cfg.vec.start_method)
    else:
        train_env = TradingEnvironment(**env_kwargs)

    val_env = None
    if val_sequences:
        val_kwargs = dict(env_kwargs)
        val_kwargs.update({
            "sequences": val_sequences,
            "keys": val_keys,
            "backtest_mode": True,
            "use_risk_management": getattr(cfg.backtest, "use_risk_management", True),
        })
        val_env = TradingEnvironment(**val_kwargs)
        if hasattr(cfg.backtest, "exec_delay_bars"):
            setattr(val_env, "exec_delay_bars", int(cfg.backtest.exec_delay_bars))

    history = defaultdict(list)
    episode_rewards_deque = deque(maxlen=cfg.trainlog.plot_moving_avg_window)
    episode_losses_deque = deque(maxlen=cfg.trainlog.plot_moving_avg_window)
    episode_win_rate_deque = deque(maxlen=cfg.trainlog.plot_moving_avg_window)

    best_val_metric, last_val_metrics, best_validation = None, {}, {}
    no_improvement_count = 0
    best_episode = None
    train_steps = 0
    
    checkpoint_manager = None
    if getattr(cfg.trainlog, "save_top_k", 0) > 0:
        checkpoint_manager = TopKCheckpointManager(
            save_dir=os.path.join(models_dir, "checkpoints"),
            top_k=cfg.trainlog.save_top_k,
            metric_key=cfg.trainlog.checkpoint_metric,
            mode=cfg.trainlog.save_mode
        )

    if num_envs > 1:
        train_env.reset()
    else:
        train_env.reset(seed=cfg.global_env_seed)

    counter = trange(1, cfg.trainlog.episodes + 1, desc="Training in episodes", leave=False)
    for ep in counter:
        if num_envs > 1:
            ep_reward, _, transitions, avg_loss, ep_info = _rollout_vectorized_episode(train_env, agent, cfg.seq.agent_session_len)
            ep_losses = [avg_loss] if avg_loss > 0 else []
            train_steps += transitions
        else:
            obs, _ = train_env.reset(seed=None, options=None)
            ep_reward, ep_losses, done = 0.0, [], False
            while not done:
                action = agent.select_action(obs, training=True)
                next_obs, reward, done, _, info = train_env.step(action)
                next_state_to_store = info.get("terminal_observation", info.get("final_observation", next_obs)) if done else next_obs
                agent.store_experience(obs, action, reward, next_state_to_store, done)
                loss = agent.learn()
                if loss: ep_losses.append(loss)
                obs = next_obs
                agent.increment_step()
                train_steps += 1
                ep_reward += reward

        history["episodes"].append(ep)
        history["rewards"].append(ep_reward)
        avg_loss = np.mean(ep_losses) if ep_losses else 0.0
        history["losses"].append(avg_loss)

        episode_rewards_deque.append(ep_reward)
        history["mean_rewards_N"].append(np.mean(episode_rewards_deque))
        episode_losses_deque.append(avg_loss)
        history["mean_losses_N"].append(np.mean(episode_losses_deque))

        eps_current = agent.eps_end + (agent.eps_start - agent.eps_end) * np.exp(-train_steps / agent.eps_frames)
        history["epsilons"].append(eps_current)

        current_win_rate = (ep_info if num_envs > 1 else info).get("episode_win_rate", 0.0)
        episode_win_rate_deque.append(current_win_rate)
        history["win_rates"].append(current_win_rate)
        history["mean_win_rates_N"].append(np.mean(episode_win_rate_deque))

        counter.desc = f"Training loss={avg_loss:.7f}, reward={ep_reward:.5f}"

        if val_env and ep % cfg.trainlog.val_freq == 0:
            metrics = evaluate_agent(val_env, agent, len(val_sequences), "Validation", ep, cfg.global_env_seed, cfg, keys=val_keys)

            sel_keys = cfg.trainlog.val_selection_metrics

            def _fetch_metric(name: str, lower_is_better: bool) -> float:
                v = metrics.get(name, None)
                if v is None:
                    logging.warning(f"[Validation] metric '{name}' is missing in metrics dict — using fallback -inf")
                    return float("-inf")
                try:
                    v = float(v)
                except Exception:
                    logging.warning(f"[Validation] metric '{name}' has non-numeric value '{v}' — fallback -inf")
                    return float("-inf")
                return -v if lower_is_better else v

            last_val_metrics = metrics

            gate = getattr(getattr(cfg, "trainlog", object()), "validation_gate", None)
            if gate is None:
                gate = getattr(cfg, "validation_gate", None)

            def _passes_gate(m: Dict[str, Any], g: Dict[str, Any] | None) -> bool:
                if not g:
                    return True
                def _f(name: str, default: float | None = None) -> float:
                    v = m.get(name, default)
                    try: return float(v)
                    except Exception: return float("-inf")
                cur_sharpe  = _f("Validation_sharpe")
                cur_sortino = _f("Validation_sortino")
                cur_pf_raw  = m.get("Validation_profit_factor", None)
                try: cur_pf = float(cur_pf_raw)
                except Exception: cur_pf = float("-inf")
                cur_dd      = _f("Validation_max_drawdown")
                cur_wr      = _f("Validation_win_rate")
                cur_trades  = int(m.get("Validation_trades", 0) or 0)
                min_sharpe     = g.get("min_sharpe", None)
                min_sortino    = g.get("min_sortino", None)
                min_pf         = g.get("min_profit_factor", None)
                max_dd_at_most = g.get("max_drawdown_at_most", None)
                min_wr         = g.get("min_win_rate", None)
                min_trades     = g.get("min_trades", None)
                deny_zero_dd   = bool(g.get("deny_zero_drawdown", False))
                deny_inf_pf    = bool(g.get("deny_inf_pf", False))
                ok = True
                if (min_sharpe  is not None) and not (cur_sharpe  >= float(min_sharpe)):         ok = False
                if (min_sortino is not None) and not (cur_sortino >= float(min_sortino)):        ok = False
                if deny_inf_pf and (isinstance(cur_pf_raw, str) and cur_pf_raw.lower() == "inf"): ok = False
                if deny_inf_pf and (cur_pf == float("inf")):                                      ok = False
                if (min_pf      is not None) and not (cur_pf      >= float(min_pf)):             ok = False
                if (max_dd_at_most is not None) and not (cur_dd   >= float(max_dd_at_most)):     ok = False
                if deny_zero_dd and cur_dd == 0.0:                                                ok = False
                if (min_wr      is not None) and not (cur_wr      >= float(min_wr)):             ok = False
                if (min_trades  is not None) and not (cur_trades  >= int(min_trades)):           ok = False
                return ok

            if _passes_gate(metrics, gate):
                if isinstance(sel_keys, (list, tuple)):
                    val_metric = tuple(_fetch_metric(k, lower_is_better=(cfg.trainlog.val_selection_direction == "min")) for k in sel_keys)
                else:
                    val_metric = _fetch_metric(str(sel_keys), lower_is_better=(cfg.trainlog.val_selection_direction == "min"))

                def _is_better(current, best):
                    if best is None: return True
                    return current > best

                if _is_better(val_metric, best_val_metric):
                    best_val_metric = val_metric
                    best_validation = dict(metrics)
                    best_episode = ep
                    if checkpoint_manager:
                        checkpoint_manager.save_checkpoint(agent, ep, metrics)
                    else:
                        agent.save_model(os.path.join(models_dir, "best.pth"))
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1

            if no_improvement_count >= cfg.trainlog.early_stopping_patience:
                logging.info(f"Early stopping at episode {ep}.")
                break

    final_path = os.path.join(models_dir, "final.pth")
    agent.save_model(final_path)
    plot_training_progress(history, plots_dir, cfg.trainlog.plot_moving_avg_window)

    train_env.close()
    if val_env:
        val_env.close()

    return best_validation, history


def main(cfg: MasterConfig = None, cfg_mod: Optional[Any] = None):
    from config import cfg as loaded_cfg
    if cfg is None:
        cfg = loaded_cfg

    timestamp = time.strftime("date_%Y%m%d_time_%H%M%S")
    session_name = f"{cfg.project_name}_{timestamp}"
    setup_logging(session_name, cfg)
    # ИСПРАВЛЕНО: детерминизм должен управляться из конфига, а не быть захардкоженным
    set_random_seed(cfg.random_seed, getattr(cfg, "deterministic", False))

    models_dir = os.path.join(cfg.paths.model_dir, session_name)
    plots_dir = os.path.join(cfg.paths.plot_dir, session_name)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    with open(os.path.join(models_dir, "config_train.json"), "w") as f:
        json.dump(cfg.model_dump(), f, indent=2, default=str)

    norm_stats_path = getattr(cfg.paths, "norm_stats_path", "norm_stats.json")
    if os.path.exists(norm_stats_path):
        with open(norm_stats_path, 'r') as f:
            norm_stats = json.load(f)
    else:
        norm_stats = compute_norm_stats(cfg.paths.train_data_path, cfg, norm_stats_path)

    if getattr(cfg, "walk_forward", None) and cfg.walk_forward.enabled:
        logging.info("Walk-Forward Validation ENABLED.")
        all_sequences = []
        for src in cfg.walk_forward.data_sources:
            if os.path.exists(src):
                seqs = load_npz_dataset(src, "merged_wfv", plots_dir)
                all_sequences.extend(seqs)
            else:
                logging.warning(f"WFV source not found: {src}")

        folds = create_walk_forward_folds(
            all_sequences,
            cfg.walk_forward.train_months,
            cfg.walk_forward.test_months,
            cfg.walk_forward.step_months
        )
        if not folds:
            logging.error("No WFV folds created. Check data dates.")
            return

        wfv_results = []
        for i, (train_s, test_s) in enumerate(folds):
            logging.info(f"=== Starting WFV Fold {i+1}/{len(folds)} ===")

            fold_train_keys = [k for k, _ in train_s]
            fold_train_data = [d for _, d in train_s]
            fold_test_keys = [k for k, _ in test_s]
            fold_test_data = [d for _, d in test_s]

            fold_models_dir = os.path.join(models_dir, f"fold_{i+1}")
            fold_plots_dir = os.path.join(plots_dir, f"fold_{i+1}")
            os.makedirs(fold_models_dir, exist_ok=True)
            os.makedirs(fold_plots_dir, exist_ok=True)

            # --- Per-Fold Normalization ---
            logging.info(f"Calculating normalization stats for Fold {i+1}...")
            fold_norm_stats = calculate_normalization_stats(
                [d for _, d in train_s],  # Raw data from the current fold
                cfg.data.datachannels,
                cfg.data.pricechannels,
                cfg.data.volumechannels,
                cfg.data.otherchannels
            )

            # Pre-process data for the current fold
            train_seqs, train_keys_prep = load_and_prep_data_from_source(fold_train_data, fold_train_keys, "Train", fold_norm_stats)
            val_seqs, val_keys_prep = load_and_prep_data_from_source(fold_test_data, fold_test_keys, "Validation", fold_norm_stats)

            best_metrics, _ = run_training_session(
                train_sequences=train_seqs,
                train_keys=train_keys_prep,
                val_sequences=val_seqs,
                val_keys=val_keys_prep,
                cfg=cfg,
                norm_stats=fold_norm_stats,
                models_dir=fold_models_dir,
                plots_dir=fold_plots_dir,
                session_name=f"{session_name}_fold_{i+1}",
                cfg_mod=cfg_mod
            )
            wfv_results.append(best_metrics)

        # Aggregate and log WFV results
        if wfv_results:
            df_results = pd.DataFrame(wfv_results)
            logging.info("\n" + "="*50 + "\nWalk-Forward Validation Summary\n" + "="*50)
            logging.info(f"Total Folds: {len(df_results)}")
            logging.info("\n" + df_results.mean().to_string())
            df_results.to_csv(os.path.join(models_dir, "wfv_results.csv"))

    else:
        # Standard training run
        allowed_assets = getattr(cfg.paper, "symbols", None)
        if allowed_assets == "ALL": allowed_assets = None

        train_seqs, train_keys = load_and_prep_data(cfg.paths.train_data_path, "Train", norm_stats, allowed_assets)
        val_seqs, val_keys = load_and_prep_data(cfg.paths.val_data_path, "Validation", norm_stats, allowed_assets)

        if not train_seqs:
            logging.error("Training data not loaded. Exiting.")
            sys.exit(1)

        episodes_per_epoch = getattr(cfg.trainlog, "episodesperepoch", None)
        if episodes_per_epoch is not None and len(train_seqs) > episodes_per_epoch:
            rng = np.random.default_rng(cfg.random_seed)
            indices = rng.choice(len(train_seqs), episodes_per_epoch, replace=False)
            indices = sorted(indices.tolist())
            train_seqs = [train_seqs[i] for i in indices]
            train_keys = [train_keys[i] for i in indices]
            logging.info("Sampled train set down to %d episodes", len(train_seqs))

        if val_seqs:
            val_seqs, val_keys = create_validation_episodes(
                val_sequences=val_seqs,
                val_keys=val_keys,
                num_episodes=cfg.trainlog.num_val_ep,
                seed=cfg.random_seed
            )
            logging.info(f"Validation set sampled: {len(val_seqs)} episodes")

        best_validation, history = run_training_session(
            train_sequences=train_seqs,
            train_keys=train_keys,
            val_sequences=val_seqs,
            val_keys=val_keys,
            cfg=cfg,
            norm_stats=norm_stats,
            models_dir=models_dir,
            plots_dir=plots_dir,
            session_name=session_name,
            cfg_mod=cfg_mod
        )

        bundle_cfg = getattr(cfg_mod, "bundle_cfg", getattr(cfg, "bundle", object()))
        bundle_enabled = getattr(bundle_cfg, "enable", True)
        if bundle_enabled:
            sel_keys = cfg.trainlog.val_selection_metrics if isinstance(cfg.trainlog.val_selection_metrics, (list, tuple)) else [cfg.trainlog.val_selection_metrics]
            best_val_metric_human = tuple(best_validation.get(k, None) for k in sel_keys) if best_validation else None

            bundle_metrics = {
                "val_selection_metric": cfg.trainlog.val_selection_metrics,
                "val_selection_direction": cfg.trainlog.val_selection_direction,
                "val_min_delta": cfg.trainlog.val_min_delta,
                "best_val_metric": best_val_metric_human,
                "best_episode": best_validation.get("episode"),
                "best_validation": best_validation,
                "history": {
                    "episodes": history.get("episodes", []),
                    "mean_rewards_N": history.get("mean_rewards_N", []),
                    "mean_losses_N": history.get("mean_losses_N", []),
                    "mean_win_rates_N": history.get("mean_win_rates_N", []),
                },
            }
            with open(os.path.join(models_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(bundle_metrics, f, indent=2, default=_numpy_json_default)
            _dump_requirements_lock(os.path.join(models_dir, "requirements-lock.txt"))
            _dump_torch_env(os.path.join(models_dir, "torch_env.txt"))
            _dump_env_flags(cfg, os.path.join(models_dir, "env_flags.json"))
            data_manifest = _build_data_manifest(cfg)
            with open(os.path.join(models_dir, "data_manifest.json"), "w", encoding="utf-8") as f:
                json.dump(data_manifest, f, indent=2)

if __name__ == "__main__":
    cfg = None
    mod = None
    if len(sys.argv) > 1:
        cfg, mod = load_config(sys.argv[1], return_module=True)
    main(cfg=cfg, cfg_mod=mod)