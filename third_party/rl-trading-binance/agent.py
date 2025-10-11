# agent.py
import datetime as dt
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import millify

from model import DuelingQNetwork
from replay_buffer import PrioritizedReplayBuffer
from config import PerformanceConfig

logger = logging.getLogger(__name__)


class D3QN_PER_Agent:
    def __init__(
        self,
        state_shape: Tuple[int, ...],
        action_dim: int,
        cnn_maps: List[int],
        cnn_kernels: List[int],
        cnn_strides: List[int],
        dense_val: List[int],
        dense_adv: List[int],
        additional_feats: int,
        dropout_model: float,
        device: torch.device,
        gamma: float,
        learning_rate: float,
        batch_size: int,
        buffer_size: int,
        target_update_freq: int,
        train_start: int,
        per_alpha: float,
        per_beta_start: float,
        per_beta_frames: int,
        eps_start: float,
        eps_end: float,
        eps_frames: int,
        epsilon: float,
        max_gradient_norm: float,
        backtest_cache_path: str = None,
        perf_cfg: PerformanceConfig = PerformanceConfig(),
    ) -> None:
        self.device = device
        model_kwargs = {
            "input_shape": state_shape,
            "action_dim": action_dim,
            "cnn_maps": cnn_maps,
            "cnn_kernels": cnn_kernels,
            "cnn_strides": cnn_strides,
            "dense_val": dense_val,
            "dense_adv": dense_adv,
            "additional_feats": additional_feats,
            "dropout_p": dropout_model,
        }

        self.policy_net = DuelingQNetwork(**model_kwargs).to(device)
        self.target_net = DuelingQNetwork(**model_kwargs).to(device)
        
        if perf_cfg.compile_mode:
            logger.info(f"Enabling torch.compile with mode='{perf_cfg.compile_mode}' and dynamic={perf_cfg.compile_dynamic}")
            self.policy_net = torch.compile(
                self.policy_net,
                mode=perf_cfg.compile_mode,
                dynamic=perf_cfg.compile_dynamic,
            )
            self.target_net = torch.compile(self.target_net,
                                    mode=perf_cfg.compile_mode,
                                    dynamic=perf_cfg.compile_dynamic)

        self.use_amp = perf_cfg.use_amp and self.device.type == 'cuda'
        if self.use_amp:
            amp_dtype_str = perf_cfg.amp_dtype
            self.amp_dtype = torch.float16 if amp_dtype_str == "float16" else torch.bfloat16
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info(f"Automatic Mixed Precision (AMP) enabled with dtype={amp_dtype_str}.")

        num_params = sum(p.numel() for p in self.policy_net.parameters())
        logger.info(f"Policy Net with {millify(num_params, precision=1)} parameters created in Agent")
        logger.info(f"Target Net with {millify(num_params, precision=1)} parameters created in Agent")

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_size,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_frames=per_beta_frames,
            epsilon=epsilon,
        )

        self.gamma = gamma
        self.batch_size = batch_size
        self.train_start = train_start
        self.target_update_freq = target_update_freq

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_frames = eps_frames
        self.total_steps = 0
        self.learn_steps = 0
        self.max_gradient_norm = max_gradient_norm

        if backtest_cache_path is not None:
            self.qval_cache: Dict[Tuple[str, dt.datetime], Union[int, np.ndarray]] = {}
            self.cache_path = os.path.join(backtest_cache_path, "qval_cache.pkl")
            self._load_disk_cache()

        if device.type == "cpu":
            torch.set_flush_denormal(True)

        logger.info("D3QN_PER_Agent initialized.")

    def select_action(
        self,
        state: np.ndarray,
        training: bool = True,
        return_qvals: bool = False,
        use_cache: bool = False,
        cache_key: Optional[Tuple[str, dt.datetime]] = None,
    ) -> Union[int, np.ndarray]:
        eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.total_steps / self.eps_frames)
        if training and np.random.rand() < eps:
            return np.random.randint(self.policy_net.action_dim)

        if use_cache and not training and cache_key is not None:
            if cache_key in self.qval_cache:
                qvals = self.qval_cache[cache_key]
            else:
                with torch.no_grad():
                    tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                    qvals = self.policy_net(tensor).cpu().numpy()
                    self.qval_cache[cache_key] = qvals
            qvals = qvals.squeeze(0)
            return qvals if return_qvals else int(np.argmax(qvals))

        with torch.no_grad():
            tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            qvals = self.policy_net(tensor).cpu().numpy().squeeze(0)
        return qvals if return_qvals else int(np.argmax(qvals))

    def select_action_batch(self, states: np.ndarray, training: bool = True) -> list[int]:
        """
        Векторизованный epsilon-greedy для батча состояний (N,H,W,C/…):
        - один forward сети на весь батч (torch.no_grad, AMP при включённом autocast)
        - argmax по действиям для каждой строки
        - случайные действия под epsilon по тем же индексам
        Возвращает список длины N.
        """
        self.policy_net.eval()  # детерминированный инференс вне MC-дропаут
        n = int(states.shape[0])
        with torch.no_grad(), torch.autocast(device_type="cuda" if self.device.type=="cuda" else "cpu",
                                             enabled=getattr(self, "amp_enabled", False)):
            x = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            q = self.policy_net(x)              # [N, action_dim]
            greedy = q.argmax(dim=1).detach().to("cpu").numpy()  # [N]
        # epsilon-greedy по батчу
        eps = float(self.epsilon if hasattr(self, "epsilon") else
                    (self.eps_end + (self.eps_start - self.eps_end)
                     * np.exp(-self.total_steps / max(1, self.eps_frames))))
        if training and eps > 0.0:
            rnd = np.random.rand(n) < eps
            if np.any(rnd):
                rand_actions = np.random.randint(0, self.action_dim, size=int(rnd.sum()))
                greedy = greedy.copy()
                greedy[rnd] = rand_actions
        return greedy.tolist()

    def predict_ensemble(
        self,
        state: np.ndarray,
        training: bool = False,
        use_cache: bool = True,
        cache_key: Optional[Tuple[str, dt.datetime]] = None,
        n_samples: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        MC-Dropout (Monte Carlo Dropout)
        """
        if training:
            raise IndexError("Predict Ensemble use for inference mode only!")

        if use_cache and cache_key is not None:
            if cache_key in self.qval_cache:
                mean_q, std_q = self.qval_cache[cache_key]
            else:
                mean_q, std_q = self.get_mean_std_q(state, n_samples)
                self.qval_cache[cache_key] = mean_q, std_q

            return mean_q, std_q

        mean_q, std_q = self.get_mean_std_q(state, n_samples)
        return mean_q, std_q

    def get_mean_std_q(self, state: np.ndarray, n_samples: int = 5) -> Tuple[float, float]:
        # Включаем стохастику (dropout), но сохраняем и восстанавливаем исходный режим
        prev_training = self.policy_net.training
        self.policy_net.train()
        x = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        q_list = []
        with torch.no_grad():
            for _ in range(n_samples):
                q = self.policy_net(x).squeeze(0).detach().cpu().numpy()
                q_list.append(q)
        # Восстанавливаем исходный режим (детерминированный инференс вне MC-оценки)
        if not prev_training:
            self.policy_net.eval()
        q_arr = np.stack(q_list, axis=0)
        return float(q_arr.mean()), float(q_arr.std(ddof=1) if n_samples > 1 else 0.0)

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

    def learn(self) -> Optional[float]:
        if len(self.replay_buffer) < self.train_start:
            return None

        (states, actions, rewards, next_states, dones, indices, weights) = self.replay_buffer.sample(self.batch_size)

        states_t = torch.from_numpy(states).float().to(self.device)
        actions_t = torch.from_numpy(actions).long().to(self.device)
        rewards_t = torch.from_numpy(rewards).float().to(self.device)
        next_states_t = torch.from_numpy(next_states).float().to(self.device)
        dones_t = torch.from_numpy(dones).bool().to(self.device)
        weights_t = torch.from_numpy(weights).float().to(self.device)

        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1)
            next_q_values = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            next_q_values[dones_t] = 0.0
            target_q_values = rewards_t + self.gamma * next_q_values

        self.optimizer.zero_grad()

        if self.use_amp:
            with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                current_q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
                loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction="none")
                weighted_loss = (weights_t * loss).mean()

            self.scaler.scale(weighted_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_gradient_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            current_q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction="none")
            weighted_loss = (weights_t * loss).mean()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_gradient_norm)
            self.optimizer.step()

        td_errors = (target_q_values - current_q_values).abs().detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)
        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(weighted_loss.item())

    def increment_step(self) -> None:
        self.total_steps += 1

    def save_model(self, path: str) -> None:
        """
        Сохраняет ПОЛНЫЙ чекпоинт для безопасного возобновления обучения:
        - policy/target state_dict
        - optimizer state_dict
        - GradScaler (если AMP включён)
        - meta (счётчики шагов/eps-параметры и UTC-время)
        Обратная совместимость: загрузка старых .pth с одним state_dict поддерживается в load_model().
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "format": "d3qn_per_agent_v1",
            "created_utc": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "policy_state": self.policy_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": (self.scaler.state_dict() if hasattr(self, "scaler") else None),
            "meta": {
                "total_steps": int(self.total_steps),
                "learn_steps": int(self.learn_steps),
                "eps_start": float(self.eps_start),
                "eps_end": float(self.eps_end),
                "eps_frames": int(self.eps_frames),
            },
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path} (policy+target+optimizer+scaler+meta).")

    def load_model(self, path: str, strict: bool = True) -> None:
        """
        Загружает либо новый чекпоинт (см. save_model), либо старый .pth с единственным state_dict.
        Аргумент strict пробрасывается в load_state_dict для гибкости при мелких несовпадениях ключей.
        """
        obj = torch.load(path, map_location=self.device)
        # Новый формат (чекпоинт)
        if isinstance(obj, dict) and "policy_state" in obj:
            self.policy_net.load_state_dict(obj["policy_state"], strict=strict)
            self.target_net.load_state_dict(obj.get("target_state", obj["policy_state"]), strict=strict)
            opt_state = obj.get("optimizer_state")
            if opt_state:
                try:
                    self.optimizer.load_state_dict(opt_state)
                except Exception as e:
                    logger.warning(f"Optimizer state load skipped: {e}")
            scaler_state = obj.get("scaler_state")
            if hasattr(self, "scaler") and scaler_state:
                try:
                    self.scaler.load_state_dict(scaler_state)
                except Exception as e:
                    logger.warning(f"GradScaler state load skipped: {e}")
            meta = obj.get("meta", {}) or {}
            self.total_steps = int(meta.get("total_steps", self.total_steps))
            self.learn_steps = int(meta.get("learn_steps", self.learn_steps))
            kind = "checkpoint"
        else:
            # Старый формат (только веса сети)
            self.policy_net.load_state_dict(obj, strict=strict)
            self.target_net.load_state_dict(obj, strict=strict)
            kind = "weights-only"
        self.policy_net.eval()
        self.target_net.eval()
        logger.info(f"Model loaded from {path} ({kind}).")

    def _load_disk_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                self.qval_cache = pickle.load(f)
            logger.info(f"\nLoaded Q-value cache from {self.cache_path} ({len(self.qval_cache)} entries).")

    def save_disk_cache(self) -> None:
        # Атомарная перезапись кэша: временный файл + os.replace
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        tmp_path = self.cache_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(self.qval_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp_path, self.cache_path)
        logger.info(f"Q-value cache saved at {self.cache_path}")

    def clear_disk_cache(self):
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)
            logger.info(f"\nCleared Q-value cache at {self.cache_path}.")
        self.qval_cache = {}
