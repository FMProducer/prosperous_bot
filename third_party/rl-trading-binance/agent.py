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

    def get_mean_std_q(self, state: np.ndarray, n_samples: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        # enable Dropout
        self.policy_net.train()
        tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_samples = [self.policy_net(tensor).cpu().numpy().squeeze(0) for _ in range(n_samples)]

        q_array = np.stack(q_samples)
        mean_q = q_array.mean(axis=0)
        std_q = q_array.std(axis=0)
        return mean_q, std_q

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

        next_actions = self.policy_net(next_states_t).argmax(dim=1)
        next_q_values = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        next_q_values[dones_t] = 0.0
        target_q_values = rewards_t + self.gamma * next_q_values

        current_q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        td_errors = (target_q_values - current_q_values).abs().detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)

        loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction="none")
        weighted_loss = (weights_t * loss).mean()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_gradient_norm)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(weighted_loss.item())

    def increment_step(self) -> None:
        self.total_steps += 1

    def save_model(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        self.policy_net.eval()
        self.target_net.eval()
        logger.info(f"Model loaded from {path}")

    def _load_disk_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                self.qval_cache = pickle.load(f)
            logger.info(f"\nLoaded Q-value cache from {self.cache_path} ({len(self.qval_cache)} entries).")

    def save_disk_cache(self):
        if not os.path.exists(self.cache_path):
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.qval_cache, f)
            logger.info(f"\nSaved Q-value cache to {self.cache_path} ({len(self.qval_cache)} entries).")

    def clear_disk_cache(self):
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)
            logger.info(f"\nCleared Q-value cache at {self.cache_path}.")
        self.qval_cache = {}
