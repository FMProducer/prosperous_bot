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
        perf_cfg: PerformanceConfig = None,
        # ── НОВОЕ: MC-dropout в обучении
        mc_enable=False, mc_n_action_samples=1, mc_action_agg="mean", mc_lcb_k=0.5,
        mc_use_for_target=False, mc_n_target_samples=1, mc_target_agg="mean_max",
        mc_uncertainty_guided_explore=False, mc_uncertainty_beta=0.0
    ) -> None:
        # Приводим к torch.device на случай, если из конфига придёт строка "cuda"/"cpu"
        self.device = torch.device(device)
        self.action_dim = action_dim
        if perf_cfg is None:
            perf_cfg = PerformanceConfig()
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

        self.policy_net = DuelingQNetwork(**model_kwargs).to(self.device)
        self.target_net = DuelingQNetwork(**model_kwargs).to(self.device)
        
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
            self.scaler = torch.amp.GradScaler("cuda")
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

        if self.device.type == "cpu":
            torch.set_flush_denormal(True)

        # ── MC-dropout настройки
        self.mc_enable = bool(mc_enable)
        self.mc_n_action_samples = int(mc_n_action_samples)
        self.mc_action_agg = mc_action_agg
        self.mc_lcb_k = float(mc_lcb_k)
        self.mc_use_for_target = bool(mc_use_for_target)
        self.mc_n_target_samples = int(mc_n_target_samples)
        self.mc_target_agg = mc_target_agg
        self.mc_uncertainty_guided_explore = bool(mc_uncertainty_guided_explore)
        self.mc_uncertainty_beta = float(mc_uncertainty_beta)

        # ── Лог один раз при инициализации: фиксируем режим обучения (MC или базовый)
        try:
            logging.info(
                "MC-Dropout training config: enable=%s, action_samples=%d, action_agg=%s, "
                "lcb_k=%.3f, target_mc=%s, target_samples=%d, target_agg=%s, "
                "uncertainty_guided=%s, uncertainty_beta=%.4f",
                self.mc_enable,
                self.mc_n_action_samples,
                self.mc_action_agg,
                self.mc_lcb_k,
                self.mc_use_for_target,
                self.mc_n_target_samples,
                self.mc_target_agg,
                self.mc_uncertainty_guided_explore,
                self.mc_uncertainty_beta,
            )
        except Exception as e:
            logging.warning(f"MC-Dropout config log failed: {e}")


        logger.info("D3QN_PER_Agent initialized.")

    def select_action(
        self,
        state: np.ndarray,
        training: bool = True,
        return_qvals: bool = False,
        use_cache: bool = False,
        cache_key: Optional[Tuple[str, dt.datetime]] = None,
    ) -> Union[int, np.ndarray]:
        # Базовая ε-жадная логика (epsilon берется из self.eps_* расписания внутри агента)
        # Если mc_enable=False или training=False — используем обычный путь как прежде.
        if not (training and self.mc_enable and self.mc_n_action_samples > 1):
            return self._select_action_base(state, training, return_qvals, use_cache, cache_key)

        # MC-dropout: ансамбль из N проходов онлайн-сети.
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        q_samples = self._q_forward_samples(self.policy_net, state_tensor, self.mc_n_action_samples)  # [N,1,A]
        q_mean = q_samples.mean(dim=0).squeeze(0)    # [A]
        q_std  = q_samples.std(dim=0, unbiased=False).squeeze(0)  # [A]

        # Неопределённостно-направляемая эксплорация (опц.)
        eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.total_steps / self.eps_frames)
        if self.mc_uncertainty_guided_explore:
            # увеличим ε пропорционально неопределённости лучшего действия
            best_a = int(torch.argmax(q_mean).item())
            unc = float(q_std[best_a].item())
            eps = min(1.0, max(0.0, eps + self.mc_uncertainty_beta * unc))

        # Выбор по агрегатору
        if self.mc_action_agg == "thompson":
            # Берём один случайный семпл и argmax в нём — стохастическая политика
            idx = np.random.randint(0, self.mc_n_action_samples)
            logits = q_samples[idx, 0]  # [A]
            action = int(torch.argmax(logits).item())
        elif self.mc_action_agg == "lcb":
            logits = q_mean - self.mc_lcb_k * q_std
            action = int(torch.argmax(logits).item())
        else:  # "mean"
            action = int(torch.argmax(q_mean).item())

        # Применяем ε-жадность поверх выбора (как раньше)
        if training and np.random.rand() < eps:
            action = np.random.randint(0, self.action_dim)
        
        if return_qvals:
            return q_mean.cpu().numpy()
        return action

    def _q_forward_samples(self, net, state_tensor, n_samples: int):
        """
        Выполнить n семплов forward с активным dropout.
        Возвращает тензор [n, batch(=1), action_dim].
        """
        q_list = []
        was_training = net.training
        try:
            net.train(True)  # включаем dropout
            with torch.no_grad():
                for _ in range(max(1, n_samples)):
                    q = net(state_tensor)  # ожидается [1, action_dim]
                    if q.dim() == 1:
                        q = q.unsqueeze(0)
                    q_list.append(q.unsqueeze(0))  # [1,1,A]
        finally:
            net.train(was_training)
        return torch.cat(q_list, dim=0)  # [n,1,A]

    def _select_action_base(self, state, training: bool, return_qvals: bool, use_cache: bool, cache_key: Optional[Tuple[str, dt.datetime]]):
        eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.total_steps / self.eps_frames)
        if training and np.random.rand() < eps:
            return np.random.randint(self.action_dim)
        
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
        with torch.no_grad(), torch.autocast(
            device_type=("cuda" if self.device.type == "cuda" else "cpu"),
            enabled=getattr(self, "use_amp", False)
        ):
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

    def get_mean_std_q(
        self, state: np.ndarray, n_samples: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        mean_q = q_arr.mean(axis=0)
        if n_samples > 1:
            std_q = q_arr.std(axis=0, ddof=1)
        else:
            std_q = np.zeros_like(mean_q)
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

        with torch.no_grad():
            # Таргеты с учётом MC-dropout (если включено)
            if self.mc_use_for_target and self.mc_n_target_samples > 1:
                was_training = self.target_net.training
                try:
                    # Переводим target_net в train() для включения dropout-масок
                    self.target_net.train(True)
                    # batch_next_states: [B, ...], прогоняем N раз и агрегируем
                    q_list = []
                    for _ in range(self.mc_n_target_samples):
                        q = self.target_net(next_states_t)  # [B, A]
                        q_list.append(q.unsqueeze(0))           # [1,B,A]
                    q_stack = torch.cat(q_list, dim=0)         # [N,B,A]
                    if self.mc_target_agg == "max_mean":
                        # max_a mean_n Q_n(s',a)
                        q_mean = q_stack.mean(dim=0)           # [B,A]
                        next_actions = self.policy_net(next_states_t).argmax(dim=1)
                        next_q_values = q_mean.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    else: # "mean_max"
                        # mean_n max_a Q_n(s',a)
                        next_actions = self.policy_net(next_states_t).argmax(dim=1)
                        next_q_values = q_stack.gather(2, next_actions.view(1, -1, 1).expand(self.mc_n_target_samples, -1, -1)).squeeze(2).mean(dim=0)
                finally:
                    self.target_net.train(was_training)
            else:
                # ── иначе: старая реализация
                next_actions = self.policy_net(next_states_t).argmax(dim=1)
                next_q_values = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            next_q_values[dones_t] = 0.0
            target_q_values = rewards_t + self.gamma * next_q_values
        self.optimizer.zero_grad()

        if self.use_amp:
            with torch.amp.autocast("cuda", dtype=self.amp_dtype):
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

        def _try_load_weights(sd, tag: str):
            sd = _unwrap_sd(sd)
            self.policy_net.load_state_dict(sd, strict=strict)
            self.target_net.load_state_dict(sd, strict=strict)
            return tag

        def _unwrap_sd(sd):
            """
            Превращает любые обёртки в «чистый» state_dict слоёв модели.
            Поддерживает: {'policy_state': ...}, {'model_state': ...}, {'state_dict': ...}, {'weights': ...}.
            Если внутри снова лежит чекпоинт, развернёт повторно.
            """
            if isinstance(sd, dict):
                # прямой новый чекпоинт
                if "policy_state" in sd and isinstance(sd["policy_state"], dict):
                    return sd["policy_state"]
                # обёртки старых форматов
                for k in ("model_state", "state_dict", "weights"):
                    if k in sd and isinstance(sd[k], dict):
                        inner = sd[k]
                        # на случай двойной обёртки
                        if isinstance(inner, dict) and "policy_state" in inner and isinstance(inner["policy_state"], dict):
                            return inner["policy_state"]
                        return inner
            return sd

        kind = None
        if isinstance(obj, dict):
            # 1) Новый полноформатный чекпоинт
            if "policy_state" in obj:
                self.policy_net.load_state_dict(_unwrap_sd(obj["policy_state"]), strict=strict)
                self.target_net.load_state_dict(_unwrap_sd(obj.get("target_state", obj["policy_state"])), strict=strict)
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
                # 2) Обёрнутые веса разных старых форматов
                wrapped_sd = obj.get("model_state", None)
                if wrapped_sd is None and "state_dict" in obj:
                    wrapped_sd = obj["state_dict"]
                if wrapped_sd is None and "weights" in obj:
                    wrapped_sd = obj["weights"]
                if wrapped_sd is not None:
                    # Если присутствует отдельный target_state — загрузим его, иначе дублируем policy
                    self.policy_net.load_state_dict(_unwrap_sd(wrapped_sd), strict=strict)
                    self.target_net.load_state_dict(_unwrap_sd(obj.get("target_state", wrapped_sd)), strict=strict)
                    # Не критично: попробуем подтянуть optimizer/scaler, если есть
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
                    kind = "wrapped-weights"
                else:
                    # 3) Попытка трактовать obj как «голые» веса (редкий случай dict-весов)
                    kind = _try_load_weights(obj, "weights-only(dict)")
        else:
            # 4) Старый «голый» state_dict как OrderedDict/Mapping
            kind = _try_load_weights(obj, "weights-only")

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
