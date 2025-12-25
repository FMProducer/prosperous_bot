# agent.py
import datetime as dt
import logging
import os
import pickle
from collections import OrderedDict
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

try:
    import onnxruntime as ort
except ImportError:
    ort = None
    # Сообщение будет выведено только при попытке использовать ONNX функции.
    # logger.warning("ONNX Runtime не найден. Для ускорения CPU-инференса, установите его: pip install onnxruntime")


class D3QN_PER_Agent:
    """A Dueling Double Deep Q-Network (D3QN) agent with Prioritized Experience Replay (PER).

    This agent learns a policy to select actions in a trading environment.
    The state is represented by a combination of historical market data and
    additional features. The reward is typically related to the change in

    portfolio value.

    Attributes:
        policy_net (DuelingQNetwork): The primary network for action selection.
        target_net (DuelingQNetwork): A lagging copy of the policy network for stable Q-value estimation.
        replay_buffer (PrioritizedReplayBuffer): Stores and samples experiences based on their TD-error.
        gamma (float): The discount factor for future rewards.
        total_steps (int): The total number of steps taken in the environment.
    """
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
        cnn_dilations: Optional[List[int]] = None,
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
        self.ort_session = None  # ONNX Runtime session
        if perf_cfg is None:
            perf_cfg = PerformanceConfig()
        model_kwargs = {
            "input_shape": state_shape,
            "action_dim": action_dim,
            "cnn_maps": cnn_maps,
            "cnn_kernels": cnn_kernels,
            "cnn_strides": cnn_strides,
            "cnn_dilations": cnn_dilations,
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
            if amp_dtype_str == "float16":
                self.amp_dtype = torch.float16
            else:
                self.amp_dtype = torch.bfloat16 # Default to bfloat16 for stability
            self.scaler = torch.amp.GradScaler("cuda")
            logger.info(f"Automatic Mixed Precision (AMP) enabled with dtype={self.amp_dtype}.")

        num_params = sum(p.numel() for p in self.policy_net.parameters())
        logger.info(f"Policy Net with {millify(num_params, precision=1)} parameters created in Agent")
        logger.info(f"Target Net with {millify(num_params, precision=1)} parameters created in Agent")

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Вычисляем размер плоского state'а, который будет храниться в буфере
        history_flat_size = state_shape[0] * state_shape[1]
        buffer_state_shape = (history_flat_size + additional_feats,)

        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_size,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_frames=per_beta_frames,
            epsilon=epsilon,
            state_shape=buffer_state_shape,
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

        self.qval_cache: OrderedDict[Tuple[str, dt.datetime], np.ndarray] = OrderedDict()
        self.max_cache_size = 10000

        if backtest_cache_path is not None:
            self.cache_path = os.path.join(backtest_cache_path, "qval_cache.pkl")
            self._load_disk_cache()

        if self.device.type == "cpu":
            torch.set_flush_denormal(True)

        # Auxiliary Value Loss parameters
        self.use_auxiliary_value_loss = True  # Can be made a config parameter
        self.aux_value_loss_weight = 0.5  # Weight for auxiliary loss

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

    def load_onnx_session(self, onnx_path: str):
        """Loads ONNX runtime session for inference."""
        if ort is None:
            logger.error("onnxruntime not installed.")
            return
        try:
            # General CPU optimizations for ONNX Runtime
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1 # Avoid contention
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.ort_session = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
            logger.info(f"ONNX session loaded from {onnx_path}")
        except Exception as e:
            logger.error(f"Failed to load ONNX session: {e}")

    def export_to_onnx(self, file_path: str):
        """Exports the policy network to the ONNX format.

        This method creates a deployable representation of the agent's policy network,
        which can be used for high-performance inference. The input tensor shape is
        derived from the model's internal configuration, not from the environment's
        observation space.

        Args:
            file_path (str): The path to save the ONNX file.
        """
        try:
            import onnx  # type: ignore
        except ImportError:
            logger.error("Невозможно экспортировать в ONNX: пакет 'onnx' не установлен.")
            logger.error("Пожалуйста, установите его командой: pip install onnx")
            return

        if ort is None:
            logger.error("Невозможно экспортировать в ONNX: onnxruntime не установлен.")
            logger.error("Пожалуйста, установите его командой: pip install onnxruntime")
            return

        self.policy_net.eval()

        # Корректное определение размера входного вектора для модели
        # self.policy_net.input_shape это (каналы, длина_истории, 1)
        model_input_shape = self.policy_net.input_shape
        # self.policy_net.additional_feats это количество доп. признаков
        additional_feats = getattr(self.policy_net, "additional_feats", 0)
        # Размер плоского вектора истории = Каналы * Длина
        history_flat_size = model_input_shape[0] * model_input_shape[1]
        total_input_size = history_flat_size + additional_feats

        # Создаем dummy_input правильной формы [batch_size, total_input_size]
        dummy_input = torch.randn(1, total_input_size, device=self.device)
        logger.info(f"Подготовка к экспорту в ONNX. Размер dummy_input: {dummy_input.shape}")

        # Убедимся, что директория существует
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        try:
            torch.onnx.export(
                self.policy_net,
                dummy_input,
                file_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            logger.info(f"✅ Модель успешно экспортирована в ONNX: {file_path}")
        except Exception as e:
            logger.error(f"❌ Ошибка при экспорте в ONNX: {e}")
            logger.error(f"  - Размер dummy_input: {dummy_input.shape}")
            logger.error(f"  - Расчетный размер: {total_input_size} (История: {history_flat_size}, Доп: {additional_feats})")
            raise

    def load_onnx_model(self, file_path: str):
        """Loads an ONNX model for accelerated CPU inference.

        If the ONNX Runtime is available, this method loads the specified model
        and prepares it for inference. This can significantly speed up action
        selection in non-training scenarios.

        Args:
            file_path (str): The path to the ONNX model file.
        """
        self.load_onnx_session(file_path)

    def select_action(
        self,
        state: np.ndarray,
        training: bool = True,
        return_qvals: bool = False,
        use_cache: bool = False,
        cache_key: Optional[Tuple[str, dt.datetime]] = None,
    ) -> Union[int, np.ndarray]:
        """Selects an action based on the current state using an epsilon-greedy policy.

        In training mode, the agent explores with a probability of epsilon, which
        decays over time. In evaluation mode, it selects the action with the highest
        estimated Q-value. The state `s_t` is a snapshot of the environment at
        time `t`, and the chosen action `a_t` leads to a new state `s_{t+1}`.

        Args:
            state (np.ndarray): The current state of the environment.
            training (bool): Whether the agent is in training mode.
            return_qvals (bool): If True, returns the Q-values for all actions.
            use_cache (bool): If True, uses a cache for Q-value lookups.
            cache_key (Optional[Tuple[str, dt.datetime]]): The key for the Q-value cache.

        Returns:
            Union[int, np.ndarray]: The selected action or an array of Q-values.
        """
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

        # --- ONNX INFERENCE PATH ---
        if self.ort_session is not None and not training:
            ort_inputs = {self.ort_session.get_inputs()[0].name: state.astype(np.float32)[np.newaxis, ...]}
            qvals = self.ort_session.run(None, ort_inputs)[0][0] # [1, A] -> [A]
            return qvals if return_qvals else int(np.argmax(qvals))
        # ---------------------------

        if use_cache and not training and cache_key is not None:
            if cache_key in self.qval_cache:
                self.qval_cache.move_to_end(cache_key)
                qvals = self.qval_cache[cache_key]
            else:
                with torch.no_grad():
                    tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                    qvals = self.policy_net(tensor).cpu().numpy().squeeze(0)

                self.qval_cache[cache_key] = qvals
                if len(self.qval_cache) > self.max_cache_size:
                    self.qval_cache.popitem(last=False)
            qvals = qvals.squeeze(0)
            return qvals if return_qvals else int(np.argmax(qvals))

        with torch.no_grad():
            tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            qvals = self.policy_net(tensor).cpu().numpy().squeeze(0)
            return qvals if return_qvals else int(np.argmax(qvals))

    def select_action_batch(self, states: np.ndarray, training: bool = True) -> list[int]:
        """Selects actions for a batch of states using a vectorized epsilon-greedy policy.

        This method performs a single forward pass on the policy network for the entire
        batch of states, making it more efficient than calling `select_action` in a loop.
        Epsilon-greedy exploration is applied to the batch, with random actions
        selected for a subset of the states.

        Args:
            states (np.ndarray): A batch of states with shape (N, ...), where N is the
                batch size.
            training (bool): Whether the agent is in training mode.

        Returns:
            list[int]: A list of selected actions for each state in the batch.
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
        """Performs Monte Carlo Dropout to estimate Q-value uncertainty.

        This method runs multiple forward passes with dropout enabled to obtain a
        distribution of Q-values. The mean and standard deviation of this
        distribution can be used to gauge the model's confidence in its
        predictions.

        Args:
            state (np.ndarray): The current state of the environment.
            training (bool): Must be False, as this is an inference-only method.
            use_cache (bool): If True, uses a cache for Q-value lookups.
            cache_key (Optional[Tuple[str, dt.datetime]]): The key for the Q-value cache.
            n_samples (int): The number of forward passes to perform.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the mean and
                standard deviation of the Q-values.
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
        """Calculates the mean and standard deviation of Q-values using MC Dropout.

        This is a helper method for `predict_ensemble`. It performs `n_samples`
        forward passes through the policy network with dropout enabled and
        computes the mean and standard deviation of the resulting Q-values.

        Args:
            state (np.ndarray): The current state of the environment.
            n_samples (int): The number of forward passes to perform.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The mean and standard deviation of the Q-values.
        """
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
        """Stores a transition in the replay buffer.

        A transition consists of the current state, the action taken, the resulting
        reward, the next state, and a done flag indicating if the episode has
        terminated. This experience is stored for later use in the learning process.

        The transition is represented as `(s_t, a_t, r_t, s_{t+1})`.

        Args:
            state (np.ndarray): The state at time `t`.
            action (int): The action taken at time `t`.
            reward (float): The reward received at time `t`.
            next_state (np.ndarray): The state at time `t+1`.
            done (bool): Whether the episode terminated at time `t+1`.
        """
        self.replay_buffer.add(state, action, reward, next_state, done)

    def learn(self) -> Optional[float]:
        """Performs a single learning step.

        This method samples a batch of experiences from the replay buffer and uses
        it to update the policy network. The target Q-values are calculated using
        the target network, and the loss is computed as the difference between the
        predicted and target Q-values. The loss is then backpropagated to update
        the policy network's weights.

        The target Q-value is calculated as:
        `Q_target(s_t, a_t) = r_t + gamma * Q_target(s_{t+1}, argmax_a Q_policy(s_{t+1}, a))`

        Returns:
            Optional[float]: The loss value for the current learning step, or None if
                learning has not yet started.
        """
        # Do not start learning until the buffer has enough transitions.
        # We take the max of train_start and batch_size to ensure the sample is valid.
        if len(self.replay_buffer) < max(self.train_start, self.batch_size):
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            weights,
        ) = self.replay_buffer.sample(self.batch_size)

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
                # Get Q, V, A from policy network
                q_out, v_out, adv_out = self.policy_net(states_t, return_components=True)
                current_q_values = q_out.gather(1, actions_t.unsqueeze(1)).squeeze(1)

                # Защита от нечисловых значений в Q: если что-то пошло не так,
                # пропускаем шаг обучения, чтобы не портить сеть и буфер.
                if not torch.isfinite(current_q_values).all() or not torch.isfinite(target_q_values).all():
                    logger.warning("Non-finite Q-values detected in AMP branch, skipping learn step")
                    # Расширенное логирование для отладки
                    def log_stats(name, tensor):
                        if not torch.is_tensor(tensor): return
                        finite_tensor = tensor[torch.isfinite(tensor)]
                        if finite_tensor.numel() == 0:
                            logger.debug(f"  {name}: all values are non-finite")
                            return
                        logger.debug(
                            f"  {name}: "
                            f"non-finite={torch.isinf(tensor).sum().item()+torch.isnan(tensor).sum().item()}, "
                            f"min={finite_tensor.min().item():.4f}, "
                            f"max={finite_tensor.max().item():.4f}, "
                            f"mean={finite_tensor.mean().item():.4f}"
                        )

                    log_stats("states_t", states_t)
                    log_stats("rewards_t", rewards_t)
                    log_stats("next_states_t", next_states_t)
                    log_stats("current_q_values", current_q_values)
                    log_stats("target_q_values", target_q_values)
                    return None

                # Main TD loss
                td_loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction="none")
                weighted_td_loss = (weights_t * td_loss).mean()
                
                # Auxiliary Value Loss: V(s) should be close to mean Q(s,a)
                if self.use_auxiliary_value_loss:
                    with torch.no_grad():
                        # Target for V(s) = mean of current Q-values (no gradient)
                        target_value = q_out.mean(dim=1, keepdim=True).detach()
                    
                    # MSE loss between V(s) and target
                    aux_value_loss = F.mse_loss(v_out, target_value)
                    
                    # Combined loss
                    weighted_loss = weighted_td_loss + self.aux_value_loss_weight * aux_value_loss
                else:
                    weighted_loss = weighted_td_loss

                self.scaler.scale(weighted_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_gradient_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            # FP32 branch - similar changes
            q_out, v_out, adv_out = self.policy_net(states_t, return_components=True)
            current_q_values = q_out.gather(1, actions_t.unsqueeze(1)).squeeze(1)

            if not torch.isfinite(current_q_values).all() or not torch.isfinite(target_q_values).all():
                logger.warning("Non-finite Q-values detected in FP32 branch, skipping learn step")
                # Расширенное логирование для отладки
                def log_stats(name, tensor):
                    if not torch.is_tensor(tensor): return
                    finite_tensor = tensor[torch.isfinite(tensor)]
                    if finite_tensor.numel() == 0:
                        logger.debug(f"  {name}: all values are non-finite")
                        return
                    logger.debug(
                        f"  {name}: "
                        f"non-finite={torch.isinf(tensor).sum().item()+torch.isnan(tensor).sum().item()}, "
                        f"min={finite_tensor.min().item():.4f}, "
                        f"max={finite_tensor.max().item():.4f}, "
                        f"mean={finite_tensor.mean().item():.4f}"
                    )

                log_stats("states_t", states_t)
                log_stats("rewards_t", rewards_t)
                log_stats("next_states_t", next_states_t)
                log_stats("current_q_values", current_q_values)
                log_stats("target_q_values", target_q_values)
                return None

            # Main TD loss
            td_loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction="none")
            weighted_td_loss = (weights_t * td_loss).mean()
            
            if self.use_auxiliary_value_loss:
                with torch.no_grad():
                    target_value = q_out.mean(dim=1, keepdim=True).detach()
                aux_value_loss = F.mse_loss(v_out, target_value)
                weighted_loss = weighted_td_loss + self.aux_value_loss_weight * aux_value_loss
            else:
                weighted_loss = weighted_td_loss
            
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_gradient_norm)
            self.optimizer.step()

        td_errors = (target_q_values - current_q_values).abs().detach().cpu().numpy()
        # Здесь td_errors гарантированно конечные (NaN/inf отфильтрованы выше)
        self.replay_buffer.update_priorities(indices, td_errors)
        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(weighted_loss.item())

    def increment_step(self) -> None:
        """Increments the total step counter.

        This method should be called after each step in the environment to keep
        track of the agent's total experience.
        """
        self.total_steps += 1

    def prepare_for_qat(self):
        """Prepares the model for Quantization-Aware Training (QAT).

        This method sets the model's qconfig and prepares it for QAT. This
        should be called before fine-tuning a model with quantization-aware
        training.
        """
        self.policy_net.train()
        self.policy_net.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
        torch.ao.quantization.prepare_qat(self.policy_net, inplace=True)
        logger.info("Model prepared for Quantization-Aware Training (QAT)")

    def optimize_for_cpu(self):
        """Applies dynamic quantization to the policy network for INT8 inference."""
        self.policy_net.eval()
        self.policy_net = torch.quantization.quantize_dynamic(
            self.policy_net, {torch.nn.Linear}, dtype=torch.qint8
        )
        logger.info("Policy network optimized for CPU inference with dynamic quantization.")

    def convert_to_cpu_optimized(self):
        """Оптимизация модели для экстремальной скорости на Ryzen (CPU)."""
        self.policy_net.eval()
        self.policy_net = torch.quantization.quantize_dynamic(
            self.policy_net,
            {torch.nn.Linear, torch.nn.Conv1d},
            dtype=torch.qint8
        )
        logger.info("🚀 Model quantized for CPU inference (INT8).")

    def save_model(self, path: str) -> None:
        """Saves a complete checkpoint of the agent's state.

        This method saves all the necessary components to resume training, including
        the policy and target network weights, the optimizer state, and other
        metadata such as the total number of steps.

        Args:
            path (str): The path to save the checkpoint file.
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
        """Loads a checkpoint of the agent's state.

        This method loads a saved checkpoint, which can be either a full checkpoint
        (including optimizer state) or a file containing only the model weights.
        This allows for both resuming training and loading pre-trained models for
        inference.

        Args:
            path (str): The path to the checkpoint file.
            strict (bool): Whether to strictly enforce that the keys in the
                checkpoint match the keys returned by this module's
                `state_dict()` function.
        """
        # Исправляем загрузку для CPU-only машин
        device_to_load = torch.device('cpu') if not torch.cuda.is_available() else self.device
        obj = torch.load(path, map_location=device_to_load)

        # Если модель была обучена с QAT, конвертируем её в инт8 после загрузки
        if hasattr(self.policy_net, 'quant'):
            self.policy_net.eval()
            torch.ao.quantization.convert(self.policy_net, inplace=True)
            logger.info("Model converted to INT8 for extreme CPU speed")

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