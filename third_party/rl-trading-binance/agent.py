# agent.py
import datetime as dt
import logging
import os
import ast
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

try:
    import msgpack  # type: ignore
    import msgpack_numpy as m  # type: ignore
except ImportError:
    msgpack = None
    m = None

import numpy as np
import torch

# For reproducibility and performance, limit torch threads
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

try:
    from safetensors import safe_open  # type: ignore
    from safetensors.torch import save_file  # type: ignore
except ImportError:
    safe_open = None
    save_file = None

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

def get_ort_session(model_path: str, intra_threads: int = 1):
    """Оптимизированная сессия с динамическим количеством потоков."""
    if not os.path.exists(model_path):
        return None

    sess_options = ort.SessionOptions()
    # Для обучения в SubprocVecEnv ставим 1, для инференса в проде - 6 или 8
    sess_options.intra_op_num_threads = intra_threads
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    return ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])

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
        config: any,
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
        self.epsilon = 1.0 # Фикс для AttributeError

        # Определяем количество потоков: 1 если мы в векторизованной среде обучения
        # Иначе берем из конфига или ставим 6 (для твоего CPU)
        num_envs = getattr(config.vec, 'num_envs', 1)
        onnx_threads = 1 if num_envs > 1 else 6

        self.ort_session = None
        model_path_onnx = getattr(config.paths, 'model_path_onnx', None)
        if model_path_onnx and os.path.exists(model_path_onnx):
            self.ort_session = get_ort_session(model_path_onnx, intra_threads=onnx_threads)

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
        # Оптимизация для Ryzen: ограничение потоков на инференс одной модели
        if self.device.type == 'cpu':
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
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
            self.cache_path = os.path.join(backtest_cache_path, "qval_cache.json")
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

    def export_onnx(self, file_path: str):
        """Exports the policy network to the ONNX format using JIT tracing.

        This method creates a deployable representation of the agent's policy network,
        which can be used for high-performance inference. The input tensor shape is
        derived from the model's internal configuration.

        Args:
            file_path (str): The path to save the ONNX file.
        """
        if ort is None:
            logger.error("Cannot export to ONNX: onnxruntime is not installed. Please run: pip install onnxruntime")
            return

        self.policy_net.eval()

        # Correctly determine the input vector size for the model
        model_input_shape = self.policy_net.input_shape
        additional_feats = getattr(self.policy_net, "additional_feats", 0)
        history_flat_size = model_input_shape[0] * model_input_shape[1]
        total_input_size = history_flat_size + additional_feats

        # Create a dummy input with the correct shape [batch_size, total_input_size]
        dummy_input = torch.randn(1, total_input_size, device=self.device)
        logger.info(f"Preparing to export to ONNX. Dummy input shape: {dummy_input.shape}")

        # Ensure the directory exists
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
            logger.info(f"✅ Model successfully exported to ONNX: {file_path}")
        except Exception as e:
            logger.error(f"❌ Error during ONNX export: {e}")
            logger.error(f"  - Dummy input shape: {dummy_input.shape}")
            logger.error(f"  - Calculated size: {total_input_size} (History: {history_flat_size}, Additional: {additional_feats})")
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
        if not (training and self.mc_enable and self.mc_n_action_samples > 1):
            return self._select_action_base(state, training, return_qvals, use_cache, cache_key)

        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        q_samples = self._q_forward_samples(self.policy_net, state_tensor, self.mc_n_action_samples)
        q_mean = q_samples.mean(dim=0).squeeze(0)
        q_std = q_samples.std(dim=0, unbiased=False).squeeze(0)

        eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.total_steps / self.eps_frames)
        if self.mc_uncertainty_guided_explore:
            best_a = int(torch.argmax(q_mean).item())
            unc = float(q_std[best_a].item())
            eps = min(1.0, max(0.0, eps + self.mc_uncertainty_beta * unc))

        if self.mc_action_agg == "thompson":
            idx = np.random.randint(0, self.mc_n_action_samples)
            logits = q_samples[idx, 0]
            action = int(torch.argmax(logits).item())
        elif self.mc_action_agg == "lcb":
            logits = q_mean - self.mc_lcb_k * q_std
            action = int(torch.argmax(logits).item())
        else:  # "mean"
            action = int(torch.argmax(q_mean).item())

        if training and np.random.rand() < eps:
            action = np.random.randint(0, self.action_dim)
        
        if return_qvals:
            return q_mean.cpu().numpy()
        return action

    def _q_forward_samples(self, net, state_tensor, n_samples: int):
        q_list = []
        was_training = net.training
        try:
            net.train(True)
            with torch.no_grad():
                for _ in range(max(1, n_samples)):
                    q = net(state_tensor)
                    if q.dim() == 1:
                        q = q.unsqueeze(0)
                    q_list.append(q.unsqueeze(0))
        finally:
            net.train(was_training)
        return torch.cat(q_list, dim=0)

    def _select_action_base(self, state: np.ndarray, training: bool, return_qvals: bool, use_cache: bool, cache_key: Optional[Tuple[str, dt.datetime]]):
        eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.total_steps / self.eps_frames)

        if training and np.random.rand() < eps:
            return np.random.randint(self.action_dim)

        # ONNX is the highest priority for inference
        if self.ort_session is not None and not training:
            state_input = state.astype(np.float32)
            if state_input.ndim == 3:
                state_input = state_input[np.newaxis, ...]

            ort_inputs = {self.ort_session.get_inputs()[0].name: state_input}
            qvals = self.ort_session.run(None, ort_inputs)[0][0]
            return qvals if return_qvals else int(np.argmax(qvals))

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
            return qvals if return_qvals else int(np.argmax(qvals))

        with torch.no_grad():
            tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            qvals = self.policy_net(tensor).cpu().numpy().squeeze(0)
            return qvals if return_qvals else int(np.argmax(qvals))

    def select_action_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Selects greedy actions for a batch of states.
        This is a pure-torch, no-grad method for performance.
        """
        self.policy_net.eval()
        with torch.no_grad(), torch.autocast(
            device_type=("cuda" if self.device.type == "cuda" else "cpu"),
            enabled=getattr(self, "use_amp", False)
        ):
            states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            q_values = self.policy_net(states_t)
            return q_values.argmax(dim=1).cpu().numpy()

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
        prev_training = self.policy_net.training
        self.policy_net.train()
        x = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        q_list = []
        with torch.no_grad():
            for _ in range(n_samples):
                q = self.policy_net(x).squeeze(0).detach().cpu().numpy()
                q_list.append(q)
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
        """Saves a complete checkpoint of the agent's state using safetensors."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if save_file is None:
            logger.warning("`safetensors` library not found. Saving model using legacy `torch.save`. "
                           "It is recommended to install safetensors for safer model serialization: `pip install safetensors`")
            if not path.endswith(".pth"):
                path = os.path.splitext(path)[0] + ".pth"
            
            checkpoint = {
                "policy_net": self.policy_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "meta": {
                    "format": "d3qn_per_agent_v3_legacy",
                    "total_steps": str(self.total_steps),
                    "learn_steps": str(self.learn_steps),
                }
            }
            if hasattr(self, "scaler"):
                checkpoint["scaler"] = self.scaler.state_dict()
            
            torch.save(checkpoint, path)
            logger.info(f"Legacy checkpoint saved to: {path}")
            return

        # Ensure the path ends with .safetensors
        if not path.endswith(".safetensors"):
            path = os.path.splitext(path)[0] + ".safetensors"

        tensors_to_save = {
            "policy_net": self.policy_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if hasattr(self, "scaler"):
            tensors_to_save["scaler"] = self.scaler.state_dict()

        # Metadata is saved inside the safetensors file
        metadata = {
            "format": "d3qn_per_agent_v4",
            "total_steps": str(self.total_steps),
            "learn_steps": str(self.learn_steps),
        }

        save_file(tensors_to_save, path, metadata=metadata)
        logger.info(f"Checkpoint saved securely to: {path}")

    def load_model(self, path: str, strict: bool = True) -> None:
        """Loads a checkpoint, supporting both new .safetensors and legacy .pth formats."""

        # Try loading new .safetensors format first
        if path.endswith(".safetensors") and os.path.exists(path):
            if safe_open is None:
                logger.error(f"Cannot load .safetensors file '{path}' because `safetensors` is not installed. Please run `pip install safetensors`.")
                return

            try:
                with safe_open(path, framework="pt", device=str(self.device)) as f:
                    # Load metadata
                    metadata = f.metadata()
                    if metadata:
                        self.total_steps = int(metadata.get("total_steps", self.total_steps))
                        self.learn_steps = int(metadata.get("learn_steps", self.learn_steps))
                        logger.info(f"Metadata loaded from {path}")

                    # Load tensors
                    self.policy_net.load_state_dict(f.get_tensor("policy_net"), strict=strict)
                    self.target_net.load_state_dict(f.get_tensor("policy_net"), strict=strict)
                    self.optimizer.load_state_dict(f.get_tensor("optimizer"))
                    if hasattr(self, "scaler") and "scaler" in f.keys():
                        self.scaler.load_state_dict(f.get_tensor("scaler"))

                logger.info(f"Safetensors checkpoint loaded from {path}")

            except Exception as e:
                logger.error(f"Failed to load safetensors checkpoint {path}: {e}")
                # Fallback to legacy might be risky, but we can try if needed
                return

        # Fallback for legacy .pth (pickle) files
        elif os.path.exists(path):
            logging.warning(f"DEPRECATION: Loading legacy '.pth' (pickle) checkpoint from {path}. Please re-save to '.safetensors' format for security.")
            device_to_load = torch.device('cpu') if not torch.cuda.is_available() else self.device
            obj = torch.load(path, map_location=device_to_load)

            # This handles various old checkpoint structures
            policy_state = obj.get("policy_state", obj.get("policy_net", obj))
            target_state = obj.get("target_state", policy_state) # Use policy if target is missing
            optimizer_state = obj.get("optimizer_state")
            scaler_state = obj.get("scaler_state")
            meta = obj.get("meta", {})

            self.policy_net.load_state_dict(policy_state, strict=strict)
            self.target_net.load_state_dict(target_state, strict=strict)

            if optimizer_state:
                self.optimizer.load_state_dict(optimizer_state)
            if scaler_state and hasattr(self, "scaler"):
                self.scaler.load_state_dict(scaler_state)

            self.total_steps = int(meta.get("total_steps", self.total_steps))
            self.learn_steps = int(meta.get("learn_steps", self.learn_steps))

            logger.info(f"Legacy checkpoint successfully loaded from {path}")

        else:
            logger.error(f"Checkpoint file not found at path: {path}")

        self.policy_net.eval()
        self.target_net.eval()

    def _load_disk_cache(self) -> None:
        if msgpack is None:
            logger.debug("msgpack not installed, disk cache is disabled.")
            return

        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    packed_data = f.read()
                    data = msgpack.unpackb(packed_data, object_hook=m.decode)
                    # Keys are stored as strings, convert them back to tuples
                    self.qval_cache = OrderedDict({ast.literal_eval(k): v for k, v in data.items()})
                logger.info(f"Loaded MessagePack Q-value cache from {self.cache_path} ({len(self.qval_cache)} entries).")
            except Exception as e:
                logger.error(f"Failed to load MessagePack cache: {e}. Starting with an empty cache.")
                self.qval_cache = OrderedDict()

    def save_disk_cache(self) -> None:
        if msgpack is None:
            return

        if not self.qval_cache:
            return
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        tmp_path = self.cache_path + ".tmp"
        try:
            # Convert tuple keys to strings for serialization
            data_to_pack = {str(k): v for k, v in self.qval_cache.items()}
            with open(tmp_path, "wb") as f:
                packed_data = msgpack.packb(data_to_pack, default=m.encode)
                f.write(packed_data)
            os.replace(tmp_path, self.cache_path)
            logger.info(f"Q-value cache saved to {self.cache_path} using MessagePack.")
        except Exception as e:
            logger.error(f"Failed to save MessagePack cache: {e}")

    def clear_disk_cache(self):
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)
            logger.info(f"\nCleared Q-value cache at {self.cache_path}.")
        self.qval_cache = OrderedDict()