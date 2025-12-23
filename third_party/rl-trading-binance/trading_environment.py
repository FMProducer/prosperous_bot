# trading_environment.py
import datetime as dt
import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from utils import apply_normalization

logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    """A custom trading environment that simulates the process of trading in a financial market.

    This environment conforms to the Gymnasium API and is designed for training
    reinforcement learning agents. It supports various features such as shaped
    rewards, risk management, and different observation formats.

    The state `s_t` is a composite of historical market data (e.g., OHLCV),
    the agent's current position, unrealized profit/loss, and the time
    remaining in the episode.

    The reward `r_t` is primarily based on the change in portfolio value (PnL)
    but can be augmented with shaped rewards to encourage desirable behaviors
    such as holding profitable positions and penalizing inaction.
    """
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}
    exit_options = np.array(["FORCED", "SL", "TP", "TSL"])

    def __init__(
        self,
        sequences: List[np.ndarray],
        stats: Dict[str, Dict[str, float]],
        keys: List[str],
        render_mode: Optional[str],
        full_seq_len: int,
        num_features: int,
        num_actions: int,
        flat_state_size: int,
        initial_balance: float,
        pre_signal_len: int,
        datachannels: List[str],
        slippage: float,
        transaction_fee: float,
        agent_session_len: int,
        agent_history_len: int,
        input_history_len: int,
        pricechannels: List[str],
        volumechannels: List[str],
        otherchannels: List[str],
        action_history_len: int,
        inaction_penalty_ratio: float,
        backtest_mode: bool = False,
        use_risk_management: bool = False,
        cnn_format: bool = False,
        position_fraction: float = 1.0,
        order_size_usdt: float = 0.0,
        bankruptcy_threshold: float = 0.0,
        bankruptcy_penalty: float = 1.0,
        # Penalties for max drawdown
        max_drawdown_threshold: float | None = None,
        max_drawdown_penalty: float = 0.0,
        max_drawdown_penalty_type: str = "absolute",
        # New reward shaping parameters
        new_equity_peak_reward: float = 0.0,
        perfect_entry_reward: float = 0.0,
        risk_reward_ratio_threshold: float = 3.0,
        risk_reward_ratio_reward: float = 0.0,
        continuous_pain_penalty_ratio: float = 0.0,

        # Shaped rewards/penalties (previously hardcoded)
        good_exit_bonus: float = 0.0,
        fast_exit_bonus: float = 0.0,
        low_balance_penalty: float = 0.0,
        bankruptcy_slippage_penalty: float = 0.0,
        holding_penalty_multiplier: float = 0.0,
        greed_penalty_multiplier: float = 0.0,
        # NEW: Asymmetric penalties
        premature_profit_exit_penalty: float = 0.0,
        holding_loss_penalty: float = 0.0,
        # OLD: Kept for compatibility (set to 0.0)
        premature_exit_penalty: float = 0.0,
        profit_holding_bonus: float = 0.0,
        
        # Thresholds for shaped rewards (previously hardcoded)
        holding_penalty_threshold: int = 15,
        greed_penalty_threshold: float = 0.50,
        exit_quality_threshold: float = 0.80,
        fast_exit_threshold: int = 20,
        premature_exit_threshold: int = 5,

        # NEW: Thresholds for asymmetric logic
        profit_exit_threshold: int = 5,
        loss_exit_threshold: int = 3,
        allow_opposite_trades: bool = True, # НОВЫЙ ПАРАМЕТР
        close_action_index: Optional[int] = None,
        seed: Optional[int] = None,
        allowed_directions: Optional[List[str]] = None,
        raw_sequences: Optional[List[np.ndarray]] = None,
        **kwargs,
    ) -> None:
        if not sequences:
            raise ValueError("`sequences` must be a non-empty list of arrays")
        if not keys:
            raise ValueError("`keys` must be a non-empty list of strings")
        if len(sequences) != len(keys):
            raise ValueError("Length of `sequences` and `keys` must be the same")

        if raw_sequences and len(sequences) != len(raw_sequences):
            raise ValueError("Length of sequences and raw_sequences must match")

        self.sequences = sequences
        self.raw_sequences = raw_sequences
        self.stats = stats
        self.keys = keys
        # FIX: Save num_features immediately
        self.num_features = num_features
        self.datachannels = datachannels

        # --- Filtering Logic ---
        if 'filter_direction' in kwargs and kwargs['filter_direction'] in ['LONG', 'SHORT']:
            filter_direction = kwargs['filter_direction']
            logging.info(f"Filtering sequences for direction: {filter_direction}")
            original_count = len(self.sequences)
            filtered_sequences = []
            filtered_keys = []
            close_idx = self.datachannels.index("close")

            for seq, key in zip(self.sequences, self.keys):
                start_price = seq[0, close_idx]
                end_price = seq[-1, close_idx]

                if filter_direction == 'LONG' and end_price > start_price:
                    filtered_sequences.append(seq)
                    filtered_keys.append(key)
                elif filter_direction == 'SHORT' and end_price < start_price:
                    filtered_sequences.append(seq)
                    filtered_keys.append(key)

            if not filtered_sequences:
                logging.warning(f"Filtering for {filter_direction} resulted in zero sequences. Disabling filter.")
            else:
                self.sequences = filtered_sequences
                self.keys = filtered_keys
                logging.info(f"Filtered sequences: {original_count} -> {len(self.sequences)}")

        self.render_mode = render_mode
        self.initial_balance = initial_balance
        self.pre_signal_len = pre_signal_len
        self.slippage = slippage
        self.transaction_fee = transaction_fee
        self.agent_session_len = agent_session_len
        self.agent_history_len = agent_history_len
        self.input_history_len = input_history_len
        self.pricechannels = pricechannels
        self.volumechannels = volumechannels
        self.otherchannels = otherchannels
        self.action_history_len = action_history_len
        self.num_actions = num_actions
        self.inaction_penalty_ratio = inaction_penalty_ratio
        self.backtest_mode = backtest_mode
        self.use_risk_management = use_risk_management
        self.cnn_format = cnn_format
        self.position_fraction = position_fraction
        self.order_size_usdt = order_size_usdt
        self.bankruptcy_threshold = bankruptcy_threshold
        self.bankruptcy_penalty = bankruptcy_penalty
        self.max_drawdown_threshold = max_drawdown_threshold
        self.max_drawdown_penalty = max_drawdown_penalty
        self.max_drawdown_penalty_type = max_drawdown_penalty_type
        # New reward shaping
        self.new_equity_peak_reward = new_equity_peak_reward
        self.perfect_entry_reward = perfect_entry_reward
        self.risk_reward_ratio_threshold = risk_reward_ratio_threshold
        self.risk_reward_ratio_reward = risk_reward_ratio_reward
        self.continuous_pain_penalty_ratio = continuous_pain_penalty_ratio
        
        # Shaped rewards/penalties (previously hardcoded)
        self.good_exit_bonus = good_exit_bonus
        self.fast_exit_bonus = fast_exit_bonus
        self.low_balance_penalty = low_balance_penalty
        self.bankruptcy_slippage_penalty = bankruptcy_slippage_penalty
        self.holding_penalty_multiplier = holding_penalty_multiplier
        self.greed_penalty_multiplier = greed_penalty_multiplier
        # NEW: Asymmetric penalties
        self.premature_profit_exit_penalty = premature_profit_exit_penalty
        self.holding_loss_penalty = holding_loss_penalty
        # OLD: Kept for compatibility
        self.premature_exit_penalty = premature_exit_penalty
        self.profit_holding_bonus = profit_holding_bonus
        

        # Thresholds
        self.holding_penalty_threshold = holding_penalty_threshold
        self.greed_penalty_threshold = greed_penalty_threshold
        self.exit_quality_threshold = exit_quality_threshold
        self.fast_exit_threshold = fast_exit_threshold
        self.premature_exit_threshold = premature_exit_threshold
        # NEW: Thresholds for asymmetric logic
        self.profit_exit_threshold = profit_exit_threshold
        self.loss_exit_threshold = loss_exit_threshold
        self.allow_opposite_trades = allow_opposite_trades
        self.allowed_directions = allowed_directions

        # Определяем индекс действия "закрыть"
        self.close_action = close_action_index
        if self.close_action is None:
            self.close_action = self.num_actions - 1 if self.num_actions > 3 else -1 # -1 если close отключен


        self.seed_value = seed
        # Cache frequently used channel index
        self.close_idx = self.datachannels.index("close")

        self.history_vector_size = num_actions * self.action_history_len
        # Validate sequence shape
        # Support both (L, C) and (C, L, 1) formats
        expected_shape = (full_seq_len, num_features)
        if self.sequences and self.sequences[0].shape != expected_shape:
            # Try to reshape (C, L, 1) to (L, C)
            if len(self.sequences[0].shape) == 3 and self.sequences[0].shape[2] == 1:
                self.sequences = [seq.squeeze(-1).T for seq in self.sequences]
                logging.info(f"Reshaped sequences from (C, L, 1) to (L, C): {self.sequences[0].shape}")
            else:
                raise ValueError(f"Expected sequence shape {expected_shape}, but got {self.sequences[0].shape}")

        # Define observation and action spaces
        self.action_space = spaces.Discrete(num_actions)
        if self.cnn_format:
            # For CNN: (num_features + extras + action_history_onehot, agent_history_len)
            # We will treat extras and action history as additional channels
            num_extra_channels = 4 + (1 if self.action_history_len > 0 else 0)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(num_features + num_extra_channels, self.agent_history_len),
                dtype=np.float32
            )
        else:
            # For MLP: flat vector
            # FIX: Ensure buffer size matches actual data generation logic
            # Calculate expected size based on what _get_obs actually produces
            # Use local argument or self.num_features (now saved)
            real_data_size = (num_features * agent_history_len) + 4 + (num_actions * action_history_len)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(real_data_size,), dtype=np.float32)

        # PERFORMANCE: Pre-allocate reusable observation buffer
        # FIX: Ensure buffer size matches actual data generation logic
        # Re-allocate with correct size if mismatch
        self._obs_buffer = np.zeros((real_data_size,), dtype=np.float32)
        # Update space to match reality

        # For shaped reward function
        self._position_entry_step = None
        self._max_unrealized_pnl = 0.0
        self._min_unrealized_pnl = 0.0

        self._init_episode_vars()
    def _init_episode_vars(self) -> None:
        self.current_seq: Optional[np.ndarray] = None
        self.current_raw_seq: Optional[np.ndarray] = None
        self.current_asset_name: Optional[str] = None
        self.step_idx: int = 0
        self.balance: float = self.initial_balance
        self.position: int = 0
        self.entry_price: float = 0.0
        self.real_entry_price: float = 0.0 # Stores real entry price for PnL calculation
        # Фиксируем размер позиции на входе и используем до закрытия
        self.position_volume: float = 0.0
        self.realized_pnl: float = 0.0
        self.closed_trades: int = 0
        self.profitable_trades: int = 0
        self.last_step: bool = False
        # Отслеживание просадки
        self.equity_peak: float = self.initial_balance
        self.current_max_drawdown: float = 0.0
        
        # Reset shaped reward tracking
        self._position_entry_step = None
        self._max_unrealized_pnl = 0.0
        self._min_unrealized_pnl = 0.0
        
        if self.backtest_mode:
            self.total_commission: float = 0.0
            self.direction: Optional[str] = None
            self.trade_dt: dt.datetime = None
            if self.use_risk_management:
                self.trailing_max_price: float = None
                self.trailing_min_price: float = None
                self.tsl_price: float = None
                self.p_at_last_tsl_update: float = 0.0

        if self.action_history_len > 0:
            self.history_actions: List[Optional[int]] = [None] * self.action_history_len

    def _get_asset_stats(self) -> Dict[str, float]:
        """DEPRECATED: Helper to get stats for the current asset. Now returns empty dict."""
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to the beginning of a new episode.

        This method selects a new trading sequence, resets the portfolio
        balance, and returns the initial observation. The state `s_0` represents
        the starting point of a new trading session.

        Args:
            seed (Optional[int]): The seed for the random number generator.
            options (Optional[dict]): Additional options for resetting the
                environment.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: A tuple containing the initial
                observation and an info dictionary.
        """
        if seed is None:
            seed = self.seed_value
        super().reset(seed=seed)
        self._init_episode_vars()
        
        # Reset buffer for safety
        self._obs_buffer.fill(0.0)

        # Reset buffer for safety
        self._obs_buffer.fill(0.0)

        idx = self.np_random.integers(0, len(self.sequences)) if options is None else options["forced_index"]
        self.current_seq = self.sequences[idx]
        if self.raw_sequences:
            self.current_raw_seq = self.raw_sequences[idx]
        else:
            self.current_raw_seq = None # Fallback logic needed?
        # Extract asset name for logging/debugging
        key = self.keys[idx]
        if isinstance(key, tuple):
            self.current_asset_name = key[0]  # (ticker, dt)
        else:
            self.current_asset_name = str(key).split('_')[0]

        obs = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_human(info, first=True)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes a single time step in the environment.

        This method processes the agent's action, updates the environment's
        state, and calculates the reward. The state transition from `s_t` to
        `s_{t+1}` is determined by the market data and the agent's action `a_t`.
        The reward `r_t` is calculated based on the resulting change in
        portfolio value and any applicable shaped rewards or penalties.

        Args:
            action (int): The action selected by the agent.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: A tuple
                containing the next observation, the reward, a flag indicating
                if the episode has terminated, a flag indicating if the episode
                has been truncated, and an info dictionary.
        """
        assert self.current_seq is not None, "reset() must be called before step()"
        prev_position = self.position

        # Определяем действие "закрыть" (3 для num_actions=4, или -1 если close отключен)
        close_action = self.close_action

        self.last_step = self.step_idx == self.agent_session_len - 1
        if self.last_step:
            if self.position == 0 and action in {1, 2}:
                action = 0
            elif self.position != 0 and action != close_action and close_action != -1:
                action = close_action

        price_idx = min(self.pre_signal_len - 1 + self.step_idx, len(self.current_seq) - 1)
        if price_idx >= len(self.current_seq):
            price_idx = len(self.current_seq) - 1
        
        # --- Denormalization Setup ---
        if self.current_raw_seq is not None:
            # Берем реальную цену из сырых данных
            real_price = float(self.current_raw_seq[price_idx, self.close_idx])
            norm_price = self.current_seq[price_idx, self.close_idx] # Keep for agent observation
        else:
            # Fallback (Broken for log-returns, but kept for legacy compatibility)
            norm_price = self.current_seq[price_idx, self.close_idx]
            close_mean = self.stats["means"].get("close", 0.0)
            close_std = self.stats["stds"].get("close", 1.0)
            real_price = norm_price * close_std + close_mean
        
        # --- НАЧАЛО ИЗМЕНЕНИЙ: Принудительный запрет противоположных сделок ---
        if not self.allow_opposite_trades:
            is_long = self.position > 0
            is_short = self.position < 0

            # Если есть LONG, запрещаем SHORT (действие 2)
            if is_long and action == 2:
                action = 0  # Заменяем на HOLD
            # Если есть SHORT, запрещаем LONG (действие 1)
            elif is_short and action == 1:
                action = 0  # Заменяем на HOLD
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---

        pnl_change = 0.0
        trade_pnl = 0.0
        reward = 0.0  # Initialize reward

        # --- Risk-based Balance Check ---
        MIN_SAFE_FRACTION = 1.2  # 20% safety buffer above bankruptcy
        if action in [1, 2] and self.position == 0:
            if self.balance < self.bankruptcy_threshold * MIN_SAFE_FRACTION:
                logging.debug(
                    f"Balance {self.balance:.2f} is too close to bankruptcy threshold "
                    f"({self.bankruptcy_threshold:.2f}). Forcing HOLD."
                )
                action = 0  # Force HOLD
                reward -= self.low_balance_penalty # Penalize attempt

        # --- Position Opening ---
        if action == 1 and self.position == 0: # OPEN LONG
            if not self.allowed_directions or 'LONG' in self.allowed_directions:
                if self.order_size_usdt > 0:
                    trade_amount = min(self.order_size_usdt, self.balance * 0.95)  # Cap at 95% of balance
                else:
                    trade_amount = self.balance * self.position_fraction
                    trade_amount = max(0.0, min(trade_amount, self.balance * 0.95)) # Ensure it's within 95% of balance

                if trade_amount > 0:
                    real_exec_price = real_price * (1 + self.slippage)
                    norm_exec_price = norm_price * (1 + self.slippage)

                    self.position = 1
                    self.entry_price = norm_exec_price      # Store NORMALIZED price
                    self.real_entry_price = real_exec_price # Store REAL price
                    
                    volume = trade_amount / real_exec_price
                    self.position_volume = volume
                    
                    fee = real_exec_price * volume * self.transaction_fee
                    pnl_change -= fee
                else:
                    # Balance too small for a trade, force HOLD and penalize
                    action = 0
                    reward = -self.low_balance_penalty

        elif action == 2 and self.position == 0: # OPEN SHORT
            if not self.allowed_directions or 'SHORT' in self.allowed_directions:
                if self.order_size_usdt > 0:
                    trade_amount = min(self.order_size_usdt, self.balance * 0.95)  # Cap at 95% of balance
                else:
                    trade_amount = self.balance * self.position_fraction
                    trade_amount = max(0.0, min(trade_amount, self.balance * 0.95)) # Ensure it's within 95% of balance

                if trade_amount > 0:
                    real_exec_price = real_price * (1 - self.slippage)
                    norm_exec_price = norm_price * (1 - self.slippage)

                    self.position = -1
                    self.entry_price = norm_exec_price      # Store NORMALIZED price
                    self.real_entry_price = real_exec_price # Store REAL price

                    volume = trade_amount / real_exec_price
                    self.position_volume = volume
                    
                    fee = real_exec_price * volume * self.transaction_fee
                    pnl_change -= fee
                else:
                    # Balance too small for a trade, force HOLD and penalize
                    action = 0
                    reward = -self.low_balance_penalty

        # --- Position Closing ---
        elif action == close_action and self.position != 0 and close_action != -1:
            volume = self.position_volume
            
            if self.position == 1: # CLOSE LONG
                real_exec_price = real_price * (1 - self.slippage)
                trade_pnl = (real_exec_price - self.real_entry_price) * volume
            else: # CLOSE SHORT
                real_exec_price = real_price * (1 + self.slippage)
                trade_pnl = (self.real_entry_price - real_exec_price) * volume
            
            fee = real_exec_price * volume * self.transaction_fee
            pnl_change += trade_pnl - fee
            
            self.closed_trades += 1
            if trade_pnl > 0:
                self.profitable_trades += 1
            self.position = 0
            self.position_volume = 0.0

        self.realized_pnl += pnl_change
        self.balance += pnl_change

        # LIQUIDATION / BANKRUPTCY CHECK
        terminated = False
        info = self._get_info()

        unrealized_pnl = self._calculate_unrealized_pnl()
        equity = self.balance + unrealized_pnl

        if self.position != 0:
            position_value = self.real_entry_price * self.position_volume
            maintenance_margin = position_value * 0.05
            if equity < maintenance_margin:
                terminated = True
                reward -= 10.0  # Жесткий штраф за ликвидацию
                info["liquidation"] = True

        if not terminated and self.balance <= self.bankruptcy_threshold:
            terminated = True
            reward = -self.bankruptcy_penalty

        if terminated:
            if self.position != 0:
                self.balance += unrealized_pnl # Realize the loss
                self.position = 0
                self.position_volume = 0.0
            
            self.balance = max(0.0, self.balance)
            info['bankruptcy'] = True # for metrics
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, reward, terminated, False, info

        if action == 0 and prev_position == 0:
            inaction_penalty = self.inaction_penalty_ratio
        else:
            inaction_penalty = 0.0

        if self.action_history_len > 0:
            self.history_actions.pop(0)
            self.history_actions.append(action)

        self.step_idx += 1
        
        terminated = self.step_idx >= self.agent_session_len
        
        # Track position metrics for shaped reward
        self._track_position_metrics(action, prev_position)
        
        # Use shaped reward
        reward = self._calculate_shaped_reward(
            pnlchange=pnl_change,
            inaction_penalty=inaction_penalty,
            action=action,
            prev_position=prev_position,
            trade_pnl=trade_pnl
        )
        
        # --- Drawdown Calculation (using real financial values) ---
        portfolio_value = self.balance
        if self.position != 0:
            m2m_price_idx = min(len(self.current_seq) - 1, self.pre_signal_len - 1 + self.step_idx)
            if self.current_raw_seq is not None:
                real_m2m_price = float(self.current_raw_seq[m2m_price_idx, self.close_idx])
            else:
                norm_m2m_price = self.current_seq[m2m_price_idx, self.close_idx]
                real_m2m_price = norm_m2m_price * close_std + close_mean # Fallback

            if self.position == 1: # LONG
                mark2market = (real_m2m_price - self.real_entry_price) * self.position_volume
            elif self.position == -1: # SHORT
                mark2market = (self.real_entry_price - real_m2m_price) * self.position_volume
            else:
                mark2market = 0.0
            portfolio_value += mark2market
     
        info = self._get_info()
     
        if portfolio_value > self.equity_peak:
            if self.new_equity_peak_reward > 0:
                reward += self.new_equity_peak_reward
            self.equity_peak = portfolio_value
     
        current_drawdown = (portfolio_value - self.equity_peak) / self.equity_peak if self.equity_peak != 0 else 0.0
        if current_drawdown < self.current_max_drawdown:
            self.current_max_drawdown = current_drawdown
     
        drawdown_penalty = 0.0
        if self.max_drawdown_threshold is not None and self.current_max_drawdown < self.max_drawdown_threshold:
            if self.max_drawdown_penalty_type == 'proportional':
                excess = abs(self.current_max_drawdown - self.max_drawdown_threshold)
                drawdown_penalty = excess * self.max_drawdown_penalty
            else:
                drawdown_penalty = self.max_drawdown_penalty
            
            # НОВОЕ: Принудительно завершить эпизод
            reward -= drawdown_penalty
            terminated = True
            info["max_drawdown_exceeded"] = True
            info["terminal_observation"] = self._get_observation()
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            
            if self.render_mode == "human":
                self._render_human(info, action, reward)
            return obs, reward, terminated, False, info
            
        if terminated:
            info["terminal_observation"] = self._get_observation()
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            info.update({
                "episode_realized_pnl": self.realized_pnl,
                "episode_win_rate": self.profitable_trades / max(1, self.closed_trades),
                "episode_closed_trades": self.closed_trades,
                "episode_max_drawdown": self.current_max_drawdown,
            })
        else:
            obs = self._get_observation()
     
        reward -= drawdown_penalty

        # НОВОЕ: Штраф за каждый шаг с открытой убыточной позицией (Continuous Penalty for unrealized losses)
        if self.position != 0 and self.continuous_pain_penalty_ratio > 0:
            unrealized_pnl = self._calculate_unrealized_pnl()
            if unrealized_pnl < 0:
                # Штраф пропорционален убытку
                pain_penalty = abs(unrealized_pnl) / self.initial_balance * self.continuous_pain_penalty_ratio
                reward -= pain_penalty
     
        if self.render_mode == "human":
            self._render_human(info, action, reward)
     
        return obs, reward, terminated, False, info

    def _track_position_metrics(self, action: int, prev_position: int):
        """Tracks metrics related to the current position for shaped rewards."""
        # Position just opened
        if self.position != 0 and prev_position == 0:
            self._position_entry_step = self.step_idx
            self._max_unrealized_pnl = 0.0
            self._min_unrealized_pnl = 0.0
            return

        # Position is open
        if self.position != 0:
            # Calculate current unrealized PnL
            price_idx = min(len(self.current_seq) - 1, self.pre_signal_len + self.step_idx - 1)
            
            if self.current_raw_seq is not None:
                real_current_price = float(self.current_raw_seq[price_idx, self.close_idx])
            else:
                # Denormalize for real PnL calculation (Fallback)
                current_price = self.current_seq[price_idx, self.close_idx]
                close_mean = self.stats["means"].get("close", 0.0)
                close_std = self.stats["stds"].get("close", 1.0)
                real_current_price = current_price * close_std + close_mean
            
            if self.position == 1: # LONG
                unrealized_pnl = (real_current_price - self.real_entry_price) * self.position_volume
            else: # SHORT
                unrealized_pnl = (self.real_entry_price - real_current_price) * self.position_volume
            
            # Update max/min unrealized PnL
            self._max_unrealized_pnl = max(self._max_unrealized_pnl, unrealized_pnl)
            self._min_unrealized_pnl = min(self._min_unrealized_pnl, unrealized_pnl)

    def _calculate_shaped_reward(self, pnlchange: float, inaction_penalty: float, action: int, prev_position: int, trade_pnl: float) -> float:
        base_reward = pnlchange / self.initial_balance
        shaped_reward = 0.0

        # 1. Бонус за HOLD в низкой волатильности (ИСПРАВЛЕНО)
        if action == 0 and prev_position == 0:  # HOLD вне позиции
            if self.step_idx >= 5:
                end_idx = self.pre_signal_len + self.step_idx
                start_idx = end_idx - 5
                
                if self.current_raw_seq is not None:
                    real_prices = self.current_raw_seq[start_idx:end_idx, self.close_idx]
                else:
                    # Денормализовать для корректного расчёта (Fallback)
                    recent_prices = self.current_seq[start_idx:end_idx, self.close_idx]
                    close_mean = self.stats["means"].get("close", 0.0)
                    close_std = self.stats["stds"].get("close", 1.0)
                    real_prices = recent_prices * close_std + close_mean
                
                # Вычислить процентное изменение
                price_change_pct = abs((real_prices[-1] - real_prices[0]) / real_prices[0])
                
                # Если изменение < 0.5% за 5 минут → флэт
                if price_change_pct < 0.005:  # < 0.5%
                    shaped_reward += 0.002  # Бонус за HOLD

        holding_duration = 0
        if self._position_entry_step is not None:
            holding_duration = self.step_idx - self._position_entry_step

        # 2. Holding Penalty (Progressive) and Bonus
        if self.position != 0 and self._position_entry_step is not None:
            unrealized_pnl = self._calculate_unrealized_pnl()
            
            # Штраф за удержание убыточной позиции
            if holding_duration > self.holding_penalty_threshold and unrealized_pnl < 0:
                penalty_factor = (holding_duration - self.holding_penalty_threshold) / self.agent_session_len
                shaped_reward -= penalty_factor * self.holding_penalty_multiplier
            
            # Бонус за удержание прибыльной позиции (УЛУЧШЕНО)
            if unrealized_pnl > 0 and holding_duration > 5:
                # Проверить, что прибыль не откатывается
                profit_retracement = (self._max_unrealized_pnl - unrealized_pnl) / max(self._max_unrealized_pnl, 1e-8)
                
                # Давать бонус только если откат < 20%
                if profit_retracement < 0.20:
                    holding_bonus = (holding_duration / self.agent_session_len) * 0.02
                    shaped_reward += holding_bonus
        
        # Penalties and bonuses applied upon closing a position
        # Определяем действие "закрыть" (3 для num_actions=4, или -1 если close отключен)
        close_action = self.close_action
        if action == close_action and prev_position != 0 and close_action != -1:

            # --- НОВАЯ АСИММЕТРИЧНАЯ ЛОГИКА ---
            # 1. Штраф за ранний выход из ПРИБЫЛЬНОЙ позиции
            if self.premature_profit_exit_penalty > 0 and trade_pnl > 0:
                if holding_duration < self.profit_exit_threshold:
                    shaped_reward -= self.premature_profit_exit_penalty
            
            # 2. Штраф за долгое удержание УБЫТОЧНОЙ позиции
            if self.holding_loss_penalty > 0 and trade_pnl < 0:
                if holding_duration > self.loss_exit_threshold:
                    shaped_reward -= self.holding_loss_penalty

            # 2. Greed Penalty + 3. Exit Bonus
            if self._max_unrealized_pnl > 0 and trade_pnl > 0:
                profit_retracement = (self._max_unrealized_pnl - trade_pnl) / self._max_unrealized_pnl
                # Greed penalty only if retracement is significant
                if self.greed_penalty_multiplier > 0 and profit_retracement > self.greed_penalty_threshold:
                    shaped_reward -= profit_retracement * self.greed_penalty_multiplier
                
                if self.good_exit_bonus > 0 and trade_pnl >= self._max_unrealized_pnl * self.exit_quality_threshold:
                    exit_bonus = self.good_exit_bonus

                    # 4. Fast Exit Bonus
                    if holding_duration < self.fast_exit_threshold:
                        exit_bonus += self.fast_exit_bonus

                    shaped_reward += exit_bonus

            # 6. NEW: Perfect Entry Reward
            if self.perfect_entry_reward > 0 and trade_pnl > 0 and self._min_unrealized_pnl >= 0:
                shaped_reward += self.perfect_entry_reward

            # 7. NEW: Risk/Reward Ratio Reward
            if self.risk_reward_ratio_reward > 0 and trade_pnl > 0:
                max_profit = self._max_unrealized_pnl
                max_loss = abs(self._min_unrealized_pnl)
                
                if max_loss > 0 and (max_profit / max_loss) > self.risk_reward_ratio_threshold:
                    shaped_reward += self.risk_reward_ratio_reward

        return base_reward + shaped_reward - inaction_penalty

    def _get_observation(self) -> np.ndarray:
        # The window from current_seq is already pre-normalized.
        # load_and_prep_data has already performed Z-normalization.
        end = self.pre_signal_len + self.step_idx
        start = end - self.agent_history_len
        raw_window = self.current_seq[start:end]  # shape: (agent_history_len, num_features)

        # For compatibility with the rest of the logic, we assume input_history_len == agent_history_len.
        # No additional normalization is performed here.
        normalized = raw_window.astype(np.float32)

        unrealized = 0.0
        if self.position != 0:
            # FIX: Use the last price from the *current* observation window, not a future price.
            # The observation window ends at `self.pre_signal_len + self.step_idx`, so the last element is at index -1 of that slice.
            price_idx = min(len(self.current_seq) - 1, self.pre_signal_len + self.step_idx - 1)
            current_price = self.current_seq[price_idx, self.close_idx]
            delta = (current_price - self.entry_price) * self.position
            unrealized = delta / self.entry_price

        time_elapsed = float(self.step_idx) / self.agent_session_len
        time_remaining = float(self.agent_session_len - self.step_idx) / self.agent_session_len
        extras = np.array(
            [
                float(self.position),
                unrealized,
                time_elapsed,
                time_remaining,
            ],
            dtype=np.float32,
        )

        # LEGACY PATH (CNN)
        if self.cnn_format:
            # For CNNs, we treat extras and history as additional channels
            # Shape (L, C) -> (C, L)
            obs = normalized.T

            # Create channels for extras and broadcast to length of sequence
            extras_channels = np.repeat(extras[:, np.newaxis], self.agent_history_len, axis=1)
            obs = np.vstack([obs, extras_channels])

            if self.action_history_len > 0:
                hist_onehot = np.zeros(self.history_vector_size, dtype=np.float32)
                for idx, action in enumerate(self.history_actions):
                    if action is not None:
                        hist_onehot[idx * self.num_actions + action] = 1.0
                # Treat the whole history vector as one channel
                history_channel = np.repeat(hist_onehot[np.newaxis, :], self.agent_history_len, axis=0).T
                # This seems complex. A simpler way is to just have one channel for the last action
                # For now, let's just append it as a flat vector, which is not ideal for CNNs.
                # A better approach might be to rethink history for CNNs.
                # Let's create a single channel representing the one-hot encoded history vector, repeated.
                # This is not standard, but it's one way to fit it in.
                # A better CNN approach would use a separate MLP head for flat features.
                # For simplicity, we will just create one channel from the flattened history.
                history_channel = np.tile(hist_onehot, (self.agent_history_len, 1)).T
                obs = np.vstack([obs, history_channel]) # This will fail if history_vector_size > 1
            return obs.astype(np.float32)

        # OPTIMIZED PATH: Avoid malloc for MLP mode by filling pre-allocated buffer
        # 1. Fill History (Zero-Copy View)
        hist_len = normalized.size
        self._obs_buffer[:hist_len] = normalized.ravel()

        # 2. Fill Extra Features
        # extras = [position, unrealized_pnl, time_elapsed, time_remaining]
        self._obs_buffer[hist_len:hist_len+4] = extras

        # 3. Fill Action History
        if self.action_history_len > 0:
            hist_onehot = np.zeros(self.history_vector_size, dtype=np.float32)
            for idx, action in enumerate(self.history_actions):
                if action is not None:
                    hist_onehot[idx * self.num_actions + action] = 1.0
            self._obs_buffer[hist_len+4:] = hist_onehot

        return self._obs_buffer

    def _get_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "step": self.step_idx,
            "balance": self.balance,
            "position": self.position,
        }

        if self.position != 0:
            exec_delay = getattr(self, "exec_delay_bars", 0)
            price_idx = min(len(self.current_seq) - 1, self.pre_signal_len - 1 + self.step_idx + exec_delay)
            current_price = self.current_seq[price_idx, self.close_idx]
            # MTM по зафиксированному объёму, а не по "текущий баланс / entry_price"
            mark2market = (current_price - self.entry_price) * self.position * self.position_volume
            info["portfolio_value"] = self.balance + mark2market
        else:
            info["portfolio_value"] = self.balance
        return info

    def _calculate_unrealized_pnl(self) -> float:
        """Calculates the unrealized profit or loss for the current open position."""
        if self.position == 0:
            return 0.0

        # Determine the correct price index for the current step
        price_idx = min(len(self.current_seq) - 1, self.pre_signal_len - 1 + self.step_idx)
        
        if self.current_raw_seq is not None:
            real_current_price = float(self.current_raw_seq[price_idx, self.close_idx])
        else:
            # Get the normalized price from the sequence
            norm_current_price = self.current_seq[price_idx, self.close_idx]
            # Denormalize the price to get the real price (Fallback)
            close_mean = self.stats["means"].get("close", 0.0)
            close_std = self.stats["stds"].get("close", 1.0)
            real_current_price = norm_current_price * close_std + close_mean
        
        # Calculate PnL based on position direction
        if self.position == 1:  # LONG
            unrealized_pnl = (real_current_price - self.real_entry_price) * self.position_volume
        elif self.position == -1:  # SHORT
            unrealized_pnl = (self.real_entry_price - real_current_price) * self.position_volume
        else:
            unrealized_pnl = 0.0
            
        return unrealized_pnl

    def _calculate_effective_trail_distance(
        self, p: float, d0: float, d_min: float, fee_buf: float
    ) -> float:
        """Calculates the effective trailing stop distance based on profit."""
        # Until fees are covered (p <= fee_buf), use the initial distance d0.
        if p <= fee_buf:
            return d0
        # As profit increases, tighten the trail distance from d0 towards d_min.
        d_eff = d0 - (p - fee_buf)
        return max(d_min, d_eff)

    def backtest_step(
        self,
        action: int,
        signal_dt: dt.datetime,
        ticker: str,
        stop_loss: float = 0.02,
        take_profit: float = 0.04,
        trailing_stop: float = 0.01,
        trailing_stop_min: float = None,
        fee_buffer_mult: float = None,
        delta_p_hysteresis: float = None,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes a single time step in backtesting mode.

        This method is similar to `step` but is designed for backtesting. It
        includes more detailed trade execution logic, such as trailing stops
        and fee calculations. The reward is typically not used in backtesting,
        as performance is evaluated based on the final portfolio value.

        Args:
            action (int): The action selected by the agent.
            signal_dt (dt.datetime): The timestamp of the signal.
            ticker (str): The ticker symbol of the asset being traded.
            stop_loss (float): The stop loss percentage.
            take_profit (float): The take profit percentage.
            trailing_stop (float): The trailing stop percentage.
            trailing_stop_min (float): The minimum trailing stop percentage.
            fee_buffer_mult (float): A multiplier for the fee buffer.
            delta_p_hysteresis (float): The hysteresis for the delta price.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: A tuple
                containing the next observation, the reward (always 0 in
                backtest mode), a termination flag, a truncation flag, and an
                info dictionary with detailed trade information.
        """
        assert self.current_seq is not None, "reset() must be called before backtest_step()"

        self.last_step = self.step_idx == self.agent_session_len - 1
        if self.last_step:
            if self.position == 0 and action in {1, 2}:
                action = 0
            elif self.position != 0 and action != 3:
                action = 3

        exec_delay = getattr(self, "exec_delay_bars", 0)
        price_idx = min(self.pre_signal_len - 1 + self.step_idx + exec_delay, len(self.current_seq) - 1)

        # --- Denormalization Setup ---
        if self.current_raw_seq is not None:
            real_price = float(self.current_raw_seq[price_idx, self.close_idx])
            norm_price = self.current_seq[price_idx, self.close_idx]
        else:
            norm_price = self.current_seq[price_idx, self.close_idx]
            close_mean = self.stats["means"].get("close", 0.0)
            close_std = self.stats["stds"].get("close", 1.0)
            real_price = norm_price * close_std + close_mean  # <--- ВАЖНО: Мы используем это!

        position_closed = False
        pnl_change = 0.0
        trade_pnl = None
        exit_reason = ""
        
        # --- Strict Balance Check ---
        MIN_SAFE_FRACTION = 1.2
        if action in [1, 2] and self.position == 0:
            if self.balance <= self.bankruptcy_threshold * MIN_SAFE_FRACTION:
                 action = 0 # Force HOLD

        # --- Risk Management (CORRECTED: Uses REAL PRICE) ---
        if self.use_risk_management and self.position != 0:
            d0 = trailing_stop
            d_min = trailing_stop_min
            fee = self.transaction_fee
            fee_buf = fee * (fee_buffer_mult or 2.0)
            
            # Работаем только с REAL PRICE для TSL
            tsl_price = self.tsl_price
            
            # Инициализация tsl_price, если None (например, после загрузки состояния)
            if tsl_price is None:
                 # Для безопасности ставим очень далекий стоп при старте,
                 # он сразу же подтянется на следующей строке
                 tsl_price = -999999.0 if self.position == 1 else 999999.0

            if self.position == 1: # LONG
                # 1. Обновляем локальный максимум цены (Highest High)
                # Используем real_price!
                current_max = getattr(self, "trailing_max_price", real_price)
                if current_max is None: current_max = real_price # Защита
                self.trailing_max_price = max(current_max, real_price)
                
                # 2. Базовый уровень TSL (расстояние d0 от макс. цены)
                base_tsl = self.trailing_max_price * (1 - d0)
                
                # 3. Подтягиваем TSL (он не может идти вниз)
                tsl_price = max(tsl_price, base_tsl)
                
                # 4. Умное сужение (Smart tightening)
                if d_min is not None:
                    # Прибыль в процентах
                    p = max(0.0, self.trailing_max_price / self.real_entry_price - 1.0)
                    
                    # Гистерезис обновления
                    if delta_p_hysteresis is None or p >= self.p_at_last_tsl_update + (delta_p_hysteresis or 0.0):
                        if delta_p_hysteresis is not None: self.p_at_last_tsl_update = p
                        
                        if p <= fee_buf: 
                            d_eff = d0
                        else: 
                            d_eff = self._calculate_effective_trail_distance(p, d0, d_min, fee_buf)
                            
                        advanced_tsl_price = self.trailing_max_price * (1 - d_eff)
                        tsl_price = max(tsl_price, advanced_tsl_price)
                
                self.tsl_price = tsl_price
                trailing_trigger = real_price <= tsl_price # Сравниваем REAL с REAL

                # DIAGNOSTIC
                # if self.step_idx % 10 == 0:
                #    logger.info(f"[TSL-L] Step={self.step_idx}, Price={real_price:.2f}, TSL={tsl_price:.2f}, Trigger={trailing_trigger}")

            else: # SHORT
                # 1. Обновляем локальный минимум цены (Lowest Low)
                current_min = getattr(self, "trailing_min_price", real_price)
                if current_min is None: current_min = real_price
                self.trailing_min_price = min(current_min, real_price)
                
                # 2. Базовый уровень TSL
                base_tsl = self.trailing_min_price * (1 + d0)
                
                # 3. Подтягиваем TSL (он не может идти вверх)
                tsl_price = min(tsl_price, base_tsl)

                if d_min is not None:
                    p = max(0.0, 1.0 - self.trailing_min_price / self.real_entry_price)
                    
                    if delta_p_hysteresis is None or p >= self.p_at_last_tsl_update + (delta_p_hysteresis or 0.0):
                        if delta_p_hysteresis is not None: self.p_at_last_tsl_update = p
                        
                        if p <= fee_buf:
                            d_eff = d0
                        else:
                            d_eff = self._calculate_effective_trail_distance(p, d0, d_min, fee_buf)
                        
                        advanced_tsl_price = self.trailing_min_price * (1 + d_eff)
                        tsl_price = min(tsl_price, advanced_tsl_price)

                self.tsl_price = tsl_price
                trailing_trigger = real_price >= tsl_price

            # --- Trigger Logic ---
            if trailing_trigger or self.last_step:
                action = 3 # CLOSE
                if trailing_trigger:
                    exit_reason = "TSL"
                    # logger.info(f"🎯 TSL TRIGGERED! Pos={self.position}, Price={real_price:.4f}, TSL={tsl_price:.4f}")
                elif self.last_step:
                    exit_reason = "FORCED"

        current_dt = signal_dt + dt.timedelta(minutes=self.step_idx)

        # --- STRICT DIRECTION CONTROL ---
        if action == 1 and self.allowed_directions and 'LONG' not in self.allowed_directions:
            action = 0  # Force HOLD
        if action == 2 and self.allowed_directions and 'SHORT' not in self.allowed_directions:
            action = 0  # Force HOLD

        # --- Position Opening ---
        if action == 1 and self.position == 0: # OPEN LONG
            real_exec_price = real_price * (1 + self.slippage)
            norm_exec_price = norm_price * (1 + self.slippage)
            
            self.position = 1
            self.entry_price = norm_exec_price      # Store NORMALIZED price for agent
            self.real_entry_price = real_exec_price # Store REAL price for PnL

            if self.order_size_usdt > 0:
                trade_amount = min(self.order_size_usdt, self.balance * 0.95)
            else:
                trade_amount = self.balance * self.position_fraction
            trade_amount = max(0.0, min(trade_amount, self.balance * 0.95))
            
            volume = trade_amount / real_exec_price
            self.position_volume = volume
            
            fee = real_exec_price * volume * self.transaction_fee
            pnl_change -= fee
            self.total_commission += fee
            self.direction = "LONG"
            self.trade_dt = current_dt
            if self.use_risk_management:
                self.trailing_max_price = real_exec_price
                self.tsl_price = None
                self.p_at_last_tsl_update = 0.0
            self._position_entry_step = self.step_idx  # Запомнить шаг входа
            logging.info(f": (LONG) BUY {volume:.8f} {ticker} for {real_exec_price:.5f} at {current_dt.strftime('%Y-%m-%d %H:%M')}")

        elif action == 2 and self.position == 0: # OPEN SHORT
            real_exec_price = real_price * (1 - self.slippage)
            norm_exec_price = norm_price * (1 - self.slippage)

            self.position = -1
            self.entry_price = norm_exec_price      # Store NORMALIZED price
            self.real_entry_price = real_exec_price # Store REAL price

            if self.order_size_usdt > 0:
                trade_amount = min(self.order_size_usdt, self.balance * 0.95)
            else:
                trade_amount = self.balance * self.position_fraction
            trade_amount = max(0.0, min(trade_amount, self.balance * 0.95))
            
            volume = trade_amount / real_exec_price
            self.position_volume = volume
            
            fee = real_exec_price * volume * self.transaction_fee
            pnl_change -= fee
            self.total_commission += fee
            self.direction = "SHORT"
            self.trade_dt = current_dt
            if self.use_risk_management:
                self.trailing_min_price = real_exec_price
                self.tsl_price = None
                self.p_at_last_tsl_update = 0.0
            self._position_entry_step = self.step_idx  # Запомнить шаг входа
            logging.info(f": (SHORT) SELL {volume:.8f} {ticker} for {real_exec_price:.5f} at {current_dt.strftime('%Y-%m-%d %H:%M')}")

        # --- Position Closing ---
        elif action == 3 and self.position != 0:
            position_closed = True
            volume = self.position_volume
            was_long = (self.position == 1)
            
            if was_long:
                real_exec_price = real_price * (1 - self.slippage)
                trade_pnl = (real_exec_price - self.real_entry_price) * volume
                close_action = "SELL"
                trade_price_delta = (real_exec_price - self.real_entry_price) / self.real_entry_price
            else: # SHORT
                real_exec_price = real_price * (1 + self.slippage)
                trade_pnl = (self.real_entry_price - real_exec_price) * volume
                close_action = "BUY"
                trade_price_delta = (self.real_entry_price - real_exec_price) / self.real_entry_price

            fee = real_exec_price * volume * self.transaction_fee
            pnl_change += trade_pnl - fee
            self.total_commission += fee

            self.position = 0
            self.position_volume = 0.0

            if self.use_risk_management:
                brk = (
                    self.real_entry_price * (1 + self.transaction_fee) / (1 - self.transaction_fee)
                    if was_long else
                    self.real_entry_price * (1 - self.transaction_fee) / (1 + self.transaction_fee)
                )
                if exit_reason == "TSL":
                    exit_reason = "TSL" if ((real_exec_price > brk) if was_long else (real_exec_price < brk)) else "TSL SL"
                elif exit_reason == "FORCED":
                    exit_reason = "TSL Time" if ((real_exec_price > brk) if was_long else (real_exec_price < brk)) else "Time SL"

        self.realized_pnl += pnl_change
        self.balance += pnl_change
        
        single_trade_realized_pnl = 0.0
        if position_closed:
            opening_fee = self.real_entry_price * volume * self.transaction_fee
            single_trade_realized_pnl = trade_pnl - fee - opening_fee
            logging.info(
                f": (CLOSE) {close_action} {exit_reason} {volume:.8f} {ticker} for {real_exec_price:.5f} at "
                f"{current_dt.strftime('%Y-%m-%d %H:%M')} PnL = {single_trade_realized_pnl:+.2f}"
            )

        if self.action_history_len > 0:
            self.history_actions.pop(0)
            self.history_actions.append(action)

        self.step_idx += 1
        terminated = self.step_idx >= self.agent_session_len
        
        info = {}
        reward = 0.0
        # НОВОЕ: Проверка банкротства (как в step())
        if not terminated:
            # Рассчитать portfolio_value
            portfolio_value = self.balance
            if self.position != 0:
                m2m_price_idx = min(len(self.current_seq) - 1, self.pre_signal_len - 1 + self.step_idx)
                if self.current_raw_seq is not None:
                    real_m2m_price = float(self.current_raw_seq[m2m_price_idx, self.close_idx])
                else:
                    norm_m2m_price = self.current_seq[m2m_price_idx, self.close_idx]
                    close_mean = self.stats["means"].get("close", 0.0)
                    close_std = self.stats["stds"].get("close", 1.0)
                    real_m2m_price = norm_m2m_price * close_std + close_mean # Fallback

                if self.position == 1: # LONG
                    mark2market = (real_m2m_price - self.real_entry_price) * self.position_volume
                elif self.position == -1: # SHORT
                    mark2market = (self.real_entry_price - real_m2m_price) * self.position_volume
                else:
                    mark2market = 0.0
                portfolio_value += mark2market
            
            # Проверка банкротства (аналогично step())
            if portfolio_value <= self.bankruptcy_threshold:
                # Принудительно закрыть позицию для корректного PnL
                if self.position != 0:
                    volume = self.position_volume
                    if self.position == 1:  # LONG
                        real_exec_price = real_price * (1 - self.slippage)
                        trade_pnl = (real_exec_price - self.real_entry_price) * volume
                    else:  # SHORT
                        real_exec_price = real_price * (1 + self.slippage)
                        trade_pnl = (self.real_entry_price - real_exec_price) * volume
                    
                    fee = real_exec_price * volume * self.transaction_fee
                    pnl_change = trade_pnl - fee
                    self.balance += pnl_change
                    self.position = 0
                    self.position_volume = 0.0
                
                # Применить штраф и завершить
                reward = -self.bankruptcy_penalty
                terminated = True
                info["bankruptcy"] = True
                info["bankruptcy_equity"] = portfolio_value
        
        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)

        if position_closed:
            holding_duration = self.step_idx - (self._position_entry_step or 0)
            # Note: single_trade_realized_pnl and opening_fee were calculated above
            trade_info = {
                "position_closed": position_closed,
                "trade_realized_pnl": single_trade_realized_pnl,
                "trade_commission": fee + opening_fee,
                "total_commission": self.total_commission,
                "trade_amount": self.real_entry_price * volume,
                "trade_price_delta": trade_price_delta,
                "correct_prediction": single_trade_realized_pnl > 0.0,
                "direction": self.direction,
                "trade_dt": self.trade_dt,
                "exit_reason": exit_reason if self.use_risk_management else "",
                "tsl_triggered": isinstance(exit_reason, str) and exit_reason.startswith("TSL"),
                "holding_duration_bars": holding_duration,
                "holding_duration_minutes": holding_duration,
            }
            info.update(trade_info)
            self.direction = None
            self.trade_dt = None
            if self.use_risk_management:
                self.trailing_max_price = None
                self.trailing_min_price = None
                self.tsl_price = None
                self.p_at_last_tsl_update = 0.0
        else:
            info["position_closed"] = position_closed

        return obs, reward, terminated, False, info

    def _render_human(
        self, info: Dict[str, Any], action: Optional[int] = None, reward: Optional[float] = None, first: bool = False
    ) -> None:
        if first:
            logger.info(f"--- Episode started | Balance={info['balance']:.2f} ---")
        else:
            logger.info(
                f"Step={info['step']} | Action={action} | Position={info['position']} | "
                f"Balance={info['balance']:.2f} | Portfolio={info['portfolio_value']:.2f} | Reward={reward:.4f}"
            )

    def close(self) -> None:
        """Closes the environment and cleans up any resources.

        This method is called when the environment is no longer needed. It can be
        used to close any open files or network connections.
        """
        logger.info("TradingEnvironment closed.")