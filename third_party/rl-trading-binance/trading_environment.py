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
        **kwargs,
    ) -> None:
        if not sequences:
            raise ValueError("`sequences` must be a non-empty list of arrays")
        if not keys:
            raise ValueError("`keys` must be a non-empty list of strings")
        if len(sequences) != len(keys):
            raise ValueError("Length of `sequences` and `keys` must be the same")

        self.sequences = sequences
        self.stats = stats
        self.keys = keys
        self.render_mode = render_mode
        self.initial_balance = initial_balance
        self.pre_signal_len = pre_signal_len
        self.datachannels = datachannels
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
                    self.observation_space = spaces.Box(
                        low=-np.inf, high=np.inf, shape=(flat_state_size + self.history_vector_size,), dtype=np.float32
                    )
            
                    # For shaped reward function
                    self._position_entry_step = None
                    self._max_unrealized_pnl = 0.0
                    self._min_unrealized_pnl = 0.0
            
                    self._init_episode_vars()
    def _init_episode_vars(self) -> None:
        self.current_seq: Optional[np.ndarray] = None
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
        """Helper to get stats for the current asset, with a fallback."""
        if not self.stats:
            raise ValueError("Normalization stats are not provided to the environment.")
        
        asset_stats = self.stats.get(self.current_asset_name)
        if asset_stats is None:
            fallback_asset = next(iter(self.stats))
            logging.warning(
                f"Stats for asset '{self.current_asset_name}' not found. "
                f"Falling back to stats of '{fallback_asset}'."
            )
            asset_stats = self.stats[fallback_asset]
        return asset_stats

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._init_episode_vars()

        idx = self.np_random.integers(0, len(self.sequences)) if options is None else options["forced_index"]
        self.current_seq = self.sequences[idx]
        try:
            self.current_asset_name = self.keys[idx].split('_')[0]
        except IndexError:
            logging.error(f"Could not parse asset name from key: {self.keys[idx]}")
            self.current_asset_name = "UNKNOWN"

        obs = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_human(info, first=True)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.current_seq is not None, "reset() must be called before step()"
        prev_position = self.position

        self.last_step = self.step_idx == self.agent_session_len - 1
        if self.last_step:
            if self.position == 0 and action in {1, 2}:
                action = 0
            elif self.position != 0 and action != 3:
                action = 3

        price_idx = min(self.pre_signal_len - 1 + self.step_idx, len(self.current_seq) - 1)
        if price_idx >= len(self.current_seq):
            price_idx = len(self.current_seq) - 1
        
        # --- Denormalization Setup ---
        asset_stats = self._get_asset_stats()
        norm_price = self.current_seq[price_idx, self.close_idx]
        close_mean = asset_stats['mean'][self.close_idx]
        close_std = asset_stats['std'][self.close_idx]
        real_price = norm_price * close_std + close_mean
        
        pnl_change = 0.0
        trade_pnl = 0.0

        # --- Position Opening ---
        if action == 1 and self.position == 0: # OPEN LONG
            real_exec_price = real_price * (1 + self.slippage)
            norm_exec_price = norm_price * (1 + self.slippage)

            self.position = 1
            self.entry_price = norm_exec_price      # Store NORMALIZED price
            self.real_entry_price = real_exec_price # Store REAL price

            if self.order_size_usdt > 0: trade_amount = self.order_size_usdt
            else: trade_amount = self.balance * self.position_fraction
            
            volume = trade_amount / real_exec_price
            self.position_volume = volume
            
            fee = real_exec_price * volume * self.transaction_fee
            pnl_change -= fee

        elif action == 2 and self.position == 0: # OPEN SHORT
            real_exec_price = real_price * (1 - self.slippage)
            norm_exec_price = norm_price * (1 - self.slippage)

            self.position = -1
            self.entry_price = norm_exec_price      # Store NORMALIZED price
            self.real_entry_price = real_exec_price # Store REAL price

            if self.order_size_usdt > 0: trade_amount = self.order_size_usdt
            else: trade_amount = self.balance * self.position_fraction
            
            volume = trade_amount / real_exec_price
            self.position_volume = volume
            
            fee = real_exec_price * volume * self.transaction_fee
            pnl_change -= fee

        # --- Position Closing ---
        elif action == 3 and self.position != 0:
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
        
        # --- Bankruptcy & Drawdown Calculation (using real financial values) ---
        portfolio_value = self.balance
        if self.position != 0:
            m2m_price_idx = min(len(self.current_seq) - 1, self.pre_signal_len - 1 + self.step_idx)
            norm_m2m_price = self.current_seq[m2m_price_idx, self.close_idx]
            real_m2m_price = norm_m2m_price * close_std + close_mean
            mark2market = (real_m2m_price - self.real_entry_price) * self.position_volume
            portfolio_value += mark2market
     
        info = self._get_info()
     
        if portfolio_value > self.equity_peak:
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
     
        if portfolio_value <= self.bankruptcy_threshold:
            if self.position != 0:
                # Force close position at current real price for accurate reward
                real_m2m_price = (self.current_seq[price_idx, self.close_idx] * close_std) + close_mean
                if self.position == 1:
                    real_exec_price = real_m2m_price * (1 - self.slippage)
                    trade_pnl = (real_exec_price - self.real_entry_price) * self.position_volume
                else:
                    real_exec_price = real_m2m_price * (1 + self.slippage)
                    trade_pnl = (self.real_entry_price - real_exec_price) * self.position_volume
                
                fee = real_exec_price * self.position_volume * self.transaction_fee
                pnl_change = trade_pnl - fee
                reward += pnl_change / self.initial_balance
            
            reward -= self.bankruptcy_penalty
            terminated = True
            info["bankruptcy"] = True
            info["bankruptcy_equity"] = portfolio_value
            
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
            current_price = self.current_seq[price_idx, self.close_idx]
            
            # Denormalize for real PnL calculation
            asset_stats = self._get_asset_stats()
            close_mean = asset_stats['mean'][self.close_idx]
            close_std = asset_stats['std'][self.close_idx]
            
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
        
        holding_duration = 0
        if self._position_entry_step is not None:
            holding_duration = self.step_idx - self._position_entry_step

        # 1. Holding Penalty (Progressive)
        # Applied when the position is still open
        if self.position != 0 and self._position_entry_step is not None:
            if holding_duration > 15:
                # Calculate unrealized PnL to check if the position is at a loss
                price_idx = min(len(self.current_seq) - 1, self.pre_signal_len + self.step_idx - 1)
                current_price = self.current_seq[price_idx, self.close_idx]
                
                asset_stats = self._get_asset_stats()
                close_mean = asset_stats['mean'][self.close_idx]
                close_std = asset_stats['std'][self.close_idx]
                real_current_price = current_price * close_std + close_mean
                
                if self.position == 1: # LONG
                    unrealized_pnl = (real_current_price - self.real_entry_price) * self.position_volume
                else: # SHORT
                    unrealized_pnl = (self.real_entry_price - real_current_price) * self.position_volume

                if unrealized_pnl < 0:  # Position at loss
                    penalty_factor = (holding_duration - 15) / self.agent_session_len
                    shaped_reward -= penalty_factor * 0.5
        
        # Penalties and bonuses applied upon closing a position
        if action == 3 and prev_position != 0:
            # 2. Greed Penalty + 3. Exit Bonus
            if self._max_unrealized_pnl > 0 and trade_pnl > 0:
                profit_retracement = (self._max_unrealized_pnl - trade_pnl) / self._max_unrealized_pnl
                if profit_retracement > 0.50:
                    shaped_reward -= profit_retracement * 0.3
                
                if trade_pnl >= self._max_unrealized_pnl * 0.80:
                    exit_bonus = 0.15
                    # 4. Fast Exit Bonus
                    if holding_duration < 20:
                        exit_bonus += 0.10
                    shaped_reward += exit_bonus
            
            # 5. Premature Exit Penalty
            if holding_duration < 5 and trade_pnl > 0:
                shaped_reward -= 0.02

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

        if self.action_history_len > 0:
            hist_onehot = np.zeros(self.history_vector_size, dtype=np.float32)
            for idx, action in enumerate(self.history_actions):
                if action is not None:
                    hist_onehot[idx * self.num_actions + action] = 1.0
            return np.concatenate([normalized.flatten(), extras, hist_onehot])
        
        return np.concatenate([normalized.flatten(), extras])

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
        asset_stats = self._get_asset_stats()
        norm_price = self.current_seq[price_idx, self.close_idx]
        close_mean = asset_stats['mean'][self.close_idx]
        close_std = asset_stats['std'][self.close_idx]
        real_price = norm_price * close_std + close_mean

        position_closed = False
        pnl_change = 0.0
        trade_pnl = None
        exit_reason = ""

        # --- Risk Management (uses normalized prices) ---
        if self.use_risk_management and self.position != 0:
            d0 = trailing_stop
            d_min = trailing_stop_min
            fee = self.transaction_fee
            fee_buf = fee * (fee_buffer_mult or 2.0)
            tsl_price = self.tsl_price

            if self.position == 1:  # LONG
                self.trailing_max_price = max(getattr(self, "trailing_max_price", norm_price), norm_price)
                base_tsl = self.trailing_max_price * (1 - d0)
                tsl_price = max(base_tsl, tsl_price) if tsl_price is not None else base_tsl

                if d_min is not None:
                    p = max(0.0, self.trailing_max_price / self.entry_price - 1.0)
                    if delta_p_hysteresis is None or p >= self.p_at_last_tsl_update + (delta_p_hysteresis or 0.0):
                        if delta_p_hysteresis is not None: self.p_at_last_tsl_update = p
                        if p <= fee_buf: d_eff = d0
                        else: d_eff = self._calculate_effective_trail_distance(p, d0, d_min, fee_buf)
                        advanced_tsl_price = self.trailing_max_price * (1 - d_eff)
                        tsl_price = max(tsl_price, advanced_tsl_price)
                trailing_trigger = norm_price <= tsl_price
            else:  # SHORT
                self.trailing_min_price = min(getattr(self, "trailing_min_price", norm_price), norm_price)
                base_tsl = self.trailing_min_price * (1 + d0)
                tsl_price = min(base_tsl, tsl_price) if tsl_price is not None else base_tsl

                if d_min is not None:
                    p = max(0.0, 1.0 - self.trailing_min_price / self.entry_price)
                    if delta_p_hysteresis is None or p >= self.p_at_last_tsl_update + (delta_p_hysteresis or 0.0):
                        if delta_p_hysteresis is not None: self.p_at_last_tsl_update = p
                        if p <= fee_buf: d_eff = d0
                        else: d_eff = self._calculate_effective_trail_distance(p, d0, d_min, fee_buf)
                        advanced_tsl_price = self.trailing_min_price * (1 + d_eff)
                        tsl_price = min(tsl_price, advanced_tsl_price) if tsl_price is not None else advanced_tsl_price
                trailing_trigger = norm_price >= tsl_price

            self.tsl_price = tsl_price
            sl_trigger = False
            tp_trigger = False

            if trailing_trigger or self.last_step:
                action = 3
                if trailing_trigger: exit_reason = "TSL"
                elif self.last_step: exit_reason = "FORCED"

        current_dt = signal_dt + dt.timedelta(minutes=self.step_idx)

        # --- Position Opening ---
        if action == 1 and self.position == 0: # OPEN LONG
            real_exec_price = real_price * (1 + self.slippage)
            norm_exec_price = norm_price * (1 + self.slippage)
            
            self.position = 1
            self.entry_price = norm_exec_price      # Store NORMALIZED price for agent
            self.real_entry_price = real_exec_price # Store REAL price for PnL

            if self.order_size_usdt > 0: trade_amount = self.order_size_usdt
            else: trade_amount = self.balance * self.position_fraction
            
            volume = trade_amount / real_exec_price
            self.position_volume = volume
            
            fee = real_exec_price * volume * self.transaction_fee
            pnl_change -= fee
            self.total_commission += fee
            self.direction = "LONG"
            self.trade_dt = current_dt
            if self.use_risk_management:
                self.trailing_max_price = norm_exec_price
                self.tsl_price = None
                self.p_at_last_tsl_update = 0.0
            logging.info(f": (LONG) BUY {volume:.8f} {ticker} for {real_exec_price:.5f} at {current_dt.strftime('%Y-%m-%d %H:%M')}")

        elif action == 2 and self.position == 0: # OPEN SHORT
            real_exec_price = real_price * (1 - self.slippage)
            norm_exec_price = norm_price * (1 - self.slippage)

            self.position = -1
            self.entry_price = norm_exec_price      # Store NORMALIZED price
            self.real_entry_price = real_exec_price # Store REAL price

            if self.order_size_usdt > 0: trade_amount = self.order_size_usdt
            else: trade_amount = self.balance * self.position_fraction
            
            volume = trade_amount / real_exec_price
            self.position_volume = volume
            
            fee = real_exec_price * volume * self.transaction_fee
            pnl_change -= fee
            self.total_commission += fee
            self.direction = "SHORT"
            self.trade_dt = current_dt
            if self.use_risk_management:
                self.trailing_min_price = norm_exec_price
                self.tsl_price = None
                self.p_at_last_tsl_update = 0.0
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
        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0

        if position_closed:
            # Note: single_trade_realized_pnl and opening_fee were calculated above
            info = {
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
            }
            self.direction = None
            self.trade_dt = None
            if self.use_risk_management:
                self.trailing_max_price = None
                self.trailing_min_price = None
                self.tsl_price = None
                self.p_at_last_tsl_update = 0.0
        else:
            info = {"position_closed": position_closed}

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
        logger.info("TradingEnvironment closed.")
