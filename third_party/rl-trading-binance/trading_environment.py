# trading_environment.py
import datetime as dt
import logging
import copy
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
        time_sl_penalty_ratio: float = 0.0,
        backtest_mode: bool = False,
        use_risk_management: bool = False,
        # TSL parameters
        trailing_stop: float = 0.07519862504113693,
        trailing_stop_min: float = 0.0008225518697224519,
        delta_p_hysteresis: float = 0.0015218098435784326,
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

        # Shaped rewards/penalties
        good_exit_bonus: float = 0.0,
        fast_exit_bonus: float = 0.0,
        low_balance_penalty: float = 0.0,
        bankruptcy_slippage_penalty: float = 0.0,
        holding_penalty_multiplier: float = 0.0,
        greed_penalty_multiplier: float = 0.0,
        # NEW: Asymmetric penalties
        premature_profit_exit_penalty: float = 0.0,
        holding_loss_penalty: float = 0.0,
        # OLD: Kept for compatibility
        premature_exit_penalty: float = 0.0,
        profit_holding_bonus: float = 0.0,
        
        # Thresholds for shaped rewards
        holding_penalty_threshold: int = 15,
        greed_penalty_threshold: float = 0.50,
        exit_quality_threshold: float = 0.80,
        fast_exit_threshold: int = 20,
        premature_exit_threshold: int = 5,

        # NEW: Thresholds for asymmetric logic
        profit_exit_threshold: int = 5,
        loss_exit_threshold: int = 3,
        allow_opposite_trades: bool = True,
        max_trades_per_episode: int = 100,
        close_action_index: Optional[int] = None,
        seed: Optional[int] = None,
        filter_direction: Optional[str] = None,
        allowed_directions: Optional[List[str]] = None,
        mirror_mode: bool = True,
        **kwargs,
    ) -> None:
        if not sequences:
            raise ValueError("`sequences` must be a non-empty list of arrays")
        if not keys:
            raise ValueError("`keys` must be a non-empty list of strings")
        if len(sequences) != len(keys):
            raise ValueError("Length of `sequences` and `keys` must be the same")

        self.stats = stats
        self.keys = keys
        self.num_features = num_features
        self.datachannels = datachannels
        self.sequences = sequences

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
        self.time_sl_penalty_ratio = time_sl_penalty_ratio
        self.backtest_mode = backtest_mode
        self.use_risk_management = use_risk_management
        self.trailing_stop = trailing_stop
        self.trailing_stop_min = trailing_stop_min
        self.delta_p_hysteresis = delta_p_hysteresis
        self.cnn_format = cnn_format
        self.position_fraction = position_fraction
        self.order_size_usdt = order_size_usdt
        self.bankruptcy_threshold = bankruptcy_threshold
        self.bankruptcy_penalty = bankruptcy_penalty
        self.max_drawdown_threshold = max_drawdown_threshold
        self.max_drawdown_penalty = max_drawdown_penalty
        self.max_drawdown_penalty_type = max_drawdown_penalty_type
        
        # Reward shaping
        self.new_equity_peak_reward = new_equity_peak_reward
        self.perfect_entry_reward = perfect_entry_reward
        self.risk_reward_ratio_threshold = risk_reward_ratio_threshold
        self.risk_reward_ratio_reward = risk_reward_ratio_reward
        self.continuous_pain_penalty_ratio = continuous_pain_penalty_ratio
        
        self.good_exit_bonus = good_exit_bonus
        self.fast_exit_bonus = fast_exit_bonus
        self.low_balance_penalty = low_balance_penalty
        self.bankruptcy_slippage_penalty = bankruptcy_slippage_penalty
        self.holding_penalty_multiplier = holding_penalty_multiplier
        self.greed_penalty_multiplier = greed_penalty_multiplier
        self.premature_profit_exit_penalty = premature_profit_exit_penalty
        self.holding_loss_penalty = holding_loss_penalty
        self.premature_exit_penalty = premature_exit_penalty
        self.profit_holding_bonus = profit_holding_bonus

        self.holding_penalty_threshold = holding_penalty_threshold
        self.greed_penalty_threshold = greed_penalty_threshold
        self.exit_quality_threshold = exit_quality_threshold
        self.fast_exit_threshold = fast_exit_threshold
        self.premature_exit_threshold = premature_exit_threshold
        self.profit_exit_threshold = profit_exit_threshold
        self.loss_exit_threshold = loss_exit_threshold
        self.allow_opposite_trades = allow_opposite_trades
        self.max_trades_per_episode = max_trades_per_episode
        self.allowed_directions = allowed_directions

        self.close_action = close_action_index
        if self.close_action is None:
            self.close_action = self.num_actions - 1 if self.num_actions > 3 else -1

        self.seed_value = seed
        self.close_idx = self.datachannels.index("close")
        self.history_vector_size = num_actions * self.action_history_len

        # 1. Validate and standardize sequence shape to (L, C)
        expected_shape = (full_seq_len, num_features)
        if self.sequences and self.sequences[0].shape != expected_shape:
            if len(self.sequences[0].shape) == 3 and self.sequences[0].shape[2] == 1:
                self.sequences = [seq.squeeze(-1).T for seq in self.sequences]
                logging.info(f"Reshaped sequences from (C, L, 1) to (L, C): {self.sequences[0].shape}")
            else:
                raise ValueError(f"Expected sequence shape {expected_shape}, but got {self.sequences[0].shape}")

        # 2. Inversion logic for Mirror Mode (SHORT specialist)
        if filter_direction == 'SHORT' and mirror_mode:
            logger.info("MIRROR MODE: Inverting sequences and stats for SHORT-only agent using geometric OHLC inversion.")
            idx = {name: i for i, name in enumerate(datachannels)}
            price_indices = [i for i, name in enumerate(datachannels) if name in pricechannels]

            mirrored = []
            for seq in self.sequences:
                m_seq = seq.copy()
                if price_indices:
                    m_seq[:, price_indices] *= -1.0

                if 'high' in idx and 'low' in idx:
                    h_idx, l_idx = idx['high'], idx['low']
                    m_seq[:, [h_idx, l_idx]] = m_seq[:, [l_idx, h_idx]]

                mirrored.append(m_seq)
            self.sequences = mirrored

            if self.stats:
                self.stats = copy.deepcopy(self.stats)
                for asset_stats in self.stats.values():
                    for i, (m, s) in enumerate(zip(asset_stats['mean'], asset_stats['std'])):
                        if i in price_indices:
                            asset_stats['mean'][i] = -m

                    if 'high' in idx and 'low' in idx:
                        h_idx, l_idx = idx['high'], idx['low']
                        asset_stats['mean'][h_idx], asset_stats['mean'][l_idx] = \
                            asset_stats['mean'][l_idx], asset_stats['mean'][h_idx]
                        asset_stats['std'][h_idx], asset_stats['std'][l_idx] = \
                            asset_stats['std'][l_idx], asset_stats['std'][h_idx]

        # 3. Filtering sequences by direction
        if filter_direction in ['LONG', 'SHORT']:
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
                elif filter_direction == 'SHORT':
                    is_trend = (end_price > start_price) if mirror_mode else (end_price < start_price)
                    if is_trend:
                        filtered_sequences.append(seq)
                        filtered_keys.append(key)

            if not filtered_sequences:
                logging.warning(f"Filtering for {filter_direction} resulted in zero sequences. Disabling filter.")
            else:
                self.sequences = filtered_sequences
                self.keys = filtered_keys
                logging.info(f"Filtered sequences: {original_count} -> {len(self.sequences)}")

        self.action_space = spaces.Discrete(num_actions)
        if self.cnn_format:
            num_extra_channels = 4 + (1 if self.action_history_len > 0 else 0)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(num_features + num_extra_channels, self.agent_history_len),
                dtype=np.float32
            )
        else:
            real_data_size = (num_features * agent_history_len) + 4 + (num_actions * action_history_len)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(real_data_size,), dtype=np.float32)

        self._obs_buffer = np.zeros((real_data_size,), dtype=np.float32)

        self._position_entry_step = None
        self._max_unrealized_pnl = 0.0
        self._min_unrealized_pnl = 0.0

        self.trailing_max_price: Optional[float] = None
        self.trailing_min_price: Optional[float] = None
        self.tsl_price: Optional[float] = None
        self.p_at_last_tsl_update: float = 0.0

        self._init_episode_vars()

    def _init_episode_vars(self) -> None:
        self.current_seq: Optional[np.ndarray] = None
        self.current_asset_name: Optional[str] = None
        self.step_idx: int = 0
        self.balance: float = self.initial_balance
        self.position: int = 0
        self.entry_price: float = 0.0
        self.real_entry_price: float = 0.0
        self.position_volume: float = 0.0
        self.realized_pnl: float = 0.0
        self.closed_trades: int = 0
        self.trades_count: int = 0
        self.profitable_trades: int = 0
        self.last_step: bool = False
        self.equity_peak: float = self.initial_balance
        self.current_max_drawdown: float = 0.0
        
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

        self.trailing_max_price = None
        self.trailing_min_price = None
        self.tsl_price = None
        self.p_at_last_tsl_update = 0.0

    def _get_asset_stats(self) -> Dict[str, float]:
        if not self.stats:
            raise ValueError("Normalization stats are not provided to the environment.")
        
        asset_stats = self.stats.get(self.current_asset_name)
        if asset_stats is None:
            raise ValueError(
                f"CRITICAL ERROR: Normalization stats for asset '{self.current_asset_name}' not found. "
                "Ensure your norm_stats.json is generated from a dataset containing ALL tickers."
            )
        return asset_stats

    def _calculate_effective_trail_distance(
        self, p: float, d0: float, d_min: float, fee_buf: float
    ) -> float:
        if p <= fee_buf:
            return d0
        d_eff = d0 - (p - fee_buf)
        return max(d_min, d_eff)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is None:
            seed = self.seed_value
        super().reset(seed=seed)
        self._init_episode_vars()
        
        self._obs_buffer.fill(0.0)

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
        close_action = self.close_action

        if self.step_idx >= self.agent_session_len - 1:
            if self.position == 0 and action in {1, 2}:
                 action = 0
            elif self.position != 0:
                 if self.close_action is not None and action != self.close_action:
                     action = self.close_action

        price_idx = min(self.pre_signal_len - 1 + self.step_idx, len(self.current_seq) - 1)
        if price_idx >= len(self.current_seq):
            price_idx = len(self.current_seq) - 1
        
        asset_stats = self._get_asset_stats()
        norm_price = self.current_seq[price_idx, self.close_idx]
        close_mean = asset_stats['mean'][self.close_idx]
        close_std = asset_stats['std'][self.close_idx]
        real_price = norm_price * close_std + close_mean
        
        if not self.allow_opposite_trades:
            is_long = self.position > 0
            is_short = self.position < 0
            if is_long and action == 2:
                action = 0
            elif is_short and action == 1:
                action = 0

        if action in [1, 2] and self.position == 0 and self.trades_count >= self.max_trades_per_episode:
            action = 0

        pnl_change = 0.0
        trade_pnl = 0.0
        reward = 0.0
        position_closed_this_step = False
        info = {}

        if self.use_risk_management and self.position != 0:
            d0 = self.trailing_stop
            d_min = self.trailing_stop_min
            delta_p = self.delta_p_hysteresis
            tsl_price = self.tsl_price

            if tsl_price is None:
                tsl_price = -np.inf if self.position == 1 else np.inf

            if self.position == 1:  # LONG
                self.trailing_max_price = max(self.trailing_max_price or real_price, real_price)
                if self.trailing_max_price < 0:
                    base_tsl = self.trailing_max_price * (1 + d0)
                else:
                    base_tsl = self.trailing_max_price * (1 - d0)
                tsl_price = max(tsl_price, base_tsl)

                p = max(0.0, self.trailing_max_price / self.real_entry_price - 1.0)
                if p >= self.p_at_last_tsl_update + delta_p:
                    self.p_at_last_tsl_update = p
                    d_eff = self._calculate_effective_trail_distance(p, d0, d_min, self.transaction_fee * 2)
                    if self.trailing_max_price < 0:
                        advanced_tsl_price = self.trailing_max_price * (1 + d_eff)
                    else:
                        advanced_tsl_price = self.trailing_max_price * (1 - d_eff)
                    tsl_price = max(tsl_price, advanced_tsl_price)

                self.tsl_price = tsl_price
                if real_price <= tsl_price and self.close_action is not None:
                    action = self.close_action
            else:  # SHORT
                self.trailing_min_price = min(self.trailing_min_price or real_price, real_price)
                if self.trailing_min_price < 0:
                    base_tsl = self.trailing_min_price * (1 - d0)
                else:
                    base_tsl = self.trailing_min_price * (1 + d0)
                tsl_price = min(tsl_price, base_tsl)

                p = max(0.0, 1.0 - self.trailing_min_price / self.real_entry_price)
                if p >= self.p_at_last_tsl_update + delta_p:
                    self.p_at_last_tsl_update = p
                    d_eff = self._calculate_effective_trail_distance(p, d0, d_min, self.transaction_fee * 2)
                    if self.trailing_min_price < 0:
                        advanced_tsl_price = self.trailing_min_price * (1 - d_eff)
                    else:
                        advanced_tsl_price = self.trailing_min_price * (1 + d_eff)
                    tsl_price = min(tsl_price, advanced_tsl_price)

                self.tsl_price = tsl_price
                if real_price >= tsl_price and self.close_action is not None:
                    action = self.close_action

        MIN_SAFE_FRACTION = 1.2
        if action in [1, 2] and self.position == 0:
            if self.balance < self.bankruptcy_threshold * MIN_SAFE_FRACTION:
                action = 0
                reward -= self.low_balance_penalty

        if action == 1 and self.position == 0:  # OPEN LONG
            if not self.allowed_directions or 'LONG' in self.allowed_directions:
                trade_amount = self.balance * self.position_fraction
                if self.order_size_usdt > 0:
                    trade_amount = min(self.order_size_usdt, self.balance * 0.95)

                if trade_amount > 0:
                    if real_price < 0:
                        real_exec_price = real_price * (1 - self.slippage)
                    else:
                        real_exec_price = real_price * (1 + self.slippage)
                    self.position = 1
                    self.entry_price = (real_exec_price - close_mean) / (close_std + 1e-8)
                    self.real_entry_price = real_exec_price
                    self.position_volume = trade_amount / abs(real_exec_price)
                    self.trades_count += 1
                    pnl_change -= self.position_volume * abs(real_exec_price) * self.transaction_fee
                    self.trailing_max_price = real_exec_price
                    self.tsl_price = None
                    self.p_at_last_tsl_update = 0.0
                else:
                    action = 0
                    reward = -self.low_balance_penalty

        elif action == 2 and self.position == 0:  # OPEN SHORT
            if not self.allowed_directions or 'SHORT' in self.allowed_directions:
                trade_amount = self.balance * self.position_fraction
                if self.order_size_usdt > 0:
                    trade_amount = min(self.order_size_usdt, self.balance * 0.95)

                if trade_amount > 0:
                    if real_price < 0:
                        real_exec_price = real_price * (1 + self.slippage)
                    else:
                        real_exec_price = real_price * (1 - self.slippage)
                    self.position = -1
                    self.entry_price = (real_exec_price - close_mean) / (close_std + 1e-8)
                    self.real_entry_price = real_exec_price
                    self.position_volume = trade_amount / abs(real_exec_price)
                    self.trades_count += 1
                    pnl_change -= self.position_volume * abs(real_exec_price) * self.transaction_fee
                    self.trailing_min_price = real_exec_price
                    self.tsl_price = None
                    self.p_at_last_tsl_update = 0.0
                else:
                    action = 0
                    reward = -self.low_balance_penalty

        elif action == close_action and self.position != 0 and close_action != -1:
            volume = self.position_volume
            if self.position == 1:  # CLOSE LONG
                if real_price < 0:
                    real_exec_price = real_price * (1 + self.slippage)
                else:
                    real_exec_price = real_price * (1 - self.slippage)
                trade_pnl = (real_exec_price - self.real_entry_price) * volume
            else:  # CLOSE SHORT
                if real_price < 0:
                    real_exec_price = real_price * (1 - self.slippage)
                else:
                    real_exec_price = real_price * (1 + self.slippage)
                trade_pnl = (self.real_entry_price - real_exec_price) * volume

            fee = abs(real_exec_price) * volume * self.transaction_fee
            pnl_change += trade_pnl - fee
            self.closed_trades += 1
            if trade_pnl > 0:
                self.profitable_trades += 1

            position_closed_this_step = True
            info.update({
                "position_closed": True,
                "trade_realized_pnl": trade_pnl - fee,
                "win_rate": 1.0 if trade_pnl > 0 else 0.0,
                "time_sl_penalty": 0.0,
            })

            self.position = 0
            self.position_volume = 0.0
            self.entry_price = 0.0
            self.real_entry_price = 0.0
            self._position_entry_step = None
            self.trailing_max_price = None
            self.trailing_min_price = None
            self.tsl_price = None

        self.realized_pnl += pnl_change
        self.balance += pnl_change

        if self.balance <= self.bankruptcy_threshold:
            logging.warning(f"BANKRUPTCY at step {self.step_idx}: balance={self.balance:.2f} USDT")
            
            if self.position != 0:
                slippage_penalty = self.bankruptcy_slippage_penalty
                if real_price < 0:
                    liquidation_price = real_price * (1 + slippage_penalty if self.position == 1 else 1 - slippage_penalty)
                else:
                    liquidation_price = real_price * (1 - slippage_penalty if self.position == 1 else 1 + slippage_penalty)
                liquidation_pnl = ((liquidation_price - self.real_entry_price) * self.position_volume 
                                   if self.position == 1 
                                   else (self.real_entry_price - liquidation_price) * self.position_volume)
                self.balance += liquidation_pnl
            
            self.balance = max(0.0, self.balance)
            reward = -self.bankruptcy_penalty
            terminated = True
            info = self._get_info()
            info['bankruptcy'] = True
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
        truncated = False

        self._track_position_metrics(action, prev_position)
        
        reward = self._calculate_shaped_reward(
            pnlchange=pnl_change,
            inaction_penalty=inaction_penalty,
            action=action,
            prev_position=prev_position,
            trade_pnl=trade_pnl
        )
        
        portfolio_value = self.balance
        if self.position != 0:
            m2m_price_idx = min(len(self.current_seq) - 1, self.pre_signal_len - 1 + self.step_idx)
            norm_m2m_price = self.current_seq[m2m_price_idx, self.close_idx]
            real_m2m_price = norm_m2m_price * close_std + close_mean
            if self.position == 1: # LONG
                mark2market = (real_m2m_price - self.real_entry_price) * self.position_volume
            elif self.position == -1: # SHORT
                mark2market = (self.real_entry_price - real_m2m_price) * self.position_volume
            else:
                mark2market = 0.0
            portfolio_value += mark2market
     
        base_info = self._get_info()
        base_info.update(info)
        info = base_info
     
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
            
            reward -= drawdown_penalty
            terminated = True
            info["max_drawdown_exceeded"] = True
            info["terminal_observation"] = self._get_observation()
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            
            if self.render_mode == "human":
                self._render_human(info, action, reward)
            return obs, reward, terminated, truncated, info
            
        if terminated:
            is_time_sl = self.position != 0 and not position_closed_this_step
            info["time_sl_penalty_applied"] = is_time_sl

            if is_time_sl:
                final_pnl = self._calculate_unrealized_pnl()
                reward -= self.time_sl_penalty_ratio
                logger.debug(f"Applied Time SL penalty: -{self.time_sl_penalty_ratio:.4f}")

                info.update({
                    "position_closed": True,
                    "trade_realized_pnl": final_pnl,
                    "win_rate": 1.0 if final_pnl > 0 else 0.0,
                    "time_sl_penalty": self.time_sl_penalty_ratio,
                })

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

        if self.position != 0 and self.continuous_pain_penalty_ratio > 0:
            unrealized_pnl = self._calculate_unrealized_pnl()
            if unrealized_pnl < 0:
                pain_penalty = abs(unrealized_pnl) / self.initial_balance * self.continuous_pain_penalty_ratio
                reward -= pain_penalty
     
        if self.render_mode == "human":
            self._render_human(info, action, reward)

        return obs, reward, terminated, truncated, info

    def _track_position_metrics(self, action: int, prev_position: int):
        if self.position != 0 and prev_position == 0:
            self._position_entry_step = self.step_idx
            self._max_unrealized_pnl = 0.0
            self._min_unrealized_pnl = 0.0
            return

        if self.position != 0:
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
            
            self._max_unrealized_pnl = max(self._max_unrealized_pnl, unrealized_pnl)
            self._min_unrealized_pnl = min(self._min_unrealized_pnl, unrealized_pnl)

    def _calculate_shaped_reward(self, pnlchange: float, inaction_penalty: float, action: int, prev_position: int, trade_pnl: float) -> float:
        base_reward = pnlchange / self.initial_balance
        shaped_reward = 0.0

        if action == 0 and prev_position == 0:
            if self.step_idx >= 5:
                end_idx = self.pre_signal_len + self.step_idx
                start_idx = end_idx - 5
                recent_prices = self.current_seq[start_idx:end_idx, self.close_idx]
                
                asset_stats = self._get_asset_stats()
                close_mean = asset_stats['mean'][self.close_idx]
                close_std = asset_stats['std'][self.close_idx]
                real_prices = recent_prices * close_std + close_mean
                
                price_change_pct = abs((real_prices[-1] - real_prices[0]) / real_prices[0])
                
                if price_change_pct < 0.005:
                    shaped_reward += 0.002

        holding_duration = 0
        if self._position_entry_step is not None:
            holding_duration = self.step_idx - self._position_entry_step

        if self.position != 0 and self._position_entry_step is not None:
            unrealized_pnl = self._calculate_unrealized_pnl()
            
            if holding_duration > self.holding_penalty_threshold and unrealized_pnl < 0:
                penalty_factor = (holding_duration - self.holding_penalty_threshold) / self.agent_session_len
                shaped_reward -= penalty_factor * self.holding_penalty_multiplier
            
            if unrealized_pnl > 0 and holding_duration > 5:
                profit_retracement = (self._max_unrealized_pnl - unrealized_pnl) / max(self._max_unrealized_pnl, 1e-8)
                
                if profit_retracement < 0.20:
                    holding_bonus = (holding_duration / self.agent_session_len) * 0.02
                    shaped_reward += holding_bonus
        
        close_action = self.close_action
        if action == close_action and prev_position != 0 and close_action != -1:
            if self.premature_profit_exit_penalty > 0 and trade_pnl > 0:
                if holding_duration < self.profit_exit_threshold:
                    shaped_reward -= self.premature_profit_exit_penalty
            
            if self.holding_loss_penalty > 0 and trade_pnl < 0:
                if holding_duration > self.loss_exit_threshold:
                    shaped_reward -= self.holding_loss_penalty

            if self._max_unrealized_pnl > 0 and trade_pnl > 0:
                profit_retracement = (self._max_unrealized_pnl - trade_pnl) / self._max_unrealized_pnl
                if self.greed_penalty_multiplier > 0 and profit_retracement > self.greed_penalty_threshold:
                    shaped_reward -= profit_retracement * self.greed_penalty_multiplier
                
                if self.good_exit_bonus > 0 and trade_pnl >= self._max_unrealized_pnl * self.exit_quality_threshold:
                    exit_bonus = self.good_exit_bonus
                    if holding_duration < self.fast_exit_threshold:
                        exit_bonus += self.fast_exit_bonus
                    shaped_reward += exit_bonus

            if self.perfect_entry_reward > 0 and trade_pnl > 0 and self._min_unrealized_pnl >= 0:
                shaped_reward += self.perfect_entry_reward

            if self.risk_reward_ratio_reward > 0 and trade_pnl > 0:
                max_profit = self._max_unrealized_pnl
                max_loss = abs(self._min_unrealized_pnl)
                
                if max_loss > 0 and (max_profit / max_loss) > self.risk_reward_ratio_threshold:
                    shaped_reward += self.risk_reward_ratio_reward

        return base_reward + shaped_reward - inaction_penalty

    def _get_observation(self) -> np.ndarray:
        max_len = len(self.current_seq)
        end = min(self.pre_signal_len + self.step_idx, max_len)
        start = end - self.agent_history_len
        if start < 0: start = 0

        raw_window = self.current_seq[start:end]

        if len(raw_window) < self.agent_history_len:
            pad_len = self.agent_history_len - len(raw_window)
            padding = np.zeros((pad_len, self.num_features), dtype=np.float32)
            raw_window = np.concatenate((padding, raw_window), axis=0)

        normalized = raw_window.astype(np.float32)

        unrealized = 0.0
        if self.position != 0:
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
            obs = normalized.T
            extras_channels = np.repeat(extras[:, np.newaxis], self.agent_history_len, axis=1)
            obs = np.vstack([obs, extras_channels])

            if self.action_history_len > 0:
                hist_onehot = np.zeros(self.history_vector_size, dtype=np.float32)
                for idx, action in enumerate(self.history_actions):
                    if action is not None:
                        action_idx = int(action)
                        if action_idx < self.num_actions:
                            target_idx = idx * self.num_actions + action_idx
                            if target_idx < len(hist_onehot):
                                hist_onehot[target_idx] = 1.0
                history_channel = np.tile(hist_onehot, (self.agent_history_len, 1)).T
                obs = np.vstack([obs, history_channel])
            return obs.astype(np.float32)

        hist_len = normalized.size
        self._obs_buffer[:hist_len] = normalized.ravel()
        self._obs_buffer[hist_len:hist_len+4] = extras

        if self.action_history_len > 0:
            hist_onehot = np.zeros(self.history_vector_size, dtype=np.float32)
            for idx, action in enumerate(self.history_actions):
                if action is not None:
                    action_idx = int(action)
                    if action_idx < self.num_actions:
                        target_idx = idx * self.num_actions + action_idx
                        if target_idx < len(hist_onehot):
                            hist_onehot[target_idx] = 1.0
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
            mark2market = (current_price - self.entry_price) * self.position * self.position_volume
            info["portfolio_value"] = self.balance + mark2market
        else:
            info["portfolio_value"] = self.balance
        return info

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

        asset_stats = self._get_asset_stats()
        norm_price = self.current_seq[price_idx, self.close_idx]
        close_mean = asset_stats['mean'][self.close_idx]
        close_std = asset_stats['std'][self.close_idx]
        real_price = norm_price * close_std + close_mean

        position_closed = False
        pnl_change = 0.0
        trade_pnl = None
        exit_reason = ""
        
        MIN_SAFE_FRACTION = 1.2
        if action in [1, 2] and self.position == 0:
            if self.balance <= self.bankruptcy_threshold * MIN_SAFE_FRACTION:
                 action = 0

        if self.use_risk_management and self.position != 0:
            d0 = trailing_stop
            d_min = trailing_stop_min
            fee = self.transaction_fee
            fee_buf = fee * (fee_buffer_mult or 2.0)
            
            tsl_price = self.tsl_price
            
            if tsl_price is None:
                 tsl_price = -999999.0 if self.position == 1 else 999999.0

            if self.position == 1: # LONG
                current_max = getattr(self, "trailing_max_price", real_price)
                if current_max is None: current_max = real_price
                self.trailing_max_price = max(current_max, real_price)
                
                if self.trailing_max_price < 0:
                    base_tsl = self.trailing_max_price * (1 + d0)
                else:
                    base_tsl = self.trailing_max_price * (1 - d0)
                
                tsl_price = max(tsl_price, base_tsl)
                
                if d_min is not None:
                    p = max(0.0, self.trailing_max_price / self.real_entry_price - 1.0)
                    
                    if delta_p_hysteresis is None or p >= self.p_at_last_tsl_update + (delta_p_hysteresis or 0.0):
                        if delta_p_hysteresis is not None: self.p_at_last_tsl_update = p
                        
                        if p <= fee_buf: 
                            d_eff = d0
                        else: 
                            d_eff = self._calculate_effective_trail_distance(p, d0, d_min, fee_buf)
                            
                        if self.trailing_max_price < 0:
                            advanced_tsl_price = self.trailing_max_price * (1 + d_eff)
                        else:
                            advanced_tsl_price = self.trailing_max_price * (1 - d_eff)
                        tsl_price = max(tsl_price, advanced_tsl_price)
                
                self.tsl_price = tsl_price
                trailing_trigger = real_price <= tsl_price

            else: # SHORT
                current_min = getattr(self, "trailing_min_price", real_price)
                if current_min is None: current_min = real_price
                self.trailing_min_price = min(current_min, real_price)
                
                if self.trailing_min_price < 0:
                    base_tsl = self.trailing_min_price * (1 - d0)
                else:
                    base_tsl = self.trailing_min_price * (1 + d0)
                
                tsl_price = min(tsl_price, base_tsl)

                if d_min is not None:
                    p = max(0.0, 1.0 - self.trailing_min_price / self.real_entry_price)
                    
                    if delta_p_hysteresis is None or p >= self.p_at_last_tsl_update + (delta_p_hysteresis or 0.0):
                        if delta_p_hysteresis is not None: self.p_at_last_tsl_update = p
                        
                        if p <= fee_buf:
                            d_eff = d0
                        else:
                            d_eff = self._calculate_effective_trail_distance(p, d0, d_min, fee_buf)
                        
                        if self.trailing_min_price < 0:
                            advanced_tsl_price = self.trailing_min_price * (1 - d_eff)
                        else:
                            advanced_tsl_price = self.trailing_min_price * (1 + d_eff)
                        tsl_price = min(tsl_price, advanced_tsl_price)

                self.tsl_price = tsl_price
                trailing_trigger = real_price >= tsl_price

            if trailing_trigger or self.last_step:
                action = 3 # CLOSE
                if trailing_trigger:
                    exit_reason = "TSL"
                elif self.last_step:
                    exit_reason = "FORCED"

        current_dt = signal_dt + dt.timedelta(minutes=self.step_idx)

        if action == 1 and self.allowed_directions and 'LONG' not in self.allowed_directions:
            action = 0
        if action == 2 and self.allowed_directions and 'SHORT' not in self.allowed_directions:
            action = 0

        if action in [1, 2] and self.position == 0 and self.trades_count >= self.max_trades_per_episode:
            action = 0

        if action == 1 and self.position == 0: # OPEN LONG
            if real_price < 0:
                real_exec_price = real_price * (1 - self.slippage)
            else:
                real_exec_price = real_price * (1 + self.slippage)
            norm_exec_price = norm_price * (1 + self.slippage)
            
            self.position = 1
            self.entry_price = norm_exec_price
            self.real_entry_price = real_exec_price

            if self.order_size_usdt > 0:
                trade_amount = min(self.order_size_usdt, self.balance * 0.95)
            else:
                trade_amount = self.balance * self.position_fraction
            trade_amount = max(0.0, min(trade_amount, self.balance * 0.95))
            
            volume = trade_amount / abs(real_exec_price)
            self.position_volume = volume
            self.trades_count += 1
            
            fee = abs(real_exec_price) * volume * self.transaction_fee
            pnl_change -= fee
            self.total_commission += fee
            self.direction = "LONG"
            self.trade_dt = current_dt
            if self.use_risk_management:
                self.trailing_max_price = real_exec_price
                self.tsl_price = None
                self.p_at_last_tsl_update = 0.0
            self._position_entry_step = self.step_idx
            logging.info(f": (LONG) BUY {volume:.8f} {ticker} for {real_exec_price:.5f} at {current_dt.strftime('%Y-%m-%d %H:%M')}")

        elif action == 2 and self.position == 0: # OPEN SHORT
            if real_price < 0:
                real_exec_price = real_price * (1 + self.slippage)
            else:
                real_exec_price = real_price * (1 - self.slippage)
            norm_exec_price = norm_price * (1 - self.slippage)

            self.position = -1
            self.entry_price = norm_exec_price
            self.real_entry_price = real_exec_price

            if self.order_size_usdt > 0:
                trade_amount = min(self.order_size_usdt, self.balance * 0.95)
            else:
                trade_amount = self.balance * self.position_fraction
            trade_amount = max(0.0, min(trade_amount, self.balance * 0.95))
            
            volume = trade_amount / abs(real_exec_price)
            self.position_volume = volume
            self.trades_count += 1
            
            fee = abs(real_exec_price) * volume * self.transaction_fee
            pnl_change -= fee
            self.total_commission += fee
            self.direction = "SHORT"
            self.trade_dt = current_dt
            if self.use_risk_management:
                self.trailing_min_price = real_exec_price
                self.tsl_price = None
                self.p_at_last_tsl_update = 0.0
            self._position_entry_step = self.step_idx
            logging.info(f": (SHORT) SELL {volume:.8f} {ticker} for {real_exec_price:.5f} at {current_dt.strftime('%Y-%m-%d %H:%M')}")

        elif action == 3 and self.position != 0:
            position_closed = True
            volume = self.position_volume
            was_long = (self.position == 1)
            
            if was_long:
                if real_price < 0:
                    real_exec_price = real_price * (1 + self.slippage)
                else:
                    real_exec_price = real_price * (1 - self.slippage)
                trade_pnl = (real_exec_price - self.real_entry_price) * volume
                close_action = "SELL"
                trade_price_delta = (real_exec_price - self.real_entry_price) / self.real_entry_price
            else: # SHORT
                if real_price < 0:
                    real_exec_price = real_price * (1 - self.slippage)
                else:
                    real_exec_price = real_price * (1 + self.slippage)
                trade_pnl = (self.real_entry_price - real_exec_price) * volume
                close_action = "BUY"
                trade_price_delta = (self.real_entry_price - real_exec_price) / self.real_entry_price

            fee = abs(real_exec_price) * volume * self.transaction_fee
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
            opening_fee = abs(self.real_entry_price) * volume * self.transaction_fee
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
        if not terminated:
            portfolio_value = self.balance
            if self.position != 0:
                m2m_price_idx = min(len(self.current_seq) - 1, self.pre_signal_len - 1 + self.step_idx)
                norm_m2m_price = self.current_seq[m2m_price_idx, self.close_idx]
                asset_stats = self._get_asset_stats()
                close_mean = asset_stats['mean'][self.close_idx]
                close_std = asset_stats['std'][self.close_idx]
                real_m2m_price = norm_m2m_price * close_std + close_mean
                if self.position == 1: # LONG
                    mark2market = (real_m2m_price - self.real_entry_price) * self.position_volume
                elif self.position == -1: # SHORT
                    mark2market = (self.real_entry_price - real_m2m_price) * self.position_volume
                else:
                    mark2market = 0.0
                portfolio_value += mark2market
            
            if portfolio_value <= self.bankruptcy_threshold:
                if self.position != 0:
                    volume = self.position_volume
                    if self.position == 1:  # LONG
                        if real_price < 0:
                            real_exec_price = real_price * (1 + self.slippage)
                        else:
                            real_exec_price = real_price * (1 - self.slippage)
                        trade_pnl = (real_exec_price - self.real_entry_price) * volume
                    else:  # SHORT
                        if real_price < 0:
                            real_exec_price = real_price * (1 - self.slippage)
                        else:
                            real_exec_price = real_price * (1 + self.slippage)
                        trade_pnl = (self.real_entry_price - real_exec_price) * volume
                    
                    fee = abs(real_exec_price) * volume * self.transaction_fee
                    pnl_change = trade_pnl - fee
                    self.balance += pnl_change
                    self.position = 0
                    self.position_volume = 0.0
                
                reward = -self.bankruptcy_penalty
                terminated = True
                info["bankruptcy"] = True
                info["bankruptcy_equity"] = portfolio_value
        
        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)

        if position_closed:
            holding_duration = self.step_idx - (self._position_entry_step or 0)
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
