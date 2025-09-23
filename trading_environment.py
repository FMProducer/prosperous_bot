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
        render_mode: Optional[str],
        full_seq_len: int,
        num_features: int,
        num_actions: int,
        flat_state_size: int,
        initial_balance: float,
        pre_signal_len: int,
        data_channels: List[str],
        slippage: float,
        transaction_fee: float,
        agent_session_len: int,
        agent_history_len: int,
        input_history_len: int,
        price_channels: List[str],
        volume_channels: List[str],
        other_channels: List[str],
        action_history_len: int,
        inaction_penalty_ratio: float,
        backtest_mode: bool = False,
        use_risk_management: bool = False,
        **kwargs,
    ) -> None:
        if not sequences:
            raise ValueError("`sequences` must be a non-empty list of arrays")

        self.sequences = sequences
        self.stats = stats
        self.render_mode = render_mode
        self.initial_balance = initial_balance
        self.pre_signal_len = pre_signal_len
        self.data_channels = data_channels
        self.slippage = slippage
        self.transaction_fee = transaction_fee
        self.agent_session_len = agent_session_len
        self.agent_history_len = agent_history_len
        self.input_history_len = input_history_len
        self.price_channels = price_channels
        self.volume_channels = volume_channels
        self.other_channels = other_channels
        self.action_history_len = action_history_len
        self.num_actions = num_actions
        self.inaction_penalty_ratio = inaction_penalty_ratio
        self.backtest_mode = backtest_mode
        self.use_risk_management = use_risk_management

        self.history_vector_size = num_actions * self.action_history_len
        expected_shape = (full_seq_len, num_features)
        if self.sequences[0].shape != expected_shape:
            raise ValueError(f"Expected sequence shape {expected_shape}, but got {self.sequences[0].shape}")

        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(flat_state_size + self.history_vector_size,),
            dtype=np.float32,
        )

        self._init_episode_vars()

    def _init_episode_vars(self) -> None:
        self.current_seq: Optional[np.ndarray] = None
        self.step_idx: int = 0
        self.balance: float = self.initial_balance
        self.position: int = 0
        self.entry_price: float = 0.0
        self.realized_pnl: float = 0.0
        self.closed_trades: int = 0
        self.profitable_trades: int = 0
        self.last_step: bool = False
        if self.backtest_mode:
            self.total_commission: float = 0.0
            self.direction: Optional[str] = None
            self.trade_dt: dt.datetime = None
            if self.use_risk_management:
                self.trailing_max_price: float = None
                self.trailing_min_price: float = None

        if self.action_history_len > 0:
            self.history_actions: List[Optional[int]] = [None] * self.action_history_len

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._init_episode_vars()

        idx = self.np_random.integers(0, len(self.sequences)) if options is None else options["forced_index"]
        self.current_seq = self.sequences[idx]
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
        price = self.current_seq[price_idx, self.data_channels.index("close")]
        pnl_change = 0.0

        if action == 1 and self.position == 0:
            exec_price = price * (1 + self.slippage)
            self.position = 1
            self.entry_price = exec_price
            volume = self.balance / exec_price
            pnl_change -= exec_price * volume * self.transaction_fee

        elif action == 2 and self.position == 0:
            exec_price = price * (1 - self.slippage)
            self.position = -1
            self.entry_price = exec_price
            volume = self.balance / exec_price
            pnl_change -= exec_price * volume * self.transaction_fee

        elif action == 3 and self.position != 0:
            volume = self.balance / self.entry_price
            if self.position == 1:
                exec_price = price * (1 - self.slippage)
                trade_pnl = (exec_price - self.entry_price) * volume
            else:
                exec_price = price * (1 + self.slippage)
                trade_pnl = (self.entry_price - exec_price) * volume
            pnl_change += trade_pnl - exec_price * volume * self.transaction_fee
            self.closed_trades += 1
            if trade_pnl > 0:
                self.profitable_trades += 1
            self.position = 0

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

        reward = (pnl_change / self.initial_balance) - inaction_penalty

        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = self._get_info()

        if terminated:
            info.update(
                {
                    "episode_realized_pnl": self.realized_pnl,
                    "episode_win_rate": self.profitable_trades / max(1, self.closed_trades),
                    "episode_closed_trades": self.closed_trades,
                }
            )

        if self.render_mode == "human":
            self._render_human(info, action, reward)

        return obs, reward, terminated, False, info

    def _get_observation(self) -> np.ndarray:
        end = self.pre_signal_len + self.step_idx
        start = end - self.agent_history_len
        window = self.current_seq[start:end]

        normalized = apply_normalization(
            window,
            self.stats,
            self.data_channels,
            self.price_channels,
            self.volume_channels,
            self.other_channels,
            self.agent_history_len,
            self.input_history_len,
        )

        unrealized = 0.0
        if self.position != 0:
            price_idx = min(len(self.current_seq) - 1, self.pre_signal_len - 1 + self.step_idx)
            current_price = self.current_seq[price_idx, self.data_channels.index("close")]
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

        if self.action_history_len > 0:
            hist_onehot = np.zeros(self.history_vector_size, dtype=np.float32)
            for idx, action in enumerate(self.history_actions):
                if action is not None:
                    hist_onehot[idx * self.num_actions + action] = 1.0
            return np.concatenate([normalized.flatten(), extras, hist_onehot])
        else:
            return np.concatenate([normalized.flatten(), extras])

    def _get_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "step": self.step_idx,
            "balance": self.balance,
            "position": self.position,
        }

        if self.position != 0:
            price_idx = min(len(self.current_seq) - 1, self.pre_signal_len - 1 + self.step_idx)
            current_price = self.current_seq[price_idx, self.data_channels.index("close")]
            mark2market = (current_price - self.entry_price) * self.position * (self.balance / self.entry_price)
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
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.current_seq is not None, "reset() must be called before backtest_step()"

        self.last_step = self.step_idx == self.agent_session_len - 1
        if self.last_step:
            if self.position == 0 and action in {1, 2}:
                action = 0
            elif self.position != 0 and action != 3:
                action = 3

        price_idx = min(self.pre_signal_len - 1 + self.step_idx, len(self.current_seq) - 1)
        price = self.current_seq[price_idx, self.data_channels.index("close")]
        position_closed = False
        pnl_change = 0.0
        exec_price = 0.0
        trade_price_delta = 0.0
        trade_pnl = None
        exit_reason = ""

        if self.use_risk_management and self.position != 0:
            if self.position == 1:
                self.trailing_max_price = max(getattr(self, "trailing_max_price"), price)
                sl_trigger = price <= self.entry_price * (1 - stop_loss)
                tp_trigger = price >= self.entry_price * (1 + take_profit)
                trailing_trigger = price <= self.trailing_max_price * (1 - trailing_stop)
            else:
                self.trailing_min_price = min(getattr(self, "trailing_min_price"), price)
                sl_trigger = price >= self.entry_price * (1 + stop_loss)
                tp_trigger = price <= self.entry_price * (1 - take_profit)
                trailing_trigger = price >= self.trailing_min_price * (1 + trailing_stop)

            if sl_trigger or tp_trigger or trailing_trigger or self.last_step:
                action = 3
                mask_exit_reason = np.array([self.last_step, sl_trigger, tp_trigger, trailing_trigger])
                exit_reason = self.exit_options[mask_exit_reason][0]

        current_dt = signal_dt + dt.timedelta(minutes=self.step_idx)

        if action == 1 and self.position == 0:
            exec_price = price * (1 + self.slippage)
            self.position = 1
            self.entry_price = exec_price
            volume = self.balance / exec_price
            fee = exec_price * volume * self.transaction_fee
            pnl_change -= fee
            self.total_commission += fee
            self.direction = "LONG"
            self.trade_dt = current_dt
            if self.use_risk_management:
                self.trailing_max_price = exec_price
            logging.info(
                f": (LONG) BUY {volume:.8f} {ticker} for {exec_price:.5f} at {current_dt.strftime('%Y-%m-%d %H:%M')}"
            )

        elif action == 2 and self.position == 0:
            exec_price = price * (1 - self.slippage)
            self.position = -1
            self.entry_price = exec_price
            volume = self.balance / exec_price
            fee = exec_price * volume * self.transaction_fee
            pnl_change -= fee
            self.total_commission += fee
            self.direction = "SHORT"
            self.trade_dt = current_dt
            if self.use_risk_management:
                self.trailing_min_price = exec_price
            logging.info(
                f": (SHORT) SELL {volume:.8f} {ticker} for {exec_price:.5f} at {current_dt.strftime('%Y-%m-%d %H:%M')}"
            )

        elif action == 3 and self.position != 0:
            position_closed = True
            volume = self.balance / self.entry_price
            if self.position == 1:
                exec_price = price * (1 - self.slippage)
                trade_pnl = (exec_price - self.entry_price) * volume
                close_action = "SELL"
                trade_price_delta = (exec_price - self.entry_price) / self.entry_price
            else:
                exec_price = price * (1 + self.slippage)
                trade_pnl = (self.entry_price - exec_price) * volume
                close_action = "BUY"
                trade_price_delta = (self.entry_price - exec_price) / self.entry_price

            fee = exec_price * volume * self.transaction_fee
            pnl_change += trade_pnl - fee
            self.total_commission += fee

            self.position = 0

        self.realized_pnl += pnl_change
        self.balance += pnl_change
        if position_closed:
            logging.info(
                f": (CLOSE) {close_action} {exit_reason if self.use_risk_management else ''} {volume:.8f} {ticker} for {exec_price:.5f} at "
                f"{current_dt.strftime('%Y-%m-%d %H:%M')} PnL = {self.realized_pnl:+.2f}"
            )

        if self.action_history_len > 0:
            self.history_actions.pop(0)
            self.history_actions.append(action)

        self.step_idx += 1
        terminated = self.step_idx >= self.agent_session_len
        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0

        if position_closed:
            info = {
                "position_closed": position_closed,
                "trade_realized_pnl": self.realized_pnl,
                "total_commission": self.total_commission,
                "trade_amount": self.initial_balance,
                "trade_price_delta": trade_price_delta,
                "max_drawdown": trade_pnl / self.initial_balance,
                "correct_prediction": trade_pnl > 0,
                "direction": self.direction,
                "trade_dt": self.trade_dt,
            }

            self.realized_pnl = 0.0
            self.total_commission = 0.0
            self.direction = None
            self.trade_dt = None
            if self.use_risk_management:
                self.trailing_max_price = None
                self.trailing_min_price = None
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
