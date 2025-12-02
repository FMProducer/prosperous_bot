"""
test_train.py - Unit-тесты для проверки полной идентичности расчётов между тренировкой и валидацией

Цель: гарантировать, что все торговые операции, PnL, комиссии, slippage, награды и штрафы
рассчитываются идентично в процессе обучения (training) и валидации (validation).
"""

import sys
import os
import unittest
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch

# Добавляем родительскую директорию в путь для импорта модулей
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_environment import TradingEnvironment
from agent import D3QN_PER_Agent
from config import cfg
import importlib.util


class TestTrainingValidationIdentity(unittest.TestCase):
    """
    Тестирует идентичность расчётов торговых операций между тренировкой и валидацией.
    """

    @classmethod
    def setUpClass(cls):
        """Загружаем конфигурацию и подготавливаем тестовое окружение."""
        # Загружаем конфигурацию alpha_seed_404_v8
        config_path = Path(__file__).parent.parent / "configs" / "alpha_seed_404_v8.py"
        spec = importlib.util.spec_from_file_location("alpha_config", config_path)
        alpha_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(alpha_config)

        cls.cfg = cfg
        cls.seed = cls.cfg.random_seed

        # Создаём минимальные синтетические данные для тестирования
        cls.test_data, cls.test_keys, cls.test_stats = cls._create_synthetic_data()

    @classmethod
    def _create_synthetic_data(cls) -> Tuple[List[np.ndarray], List[str], Dict]:
        """
        Создаёт синтетические нормализованные данные для тестирования.

        Returns:
            sequences: список массивов формы (C, L, 1)
            keys: список ключей активов
            stats: статистики нормализации
        """
        num_sequences = 5
        num_channels = cls.cfg.num_channels
        seq_len = cls.cfg.seq.full_seq_len

        sequences = []
        keys = []

        np.random.seed(cls.seed)

        for i in range(num_sequences):
            # Генерируем реалистичные цены (синус с трендом и шумом)
            base_price = 50000 + i * 1000
            trend = np.linspace(0, 1000, seq_len)
            noise = np.random.randn(seq_len) * 100
            close_prices = base_price + trend + noise

            # OHLCV данные
            seq = np.zeros((seq_len, num_channels), dtype=np.float32)
            seq[:, 0] = close_prices * (1 + np.random.randn(seq_len) * 0.001)  # open
            seq[:, 1] = close_prices * (1 + np.abs(np.random.randn(seq_len)) * 0.002)  # high
            seq[:, 2] = close_prices * (1 - np.abs(np.random.randn(seq_len)) * 0.002)  # low
            seq[:, 3] = close_prices  # close
            seq[:, 4] = np.random.uniform(100, 1000, seq_len)  # volume
            seq[:, 5] = seq[:, 4] * close_prices  # quote_volume
            seq[:, 6] = np.random.uniform(100, 500, seq_len)  # num_trades
            seq[:, 7] = seq[:, 4] * 0.5  # taker_base
            seq[:, 8] = seq[:, 5] * 0.5  # taker_quote
            seq[:, 9] = close_prices  # vwap

            # Z-нормализация
            means = seq.mean(axis=0)
            stds = seq.std(axis=0) + 1e-8
            seq_normalized = (seq - means) / stds

            # Reshape в (C, L, 1)
            seq_normalized = seq_normalized.T
            seq_normalized = np.expand_dims(seq_normalized, -1)

            sequences.append(seq_normalized.astype(np.float32))
            keys.append(f"BTCUSDT_{i:04d}")

        # Создаём статистики нормализации
        stats = {
            "BTCUSDT": {
                "mean": sequences[0][:, :, 0].mean(axis=1).tolist(),
                "std": sequences[0][:, :, 0].std(axis=1).tolist()
            }
        }

        return sequences, keys, stats

    def _create_test_environment(self, mode: str = "train") -> TradingEnvironment:
        """
        Создаёт тестовое окружение с фиксированными параметрами.

        Args:
            mode: "train" или "validation"
        """
        env_kwargs = {
            "sequences": self.test_data,
            "stats": self.test_stats,
            "keys": self.test_keys,
            "render_mode": None,
            "full_seq_len": self.cfg.seq.full_seq_len,
            "num_features": self.cfg.num_channels,
            "num_actions": self.cfg.market.num_actions,
            "flat_state_size": self.cfg.seq.flat_state_size,
            "initial_balance": self.cfg.market.initial_balance,
            "pre_signal_len": self.cfg.seq.pre_signal_len,
            "datachannels": self.cfg.data.datachannels,
            "slippage": self.cfg.market.slippage,
            "transaction_fee": self.cfg.market.transaction_fee,
            "agent_session_len": self.cfg.seq.agent_session_len,
            "agent_history_len": self.cfg.seq.agent_history_len,
            "input_history_len": self.cfg.seq.input_history_len,
            "pricechannels": self.cfg.data.pricechannels,
            "volumechannels": self.cfg.data.volumechannels,
            "otherchannels": self.cfg.data.otherchannels,
            "action_history_len": self.cfg.seq.action_history_len,
            "inaction_penalty_ratio": self.cfg.market.inaction_penalty_ratio,
            "backtest_mode": (mode == "validation"),
            "use_risk_management": False,
            "cnn_format": True,
            "position_fraction": self.cfg.market.position_fraction,
            "order_size_usdt": 0.0,
            "bankruptcy_threshold": self.cfg.market.bankruptcy_threshold,
            "bankruptcy_penalty": self.cfg.market.bankruptcy_penalty,
            "max_drawdown_threshold": self.cfg.market.max_drawdown_threshold,
            "max_drawdown_penalty": self.cfg.market.max_drawdown_penalty,
            "max_drawdown_penalty_type": self.cfg.market.max_drawdown_penalty_type,
            "new_equity_peak_reward": self.cfg.market.new_equity_peak_reward,
            "perfect_entry_reward": self.cfg.market.perfect_entry_reward,
            "risk_reward_ratio_threshold": self.cfg.market.risk_reward_ratio_threshold,
            "risk_reward_ratio_reward": self.cfg.market.risk_reward_ratio_reward,
            "continuous_pain_penalty_ratio": self.cfg.market.continuous_pain_penalty_ratio,
            "good_exit_bonus": self.cfg.market.good_exit_bonus,
            "fast_exit_bonus": self.cfg.market.fast_exit_bonus,
            "low_balance_penalty": self.cfg.market.low_balance_penalty,
            "bankruptcy_slippage_penalty": self.cfg.market.bankruptcy_slippage_penalty,
            "holding_penalty_multiplier": self.cfg.market.holding_penalty_multiplier,
            "greed_penalty_multiplier": self.cfg.market.greed_penalty_multiplier,
            "premature_exit_penalty": self.cfg.market.premature_exit_penalty,
            "holding_penalty_threshold": self.cfg.market.holding_penalty_threshold,
            "greed_penalty_threshold": self.cfg.market.greed_penalty_threshold,
            "exit_quality_threshold": self.cfg.market.exit_quality_threshold,
            "fast_exit_threshold": self.cfg.market.fast_exit_threshold,
            "premature_exit_threshold": self.cfg.market.premature_exit_threshold,
            "seed": self.seed,
        }

        return TradingEnvironment(**env_kwargs)

    def test_single_trade_long_identity(self):
        """Тест идентичности расчётов для одной LONG сделки."""
        env_train = self._create_test_environment("train")
        env_val = self._create_test_environment("validation")

        # Фиксируем seed и индекс эпизода
        obs_train, info_train = env_train.reset(seed=self.seed, options={"forced_index": 0})
        obs_val, info_val = env_val.reset(seed=self.seed, options={"forced_index": 0})

        # Проверяем идентичность начального состояния
        np.testing.assert_array_almost_equal(obs_train, obs_val, decimal=6,
                                              err_msg="Initial observations differ")
        self.assertEqual(info_train["balance"], info_val["balance"],
                        "Initial balance differs")

        # Выполняем одинаковую последовательность действий
        actions = [1, 0, 0, 3]  # LONG -> HOLD -> HOLD -> CLOSE

        train_trajectory = []
        val_trajectory = []

        for action in actions:
            obs_t, reward_t, done_t, trunc_t, info_t = env_train.step(action)
            obs_v, reward_v, done_v, trunc_v, info_v = env_val.step(action)

            train_trajectory.append({
                "obs": obs_t.copy(),
                "reward": reward_t,
                "balance": info_t.get("balance", 0),
                "position": info_t.get("position", 0),
                "realized_pnl": info_t.get("realized_pnl", 0),
                "closed_trades": info_t.get("closed_trades", 0),
            })

            val_trajectory.append({
                "obs": obs_v.copy(),
                "reward": reward_v,
                "balance": info_v.get("balance", 0),
                "position": info_v.get("position", 0),
                "realized_pnl": info_v.get("realized_pnl", 0),
                "closed_trades": info_v.get("closed_trades", 0),
            })

        # Сравниваем траектории
        for i, (t, v) in enumerate(zip(train_trajectory, val_trajectory)):
            np.testing.assert_array_almost_equal(
                t["obs"], v["obs"], decimal=5,
                err_msg=f"Step {i}: Observations differ"
            )
            self.assertAlmostEqual(
                t["reward"], v["reward"], places=6,
                msg=f"Step {i}: Rewards differ: train={t['reward']}, val={v['reward']}"
            )
            self.assertAlmostEqual(
                t["balance"], v["balance"], places=6,
                msg=f"Step {i}: Balances differ: train={t['balance']}, val={v['balance']}"
            )
            self.assertEqual(
                t["position"], v["position"],
                msg=f"Step {i}: Positions differ: train={t['position']}, val={v['position']}"
            )
            self.assertAlmostEqual(
                t["realized_pnl"], v["realized_pnl"], places=6,
                msg=f"Step {i}: Realized PnL differs"
            )

    def test_single_trade_short_identity(self):
        """Тест идентичности расчётов для одной SHORT сделки."""
        env_train = self._create_test_environment("train")
        env_val = self._create_test_environment("validation")

        obs_train, _ = env_train.reset(seed=self.seed, options={"forced_index": 1})
        obs_val, _ = env_val.reset(seed=self.seed, options={"forced_index": 1})

        # SHORT -> HOLD -> CLOSE
        actions = [2, 0, 3]

        for action in actions:
            obs_t, reward_t, done_t, trunc_t, info_t = env_train.step(action)
            obs_v, reward_v, done_v, trunc_v, info_v = env_val.step(action)

            self.assertAlmostEqual(reward_t, reward_v, places=6,
                                  msg=f"Action {action}: Rewards differ")
            self.assertAlmostEqual(info_t["balance"], info_v["balance"], places=6,
                                  msg=f"Action {action}: Balances differ")

    def test_fee_and_slippage_calculation(self):
        """Тест идентичности расчёта комиссий и проскальзывания."""
        env_train = self._create_test_environment("train")
        env_val = self._create_test_environment("validation")

        # Открываем и сразу закрываем позицию - максимизируем комиссии
        obs_train, _ = env_train.reset(seed=self.seed, options={"forced_index": 0})
        obs_val, _ = env_val.reset(seed=self.seed, options={"forced_index": 0})

        # LONG -> CLOSE
        for action in [1, 3]:
            obs_t, reward_t, done_t, trunc_t, info_t = env_train.step(action)
            obs_v, reward_v, done_v, trunc_v, info_v = env_val.step(action)

        # Проверяем, что комиссии учтены одинаково
        final_balance_train = info_t["balance"]
        final_balance_val = info_v["balance"]

        self.assertAlmostEqual(final_balance_train, final_balance_val, places=6,
                              msg="Final balances differ after fees/slippage")

        # Баланс должен быть меньше начального из-за комиссий
        self.assertLess(final_balance_train, self.cfg.market.initial_balance,
                       msg="Balance should decrease due to fees")

    def test_shaped_rewards_identity(self):
        """Тест идентичности расчёта shaped rewards."""
        env_train = self._create_test_environment("train")
        env_val = self._create_test_environment("validation")

        obs_train, _ = env_train.reset(seed=self.seed, options={"forced_index": 2})
        obs_val, _ = env_val.reset(seed=self.seed, options={"forced_index": 2})

        # Последовательность, которая должна вызвать shaped rewards
        # LONG -> удерживаем долго -> закрываем
        actions = [1] + [0] * 20 + [3]

        all_rewards_train = []
        all_rewards_val = []

        for action in actions:
            _, reward_t, _, _, _ = env_train.step(action)
            _, reward_v, _, _, _ = env_val.step(action)

            all_rewards_train.append(reward_t)
            all_rewards_val.append(reward_v)

        # Сравниваем все награды
        for i, (rt, rv) in enumerate(zip(all_rewards_train, all_rewards_val)):
            self.assertAlmostEqual(rt, rv, places=6,
                                  msg=f"Step {i}: Shaped rewards differ: train={rt}, val={rv}")

    def test_multiple_trades_identity(self):
        """Тест идентичности для нескольких последовательных сделок."""
        env_train = self._create_test_environment("train")
        env_val = self._create_test_environment("validation")

        obs_train, _ = env_train.reset(seed=self.seed, options={"forced_index": 3})
        obs_val, _ = env_val.reset(seed=self.seed, options={"forced_index": 3})

        # Несколько сделок подряд
        actions = [
            1, 0, 0, 3,  # LONG trade 1
            2, 0, 3,     # SHORT trade 2
            1, 0, 0, 0, 3  # LONG trade 3
        ]

        for i, action in enumerate(actions):
            obs_t, reward_t, done_t, trunc_t, info_t = env_train.step(action)
            obs_v, reward_v, done_v, trunc_v, info_v = env_val.step(action)

            # Проверяем все ключевые метрики
            self.assertAlmostEqual(reward_t, reward_v, places=6,
                                  msg=f"Step {i}: Rewards differ")
            self.assertAlmostEqual(info_t["balance"], info_v["balance"], places=6,
                                  msg=f"Step {i}: Balances differ")
            self.assertEqual(info_t["position"], info_v["position"],
                           msg=f"Step {i}: Positions differ")
            # Используем .get() для ключей, которые могут отсутствовать
            self.assertAlmostEqual(info_t.get("realized_pnl", 0), info_v.get("realized_pnl", 0), places=6,
                                  msg=f"Step {i}: Realized PnL differs")
            self.assertEqual(info_t.get("closed_trades", 0), info_v.get("closed_trades", 0),
                           msg=f"Step {i}: Number of closed trades differs")

    def test_drawdown_penalty_identity(self):
        """Тест идентичности расчёта штрафа за просадку."""
        env_train = self._create_test_environment("train")
        env_val = self._create_test_environment("validation")

        obs_train, _ = env_train.reset(seed=self.seed + 1, options={"forced_index": 0})
        obs_val, _ = env_val.reset(seed=self.seed + 1, options={"forced_index": 0})

        # Проверяем отслеживание просадки
        max_steps = min(30, self.cfg.seq.agent_session_len)

        for i in range(max_steps):
            action = np.random.randint(0, self.cfg.market.num_actions)

            obs_t, reward_t, done_t, trunc_t, info_t = env_train.step(action)
            obs_v, reward_v, done_v, trunc_v, info_v = env_val.step(action)

            # Проверяем просадку
            if "current_max_drawdown" in info_t and "current_max_drawdown" in info_v:
                self.assertAlmostEqual(
                    info_t["current_max_drawdown"], 
                    info_v["current_max_drawdown"], 
                    places=6,
                    msg=f"Step {i}: Max drawdown differs"
                )

            if done_t or done_v:
                break

    def test_bankruptcy_logic_identity(self):
        """Тест идентичности логики банкротства."""
        # Создаём среду с малым начальным балансом для провоцирования банкротства
        cfg_modified = self.cfg
        original_balance = cfg_modified.market.initial_balance
        cfg_modified.market.initial_balance = 100.0  # Малый баланс

        env_train = self._create_test_environment("train")
        env_val = self._create_test_environment("validation")

        obs_train, _ = env_train.reset(seed=self.seed + 2, options={"forced_index": 0})
        obs_val, _ = env_val.reset(seed=self.seed + 2, options={"forced_index": 0})

        # Агрессивная торговля для провоцирования банкротства
        done_train = False
        done_val = False
        step = 0

        while not (done_train or done_val) and step < 50:
            action = 1 if step % 6 < 3 else 2  # Чередуем LONG/SHORT

            obs_t, reward_t, done_t, trunc_t, info_t = env_train.step(action)
            obs_v, reward_v, done_v, trunc_v, info_v = env_val.step(action)

            # Если произошло банкротство, проверяем идентичность
            if info_t.get("bankruptcy", False) or info_v.get("bankruptcy", False):
                self.assertEqual(info_t.get("bankruptcy", False), 
                               info_v.get("bankruptcy", False),
                               "Bankruptcy flag differs")
                self.assertAlmostEqual(reward_t, reward_v, places=6,
                                      msg="Bankruptcy penalty differs")
                break

            done_train = done_t
            done_val = done_v
            step += 1

        # Восстанавливаем баланс
        cfg_modified.market.initial_balance = original_balance

    def test_full_episode_identity(self):
        """Тест идентичности полного эпизода."""
        env_train = self._create_test_environment("train")
        env_val = self._create_test_environment("validation")

        obs_train, _ = env_train.reset(seed=self.seed + 3, options={"forced_index": 4})
        obs_val, _ = env_val.reset(seed=self.seed + 3, options={"forced_index": 4})

        np.random.seed(self.seed + 3)

        total_reward_train = 0.0
        total_reward_val = 0.0

        for step in range(self.cfg.seq.agent_session_len):
            # Случайные действия с фиксированным seed
            action = np.random.randint(0, self.cfg.market.num_actions)

            obs_t, reward_t, done_t, trunc_t, info_t = env_train.step(action)
            obs_v, reward_v, done_v, trunc_v, info_v = env_val.step(action)

            total_reward_train += reward_t
            total_reward_val += reward_v

            # Проверяем ключевые метрики на каждом шаге
            self.assertAlmostEqual(reward_t, reward_v, places=6,
                                  msg=f"Step {step}: Rewards differ")
            self.assertAlmostEqual(info_t["balance"], info_v["balance"], places=6,
                                  msg=f"Step {step}: Balances differ")

            if done_t or done_v:
                self.assertEqual(done_t, done_v, "Episode termination differs")
                break

        # Проверяем итоговые метрики эпизода
        self.assertAlmostEqual(total_reward_train, total_reward_val, places=5,
                              msg="Total episode reward differs")
        self.assertAlmostEqual(info_t.get("episode_realized_pnl", 0),
                              info_v.get("episode_realized_pnl", 0), places=6,
                              msg="Episode realized PnL differs")
        self.assertAlmostEqual(info_t.get("episode_win_rate", 0),
                              info_v.get("episode_win_rate", 0), places=6,
                              msg="Episode win rate differs")
        self.assertEqual(info_t.get("episode_closed_trades", 0),
                        info_v.get("episode_closed_trades", 0),
                        msg="Number of closed trades differs")

    def test_deterministic_reset(self):
        """Тест детерминированности reset() с одинаковым seed."""
        env_train = self._create_test_environment("train")
        env_val = self._create_test_environment("validation")

        # Множественные reset с одним seed должны давать идентичные результаты
        for trial in range(5):
            obs_t, info_t = env_train.reset(seed=self.seed + trial, options={"forced_index": 0})
            obs_v, info_v = env_val.reset(seed=self.seed + trial, options={"forced_index": 0})

            np.testing.assert_array_almost_equal(obs_t, obs_v, decimal=6,
                                                  err_msg=f"Trial {trial}: Reset observations differ")
            self.assertEqual(info_t["balance"], info_v["balance"],
                           f"Trial {trial}: Reset balance differs")


class TestEnvironmentInvariants(unittest.TestCase):
    """
    Дополнительные тесты для проверки инвариантов окружения.
    """

    @classmethod
    def setUpClass(cls):
        """Подготовка тестового окружения."""
        cls.cfg = cfg
        cls.seed = cls.cfg.random_seed
        cls.test_data, cls.test_keys, cls.test_stats = TestTrainingValidationIdentity._create_synthetic_data()

    def _create_test_environment(self, **overrides):
        """Создаёт тестовое окружение с возможностью переопределения параметров."""
        base_kwargs = {
            "sequences": self.test_data,
            "stats": self.test_stats,
            "keys": self.test_keys,
            "render_mode": None,
            "full_seq_len": self.cfg.seq.full_seq_len,
            "num_features": self.cfg.num_channels,
            "num_actions": self.cfg.market.num_actions,
            "flat_state_size": self.cfg.seq.flat_state_size,
            "initial_balance": self.cfg.market.initial_balance,
            "pre_signal_len": self.cfg.seq.pre_signal_len,
            "datachannels": self.cfg.data.datachannels,
            "slippage": self.cfg.market.slippage,
            "transaction_fee": self.cfg.market.transaction_fee,
            "agent_session_len": self.cfg.seq.agent_session_len,
            "agent_history_len": self.cfg.seq.agent_history_len,
            "input_history_len": self.cfg.seq.input_history_len,
            "pricechannels": self.cfg.data.pricechannels,
            "volumechannels": self.cfg.data.volumechannels,
            "otherchannels": self.cfg.data.otherchannels,
            "action_history_len": self.cfg.seq.action_history_len,
            "inaction_penalty_ratio": self.cfg.market.inaction_penalty_ratio,
            "backtest_mode": False,
            "use_risk_management": False,
            "cnn_format": True,
            "position_fraction": self.cfg.market.position_fraction,
            "order_size_usdt": 0.0,
            "bankruptcy_threshold": self.cfg.market.bankruptcy_threshold,
            "bankruptcy_penalty": self.cfg.market.bankruptcy_penalty,
            "max_drawdown_threshold": self.cfg.market.max_drawdown_threshold,
            "max_drawdown_penalty": self.cfg.market.max_drawdown_penalty,
            "max_drawdown_penalty_type": self.cfg.market.max_drawdown_penalty_type,
            "new_equity_peak_reward": self.cfg.market.new_equity_peak_reward,
            "perfect_entry_reward": self.cfg.market.perfect_entry_reward,
            "risk_reward_ratio_threshold": self.cfg.market.risk_reward_ratio_threshold,
            "risk_reward_ratio_reward": self.cfg.market.risk_reward_ratio_reward,
            "continuous_pain_penalty_ratio": self.cfg.market.continuous_pain_penalty_ratio,
            "good_exit_bonus": self.cfg.market.good_exit_bonus,
            "fast_exit_bonus": self.cfg.market.fast_exit_bonus,
            "low_balance_penalty": self.cfg.market.low_balance_penalty,
            "bankruptcy_slippage_penalty": self.cfg.market.bankruptcy_slippage_penalty,
            "holding_penalty_multiplier": self.cfg.market.holding_penalty_multiplier,
            "greed_penalty_multiplier": self.cfg.market.greed_penalty_multiplier,
            "premature_exit_penalty": self.cfg.market.premature_exit_penalty,
            "holding_penalty_threshold": self.cfg.market.holding_penalty_threshold,
            "greed_penalty_threshold": self.cfg.market.greed_penalty_threshold,
            "exit_quality_threshold": self.cfg.market.exit_quality_threshold,
            "fast_exit_threshold": self.cfg.market.fast_exit_threshold,
            "premature_exit_threshold": self.cfg.market.premature_exit_threshold,
            "seed": self.seed,
        }
        base_kwargs.update(overrides)
        return TradingEnvironment(**base_kwargs)

    def test_balance_non_negative(self):
        """Баланс не должен становиться отрицательным."""
        env = self._create_test_environment()
        obs, _ = env.reset(seed=self.seed)

        for _ in range(self.cfg.seq.agent_session_len):
            action = np.random.randint(0, self.cfg.market.num_actions)
            obs, reward, done, trunc, info = env.step(action)

            self.assertGreaterEqual(info["balance"], 0.0,
                                   "Balance became negative")

            if done:
                break

    def test_position_consistency(self):
        """Позиция должна быть всегда в [-1, 0, 1]."""
        env = self._create_test_environment()
        obs, _ = env.reset(seed=self.seed)

        for _ in range(self.cfg.seq.agent_session_len):
            action = np.random.randint(0, self.cfg.market.num_actions)
            obs, reward, done, trunc, info = env.step(action)

            self.assertIn(info["position"], [-1, 0, 1],
                         f"Invalid position value: {info['position']}")

            if done:
                break

    def test_no_position_overlap(self):
        """Нельзя открыть позицию, если уже есть открытая."""
        env = self._create_test_environment()
        obs, info = env.reset(seed=self.seed, options={"forced_index": 0})

        # Открываем LONG
        obs, reward, done, trunc, info = env.step(1)
        self.assertEqual(info["position"], 1, "Failed to open LONG")

        # Пытаемся открыть ещё LONG (должно игнорироваться)
        obs, reward, done, trunc, info = env.step(1)
        self.assertEqual(info["position"], 1, "Position should remain LONG")

        # Пытаемся открыть SHORT (должно игнорироваться)
        obs, reward, done, trunc, info = env.step(2)
        self.assertEqual(info["position"], 1, "Position should still be LONG")


if __name__ == "__main__":
    # Запуск тестов с подробным выводом
    unittest.main(verbosity=2)
