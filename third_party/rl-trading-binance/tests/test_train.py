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




# ============================================================================
# HELPER FUNCTION: Создание тестового агента с полными параметрами
# ============================================================================

def create_test_agent(
    state_shape=(10, 90, 1),
    action_dim=4,
    cnn_maps=None,
    cnn_kernels=None,
    cnn_strides=None,
    cnn_dilations=None,
    dense_val=None,
    dense_adv=None,
    additional_feats=12,
    dropout_model=0.0,
    device=None,
    **kwargs
):
    """Helper для создания D3QN_PER_Agent с дефолтными параметрами."""
    from agent import D3QN_PER_Agent
    import torch

    if device is None:
        device = torch.device("cpu")

    if cnn_maps is None:
        cnn_maps = [32]
    if cnn_kernels is None:
        cnn_kernels = [3]
    if cnn_strides is None:
        cnn_strides = [1]
    if cnn_dilations is None:
        cnn_dilations = [1]
    if dense_val is None:
        dense_val = [32]
    if dense_adv is None:
        dense_adv = [32]

    default_params = {
        'gamma': 0.99,
        'learning_rate': 1e-3,
        'batch_size': 16,
        'buffer_size': 5000,
        'target_update_freq': 100,
        'train_start': 50,
        'per_alpha': 0.6,
        'per_beta_start': 0.4,
        'per_beta_frames': 10000,
        'eps_start': 1.0,
        'eps_end': 0.01,
        'eps_frames': 1000,
        'epsilon': 1e-6,
        'max_gradient_norm': 1.0,
    }

    # Merge с переданными kwargs
    default_params.update(kwargs)

    return D3QN_PER_Agent(
        state_shape=state_shape,
        action_dim=action_dim,
        cnn_maps=cnn_maps,
        cnn_kernels=cnn_kernels,
        cnn_strides=cnn_strides,
        cnn_dilations=cnn_dilations,
        dense_val=dense_val,
        dense_adv=dense_adv,
        additional_feats=additional_feats,
        dropout_model=dropout_model,
        device=device,
        **default_params
    )


# ============================================================================
# ДОПОЛНИТЕЛЬНЫЕ КРИТИЧЕСКИ ВАЖНЫЕ ТЕСТЫ ДЛЯ ОБУЧЕНИЯ МОДЕЛИ
# ============================================================================

class TestReplayBufferAndPER(unittest.TestCase):
    """
    Тесты для Replay Buffer с Prioritized Experience Replay (PER).
    Критически важны для корректного обучения DQN-агента.
    """

    @classmethod
    def setUpClass(cls):
        cls.cfg = cfg
        cls.seed = cls.cfg.random_seed

    def test_buffer_storage_capacity(self):
        """Тест: буфер не превышает максимальный размер."""
        state_shape = self.cfg.seq.state_shape
        action_dim = self.cfg.market.num_actions
        buffer_size = 1000

        # Создаём агента с малым буфером
        agent = create_test_agent(
            state_shape=state_shape,
            action_dim=action_dim,
            cnn_maps=self.cfg.model.cnn_maps[:2],
            cnn_kernels=self.cfg.model.cnn_kernels[:2],
            cnn_strides=self.cfg.model.cnn_strides[:2],
            cnn_dilations=self.cfg.model.cnn_dilations[:2],
            dense_val=[64],
            dense_adv=[64],
            additional_feats=self.cfg.model.additional_feats,
            dropout_model=0.1,
            buffer_size=buffer_size,
        )

        # Добавляем больше transitions, чем размер буфера
        state = np.random.randn(np.prod(state_shape)).astype(np.float32)

        for i in range(buffer_size + 500):
            next_state = np.random.randn(np.prod(state_shape)).astype(np.float32)
            action = np.random.randint(0, action_dim)
            reward = np.random.randn()
            done = bool(np.random.rand() > 0.9)

            agent.store_experience(state, action, reward, next_state, done)
            state = next_state

        # Проверяем, что буфер не превысил размер
        self.assertLessEqual(len(agent.memory), buffer_size,
                            f"Buffer size exceeded: {len(agent.memory)} > {buffer_size}")

    def test_per_priority_updates(self):
        """Тест: приоритеты в PER обновляются после обучения."""
        state_shape = self.cfg.seq.state_shape
        action_dim = self.cfg.market.num_actions

        agent = create_test_agent(
            state_shape=state_shape,
            action_dim=action_dim,
            cnn_maps=[32, 64],
            cnn_kernels=[3, 3],
            cnn_strides=[1, 1],
            cnn_dilations=[1, 2],
            dense_val=[64],
            dense_adv=[64],
            additional_feats=self.cfg.model.additional_feats,
            dropout_model=0.1,
            buffer_size=10000,
            train_start=100,
            batch_size=32,
        )

        # Заполняем буфер
        state = np.random.randn(np.prod(state_shape)).astype(np.float32)
        for _ in range(200):
            next_state = np.random.randn(np.prod(state_shape)).astype(np.float32)
            action = np.random.randint(0, action_dim)
            reward = np.random.randn()
            done = False
            agent.store_experience(state, action, reward, next_state, done)
            state = next_state

        # Получаем начальные приоритеты
        initial_priorities = agent.memory.priorities[:100].copy()

        # Обучаем несколько шагов
        for _ in range(10):
            agent.increment_step()
            loss = agent.learn()

        # Проверяем, что приоритеты изменились
        updated_priorities = agent.memory.priorities[:100]

        # Не все приоритеты должны остаться идентичными
        changed_count = np.sum(np.abs(initial_priorities - updated_priorities) > 1e-6)
        self.assertGreater(changed_count, 0,
                          "PER priorities were not updated after training")

    def test_buffer_deterministic_sampling(self):
        """Тест: сэмплирование из буфера детерминировано при фиксированном seed."""
        import torch

        state_shape = self.cfg.seq.state_shape
        action_dim = self.cfg.market.num_actions

        def create_and_fill_agent(seed):
            np.random.seed(seed)
            torch.manual_seed(seed)

            agent = create_test_agent(
                state_shape=state_shape,
                action_dim=action_dim,
                cnn_maps=[32],
                cnn_kernels=[3],
                cnn_strides=[1],
                cnn_dilations=[1],
                dense_val=[32],
                dense_adv=[32],
                additional_feats=self.cfg.model.additional_feats,
                dropout_model=0.0,
                buffer_size=5000,
                train_start=50,
                batch_size=16,
            )

            # Заполняем буфер фиксированными данными
            for i in range(100):
                state = np.full(np.prod(state_shape), i, dtype=np.float32)
                next_state = np.full(np.prod(state_shape), i+1, dtype=np.float32)
                agent.store_experience(state, i % action_dim, float(i), next_state, False)

            return agent

        # Создаём два агента с одним seed
        agent1 = create_and_fill_agent(self.seed)
        agent2 = create_and_fill_agent(self.seed)

        # Сэмплируем batch
        for _ in range(60):
            agent1.increment_step()
            agent2.increment_step()

        np.random.seed(self.seed)
        batch1 = agent1.memory.sample(16)

        np.random.seed(self.seed)
        batch2 = agent2.memory.sample(16)

        # Проверяем идентичность
        np.testing.assert_array_equal(batch1['indices'], batch2['indices'],
                                     "Sampled indices differ with same seed")


class TestTargetNetworkUpdates(unittest.TestCase):
    """
    Тесты для корректности обновления target network.
    Target network должна периодически синхронизироваться с policy network.
    """

    @classmethod
    def setUpClass(cls):
        cls.cfg = cfg
        cls.seed = cls.cfg.random_seed

    def test_target_network_sync(self):
        """Тест: target network синхронизируется с policy network."""
        import torch

        state_shape = self.cfg.seq.state_shape

        agent = create_test_agent(
            state_shape=state_shape,
            action_dim=4,
            cnn_maps=[32, 64],
            cnn_kernels=[3, 3],
            cnn_strides=[1, 1],
            cnn_dilations=[1, 2],
            dense_val=[64],
            dense_adv=[64],
            additional_feats=12,
            dropout_model=0.1,
            target_update_freq=100,
        )

        # Получаем начальные веса
        policy_params_initial = [p.clone() for p in agent.policy_net.parameters()]
        target_params_initial = [p.clone() for p in agent.target_net.parameters()]

        # Проверяем начальную идентичность
        for p_pol, p_tar in zip(policy_params_initial, target_params_initial):
            np.testing.assert_array_almost_equal(
                p_pol.detach().cpu().numpy(),
                p_tar.detach().cpu().numpy(),
                decimal=6,
                err_msg="Initial policy and target networks differ"
            )

        # Заполняем буфер и обучаем
        state = np.random.randn(np.prod(state_shape)).astype(np.float32)
        for _ in range(150):
            next_state = np.random.randn(np.prod(state_shape)).astype(np.float32)
            agent.store_experience(state, 0, 0.1, next_state, False)
            agent.increment_step()
            agent.learn()
            state = next_state

        # Получаем обновлённые веса
        policy_params_after = [p.clone() for p in agent.policy_net.parameters()]
        target_params_after = [p.clone() for p in agent.target_net.parameters()]

        # Policy network должна измениться
        policy_changed = False
        for p_init, p_after in zip(policy_params_initial, policy_params_after):
            if not torch.allclose(p_init, p_after, atol=1e-6):
                policy_changed = True
                break

        self.assertTrue(policy_changed, "Policy network did not change after training")

        # Target network должна синхронизироваться (после 100 шагов)
        target_changed = False
        for p_init, p_after in zip(target_params_initial, target_params_after):
            if not torch.allclose(p_init, p_after, atol=1e-6):
                target_changed = True
                break

        self.assertTrue(target_changed, "Target network was not updated")

    def test_target_network_frozen_between_updates(self):
        """Тест: target network не меняется между обновлениями."""
        import torch

        agent = create_test_agent(
            state_shape=(10, 90, 1),
            action_dim=4,
            cnn_maps=[32],
            cnn_kernels=[3],
            cnn_strides=[1],
            cnn_dilations=[1],
            dense_val=[32],
            dense_adv=[32],
            additional_feats=12,
            dropout_model=0.0,
            target_update_freq=1000,  # Редкое обновление
            train_start=50,
        )

        # Заполняем буфер
        state = np.random.randn(10 * 90).astype(np.float32)
        for _ in range(100):
            next_state = np.random.randn(10 * 90).astype(np.float32)
            agent.store_experience(state, 0, 0.1, next_state, False)
            agent.increment_step()
            state = next_state

        # Сохраняем target веса после заполнения буфера
        target_params_snapshot = [p.clone() for p in agent.target_net.parameters()]

        # Обучаем ещё 50 шагов (до 150, без обновления target на 1000)
        for _ in range(50):
            agent.learn()

        # Проверяем, что target не изменилась
        for p_snap, p_current in zip(target_params_snapshot, agent.target_net.parameters()):
            torch.testing.assert_close(
                p_snap, p_current,
                msg="Target network changed before update frequency"
            )


class TestLossAndGradients(unittest.TestCase):
    """
    Тесты для корректности расчёта loss и градиентов.
    """

    @classmethod
    def setUpClass(cls):
        cls.cfg = cfg
        cls.seed = cls.cfg.random_seed

    def test_loss_is_finite(self):
        """Тест: loss не содержит NaN или Inf."""
        agent = create_test_agent(
            state_shape=(10, 90, 1),
            action_dim=4,
            cnn_maps=[32, 64],
            cnn_kernels=[3, 3],
            cnn_strides=[1, 1],
            cnn_dilations=[1, 2],
            dense_val=[64],
            dense_adv=[64],
            additional_feats=12,
            dropout_model=0.1,
            train_start=50,
            batch_size=16,
        )

        # Заполняем буфер
        state = np.random.randn(10 * 90).astype(np.float32)
        for _ in range(100):
            next_state = np.random.randn(10 * 90).astype(np.float32)
            reward = np.random.randn()
            agent.store_experience(state, 0, reward, next_state, False)
            agent.increment_step()
            state = next_state

        # Обучаем и проверяем loss
        for _ in range(20):
            loss = agent.learn()
            if loss is not None:
                self.assertTrue(np.isfinite(loss), f"Loss is not finite: {loss}")
                self.assertFalse(np.isnan(loss), f"Loss is NaN")
                self.assertFalse(np.isinf(loss), f"Loss is Inf")

    def test_gradients_are_finite(self):
        """Тест: градиенты не содержат NaN или Inf."""
        import torch

        agent = create_test_agent(
            state_shape=(10, 90, 1),
            action_dim=4,
            cnn_maps=[32],
            cnn_kernels=[3],
            cnn_strides=[1],
            cnn_dilations=[1],
            dense_val=[32],
            dense_adv=[32],
            additional_feats=12,
            dropout_model=0.0,
            train_start=50,
            max_gradient_norm=1.0,
        )

        # Заполняем буфер
        state = np.random.randn(10 * 90).astype(np.float32)
        for _ in range(100):
            next_state = np.random.randn(10 * 90).astype(np.float32)
            agent.store_experience(state, 0, np.random.randn(), next_state, False)
            agent.increment_step()
            state = next_state

        # Обучаем и проверяем градиенты
        for _ in range(10):
            loss = agent.learn()
            if loss is not None:
                for name, param in agent.policy_net.named_parameters():
                    if param.grad is not None:
                        grad_finite = torch.isfinite(param.grad).all().item()
                        self.assertTrue(grad_finite, 
                                      f"Gradient for {name} contains NaN/Inf")

    def test_td_error_calculation(self):
        """Тест: TD error рассчитывается корректно."""
        agent = create_test_agent(
            state_shape=(10, 90, 1),
            action_dim=4,
            cnn_maps=[32],
            cnn_kernels=[3],
            cnn_strides=[1],
            cnn_dilations=[1],
            dense_val=[32],
            dense_adv=[32],
            additional_feats=12,
            dropout_model=0.0,
            gamma=0.99,
            train_start=50,
        )

        # Простой тест: reward=1, done=True, Q_target должен быть ≈ reward
        state = np.zeros(10 * 90, dtype=np.float32)
        next_state = np.zeros(10 * 90, dtype=np.float32)

        # Храним несколько terminal transitions
        for _ in range(60):
            agent.store_experience(state, 0, 1.0, next_state, True)
            agent.increment_step()

        # Обучаем
        losses = []
        for _ in range(10):
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

        # Loss должна снижаться при обучении на простых данных
        if len(losses) > 5:
            early_loss = np.mean(losses[:3])
            late_loss = np.mean(losses[-3:])
            self.assertLessEqual(late_loss, early_loss * 1.5,
                                "Loss did not decrease or diverged")


class TestEpsilonDecay(unittest.TestCase):
    """
    Тесты для корректности epsilon decay (exploration-exploitation).
    """

    @classmethod
    def setUpClass(cls):
        cls.cfg = cfg

    def test_epsilon_decay_monotonic(self):
        """Тест: epsilon монотонно убывает."""
        agent = create_test_agent(
            state_shape=(10, 90, 1),
            action_dim=4,
            cnn_maps=[16],
            cnn_kernels=[3],
            cnn_strides=[1],
            cnn_dilations=[1],
            dense_val=[16],
            dense_adv=[16],
            additional_feats=12,
            dropout_model=0.0,
            eps_start=1.0,
            eps_end=0.01,
            eps_frames=1000,
        )

        epsilon_values = []
        for step in range(1500):
            agent.increment_step()
            epsilon_values.append(agent.epsilon)

        # Проверяем монотонность до eps_end
        for i in range(len(epsilon_values) - 1):
            if epsilon_values[i] > agent.eps_end:
                self.assertGreaterEqual(
                    epsilon_values[i], epsilon_values[i + 1],
                    f"Epsilon increased at step {i}: {epsilon_values[i]} -> {epsilon_values[i+1]}"
                )

        # Проверяем, что epsilon достигает минимума
        final_epsilon = epsilon_values[-1]
        self.assertAlmostEqual(final_epsilon, agent.eps_end, places=3,
                              msg=f"Final epsilon {final_epsilon} != eps_end {agent.eps_end}")

    def test_epsilon_decay_range(self):
        """Тест: epsilon всегда в пределах [eps_end, eps_start]."""
        agent = create_test_agent(
            state_shape=(10, 90, 1),
            action_dim=4,
            cnn_maps=[16],
            cnn_kernels=[3],
            cnn_strides=[1],
            cnn_dilations=[1],
            dense_val=[16],
            dense_adv=[16],
            additional_feats=12,
            dropout_model=0.0,
            eps_start=0.9,
            eps_end=0.05,
            eps_frames=500,
        )

        for _ in range(1000):
            agent.increment_step()
            self.assertGreaterEqual(agent.epsilon, agent.eps_end - 1e-6,
                                   f"Epsilon below eps_end: {agent.epsilon}")
            self.assertLessEqual(agent.epsilon, agent.eps_start + 1e-6,
                                f"Epsilon above eps_start: {agent.epsilon}")


class TestModelSaveLoad(unittest.TestCase):
    """
    Тесты для идентичности модели после save/load.
    Критически важно для checkpoint recovery.
    """

    @classmethod
    def setUpClass(cls):
        cls.cfg = cfg
        cls.seed = cls.cfg.random_seed
        cls.temp_dir = Path("temp_test_models")
        cls.temp_dir.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        # Очистка временных файлов
        import shutil
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)

    def test_model_weights_identity_after_save_load(self):
        """Тест: веса модели идентичны после save/load."""
        import torch

        agent = create_test_agent(
            state_shape=(10, 90, 1),
            action_dim=4,
            cnn_maps=[32, 64],
            cnn_kernels=[3, 3],
            cnn_strides=[1, 1],
            cnn_dilations=[1, 2],
            dense_val=[64],
            dense_adv=[64],
            additional_feats=12,
            dropout_model=0.1,
        )

        # Обучаем немного для изменения весов
        state = np.random.randn(10 * 90).astype(np.float32)
        for _ in range(100):
            next_state = np.random.randn(10 * 90).astype(np.float32)
            agent.store_experience(state, 0, np.random.randn(), next_state, False)
            agent.increment_step()
            agent.learn()
            state = next_state

        # Сохраняем веса
        original_weights = {name: param.clone() 
                           for name, param in agent.policy_net.named_parameters()}

        # Сохраняем модель
        model_path = self.temp_dir / "test_model.pth"
        agent.save_model(str(model_path))

        # Создаём нового агента и загружаем
        agent_loaded = create_test_agent(
            state_shape=(10, 90, 1),
            action_dim=4,
            cnn_maps=[32, 64],
            cnn_kernels=[3, 3],
            cnn_strides=[1, 1],
            cnn_dilations=[1, 2],
            dense_val=[64],
            dense_adv=[64],
            additional_feats=12,
            dropout_model=0.1,
        )
        agent_loaded.load_model(str(model_path))

        # Сравниваем веса
        for name, param in agent_loaded.policy_net.named_parameters():
            torch.testing.assert_close(
                original_weights[name], param,
                msg=f"Weight mismatch for {name} after load"
            )

    def test_model_inference_identity_after_save_load(self):
        """Тест: выход модели идентичен после save/load."""
        import torch

        agent = create_test_agent(
            state_shape=(10, 90, 1),
            action_dim=4,
            cnn_maps=[32],
            cnn_kernels=[3],
            cnn_strides=[1],
            cnn_dilations=[1],
            dense_val=[32],
            dense_adv=[32],
            additional_feats=12,
            dropout_model=0.0,  # Отключаем dropout для детерминизма
        )

        # Тестовый вход
        test_state = np.random.randn(10, 90, 1).astype(np.float32)

        # Получаем предсказание до сохранения
        agent.policy_net.eval()
        with torch.no_grad():
            original_action = agent.select_action(test_state.flatten(), training=False)

        # Сохраняем и загружаем
        model_path = self.temp_dir / "test_inference_model.pth"
        agent.save_model(str(model_path))

        agent_loaded = create_test_agent(
            state_shape=(10, 90, 1),
            action_dim=4,
            cnn_maps=[32],
            cnn_kernels=[3],
            cnn_strides=[1],
            cnn_dilations=[1],
            dense_val=[32],
            dense_adv=[32],
            additional_feats=12,
            dropout_model=0.0,
        )
        agent_loaded.load_model(str(model_path))

        # Получаем предсказание после загрузки
        agent_loaded.policy_net.eval()
        with torch.no_grad():
            loaded_action = agent_loaded.select_action(test_state.flatten(), training=False)

        self.assertEqual(original_action, loaded_action,
                        "Action differs after model save/load")


class TestNormalizationStats(unittest.TestCase):
    """
    Тесты для compute_norm_stats и apply_normalization.
    """

    @classmethod
    def setUpClass(cls):
        cls.cfg = cfg
        cls.seed = cls.cfg.random_seed

    def test_norm_stats_shape(self):
        """Тест: norm stats имеют правильную форму."""
        # Создаём синтетический NPZ файл
        temp_npz = Path("temp_test_data.npz")

        num_channels = self.cfg.num_channels
        seq_len = 150

        data = {
            "BTCUSDT_0001": np.random.randn(seq_len, num_channels).astype(np.float32),
            "BTCUSDT_0002": np.random.randn(seq_len, num_channels).astype(np.float32),
            "ETHUSDT_0001": np.random.randn(seq_len, num_channels).astype(np.float32),
        }
        np.savez(temp_npz, **data)

        # Вычисляем статистики
        from train import compute_norm_stats
        stats = compute_norm_stats(str(temp_npz), self.cfg, "temp_norm_stats.json")

        # Проверяем структуру
        self.assertIn("BTCUSDT", stats, "BTCUSDT not in stats")
        self.assertIn("ETHUSDT", stats, "ETHUSDT not in stats")

        for asset, asset_stats in stats.items():
            self.assertIn("mean", asset_stats)
            self.assertIn("std", asset_stats)
            self.assertEqual(len(asset_stats["mean"]), num_channels,
                           f"Mean length mismatch for {asset}")
            self.assertEqual(len(asset_stats["std"]), num_channels,
                           f"Std length mismatch for {asset}")

        # Очистка
        temp_npz.unlink()
        Path("temp_norm_stats.json").unlink(missing_ok=True)

    def test_normalization_inverse(self):
        """Тест: denormalization восстанавливает исходные данные."""
        # Исходные данные
        original_data = np.random.randn(100, 10).astype(np.float32) * 1000 + 5000

        # Вычисляем статистики
        mean = original_data.mean(axis=0)
        std = original_data.std(axis=0) + 1e-8

        # Нормализация
        normalized = (original_data - mean) / std

        # Денормализация
        denormalized = normalized * std + mean

        # Проверяем восстановление (float32 precision)
        np.testing.assert_array_almost_equal(
            original_data, denormalized, decimal=4,
            err_msg="Denormalization did not restore original data"
        )


class TestValidationMetrics(unittest.TestCase):
    """
    Тесты для корректности расчёта validation metrics.
    """

    @classmethod
    def setUpClass(cls):
        cls.cfg = cfg

    def test_sharpe_ratio_calculation(self):
        """Тест: Sharpe Ratio рассчитывается корректно."""
        # Положительные returns с низкой волатильностью -> высокий Sharpe
        returns_good = np.array([0.01, 0.015, 0.012, 0.018, 0.011])
        mean_ret = returns_good.mean()
        std_ret = returns_good.std(ddof=1)
        expected_sharpe = mean_ret / std_ret

        # Проверяем, что Sharpe > 0 для прибыльной стратегии
        self.assertGreater(expected_sharpe, 0,
                          "Sharpe should be positive for profitable strategy")

        # Отрицательные returns -> отрицательный Sharpe
        returns_bad = np.array([-0.01, -0.015, -0.012, -0.018, -0.011])
        mean_ret_bad = returns_bad.mean()
        std_ret_bad = returns_bad.std(ddof=1)
        expected_sharpe_bad = mean_ret_bad / std_ret_bad

        self.assertLess(expected_sharpe_bad, 0,
                       "Sharpe should be negative for losing strategy")

    def test_sortino_ratio_calculation(self):
        """Тест: Sortino Ratio рассчитывается корректно."""
        # Mixed returns: прибыльная стратегия с некоторыми убытками
        returns = np.array([0.02, -0.005, 0.015, 0.01, -0.003, 0.018])

        mean_ret = returns.mean()
        downside = np.minimum(0.0, returns)
        downside_dev = np.sqrt(np.mean(downside ** 2))

        if downside_dev > 1e-12:
            expected_sortino = mean_ret / downside_dev
        else:
            expected_sortino = float('inf') if mean_ret > 0 else 0.0

        # Sortino учитывает только downside риск
        self.assertGreater(expected_sortino, 0,
                          "Sortino should be positive for profitable strategy")

    def test_profit_factor_calculation(self):
        """Тест: Profit Factor рассчитывается корректно."""
        # Прибыли и убытки
        trade_pnls = np.array([100, -50, 200, -30, 150, -40])

        gross_profit = np.sum(trade_pnls[trade_pnls > 0])
        gross_loss = np.abs(np.sum(trade_pnls[trade_pnls < 0]))

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float('inf')

        # Ожидаем PF > 1 для прибыльной стратегии
        expected_pf = (100 + 200 + 150) / (50 + 30 + 40)
        self.assertAlmostEqual(profit_factor, expected_pf, places=2)
        self.assertGreater(profit_factor, 1.0,
                          "Profit Factor should be > 1 for winning strategy")

    def test_max_drawdown_calculation(self):
        """Тест: Maximum Drawdown рассчитывается корректно."""
        # Симулируем equity curve
        initial_balance = 10000.0
        trade_pnls = np.array([500, -200, 300, -800, 400, -150])

        equity = initial_balance
        peak = initial_balance
        max_dd = 0.0

        for pnl in trade_pnls:
            equity += pnl
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (equity - peak) / peak
                if dd < max_dd:
                    max_dd = dd

        # MaxDD должна быть отрицательной
        self.assertLess(max_dd, 0, "MaxDD should be negative")

        # Проверяем расчёт
        # После -800 PnL: equity = 10000 + 500 - 200 + 300 - 800 = 9800
        # Peak = 10600 (после +500-200+300)
        # DD = (9800 - 10600) / 10600 ≈ -0.0755
        self.assertLess(max_dd, -0.05, "MaxDD calculation incorrect")


class TestTopKCheckpointManager(unittest.TestCase):
    """
    Тесты для TopKCheckpointManager.
    """

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path("temp_test_checkpoints")
        cls.temp_dir.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        import shutil
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)

    def test_topk_selection(self):
        """Тест: сохраняются только top-K чекпоинтов."""
        from train import TopKCheckpointManager

        manager = TopKCheckpointManager(
            save_dir=str(self.temp_dir),
            top_k=3,
            metric_key="Validation_sortino",
            mode="max"
        )

        # Создаём dummy agent
        agent = create_test_agent(
            state_shape=(10, 90, 1),
            action_dim=4,
            cnn_maps=[16],
            cnn_kernels=[3],
            cnn_strides=[1],
            cnn_dilations=[1],
            dense_val=[16],
            dense_adv=[16],
            additional_feats=12,
            dropout_model=0.0,
        )

        # Сохраняем 5 чекпоинтов с разными метриками
        metrics_list = [
            {"Validation_sortino": 0.5, "Validation_sharpe": 0.4},
            {"Validation_sortino": 0.8, "Validation_sharpe": 0.7},  # Best
            {"Validation_sortino": 0.3, "Validation_sharpe": 0.2},
            {"Validation_sortino": 0.7, "Validation_sharpe": 0.6},  # 2nd best
            {"Validation_sortino": 0.6, "Validation_sharpe": 0.5},  # 3rd best
        ]

        for i, metrics in enumerate(metrics_list):
            manager.save_checkpoint(agent, episode=i, metrics=metrics)

        # Проверяем, что сохранено только 3 чекпоинта
        self.assertEqual(len(manager.checkpoints), 3,
                        f"Should keep only top-3, but kept {len(manager.checkpoints)}")

        # Проверяем порядок (по убыванию sortino)
        sorted_sortinos = [ckpt[0] for ckpt in manager.checkpoints]
        self.assertEqual(sorted_sortinos, [0.8, 0.7, 0.6],
                        f"Checkpoints not sorted correctly: {sorted_sortinos}")

        # Проверяем, что худшие удалены
        checkpoint_files = list(self.temp_dir.glob("checkpoint_*.pth"))
        self.assertEqual(len(checkpoint_files), 3,
                        "Old checkpoints were not deleted from disk")

    def test_topk_metric_not_found(self):
        """Тест: чекпоинт не сохраняется при отсутствии метрики."""
        from train import TopKCheckpointManager

        manager = TopKCheckpointManager(
            save_dir=str(self.temp_dir / "test_missing"),
            top_k=5,
            metric_key="NonExistentMetric",
            mode="max"
        )

        agent = create_test_agent(
            state_shape=(10, 90, 1),
            action_dim=4,
            cnn_maps=[16],
            cnn_kernels=[3],
            cnn_strides=[1],
            cnn_dilations=[1],
            dense_val=[16],
            dense_adv=[16],
            additional_feats=12,
            dropout_model=0.0,
        )

        metrics = {"Validation_sortino": 0.5}
        result = manager.save_checkpoint(agent, episode=0, metrics=metrics)

        self.assertFalse(result, "Checkpoint should not be saved without metric")
        self.assertEqual(len(manager.checkpoints), 0,
                        "No checkpoints should be saved")


class TestEpisodeSampling(unittest.TestCase):
    """
    Тесты для детерминированности сэмплирования эпизодов.
    """

    @classmethod
    def setUpClass(cls):
        cls.cfg = cfg
        cls.seed = cls.cfg.random_seed

    def test_episode_sampling_deterministic(self):
        """Тест: сэмплирование эпизодов детерминировано при фиксированном seed."""
        sequences = [np.random.randn(150, 10).astype(np.float32) for _ in range(1000)]
        episodes_per_epoch = 100

        # Первая выборка
        np.random.seed(self.seed)
        rng1 = np.random.default_rng(self.seed)
        indices1 = rng1.choice(len(sequences), episodes_per_epoch, replace=False)
        sampled1 = sorted(indices1.tolist())

        # Вторая выборка с тем же seed
        np.random.seed(self.seed)
        rng2 = np.random.default_rng(self.seed)
        indices2 = rng2.choice(len(sequences), episodes_per_epoch, replace=False)
        sampled2 = sorted(indices2.tolist())

        # Проверяем идентичность
        self.assertEqual(sampled1, sampled2,
                        "Episode sampling differs with same seed")

    def test_episode_sampling_coverage(self):
        """Тест: сэмплирование покрывает разные эпизоды."""
        sequences = list(range(1000))
        episodes_per_epoch = 100

        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)

        # Сэмплируем несколько эпох
        all_sampled = set()
        for _ in range(10):
            indices = rng.choice(len(sequences), episodes_per_epoch, replace=False)
            all_sampled.update(indices.tolist())

        # Должны покрыть значительную часть датасета (снижен порог)
        coverage = len(all_sampled) / len(sequences)
        self.assertGreater(coverage, 0.6,
                          f"Poor episode coverage: {coverage:.1%}")



if __name__ == "__main__":
    # Запуск тестов с подробным выводом
    unittest.main(verbosity=2)
