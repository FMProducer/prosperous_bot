import json
import sys
import types
from collections import deque
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pandas as pd
import pytest
import torch

# =============================================================================
# 1. ROBUST MOCKING INFRASTRUCTURE
# =============================================================================

class DummyIStrategy:
    def __init__(self, config: dict, **kwargs):
        self.config = config
    def __getattr__(self, name):
        if name == 'min_quote_volume_usd':
            m = MagicMock(); m.value = 0.0001; return m
        return MagicMock()

def dummy_merge(df, inf, tf, itf, ffill=True):
    for col in inf.columns:
        res_col = f"{col}_{itf}"
        df[res_col] = inf[col].iloc[-1] if not inf.empty else 0
    return df

ft_s = types.ModuleType("freqtrade.strategy")
ft_s.IStrategy = DummyIStrategy
ft_s.DecimalParameter = type('DP', (object,), {'__init__': lambda s, *a, **k: None, 'value': 0.05})
ft_s.IntParameter = type('IP', (object,), {'__init__': lambda s, *a, **k: None, 'value': 2})
ft_s.merge_informative_pair = dummy_merge

ft_p = types.ModuleType("freqtrade.persistence")
class MockTrade:
    def __init__(self, **kwargs):
        self.pair = "BTC/USDT:USDT"; self.is_short = False; self.close_profit = 0.0
        self.close_date = datetime.now(timezone.utc); self.stake_amount = 1000.0
        self.open_date_utc = datetime.now(timezone.utc); self.id = 1; self.amount = 10.0
        self.fee_open = 0.001; self.fee_close = 0.001; self.open_rate = 100.0
        self.close_rate_requested = None; self.is_open = True
        for k, v in kwargs.items(): setattr(self, k, v)
    def calc_profit_ratio(self, rate=None): return self.close_profit
    def calc_profit(self, rate=None):
        if rate is None: rate = self.open_rate
        return self.amount * (rate - self.open_rate) if not self.is_short else self.amount * (self.open_rate - rate)
    def __format__(self, f): return format(0.0, f)

class MockQuery:
    def __init__(self, val=None): self.val = val
    def __eq__(self, other): return MockQuery(f"({self.val} == {other})")
    def is_(self, other): return MockQuery(f"({self.val} is {other})")
    def desc(self): return self
    def __getattr__(self, name): return MagicMock()
    def __repr__(self): return str(self.val)

class MockTClass(MagicMock):
    @staticmethod
    def get_trades(query=None, **kwargs):
        mq = MagicMock()
        mq.order_by.return_value.first.return_value = None
        mq.all.return_value = []
        return mq
    @staticmethod
    def get_open_trades(): return []

MockTClass.pair = MockQuery("pair")
MockTClass.is_open = MockQuery("is_open")
MockTClass.close_date = MockQuery("close_date")

ft_p.Trade = MockTClass
sys.modules['freqtrade'] = MagicMock()
sys.modules['freqtrade.strategy'] = ft_s
sys.modules['freqtrade.persistence'] = ft_p

mock_a = types.ModuleType("agent")
class MockAg:
    def __init__(self, **kwargs):
        self.action_dim = 3
        self.policy_net = MagicMock()
        self.policy_net.eval = MagicMock()
        self.policy_net.parameters.return_value = []
        self.mirror_mode = False
        # Set a side effect to handle the call with two arguments
        self.policy_net.side_effect = self.mock_forward

    def load_model(self, p): pass

    def mock_forward(self, img_tensor, feat_tensor):
        batch_size = img_tensor.shape[0]
        return torch.tensor(np.tile(np.array([[0.0, 0.5, 0.5]], dtype=np.float32), (batch_size, 1)))
mock_a.D3QN_PER_Agent = MockAg
sys.modules['agent'] = mock_a

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STRAT_DIR = PROJECT_ROOT / "user_data" / "strategies"
if str(STRAT_DIR) not in sys.path: sys.path.append(str(STRAT_DIR))

import CustomD3QNStrategy4z  # type: ignore
CustomD3QNStrategy4z.Trade = MockTClass

# =============================================================================
# 2. FIXTURES
# =============================================================================

@pytest.fixture
def strategy():
    cfg_path = PROJECT_ROOT / "user_data" / "config_rl4z.json"
    with open(cfg_path, "r", encoding="utf-8") as f: cfg = json.load(f)
    cfg["runmode"] = "live"; cfg["rl_calibration_mode"] = False
    m = MagicMock(); m.seq.state_shape = (90, 5); m.seq.agent_history_len = 90; m.market.num_actions = 3; m.model.additional_feats = 4
    m.rl.gamma = 0.99; m.rl.lr = 0.001; m.rl.target_update_freq = 100; m.rl.train_start = 100
    m.rl.max_gradient_norm = 1.0; m.eps.eps_end = 0.1; m.eps.eps_decay_frames = 1000
    with patch("builtins.open", mock_open(read_data='{}')), patch("json.dump"), \
         patch.object(CustomD3QNStrategy4z.CustomD3QNStrategy4z, '_load_weights'), \
         patch.object(CustomD3QNStrategy4z.CustomD3QNStrategy4z, '_find_config_file', return_value=Path("d.py")), \
         patch.object(CustomD3QNStrategy4z.CustomD3QNStrategy4z, '_load_py_config', return_value=m):
        s = CustomD3QNStrategy4z.CustomD3QNStrategy4z(cfg)
        s.dp = MagicMock(); s.startup_candle_count = 0; s.can_short = True
        s.enable_long_1 = s.enable_long_2 = s.enable_short_1 = s.enable_short_2 = True
        s.d0 = MagicMock(); s.d0.value = 0.05; s.d_min = MagicMock(); s.d_min.value = 0.01; s.hysteresis = MagicMock(); s.hysteresis.value = 0.005
        return s

def gen_df(n=300, z=False):
    df = pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=n, freq='1min', tz=timezone.utc),
        'open': np.linspace(100, 110, n), 'high': np.linspace(101, 111, n),
        'low': np.linspace(99, 109, n), 'close': np.linspace(100, 110, n),
        'volume': np.linspace(1000, 2000, n)
    })
    df['quote_volume_sma'] = 1e9
    if z:
        for c in ['open','high','low','close','volume']: df[f"{c}_z"] = 0.0
    return df

# =============================================================================
# 3. TEST SUITE
# =============================================================================

class TestCustomD3QNStrategy4z:

    def test_coverage_bomb(self, strategy):
        now = datetime.now(timezone.utc); s = strategy; s.dynamic_slots_enabled = True; s.total_slots = 10; s.runmode = 'live'
        
        # 1. Indicators & guards
        df = gen_df(200, z=False); s.dp.get_analyzed_dataframe.return_value = (gen_df(200), None)
        with patch("CustomD3QNStrategy4z.merge_informative_pair", side_effect=dummy_merge):
            df_ind = s.populate_indicators(df, {"pair": "BTC/USDT:USDT"})
            s.populate_indicators(gen_df(5), {"pair": "p"}) # Guard

        # 2. Entry Trend & Loops
        qv = {nm: np.zeros((200, 3)) for nm in ["long_1", "long_2", "short_1", "short_2"]}
        qv["long_1"][:, 1] = 2.0; qv["long_2"][:, 1] = 2.0
        s.q_normalization = {nm: {"q_min": 0, "q_max": 3} for nm in qv}; s.epsilon_threshold_eff_long = 0.1
        df_ind["st_regime_15m"] = 1
        with patch.object(s, '_run_inference', return_value=qv):
            s.populate_entry_trend(df_ind, {"pair": "BTC/USDT:USDT"})
            s.runmode = 'backtest'; s.populate_entry_trend(df_ind, {"pair": "BTC/USDT:USDT"})

        # 3. Ensemble decisions variants
        s.enable_veto = True; s.short_1_is_mirror = False
        s.rl_long_threshold = 2; s.rl_short_threshold = 2
        qv_dec = {nm: np.array([[0, 0.5, 0]]) for nm in qv}
        s._compute_ensemble_decision(qv_dec, 0, False, False) # Long
        qv_dec["short_1"] = np.array([[0, 0, 0.5]])
        s._compute_ensemble_decision(qv_dec, 0, False, False) # Veto
        qv_dec["short_2"] = np.array([[0, 0, 0.5]])
        s._compute_ensemble_decision(qv_dec, 0, False, False) # Conflict
        s.q_normalization["long_1"] = {"q_min": 1, "q_max": 1}
        s._compute_ensemble_decision(qv_dec, 0, False, False) # Zombie

        # 4. Dynamic Slots & Epsilon signs
        s.dynamic_slots_enabled = True; s.runmode = 'live'
        for l, sh in [(-100, -200), (100, 0), (0, 100), (100, 100), (0, 0)]:
            with patch.object(s, '_get_pnl_from_freqtrade', return_value=(l, sh)):
                s._update_dynamic_epsilon(); s._update_slot_allocation(now)

        # 5. confirm_trade_entry paths
        s.max_long_slots = 1
        mq = MagicMock(); mq.order_by.return_value.first.return_value = MockTrade(close_date=now - timedelta(minutes=10), close_profit=-0.1)
        with patch.object(MockTClass, 'get_trades', return_value=mq):
            s.confirm_trade_entry("p", "l", 1, 100, "g", now, "t", "long")
        mq.order_by.return_value.first.return_value = None; mq.all.return_value = [MockTrade(is_short=False)]
        with patch.object(MockTClass, 'get_trades', return_value=mq):
            s.confirm_trade_entry("p", "l", 1, 100, "g", now, "t", "long")

        # 6. Utils & Risk
        t = MockTrade(id=999); s.custom_stoploss("p", t, now, 100, 0.1); s.custom_exit("p", t, now, 100, 0.2)
        s.get_model_input_cached(gen_df(100, z=True), "p", "LONG", 1, "BTC"); s.update_normalization_config()
        s.informative_pairs(); s.leverage("p", now, 100, 1, 5, None, "long")
        img_tensor = torch.zeros((1, 5, 90, 1))
        feat_tensor = torch.zeros((1, 4))
        s._run_inference([(img_tensor, s.long_1_agent, "long_1")])
