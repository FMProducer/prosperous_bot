import sys
import os
sys.path.insert(0, os.path.abspath('third_party/rl-trading-binance'))

import types
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Подменяем db_provider:get_feed
db_mod = types.ModuleType("db_provider")
def _mk_df(start, end):
    idx = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")
    close = np.full(len(idx), 100.0)
    # create a spike of 5% in the middle of the dataframe
    spike_start_index = len(idx) // 2
    close[spike_start_index:spike_start_index+10] = np.linspace(100, 105, 10)
    return pd.DataFrame({"close": close}, index=idx)
def get_feed(symbols, start_iso, end_iso):
    start = pd.Timestamp(start_iso).tz_convert("UTC")
    end = pd.Timestamp(end_iso).tz_convert("UTC")
    return {symbols[0]: _mk_df(start, end)}
db_mod.get_feed = get_feed
sys.modules["db_provider"] = db_mod

# Подменяем inference_adapter:load_policy
inf_mod = types.ModuleType("inference_adapter")
class _DummyPolicy:
    def predict_side(self, df_ctx):
        # Всегда BUY
        return "BUY"
def load_policy(ckpt_path):
    return _DummyPolicy()
inf_mod.load_policy = load_policy
sys.modules["inference_adapter"] = inf_mod

import importlib.util
spec = importlib.util.spec_from_file_location("paper_trader", "third_party/rl-trading-binance/paper_trader.py")
paper_trader = importlib.util.module_from_spec(spec)
sys.modules['paper_trader'] = paper_trader
spec.loader.exec_module(paper_trader)
main = paper_trader.main

def test_paper_trader_runs_with_inference(tmp_path, monkeypatch):
    # Готовим временный конфиг
    checkpoint_path_str = (tmp_path / "dummy.pth").as_posix()
    cfg_text = f'''
data = {{
  "source": "stream_sim_db",
  "time_range": {{
    "start_utc": "2025-03-01T00:00:00Z",
    "end_utc": "2025-03-01T06:00:00Z"
  }},
  "ctx_minutes": 30,
  "session_minutes": 10,
  "build_index_from_db": True,
  "symbols": ["BTCUSDT"],
  "detector": {{
    "context_minutes": 90,
    "window_minutes": 10,
    "use_lookahead": True,
    "abs_change_pct": 1.0,
    "contrast_min": 1.0,
    "cooldown_minutes": 60
  }},
  "db_provider": "db_provider:get_feed",
  "inference": {{
    "policy_loader": "inference_adapter:load_policy",
    "checkpoint_path": "{checkpoint_path_str}",
    "strict": True
  }},
  "paper_trader": {{"mode": "asap", "cap_windows_per_symbol": 1}},
  "exec": {{
    "base_capital_usdt": 1000.0,
    "risk_per_trade_pct": 1.0,
    "fee_bps": 0.0,
    "slippage_bps": 0.0
  }}
}}
'''
    cfg_file = tmp_path / "alpha.py"
    cfg_file.write_text(cfg_text, encoding="utf-8")

    # Запуск
    rc = main(["paper_trader.py", str(cfg_file)])
    assert rc == 0
    out_dir = Path("third_party") / "rl-trading-binance" / "output" / "alpha"
    # Файлы с результатами должны появиться
    assert (out_dir / "trades.csv").exists()
    assert (out_dir / "metrics.json").exists()