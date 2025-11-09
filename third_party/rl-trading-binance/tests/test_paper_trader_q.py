import datetime as dt
import json
import logging
import os
import sys
from collections import deque
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, mock_open, patch

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text

# This sys.path manipulation is necessary for pytest to find modules in the parent directory.
# We suppress the Pylance warning because this is a valid approach for this project structure at runtime.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.utils import get_config_mod_from_str # type: ignore
from paper_trader_q import PaperTrader, get_engine # type: ignore
from config import MasterConfig # type: ignore

# Default config for tests
def get_test_config() -> Dict[str, Any]:
    # Using a function to ensure a fresh dict for each test
    return {
        "random_seed": 42,
        "paths": {
            "model_path": "mock/model.pth",
            "norm_stats_path": "mock/stats.json",
            "output_dir": "mock/output",
            "log_dir": "mock/output/paper_trader/logs",
        },
        "paper": {
            "symbols": ["BTCUSDT"],
            "source": "database",
            "leverage": 1.0,
        },
        "db": { "dsn": "sqlite:///:memory:" },
        "seq": {
            "pre_signal_len": 10, "post_signal_len": 5, "full_seq_len": 15,
            "agent_session_len": 5, "agent_history_len": 10, "input_history_len": 10,
            "action_history_len": 10, "num_features": 7, "flat_state_size": 15 * 7,
        },
        "data": {
            "expected_channels": ["open", "high", "low", "close", "volume", "volume_weighted_average", "num_trades"],
            "data_channels": ["open", "high", "low", "close", "volume", "volume_weighted_average", "num_trades"],
            "price_channels": ["open", "high", "low", "close"], "volume_channels": ["volume"],
            "other_channels": ["volume_weighted_average", "num_trades"],
        },
        "detector": {
            "context_minutes": 10, "window_minutes": 5, "abs_change_pct": 1.0,
            "contrast_min": 2.0, "cooldown_minutes": 30,
        },
        "market": { "initial_balance": 10000.0, "num_actions": 3, "transaction_fee": 0.001 },
        "backtest": {
            "max_parallel_sessions": 5, "position_fraction": 0.1,
            "selection_strategy": "advantage_based_filter", "long_action_threshold": 0.1,
            "short_action_threshold": 0.1, "use_risk_management": False, "exec_delay_bars": 0,
            "fee_buffer_mult": 2.0, "trailing_stop": 0.02, "trailing_stop_min": 0.005,
            "delta_p_hysteresis": 0.001, "ensemble_n_samples": 10, "ensemble_max_sigma": 0.1,
            "time_range": { "start_utc": "2023-01-01T00:00:00Z", "end_utc": "2023-01-01T02:00:00Z" }
        },
    }

@pytest.fixture
def mock_config() -> MasterConfig:
    return MasterConfig(**get_test_config())

@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.select_action.return_value = 0
    return agent

def paper_trader_factory(config: MasterConfig, agent: MagicMock = None, model_path_override: str = None):
    log_dir = config.paths.log_dir or "mock/logs"
    os.makedirs(log_dir, exist_ok=True)
    return PaperTrader(cfg=config, cfg_mod=None, model_path_override=model_path_override)

@pytest.fixture
def trader(mock_config, mock_agent) -> PaperTrader:
    with patch('paper_trader_q.init_agent', return_value=mock_agent), \
         patch('os.path.exists', return_value=True), patch(
        'builtins.open', mock_open(read_data=json.dumps({"means": {}, "stds": {}}))
    ), patch('paper_trader_q.setup_logging'):
            return paper_trader_factory(config=mock_config, agent=mock_agent)

@pytest.fixture
def sample_kline_df() -> pd.DataFrame:
    base_time = dt.datetime(2023, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    timestamps = [base_time + dt.timedelta(minutes=i) for i in range(200)]
    data = {
        "open": np.linspace(100, 110, 200), "high": np.linspace(101, 111, 200),
        "low": np.linspace(99, 109, 200), "close": np.linspace(100.5, 110.5, 200),
        "volume": np.random.rand(200) * 100, "volume_weighted_average": np.linspace(100.2, 110.2, 200),
        "num_trades": np.random.randint(10, 100, 200),
    }
    return pd.DataFrame(data, index=pd.Index(timestamps, name="ts"))

class TestInitialization:
    @patch('paper_trader_q.setup_logging')
    @patch('paper_trader_q.init_agent')
    def test_init_missing_model_path(self, mock_init, mock_logging, mock_config):
        mock_config.paths.model_path = None
        with patch('os.path.exists', return_value=False):
            with pytest.raises(FileNotFoundError, match="Model path not specified"):
                PaperTrader(cfg=mock_config, cfg_mod=None)

    @patch('paper_trader_q.setup_logging')
    @patch('paper_trader_q.init_agent')
    def test_init_with_cfg_mod(self, mock_init, mock_logging, mock_config):
        cfg_mod_str = "paper.leverage=2.0"
        cfg_mod = get_config_mod_from_str(cfg_mod_str)
        with patch('os.path.exists', return_value=True), patch(
            'builtins.open', mock_open(read_data=json.dumps({"means": {}, "stds": {}}))
        ):
            trader = PaperTrader(cfg=mock_config, cfg_mod=cfg_mod)
        assert trader.cfg.paper.leverage == 2.0

    @patch('paper_trader_q.setup_logging')
    @patch('paper_trader_q.init_agent')
    def test_init_missing_stats_file(self, mock_init, mock_logging, mock_config):
        with patch('os.path.exists', lambda path: 'stats.json' not in path):
            with pytest.raises(RuntimeError, match="Normalization stats not found"):
                PaperTrader(cfg=mock_config, cfg_mod=None)

    @patch('paper_trader_q.init_agent')
    def test_model_path_override(self, mock_init_agent, mock_agent):
        override_path = "override/path/model.pth"
        config = MasterConfig(**get_test_config())
        with (patch('os.path.exists', return_value=True),
              patch('builtins.open', mock_open(read_data=json.dumps({"means": {}, "stds": {}}))),
              patch('paper_trader_q.setup_logging')):
            paper_trader_factory(config=config, agent=mock_agent, model_path_override=override_path)
        mock_init_agent.assert_called_with(override_path, config, None)

    def test_symbol_loading_from_file(self, mock_agent):
        config = MasterConfig(**get_test_config())
        config.paper.symbols = []
        
        mock_files = {
            "data/tickers.txt": "BTCUSDT\nETHUSDT",
            config.paths.norm_stats_path: json.dumps({"means": {}, "stds": {}})
        }
        
        def open_side_effect(file, mode='r'):
            if file in mock_files:
                return mock_open(read_data=mock_files[file])().__enter__()
            else:
                raise FileNotFoundError(f"File not found: {file}") # pragma: no cover

        with patch('builtins.open', side_effect=open_side_effect), \
             patch('os.path.exists', return_value=True), \
             patch('paper_trader_q.init_agent', return_value=mock_agent), \
             patch('paper_trader_q.setup_logging'):
            trader = paper_trader_factory(config=config, agent=mock_agent)
        assert trader.symbols_to_trade == ["BTCUSDT", "ETHUSDT"]

    def test_symbol_loading_no_file_no_config(self, mock_agent):
        config = MasterConfig(**get_test_config())
        config.paper.symbols = []
        with patch('os.path.exists', lambda p: 'stats.json' in p), \
             patch('builtins.open', side_effect=FileNotFoundError), \
             patch('paper_trader_q.init_agent', return_value=mock_agent), \
             patch('paper_trader_q.setup_logging'):
            with pytest.raises(SystemExit):
                paper_trader_factory(config=config, agent=mock_agent)

    @patch('paper_trader_q.PaperTrader._get_all_symbols_from_db', return_value=["BTCUSDT", "ETHUSDT"])
    def test_symbol_loading_all_from_db(self, mock_get_symbols, mock_agent):
        config = MasterConfig(**get_test_config())
        config.paper.symbols = ["ALL"]
        with patch('paper_trader_q.init_agent', return_value=mock_agent), \
             patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps({"means": {}, "stds": {}}))), \
             patch('paper_trader_q.setup_logging'):
            trader = paper_trader_factory(config=config, agent=mock_agent)
        assert trader.symbols_to_trade == ["BTCUSDT", "ETHUSDT"]

class TestPositionManagement:
    def test_close_position_tsl_long_profit(self, trader: PaperTrader):
        trader.cfg.backtest.use_risk_management = True
        entry_price = 50000
        break_even_price = entry_price * (1 + trader.cfg.market.transaction_fee) / (1 - trader.cfg.market.transaction_fee)
        
        trader.open_positions["BTCUSDT"] = {
            "direction": "LONG", "entry_price": entry_price, "size": 1000,
            "trailing_max_price": 51000, "entry_time": dt.datetime.now(dt.timezone.utc),
            "close_time": dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=10)
        }
        
        tsl_price = 51000 * (1 - trader.cfg.backtest.trailing_stop)
        close_price = (tsl_price + break_even_price) / 2
        trader.buffers["BTCUSDT"] = deque([{"ts": dt.datetime.now(dt.timezone.utc), "close": close_price}], maxlen=10)
        trader.cfg.paper.source = "websocket"
        
        trader._update_and_close_positions()
        
        assert "BTCUSDT" not in trader.open_positions
        assert trader.trades_log[0]["exit_reason"] == "TSL"

    def test_close_position_tsl_long_loss(self, trader: PaperTrader):
        trader.cfg.backtest.use_risk_management = True
        entry_price = 50000
        
        trader.open_positions["BTCUSDT"] = {
            "direction": "LONG", "entry_price": entry_price, "size": 1000,
            "trailing_max_price": entry_price, "entry_time": dt.datetime.now(dt.timezone.utc),
            "close_time": dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=10)
        }
        
        tsl_price = entry_price * (1 - trader.cfg.backtest.trailing_stop)
        trader.buffers["BTCUSDT"] = deque([{"ts": dt.datetime.now(dt.timezone.utc), "close": tsl_price - 1}], maxlen=10)
        trader._update_and_close_positions()
        assert "BTCUSDT" not in trader.open_positions
        assert trader.trades_log[0]["exit_reason"] == "TSL"

    def test_close_position_time_exit_profit(self, trader: PaperTrader):
        now = dt.datetime.now(dt.timezone.utc)
        entry_price = 50000
        break_even_price = entry_price * (1 + trader.cfg.market.transaction_fee) / (1 - trader.cfg.market.transaction_fee)
        
        trader.open_positions["BTCUSDT"] = {
            "direction": "LONG", "entry_price": entry_price, "size": 1000,
            "entry_time": now - dt.timedelta(minutes=10), "close_time": now - dt.timedelta(seconds=1)
        }
        trader.buffers["BTCUSDT"] = deque([{"ts": now, "close": break_even_price + 1}], maxlen=10)
        trader.cfg.paper.source = "websocket"
        
        trader._update_and_close_positions()
        
        assert "BTCUSDT" not in trader.open_positions
        assert trader.trades_log[0]["exit_reason"] == "TSL Time"

    def test_close_position_tsl_short_profit(self, trader: PaperTrader):
        trader.cfg.backtest.use_risk_management = True
        entry_price = 50000
        
        trader.open_positions["BTCUSDT"] = {
            "direction": "SHORT", "entry_price": entry_price, "size": 1000,
            "trailing_min_price": 49000, "entry_time": dt.datetime.now(dt.timezone.utc),
            "close_time": dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=10)
        }
        
        tsl_price = 49000 * (1 + trader.cfg.backtest.trailing_stop)
        trader.buffers["BTCUSDT"] = deque([{"ts": dt.datetime.now(dt.timezone.utc), "close": tsl_price + 1}], maxlen=10)
        trader.cfg.paper.source = "websocket"
        
        trader._update_and_close_positions()
        
        assert "BTCUSDT" not in trader.open_positions
        assert trader.trades_log[0]["exit_reason"] == "TSL"

    def test_close_position_tsl_short_loss(self, trader: PaperTrader):
        trader.cfg.backtest.use_risk_management = True
        entry_price = 50000
        
        trader.open_positions["BTCUSDT"] = {
            "direction": "SHORT", "entry_price": entry_price, "size": 1000,
            "trailing_min_price": entry_price, "entry_time": dt.datetime.now(dt.timezone.utc),
            "close_time": dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=10)
        }
        
        tsl_price = entry_price * (1 + trader.cfg.backtest.trailing_stop)
        trader.buffers["BTCUSDT"] = deque([{"ts": dt.datetime.now(dt.timezone.utc), "close": tsl_price + 1}], maxlen=10)
        trader._update_and_close_positions()
        assert "BTCUSDT" not in trader.open_positions
        assert trader.trades_log[0]["exit_reason"] == "TSL"

    def test_close_position_no_risk_management(self, trader: PaperTrader):
        trader.cfg.backtest.use_risk_management = False
        now = dt.datetime.now(dt.timezone.utc)
        entry_price = 50000
        
        trader.open_positions["BTCUSDT"] = {
            "direction": "LONG", "entry_price": entry_price, "size": 1000,
            "entry_time": now - dt.timedelta(minutes=10), "close_time": now - dt.timedelta(seconds=1)
        }
        trader.buffers["BTCUSDT"] = deque([{"ts": now, "close": entry_price - 100}], maxlen=10) # Price drops
        trader.cfg.paper.source = "websocket"
        
        trader._update_and_close_positions()
        
        assert "BTCUSDT" not in trader.open_positions
        assert trader.trades_log[0]["exit_reason"] == "Time" # Not TSL

    def test_update_positions_no_open_positions(self, trader: PaperTrader):
        trader.open_positions = {}
        trader._update_and_close_positions()
        assert len(trader.trades_log) == 0

    def test_update_positions_no_buffer_data(self, trader: PaperTrader):
        trader.open_positions["BTCUSDT"] = {
            "direction": "LONG", "entry_price": 50000, "size": 1000,
            "entry_time": dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=10),
            "close_time": dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)
        }
        trader.buffers["BTCUSDT"] = deque(maxlen=10)
        trader._update_and_close_positions()
        assert "BTCUSDT" in trader.open_positions # Position not closed

@pytest.fixture
def trader_for_full_suite(mock_config, mock_agent) -> PaperTrader:
    with (patch('paper_trader_q.init_agent', return_value=mock_agent),
          patch('os.path.exists', return_value=True),
          patch('builtins.open', mock_open(read_data=json.dumps({"means": {}, "stds": {}}))),
          patch('paper_trader_q.setup_logging')):
        return paper_trader_factory(config=mock_config, agent=mock_agent)

@patch('paper_trader_q.get_engine')
def test_get_all_symbols_from_db_failure(mock_get_engine, trader_for_full_suite: PaperTrader, caplog):
    mock_get_engine.side_effect = Exception("DB error")
    trader_for_full_suite.cfg.db.dsn = "some_dsn"
    with caplog.at_level(logging.ERROR):
        result = trader_for_full_suite._get_all_symbols_from_db()
        assert result == []
        assert "Failed to fetch all symbols" in caplog.text

def test_get_engine_caching():
    dsn1 = "sqlite:///test1.db"
    dsn2 = "sqlite:///test2.db"
    engine1_first_call = get_engine(dsn1)
    engine1_second_call = get_engine(dsn1)
    engine2_first_call = get_engine(dsn2)
    assert engine1_first_call is engine1_second_call
    assert engine1_first_call is not engine2_first_call

def test_on_message_valid_kline(trader_for_full_suite: PaperTrader):
    trader_for_full_suite.buffers["BTCUSDT"] = deque(maxlen=100)
    msg = json.dumps({
        "stream": "btcusdt@kline_1m",
        "data": {"k": {"t": 1672531200000, "s": "BTCUSDT", "x": True, "o": 1, "h": 2, "l": 0, "c": 1.5, "v": 100, "q": 150, "n": 10}}
    })
    with patch.object(trader_for_full_suite, '_find_and_process_spikes') as mock_process:
        trader_for_full_suite._on_message(None, msg)
        assert len(trader_for_full_suite.buffers["BTCUSDT"]) == 1
        mock_process.assert_called_once()

def test_on_message_kline_not_closed(trader_for_full_suite: PaperTrader):
    msg = json.dumps({"stream": "btcusdt@kline_1m", "data": {"k": {"x": False}}})
    with patch.object(trader_for_full_suite, '_find_and_process_spikes') as mock_process:
        trader_for_full_suite._on_message(None, msg)
        mock_process.assert_not_called()

def test_on_message_invalid_json(trader_for_full_suite: PaperTrader, caplog):
    with caplog.at_level(logging.ERROR):
        trader_for_full_suite._on_message(None, "not a json")
        assert "Error in _on_message" in caplog.text

def test_websocket_callbacks(trader_for_full_suite: PaperTrader, caplog):
    with caplog.at_level(logging.INFO):
        trader_for_full_suite._on_open(None)
        assert "WebSocket connection opened" in caplog.text
    with caplog.at_level(logging.WARNING):
        trader_for_full_suite._on_close(None, 1006, "Closed abnormally")
        assert "WebSocket connection closed" in caplog.text
    with caplog.at_level(logging.ERROR):
        trader_for_full_suite._on_error(None, "Connection error")
        assert "WebSocket error" in caplog.text

@patch('paper_trader_q.find_spike_windows', return_value=[(None, dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=1), None, None)])
def test_find_and_process_spikes(mock_find_spikes, trader_for_full_suite: PaperTrader, sample_kline_df):
    with patch.object(trader_for_full_suite, '_process_signal') as mock_process_signal:
        trader_for_full_suite._find_and_process_spikes("BTCUSDT", sample_kline_df, dt.datetime.now(dt.timezone.utc))
        mock_process_signal.assert_called_once()

@patch('paper_trader_q.find_spike_windows', return_value=[])
def test_find_and_process_spikes_no_spikes(mock_find_spikes, trader_for_full_suite: PaperTrader, sample_kline_df):
    with patch.object(trader_for_full_suite, '_process_signal') as mock_process_signal:
        trader_for_full_suite._find_and_process_spikes("BTCUSDT", sample_kline_df, dt.datetime.now(dt.timezone.utc))
        mock_process_signal.assert_not_called()

def test_process_signal_cooldown(trader_for_full_suite: PaperTrader, sample_kline_df):
    now = dt.datetime.now(dt.timezone.utc)
    trader_for_full_suite.cooldowns["BTCUSDT"] = now + dt.timedelta(minutes=10)
    with patch.object(trader_for_full_suite, '_get_agent_action') as mock_get_action:
        trader_for_full_suite._process_signal("BTCUSDT", now, sample_kline_df, now)
        mock_get_action.assert_not_called()

def test_process_signal_insufficient_data(trader_for_full_suite: PaperTrader, sample_kline_df, caplog):
    signal_dt = sample_kline_df.index[5]
    with caplog.at_level(logging.WARNING):
        trader_for_full_suite._process_signal("BTCUSDT", signal_dt, sample_kline_df, signal_dt)
        assert "Could not form full sequence" in caplog.text

def test_process_signal_position_already_open(trader_for_full_suite: PaperTrader, sample_kline_df, caplog):
    now = dt.datetime.now(dt.timezone.utc)
    trader_for_full_suite.open_positions["BTCUSDT"] = {"direction": "LONG"}
    with caplog.at_level(logging.DEBUG), \
         patch.object(trader_for_full_suite, '_get_agent_action') as mock_get_action:
        trader_for_full_suite._process_signal("BTCUSDT", now, sample_kline_df, now)
        assert "Signal for BTCUSDT ignored, position already open" in caplog.text
        mock_get_action.assert_not_called()


@patch('paper_trader_q.TradingEnvironment')
def test_get_agent_action_advantage_filter(mock_env, trader_for_full_suite: PaperTrader, mock_agent):
    mock_env.return_value.reset.return_value = (np.zeros(1), {})
    trader_for_full_suite.cfg.backtest.selection_strategy = "advantage_based_filter"
    trader_for_full_suite.cfg.backtest.long_action_threshold = 0.2
    mock_agent.select_action.return_value = np.array([0.5, 0.8, 0.4])
    action = trader_for_full_suite._get_agent_action(np.random.rand(trader_for_full_suite.cfg.seq.full_seq_len, trader_for_full_suite.cfg.seq.num_features))
    assert action == 1

@patch('paper_trader_q.TradingEnvironment')
def test_get_agent_action_ensemble_filter(mock_env, trader_for_full_suite: PaperTrader, mock_agent):
    mock_env.return_value.reset.return_value = (np.zeros(1), {})
    trader_for_full_suite.cfg.backtest.selection_strategy = "ensemble_q_filter"
    trader_for_full_suite.cfg.backtest.long_action_threshold = 0.2
    trader_for_full_suite.cfg.backtest.ensemble_max_sigma = 0.05
    q_mean = np.array([0.5, 0.8, 0.4])
    q_std = np.array([0.01, 0.02, 0.03])
    mock_agent.predict_ensemble.return_value = (q_mean, q_std)
    action = trader_for_full_suite._get_agent_action(np.random.rand(1, 1))
    assert action == 1

@patch('paper_trader_q.TradingEnvironment')
def test_get_agent_action_ensemble_filter_high_uncertainty(mock_env, trader_for_full_suite: PaperTrader, mock_agent, caplog):
    mock_env.return_value.reset.return_value = (np.zeros(1), {})
    trader_for_full_suite.cfg.backtest.selection_strategy = "ensemble_q_filter"
    trader_for_full_suite.cfg.backtest.long_action_threshold = 0.2
    trader_for_full_suite.cfg.backtest.ensemble_max_sigma = 0.05
    q_mean = np.array([0.5, 0.8, 0.4])
    q_std = np.array([0.01, 0.08, 0.03])
    mock_agent.predict_ensemble.return_value = (q_mean, q_std)
    with caplog.at_level(logging.DEBUG):
        action = trader_for_full_suite._get_agent_action(np.random.rand(1, 1))
        assert "Action 1 rejected" in caplog.text
        assert "Uncertainty OK: False" in caplog.text
    assert action == 0

@patch('paper_trader_q.TradingEnvironment')
def test_get_agent_action_no_valid_action(mock_env, trader_for_full_suite: PaperTrader, mock_agent):
    mock_env.return_value.reset.return_value = (np.zeros(1), {})
    trader_for_full_suite.cfg.backtest.selection_strategy = "advantage_based_filter"
    trader_for_full_suite.cfg.backtest.long_action_threshold = 0.9
    mock_agent.select_action.return_value = np.array([0.5, 0.8, 0.4])
    action = trader_for_full_suite._get_agent_action(np.random.rand(1, 1))
    assert action == 0 # HOLD

def test_execute_trade_max_parallel(trader_for_full_suite: PaperTrader, caplog):
    trader_for_full_suite.cfg.backtest.max_parallel_sessions = 1
    trader_for_full_suite.open_positions["ETHUSDT"] = {}
    with caplog.at_level(logging.WARNING):
        trader_for_full_suite._execute_trade("BTCUSDT", 1, dt.datetime.now(dt.timezone.utc), 50000, dt.datetime.now(dt.timezone.utc), 0)
        assert "Max parallel sessions reached" in caplog.text
    assert "BTCUSDT" not in trader_for_full_suite.open_positions

def test_execute_trade_hold_action(trader_for_full_suite: PaperTrader):
    # Action 0 is HOLD
    trader_for_full_suite._execute_trade("BTCUSDT", 0, dt.datetime.now(dt.timezone.utc), 50000, dt.datetime.now(dt.timezone.utc), 0)
    assert "BTCUSDT" not in trader_for_full_suite.open_positions
    assert len(trader_for_full_suite.trades_log) == 0

def test_execute_trade_short_action(trader_for_full_suite: PaperTrader):
    # Action 2 is SHORT
    trader_for_full_suite._execute_trade("BTCUSDT", 2, dt.datetime.now(dt.timezone.utc), 50000, dt.datetime.now(dt.timezone.utc), 0)
    assert trader_for_full_suite.open_positions["BTCUSDT"]["direction"] == "SHORT"

def test_execute_trade_with_order_size_usdt(trader_for_full_suite: PaperTrader):
    trader_for_full_suite.cfg.backtest.order_size_usdt = 500
    trader_for_full_suite._execute_trade("BTCUSDT", 1, dt.datetime.now(dt.timezone.utc), 50000, dt.datetime.now(dt.timezone.utc), 0)
    assert trader_for_full_suite.open_positions["BTCUSDT"]["size"] == 500

def test_execute_trade_with_order_size_usdt_clipped(trader_for_full_suite: PaperTrader, caplog):
    trader_for_full_suite.cfg.backtest.order_size_usdt = 12000
    trader_for_full_suite.balance = 10000
    with caplog.at_level(logging.WARNING):
        trader_for_full_suite._execute_trade("BTCUSDT", 1, dt.datetime.now(dt.timezone.utc), 50000, dt.datetime.now(dt.timezone.utc), 0)
        assert "exceeds balance" in caplog.text
    assert trader_for_full_suite.open_positions["BTCUSDT"]["size"] == 10000

def test_close_position_liquidation_long(trader_for_full_suite: PaperTrader):
    trader_for_full_suite.cfg.paper.leverage = 10
    entry_price = 50000
    liquidation_price = entry_price * (1 - 1 / trader_for_full_suite.cfg.paper.leverage)
    trader_for_full_suite.open_positions["BTCUSDT"] = {
        "direction": "LONG", "entry_price": entry_price, "size": 1000,
        "entry_time": dt.datetime.now(dt.timezone.utc),
        "close_time": dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=10)
    }
    trader_for_full_suite.buffers["BTCUSDT"] = deque([{"ts": dt.datetime.now(dt.timezone.utc), "close": liquidation_price - 1}], maxlen=10)
    trader_for_full_suite.cfg.paper.source = "websocket"
    initial_balance = trader_for_full_suite.balance
    trader_for_full_suite._update_and_close_positions()
    assert "BTCUSDT" not in trader_for_full_suite.open_positions
    trade = trader_for_full_suite.trades_log[0]
    assert trade["exit_reason"] == "LIQUIDATION"
    fees = 1000 * trader_for_full_suite.cfg.paper.leverage * trader_for_full_suite.cfg.market.transaction_fee * 2
    assert trade["pnl"] == -1000 - fees
    assert trader_for_full_suite.balance == initial_balance - 1000 - fees

def test_close_position_liquidation_short(trader_for_full_suite: PaperTrader):
    trader_for_full_suite.cfg.paper.leverage = 10
    entry_price = 50000
    liquidation_price = entry_price * (1 + 1 / trader_for_full_suite.cfg.paper.leverage)
    trader_for_full_suite.open_positions["BTCUSDT"] = {
        "direction": "SHORT", "entry_price": entry_price, "size": 1000,
        "entry_time": dt.datetime.now(dt.timezone.utc),
        "close_time": dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=10)
    }
    trader_for_full_suite.buffers["BTCUSDT"] = deque([{"ts": dt.datetime.now(dt.timezone.utc), "close": liquidation_price + 1}], maxlen=10)
    trader_for_full_suite.cfg.paper.source = "websocket"
    initial_balance = trader_for_full_suite.balance
    trader_for_full_suite._update_and_close_positions()
    assert "BTCUSDT" not in trader_for_full_suite.open_positions
    trade = trader_for_full_suite.trades_log[0]
    assert trade["exit_reason"] == "LIQUIDATION"
    fees = 1000 * trader_for_full_suite.cfg.paper.leverage * trader_for_full_suite.cfg.market.transaction_fee * 2
    assert trader_for_full_suite.balance < initial_balance # PNL is negative

def test_run_websocket_mode(trader_for_full_suite: PaperTrader):
    with patch.object(trader_for_full_suite, '_run_from_websocket') as mock_run_ws:
        trader_for_full_suite.cfg.paper.source = "websocket"
        trader_for_full_suite.run()
        mock_run_ws.assert_called_once()

def test_run_database_mode(trader_for_full_suite: PaperTrader):
    with patch.object(trader_for_full_suite, '_run_from_database') as mock_run_db:
        trader_for_full_suite.cfg.paper.source = "database"
        trader_for_full_suite.run()
        mock_run_db.assert_called_once()

def test_run_unknown_source(trader_for_full_suite: PaperTrader, caplog):
    trader_for_full_suite.cfg.paper.source = "invalid_source"
    with caplog.at_level(logging.ERROR):
        trader_for_full_suite.run()
        assert "Unknown paper trader source" in caplog.text

@patch('pandas.DataFrame.to_csv')
@patch('os.makedirs')
def test_shutdown(mock_makedirs, mock_to_csv, trader_for_full_suite: PaperTrader):
    trader_for_full_suite.trades_log = [{"pnl": 100}]
    trader_for_full_suite.equity_curve = [{"balance": 10100}]
    trader_for_full_suite.shutdown()
    output_dir = os.path.join(trader_for_full_suite.cfg.paths.output_dir, "paper_trader")
    mock_makedirs.assert_called_with(output_dir, exist_ok=True)
    assert mock_to_csv.call_count == 2

@patch('pandas.DataFrame.to_csv')
@patch('os.makedirs')
def test_shutdown_no_logs(mock_makedirs, mock_to_csv, trader_for_full_suite: PaperTrader):
    trader_for_full_suite.trades_log = []
    trader_for_full_suite.equity_curve = []
    trader_for_full_suite.shutdown()
    mock_to_csv.assert_not_called()

@patch('paper_trader_q.websocket.WebSocketApp')
@patch('threading.Thread')
@patch('time.sleep', side_effect=InterruptedError)
def test_run_from_websocket_loop(mock_sleep, mock_thread, mock_ws_app, trader_for_full_suite: PaperTrader):
    with pytest.raises(InterruptedError):
        trader_for_full_suite._run_from_websocket()
    mock_ws_app.assert_called_once()
    mock_thread.return_value.start.assert_called_once()
    mock_sleep.assert_called()

@patch('paper_trader_q.pd.read_sql')
@patch('paper_trader_q.get_engine')
def test_run_from_database_no_time_range(mock_get_engine, mock_read_sql, trader_for_full_suite: PaperTrader, caplog):
    trader_for_full_suite.cfg.backtest.time_range = None
    with caplog.at_level(logging.ERROR):
        trader_for_full_suite._run_from_database()
        assert "`cfg.backtest.time_range` is not defined" in caplog.text

@patch('paper_trader_q.pd.read_sql')
@patch('paper_trader_q.get_engine')
def test_run_from_database_with_data(mock_get_engine, mock_read_sql, trader_for_full_suite: PaperTrader, sample_kline_df):
    trader_for_full_suite.symbols_to_trade = ["BTCUSDT"]
    mock_read_sql.return_value = sample_kline_df
    with patch.object(trader_for_full_suite, '_find_and_process_spikes') as mock_process:
        trader_for_full_suite._run_from_database()
        # Check if processing was triggered for the loaded data
        assert mock_process.call_count > 0

@patch('paper_trader_q.pd.read_sql', side_effect=Exception("DB read error"))
@patch('paper_trader_q.get_engine')
def test_run_from_database_db_error(mock_get_engine, mock_read_sql, trader_for_full_suite: PaperTrader, caplog):
    with caplog.at_level(logging.ERROR):
        trader_for_full_suite._run_from_database()
        assert "Failed to fetch data for symbol BTCUSDT" in caplog.text

if __name__ == "__main__":
    pytest.main([__file__])