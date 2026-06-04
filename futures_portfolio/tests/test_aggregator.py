import pytest
import json
import os
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, mock_open
from aggregator import StatusAggregator

@pytest.fixture
def aggregator():
    with patch("aggregator.os.makedirs"):
        return StatusAggregator(config_path="test_config.json")

def test_load_config_valid(aggregator):
    config_data = {"key": "value"}
    with patch("builtins.open", mock_open(read_data=json.dumps(config_data))):
        res = aggregator.load_config()
        assert res == config_data

def test_load_config_missing(aggregator):
    with patch("builtins.open", side_effect=FileNotFoundError):
        res = aggregator.load_config()
        assert res == {}

@pytest.mark.asyncio
async def test_process_telegram_queue(aggregator):
    # Mocking files in queue_dir
    msg_data = {"type": "text", "text": "hello"}
    aggregator.notifier = AsyncMock()
    aggregator.notifier.send_message.return_value = True
    
    with patch("aggregator.glob.glob", return_value=["signals/telegram_queue/1.json"]), \
         patch("builtins.open", mock_open(read_data=json.dumps(msg_data))), \
         patch("aggregator.os.path.exists", return_value=True), \
         patch("aggregator.os.remove") as mock_remove:
        
        # We need to break the infinite loop
        with patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError]):
            try:
                await aggregator.process_telegram_queue()
            except asyncio.CancelledError:
                pass
        
        aggregator.notifier.send_message.assert_called_with("hello", force_direct=True)
        mock_remove.assert_called_with("signals/telegram_queue/1.json")

@pytest.mark.asyncio
async def test_generate_swarm_section(aggregator):
    files = ["paper_state_BTCUSDT.json"]
    live_swarm = []
    active_tickers = ["BTCUSDT"]
    
    state_data = {
        "base_ticker": "BTCUSDT",
        "last_profit": 10.0,
        "rebalance_cycles": 50,
        "siphoning_reserve": 5.0
    }
    
    with patch("aggregator.safe_load_json", AsyncMock(return_value=state_data)), \
         patch.object(aggregator, "load_config", return_value={"min_cycles_for_rank": 20}):
        
        text, profit, safe = await aggregator._generate_swarm_section(files, live_swarm, active_tickers, "INCUBATOR")
        
        assert profit == 10.0
        assert safe == 5.0
        assert "BTCUSDT" in text
        assert "+10.00" in text

@pytest.mark.asyncio
async def test_collect_and_send(aggregator):
    config = {
        "telegram_enabled": True,
        "api_key": "key",
        "secret_key": "secret",
        "testnet": True,
        "portfolios": [{"initial_capital": 100.0}],
        "live_swarm": ["BTCUSDT"],
        "tickers": ["BTCUSDT"]
    }
    
    aggregator.notifier = AsyncMock()
    
    with patch.object(aggregator, "load_config", return_value=config), \
         patch("aggregator.BinanceConnector") as mock_conn_cls, \
         patch("aggregator.glob.glob", side_effect=[["paper_state_BTCUSDT.json"], ["real_state_BTCUSDT.json"]]), \
         patch.object(aggregator, "_generate_swarm_section", AsyncMock(side_effect=[
             ("COMBAT TEXT", 20.0, 0.0),
             ("INCUBATOR TEXT", 10.0, 0.0)
         ])):
        
        mock_conn = mock_conn_cls.return_value
        mock_conn.verify_connection = AsyncMock()
        mock_conn.get_free_balance = AsyncMock(return_value=1000.0)
        mock_conn.get_bnb_balance = AsyncMock(return_value=1.0)
        
        await aggregator.collect_and_send()
        
        aggregator.notifier.send_message.assert_called()
        args = aggregator.notifier.send_message.call_args[0][0]
        assert "COMBAT TEXT" in args
        assert "INCUBATOR TEXT" in args
        assert "ROI: <code>20.00%</code>" in args

@pytest.mark.asyncio
async def test_aggregator_run(aggregator):
    with patch.object(aggregator, "load_config", return_value={"telegram_summary_interval_min": 0.001}), \
         patch.object(aggregator, "collect_and_send", AsyncMock()) as mock_collect, \
         patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError]):
        try:
            await aggregator.run()
        except asyncio.CancelledError:
            pass
    mock_collect.assert_called()

@pytest.mark.asyncio
async def test_process_telegram_queue_photo(aggregator):
    msg_data = {"type": "photo", "path": "test.jpg", "caption": "test photo"}
    aggregator.notifier = AsyncMock()
    aggregator.notifier.send_photo.return_value = True
    
    with patch("aggregator.glob.glob", return_value=["signals/telegram_queue/1.json"]), \
         patch("builtins.open", mock_open(read_data=json.dumps(msg_data))), \
         patch("aggregator.os.path.exists", return_value=True), \
         patch("aggregator.os.remove") as mock_remove:
        
        with patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError]):
            try:
                await aggregator.process_telegram_queue()
            except asyncio.CancelledError:
                pass
        
        aggregator.notifier.send_photo.assert_called_with("test.jpg", "test photo", force_direct=True)
        mock_remove.assert_called_with("signals/telegram_queue/1.json")
