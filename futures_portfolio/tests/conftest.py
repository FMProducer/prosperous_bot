import sys
import os
import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch
from decimal import Decimal

# Добавляем родительскую директорию (futures_portfolio/) в sys.path,
# чтобы тесты могли импортировать main, supervisor и т.д.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

@pytest.fixture
def mock_config():
    return {
        "binance_api_key": "test_key",
        "binance_api_secret": "test_secret",
        "telegram_token": "test_token",
        "telegram_chat_id": "test_chat_id",
        "initial_capital": 1000.0,
        "leverage": 1,
        "rebalance_threshold": 0.01,
        "rebalance_threshold_surplus": 0.01,
        "rebalance_threshold_deficit": 0.01,
        "tickers": ["BTCUSDT", "ETHUSDT"],
        "share_long": 40.0,
        "share_short": 40.0,
        "share_virt": 20.0,
        "min_cycles_for_rank": 10,
        "max_replace_per_cycle": 2,
        "tpv_safety_floor": 10.0,
        "limit_order_enabled": False,
        "limit_offset_pct": 0.01,
        "limit_timeout_sec": 60,
        "ticker_thresholds": {}
    }

@pytest.fixture
def mock_targets():
    return {
        "long": 40.0,
        "short": 40.0,
        "virt": 20.0
    }

@pytest.fixture
def mock_connector():
    connector = MagicMock()
    connector.client = MagicMock()
    connector.verify_connection = AsyncMock(return_value=True)
    connector.get_mark_prices = AsyncMock(return_value={"BTCUSDT": 50000.0, "ETHUSDT": 3000.0})
    connector.get_futures_balance = AsyncMock(return_value=1000.0)
    connector.get_futures_positions = AsyncMock(return_value=[])
    return connector

@pytest.fixture(autouse=True)
def mock_binance_client():
    with patch("binance.client.Client") as mock:
        yield mock
