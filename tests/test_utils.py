import pytest
from prosperous_bot.utils import to_gate_pair, to_binance_symbol, _qty_for_tests, ensure_directory, save_to_csv, get_lot_step
import pandas as pd
from pathlib import Path
import os
from unittest.mock import patch, MagicMock

# Tests for to_gate_pair
@pytest.mark.parametrize("test_input, expected_output", [
    ("BTCUSDT", "BTC_USDT"),      # Binance style
    ("ETH_USDT", "ETH_USDT"),    # Gate.io style
    ("solusdt", "SOL_USDT"),      # Lowercase
    ("BnbUsdt", "BNB_USDT"),      # Mixed case
    ("ADABTC", "ADABTC"),        # Non-USDT pair
    ("", ""),                    # Empty string
    (None, None),                # None input
    ("SUSHI_USDT", "SUSHI_USDT"), # Already Gate style with underscore in name
    ("PEPEUSDT", "PEPE_USDT"),     # Binance style
])
def test_to_gate_pair(test_input, expected_output):
    assert to_gate_pair(test_input) == expected_output

# Tests for to_binance_symbol
@pytest.mark.parametrize("test_input, expected_output", [
    ("BTC_USDT", "BTCUSDT"),      # Gate.io style
    ("ETHUSDT", "ETHUSDT"),      # Binance style
    ("sol_usdt", "SOLUSDT"),      # Lowercase
    ("Bnb_Usdt", "BNBUSDT"),      # Mixed case
    ("ADA_BTC", "ADA_BTC"),      # Non-USDT pair (current behavior)
    ("PEPE_USDT", "PEPEUSDT"),   # Gate.io style
    ("", ""),                    # Empty string
    (None, None),                # None input
    ("SUSHI_USDT", "SUSHIUSDT"), # Gate style with underscore in name (becomes SUSHIUSDT)
])
def test_to_binance_symbol(test_input, expected_output):
    assert to_binance_symbol(test_input) == expected_output

# Tests for _qty_for_tests
@pytest.mark.parametrize("asset_key, delta_usdt, p_spot, expected_qty", [
    ("spot", 100, 50000, 100 / 50000),
    ("BTCUSDT", 100, 50000, 100 / 50000 / 0.001),
    ("unknown", 100, 50000, max(100 / 50000, 1)),
])
def test_qty_for_tests(asset_key, delta_usdt, p_spot, expected_qty):
    assert _qty_for_tests(asset_key, delta_usdt, p_spot) == expected_qty

# Test for ensure_directory
def test_ensure_directory(tmp_path):
    d = tmp_path / "sub"
    ensure_directory(d)
    assert d.is_dir()

# Test for save_to_csv
def test_save_to_csv(tmp_path):
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]}, index=pd.to_datetime(['2023-01-01', '2023-01-02']))
    df.index.name = 'my_timestamp'
    file_path = tmp_path / "test.csv"
    save_to_csv(df, file_path)
    assert file_path.is_file()
    df_read = pd.read_csv(file_path)
    assert 'timestamp' in df_read.columns
    assert df_read['a'].tolist() == [1, 2]

# Tests for get_lot_step
@patch('prosperous_bot.exchange_gate.gate_client')
def test_get_lot_step_api_success(mock_gate_client):
    mock_pair = MagicMock()
    mock_pair.min_base_amount = '0.001'
    mock_gate_client.get_spot_pairs.return_value = [mock_pair]
    assert get_lot_step("BTC") == 0.001
    mock_gate_client.get_spot_pairs.assert_called_with(pair="BTC_USDT")

@patch('prosperous_bot.exchange_gate.gate_client', side_effect=Exception("API Error"))
def test_get_lot_step_api_fail_fallback(mock_gate_client):
    assert get_lot_step("BTC") == 0.0001
    assert get_lot_step("UNKNOWN") == 1e-8
