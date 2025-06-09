import pytest
from src.prosperous_bot.utils import to_gate_pair, to_binance_symbol

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
