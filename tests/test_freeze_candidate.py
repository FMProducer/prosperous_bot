# -*- coding: utf-8 -*-
"""
Lightweight tests for tools.freeze_candidate parsing utilities.
No external files; uses synthetic log snippets.
"""
from pathlib import Path
import json

from third_party.rl_trading_binance_source_code.tools.freeze_candidate import (
    parse_seed_from_log,
    extract_final_metrics_block,
    parse_metrics_kv
)


def test_parse_seed_from_log():
    txt = "2025-10-18 23:23:00,001 [INFO] Random seed set to 25\n"
    assert parse_seed_from_log(txt) == "25"


def test_extract_final_metrics_block_and_kv_equal():
    snippet = """
some line
[Final Metrics]:
2025-10-18 23:23:57,537 [INFO] :    final_balance_change = 210.79%
2025-10-18 23:23:57,537 [INFO] :                  sharpe = 2.87
2025-10-18 23:23:57,537 [INFO] :            max_drawdown = -17.07%
2025-10-18 23:23:57,537 [INFO] :                accuracy = 69.3%

after block
"""
    block = extract_final_metrics_block(snippet)
    kv = parse_metrics_kv(block)
    assert kv["final_balance_change"] == "210.79%"
    assert kv["sharpe"] == "2.87"
    assert kv["max_drawdown"] == "-17.07%"
    assert kv["accuracy"] == "69.3%"


def test_parse_metrics_kv_colon_style():
    snippet = """
[Final Metrics]:
2025-10-18 22:37:28,864 [INFO] :          trades_per_day: 1.22
2025-10-18 22:37:28,864 [INFO] :        avg_trade_amount: 5810.88

"""
    block = extract_final_metrics_block(snippet)
    kv = parse_metrics_kv(block)
    assert kv["trades_per_day"] == "1.22"
    assert kv["avg_trade_amount"] == "5810.88"
