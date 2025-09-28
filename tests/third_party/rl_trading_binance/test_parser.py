import json
import sys
import os
import asyncio
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath('third_party/rl-trading-binance'))
from backends.postgres_ws.collector import Collector

def test_parse_events_offline():
    cfg = {
        "binance": {
            "base_url": "wss://fstream.binance.com",
            "channels": {
                "klines_1m": True,
                "agg_trade": True
            },
            "shards": {
                "kline_streams_per_conn": 600,
                "agg_trade_streams_per_conn": 600
            }
        },
        "universe": {
            "symbols": ["BTCUSDT"]
        },
        "storage": {
            "dsn": "postgresql://user:pass@host:5432/db",
            "batch_size": 10,
            "write_timeout_ms": 5000
        },
        "housekeeping": {
            "log_level": "INFO"
        }
    }

    # Mock the event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    c = Collector(cfg)
    
    # Mock the writer
    c.writer = MagicMock()
    c.writer.kbuf = []
    c.writer.tbuf = []
    c.writer.batch_rows = 10

    # Mock the websocket message
    kline_msg = {"stream":"btcusdt@kline_1m","data":{"e":"kline","E":1700000000000,"s":"BTCUSDT",
         "k":{"t":1700000000000,"T":1700000059999,"i":"1m","x":True,"o":"1","h":"2","l":"0.5","c":"1.5","v":"10","q":"15","n":42,"V":"5","Q":"7.5"}}}
    agg_trade_msg = {"stream":"btcusdt@aggTrade","data":{"e":"aggTrade","E":1700000000500,"s":"BTCUSDT",
         "a":123,"p":"30000.1","q":"0.02","f":1000,"l":1001,"T":1700000000400,"m":True}}

    # Call the _on_msg method
    c._on_msg(None, json.dumps(kline_msg))
    c._on_msg(None, json.dumps(agg_trade_msg))

    # Assert that the data is correctly parsed and appended to the buffers
    assert len(c.writer.kbuf) == 1
    assert len(c.writer.tbuf) == 1
    
    kline_row = c.writer.kbuf[0]
    assert kline_row[0] == "btcusdt"
    assert kline_row[1] == 1700000000000
    assert kline_row[11] is True

    agg_trade_row = c.writer.tbuf[0]
    assert agg_trade_row[0] == "btcusdt"
    assert agg_trade_row[1] == 123
    assert agg_trade_row[7] is True

    # Clean up the event loop
    loop.close()
