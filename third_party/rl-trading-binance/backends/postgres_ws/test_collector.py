import json
import sys
import os
import asyncio
import time
import threading
from unittest.mock import MagicMock, patch, mock_open
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from collector import Collector, PgWriter, run_ws, STOP, load_config, main

@pytest.mark.asyncio
async def test_parse_events_offline():
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

@pytest.mark.asyncio
async def test_pg_writer():
    # Mock the connection pool
    pool_mock = MagicMock()
    close_future = asyncio.Future()
    close_future.set_result(None)
    pool_mock.close.return_value = close_future
    
    # Mock the connection
    con_mock = MagicMock()
    future = asyncio.Future()
    future.set_result(None)
    con_mock.executemany.return_value = future
    pool_mock.acquire.return_value.__aenter__.return_value = con_mock

    with patch('asyncpg.create_pool', return_value=asyncio.Future()) as create_pool_mock:
        create_pool_mock.return_value.set_result(pool_mock)

        writer = PgWriter("postgresql://user:pass@host:5432/db", 10, 5000)
        await writer.start()

        # Add some data to the buffers
        writer.kbuf.append(("btcusdt", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, True))
        writer.tbuf.append(("btcusdt", 1, 2, 3, 4, 5, 6, True))

        # Flush the buffers
        await writer.flush()

        # Assert that the executemany method was called with the correct data
        assert con_mock.executemany.call_count == 2
        
        # Stop the writer
        await writer.stop_and_close()
        assert writer.stop.is_set()

@patch('collector.Collector.run_shards')
@pytest.mark.asyncio
async def test_collector_run(mock_run_shards):
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
            "symbols": ["BTCUSDT", "ETHUSDT"]
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

    collector = Collector(cfg)
    collector.writer = MagicMock()
    
    # Mock the start method of the writer
    start_future = asyncio.Future()
    start_future.set_result(None)
    collector.writer.start.return_value = start_future

    await collector.run()

    # Assert that run_shards was called with the correct arguments
    mock_run_shards.assert_called_once()
    
    # Assert that the writer's start method was called
    collector.writer.start.assert_called_once()

@patch('builtins.open', new_callable=mock_open, read_data='key: value')
def test_load_config(mock_file):
    cfg = load_config("dummy_path")
    assert cfg == {'key': 'value'}

@patch('websocket.WebSocketApp')
def test_run_ws(mock_ws_app):
    # Mock the WebSocketApp instance
    ws_instance_mock = MagicMock()
    mock_ws_app.return_value = ws_instance_mock

    # Mock the on_message callback
    on_msg_mock = MagicMock()

    # Set STOP event after 2 iterations
    def side_effect(*args, **kwargs):
        if ws_instance_mock.run_forever.call_count == 2:
            STOP.set()
        return None
    ws_instance_mock.run_forever.side_effect = side_effect

    # Run the run_ws function
    run_ws("ws://test.com", on_msg_mock)

    # Assert that WebSocketApp was called with the correct arguments
    mock_ws_app.assert_called_with(
        "ws://test.com",
        header=[],
        on_message=on_msg_mock,
        on_open=None,
        on_close=None,
    )

    # Assert that run_forever was called
    assert ws_instance_mock.run_forever.call_count > 0

@patch('collector.Collector.run_shard')
def test_collector_run_sharding(mock_run_shard):
    cfg = {
        "binance": {
            "base_url": "wss://fstream.binance.com",
            "channels": {
                "klines_1m": True,
                "agg_trade": True
            },
            "shards": {
                "kline_streams_per_conn": 1,
                "agg_trade_streams_per_conn": 1
            }
        },
        "universe": {
            "symbols": ["BTCUSDT", "ETHUSDT"]
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

    collector = Collector(cfg)
    collector.writer = MagicMock()
    
    kline_streams = [f"{s.lower()}@kline_1m" for s in cfg['universe']['symbols']]
    agg_trade_streams = [f"{s.lower()}@aggTrade" for s in cfg['universe']['symbols']]
    
    collector.run_shards(kline_streams, agg_trade_streams)
    assert mock_run_shard.call_count == 4 # 2 for kline, 2 for agg_trade

@pytest.mark.asyncio
async def test_pg_writer_flusher():
    with patch('asyncpg.create_pool', return_value=asyncio.Future()) as create_pool_mock:
        # Mock the connection pool
        pool_mock = MagicMock()
        close_future = asyncio.Future()
        close_future.set_result(None)
        pool_mock.close.return_value = close_future
        create_pool_mock.return_value.set_result(pool_mock)

        # Mock the connection
        con_mock = MagicMock()
        future = asyncio.Future()
        future.set_result(None)
        con_mock.executemany.return_value = future
        pool_mock.acquire.return_value.__aenter__.return_value = con_mock

        writer = PgWriter("postgresql://user:pass@host:5432/db", 1, 10)
        await writer.start()
        
        writer.kbuf.append(("btcusdt", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, True))
        
        await asyncio.sleep(0.02) # wait for the flusher to run
        
        assert con_mock.executemany.call_count == 1
        
        await writer.stop_and_close()

@patch('time.sleep')
@patch('websocket.WebSocketApp')
def test_run_ws_exception(mock_ws_app, mock_sleep):
    # Mock the WebSocketApp instance
    ws_instance_mock = MagicMock()
    ws_instance_mock.run_forever.side_effect = [Exception("test"), None]
    mock_ws_app.return_value = ws_instance_mock

    # Mock the on_message callback
    on_msg_mock = MagicMock()

    # Stop the loop after the first iteration
    STOP.clear()
    def stop_loop(*args, **kwargs):
        STOP.set()
    mock_sleep.side_effect = stop_loop

    # Run the run_ws function
    run_ws("ws://test.com", on_msg_mock)

    assert mock_sleep.call_count > 0

def test_collector_on_open_on_close():
    cfg = {
        "binance": {},
        "universe": {},
        "storage": {"dsn": "postgresql://user:pass@host:5432/db"}
    }
    collector = Collector(cfg)
    with patch('logging.info') as mock_info, patch('logging.warning') as mock_warning:
        collector.on_open(None)
        mock_info.assert_called_once_with("WebSocket connection opened.")
        
        collector.on_close(None, 1000, "test")
        mock_warning.assert_called_once_with("WebSocket connection closed: 1000 test")

@patch('collector.load_config')
@patch('collector.Collector')
@patch('asyncio.run')
def test_main(mock_asyncio_run, mock_collector, mock_load_config):
    # Mock the config
    mock_load_config.return_value = {}
    
    # Mock the collector instance
    collector_instance = MagicMock()
    mock_collector.return_value = collector_instance
    
    # Call the main function
    main()
    
    # Assert that the mocks were called
    mock_load_config.assert_called_once()
    mock_collector.assert_called_once()
    mock_asyncio_run.assert_called_once_with(collector_instance.run())
