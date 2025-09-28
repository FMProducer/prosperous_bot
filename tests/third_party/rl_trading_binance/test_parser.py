import json
import sys
import os
sys.path.append(os.path.abspath('third_party/rl-trading-binance'))
from backends.postgres_ws.collector import Collector, _iso

def test_iso_zero():
    assert _iso(0) == "1970-01-01T00:00:00.000000Z"

def test_parse_events_offline():
    cfg = {"ws":{"channels":["kline_1m","aggTrade"],"symbols":["btcusdt"]},
           "db":{"dsn":"postgres://u:p@h:5432/db","batch_rows":999,"flush_ms":999999},
           "runtime":{"log_level":"ERROR"}}
    c = Collector(cfg)
    class DummyWriter:
        def __init__(self):
            self.kbuf=[]
            self.tbuf=[]
            self.batch_rows=999
        async def start(self): pass
        async def stop_and_close(self): pass
        async def flush(self): pass
    c.writer = DummyWriter()
    k = {"stream":"btcusdt@kline_1m","data":{"e":"kline","E":1700000000000,"s":"BTCUSDT",
         "k":{"t":1700000000000,"T":1700000059999,"i":"1m","x":True,"o":"1","h":"2","l":"0.5","c":"1.5","v":"10","n":42}}}
    c._on_msg(json.dumps(k))
    a = {"stream":"btcusdt@aggTrade","data":{"e":"aggTrade","E":1700000000500,"s":"BTCUSDT",
         "a":123,"p":"30000.1","q":"0.02","f":1000,"l":1001,"T":1700000000400,"m":True}}
    c._on_msg(json.dumps(a))
    assert len(c.writer.kbuf) == 1
    assert len(c.writer.tbuf) == 1