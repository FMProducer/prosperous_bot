import asyncio, json, logging, os, random, ssl, time
from datetime import datetime, timezone
from typing import Any, Dict, List
import asyncpg, yaml
import websocket # websocket-client
import argparse

STOP = asyncio.Event()

def run_ws(url: str, on_msg, on_open=None, on_close=None, headers=None):
    backoff = 1
    while not STOP.is_set():
        ws = websocket.WebSocketApp(
            url,
            header=headers or [],
            on_message=on_msg,
            on_open=on_open,
            on_close=on_close,
        )
        try:
            # ping_interval поддерживает keep-alive; run_forever сам не «умный» — цикл выше
            ws.run_forever(
                sslopt={"cert_reqs": ssl.CERT_NONE},
                ping_interval=60,
                ping_timeout=10,
                http_proxy_host=None,
                http_proxy_port=None,
            )
        except Exception as e:
            logging.exception(f"WebSocket error: {e}")
        # экспоненциальный backoff с джиттером и потолком
        time.sleep(backoff + random.random())
        backoff = min(backoff * 2, 60)

class PgWriter:
    def __init__(self, dsn:str, batch_rows:int, flush_ms:int):
        self.dsn, self.batch_rows, self.flush_ms = dsn, batch_rows, flush_ms
        self.pool=None; self.kbuf:List[tuple]=[]; self.tbuf:List[tuple]=[]; self.stop=asyncio.Event()
    async def start(self): self.pool=await asyncpg.create_pool(dsn=self.dsn, min_size=1, max_size=4); asyncio.create_task(self._flusher())
    async def _flusher(self):
        while not self.stop.is_set():
            await asyncio.sleep(self.flush_ms/1000)
            await self.flush()
    async def flush(self):
        if not self.pool: return
        async with self.pool.acquire() as con:
            if self.kbuf:
                rows=self.kbuf[:] ; self.kbuf.clear()
                await con.executemany("""INSERT INTO klines_1m(symbol,open_time_ms,open_price,high_price,low_price,close_price,base_volume,quote_volume,trade_count,taker_base,taker_quote,is_closed)
                    VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                    ON CONFLICT(symbol,open_time_ms) DO UPDATE SET
                        close_price = EXCLUDED.close_price,
                        high_price = GREATEST(klines_1m.high_price, EXCLUDED.high_price),
                        low_price = LEAST(klines_1m.low_price, EXCLUDED.low_price),
                        base_volume = EXCLUDED.base_volume,
                        quote_volume = EXCLUDED.quote_volume,
                        trade_count = EXCLUDED.trade_count,
                        taker_base = EXCLUDED.taker_base,
                        taker_quote = EXCLUDED.taker_quote,
                        is_closed = EXCLUDED.is_closed,
                        ingest_ts = now()
                    """, rows)
            if self.tbuf:
                rows=self.tbuf[:] ; self.tbuf.clear()
                await con.executemany("""INSERT INTO agg_trades(symbol,agg_id,price,quantity,first_id,last_id,trade_time_ms,is_maker)
                    VALUES($1,$2,$3,$4,$5,$6,$7,$8)
                    ON CONFLICT(symbol,agg_id) DO NOTHING""", rows)
    async def stop_and_close(self):
        self.stop.set()
        if self.pool: await self.pool.close()

class Collector:
    def __init__(self, cfg:Dict[str,Any]):
        self.cfg=cfg
        self.binance_cfg = cfg['binance']
        self.universe_cfg = cfg['universe']
        self.storage_cfg = cfg['storage']
        self.writer=PgWriter(self.storage_cfg["dsn"], int(self.storage_cfg.get("batch_size",1000)), int(self.storage_cfg.get("write_timeout_ms",5000)))

    def _on_msg(self, ws, message:str):
        try: obj=json.loads(message)
        except Exception: logging.exception("bad json"); return
        
        stream = obj.get("stream")
        if not stream:
            logging.warning(f"Message without stream: {obj}")
            return
        
        data=obj.get("data", {})
        ev=data.get("e")

        if ev=="kline":
            k=data["k"]
            s=data["s"].lower()
            row=(s,int(k["t"]),k["o"],k["h"],k["l"],k["c"],k["v"],k["q"],int(k["n"]),k["V"],k["Q"],bool(k["x"]))
            self.writer.kbuf.append(row)
        elif ev=="aggTrade":
            s=data["s"].lower()
            row=(s,int(data["a"]),data["p"],data["q"],int(data["f"]),int(data["l"]),int(data["T"]),bool(data["m"]))
            self.writer.tbuf.append(row)
        
        if len(self.writer.kbuf)>=self.writer.batch_rows or len(self.writer.tbuf)>=self.writer.batch_rows:
            asyncio.run_coroutine_threadsafe(self.writer.flush(), asyncio.get_running_loop())

    def on_open(self, ws):
        logging.info("WebSocket connection opened.")

    def on_close(self, ws, close_status_code, close_msg):
        logging.warning(f"WebSocket connection closed: {close_status_code} {close_msg}")

    def run_shard(self, streams):
        if not streams:
            return
        url = f"{self.binance_cfg['base_url']}/stream?streams={'/'.join(streams)}"
        run_ws(url, on_msg=self._on_msg, on_open=self.on_open, on_close=self.on_close)

    def run_shards(self, kline_streams, agg_trade_streams):
        kline_shard_size = self.binance_cfg['shards']['kline_streams_per_conn']
        agg_trade_shard_size = self.binance_cfg['shards']['agg_trade_streams_per_conn']

        for i in range(0, len(kline_streams), kline_shard_size):
            self.run_shard(kline_streams[i:i + kline_shard_size])

        for i in range(0, len(agg_trade_streams), agg_trade_shard_size):
            self.run_shard(agg_trade_streams[i:i + agg_trade_shard_size])

    async def run(self):
        await self.writer.start()
        logging.basicConfig(level=getattr(logging, self.cfg.get("housekeeping",{}).get("log_level","INFO")))
        
        # Shard streams
        kline_streams = []
        agg_trade_streams = []
        
        symbols = [s.lower() for s in self.universe_cfg["symbols"]]
        
        if self.binance_cfg["channels"]["klines_1m"]:
            for s in symbols: kline_streams.append(f"{s}@kline_1m")
        
        if self.binance_cfg["channels"]["agg_trade"]:
            for s in symbols: agg_trade_streams.append(f"{s}@aggTrade")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.run_shards, kline_streams, agg_trade_streams)


def load_config(path:str)->Dict[str,Any]:
    with open(path,"r",encoding="utf-8") as f: return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Binance WebSocket data collector.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    collector = Collector(cfg)
    
    try:
        asyncio.run(collector.run())
    except KeyboardInterrupt:
        STOP.set()
        if collector.writer:
            asyncio.run(collector.writer.stop_and_close())

if __name__=="__main__":
    main()