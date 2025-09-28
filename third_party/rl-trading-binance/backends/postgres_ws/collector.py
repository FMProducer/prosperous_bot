import asyncio, json, logging, os
from datetime import datetime, timezone
from typing import Any, Dict, List
import asyncpg, yaml
from websocket import create_connection

def _iso(ms:int)->str:
    return datetime.fromtimestamp(ms/1000, tz=timezone.utc).isoformat(timespec='microseconds').replace("+00:00","Z")

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
                await con.executemany("""INSERT INTO rlref.klines_1m(symbol,open_time,close_time,open,high,low,close,volume,trades,event_time,is_closed)
                    VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                    ON CONFLICT(symbol,open_time) DO NOTHING""", rows)
            if self.tbuf:
                rows=self.tbuf[:] ; self.tbuf.clear()
                await con.executemany("""INSERT INTO rlref.agg_trades(symbol,agg_id,price,quantity,first_id,last_id,trade_time,is_maker,event_time)
                    VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9)
                    ON CONFLICT(symbol,agg_id) DO NOTHING""", rows)
    async def stop_and_close(self):
        self.stop.set()
        if self.pool: await self.pool.close()

class Collector:
    def __init__(self, cfg:Dict[str,Any]):
        self.cfg=cfg; self.ws_cfg=cfg["ws"]; self.db_cfg=cfg["db"]
        self.writer=PgWriter(self.db_cfg["dsn"], int(self.db_cfg.get("batch_rows",500)), int(self.db_cfg.get("flush_ms",500)))
        self.ws=None
    def _on_msg(self, message:str):
        try: obj=json.loads(message)
        except Exception: logging.exception("bad json"); return
        stream = obj.get("stream")
        data=obj.get("data", obj)
        if not stream:
            logging.warning(f"Message without stream: {obj}")
            return
            
        ev=data.get("e")
        if ev=="kline":
            k=data["k"]; s=data["s"].lower()
            row=(s,_iso(k["t"]),_iso(k["T"]),k["o"],k["h"],k["l"],k["c"],k["v"],k.get("n",0),_iso(data.get("E",k["T"])),bool(k["x"]))
            self.writer.kbuf.append(row)
        elif ev=="aggTrade":
            s=data["s"].lower()
            row=(s,int(data["a"]),data["p"],data["q"],int(data["f"]),int(data["l"]),_iso(int(data["T"])),bool(data["m"]),_iso(int(data.get("E",data["T"]))))
            self.writer.tbuf.append(row)
        if len(self.writer.kbuf)>=self.writer.batch_rows or len(self.writer.tbuf)>=self.writer.batch_rows:
            asyncio.run_coroutine_threadsafe(self.writer.flush(), asyncio.get_event_loop())

    async def run(self):
        await self.writer.start()
        logging.basicConfig(level=getattr(logging, self.cfg.get("runtime",{}).get("log_level","INFO")))
        
        streams = []
        syms=[s.lower() for s in self.ws_cfg["symbols"]]
        if "kline_1m" in self.ws_cfg["channels"]:
            for s in syms: streams.append(f"{s}@kline_1m")
        if "aggTrade" in self.ws_cfg["channels"]:
            for s in syms: streams.append(f"{s}@aggTrade")
        
        url = f"wss://fstream.binance.com/stream?streams={'/'.join(streams)}"
        self.ws = create_connection(url)
        
        try:
            while True:
                msg = self.ws.recv()
                self._on_msg(msg)
        finally:
            await self.writer.stop_and_close()
            if self.ws: self.ws.close()

def load_config(path:str)->Dict[str,Any]:
    with open(path,"r",encoding="utf-8") as f: return yaml.safe_load(f)

if __name__=="__main__":
    cfg_path=os.path.join(os.path.dirname(__file__),"ref_config.yml")
    cfg=load_config(cfg_path)
    asyncio.run(Collector(cfg).run())