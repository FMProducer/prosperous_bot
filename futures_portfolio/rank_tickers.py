import asyncio
import aiohttp
import logging
import time
import json
import os
from typing import List, Dict, Optional, Callable, Any
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("Scanner")

CACHE_FILE = "scan_cache.json"
CONFIG_FILE = "config.json"

def retry_on_network_error(retries: int = 3, delay: float = 1.0):
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt < retries - 1:
                        await asyncio.sleep(delay * (attempt + 1))
                    else:
                        logger.error(f"Max retries reached for {func.__name__}: {e}")
            return None
        return wrapper
    return decorator

class TickerScanner:
    def __init__(self, concurrent_requests: int = 15, rebalance_threshold: float = 0.02):
        self.base_url = "https://fapi.binance.com"
        self.rebalance_threshold = rebalance_threshold
        self.semaphore = asyncio.Semaphore(concurrent_requests)
        
    @retry_on_network_error(retries=3)
    async def fetch(self, session: aiohttp.ClientSession, endpoint: str, params: dict = None):
        async with self.semaphore:
            try:
                # Отключаем прокси только для этого конкретного запроса к Binance
                async with session.get(f"{self.base_url}{endpoint}", params=params, timeout=20, proxy=None) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 5))
                        await asyncio.sleep(retry_after)
                        return await self.fetch(session, endpoint, params)
                    if response.status != 200:
                        return None
                    return await response.json()
            except Exception as e:
                return None

    async def analyze_ticker(self, session: aiohttp.ClientSession, symbol: str, funding_rate: float) -> Optional[Dict]:
        now_ms = int(time.time() * 1000)
        params = {"symbol": symbol, "interval": "1m", "limit": 1440, "endTime": now_ms}
        klines = await self.fetch(session, "/fapi/v1/klines", params)

        if not klines or len(klines) < 100:
            return None
        
        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        
        cycles = 0
        basis = closes[0]
        for price in closes:
            diff_pct = abs(price - basis) / basis
            if diff_pct >= self.rebalance_threshold:
                cycles += 1
                basis = price

        max_hourly_spurt = 0.0
        for h in range(0, len(klines), 60):
            window = klines[h:h+60]
            if not window: continue
            w_high = max(float(k[2]) for k in window)
            w_low = min(float(k[3]) for k in window)
            w_open = float(window[0][1])
            spurt = (w_high - w_low) / w_open * 100
            max_hourly_spurt = max(max_hourly_spurt, spurt)

        total_path = sum(abs(closes[i] - closes[i-1]) for i in range(1, len(closes)))
        net_move_abs = abs(closes[-1] - closes[0])
        trend_ratio_pct = (net_move_abs / total_path * 100) if total_path > 0 else 0
        net_change_pct = (closes[-1] / closes[0] - 1) * 100
        
        return {
            "symbol": symbol,
            "cycles": cycles,
            "net_change": net_change_pct,
            "max_spurt": max_hourly_spurt,
            "trend": trend_ratio_pct,
            "funding": funding_rate * 100
        }

    async def get_top_tickers(self, min_volume: float = 10_000_000):
        logger.info(f"Market Scan (Min Vol: {min_volume/1e6:.0f}M, Threshold: {self.rebalance_threshold*100}%)...")
        
        # Load White List
        white_list = set()
        if os.path.exists("tickers.txt"):
            try:
                with open("tickers.txt", "r", encoding="utf-8") as f:
                    white_list = {line.strip() for line in f if line.strip()}
            except Exception as e:
                logger.error(f"Failed to load tickers.txt: {e}")

        # Load Black List from config
        black_list = set()
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    black_list = set(cfg.get("black_list", []))
            except Exception as e:
                logger.error(f"Failed to load black_list from config: {e}")

        # trust_env=True позволяет aiohttp использовать системные прокси (важно для Telegram)
        async with aiohttp.ClientSession(trust_env=True) as session:
            tickers_24h = await self.fetch(session, "/fapi/v1/ticker/24hr")
            premium_info = await self.fetch(session, "/fapi/v1/premiumIndex")
            if not tickers_24h or not premium_info:
                logger.error("Failed to fetch initial market data.")
                return []

            funding_map = {item['symbol']: float(item['lastFundingRate']) for item in premium_info}
            
            candidates = []
            for t in tickers_24h:
                symbol = t['symbol']
                if not symbol.endswith("USDT"): continue
                if float(t['quoteVolume']) < min_volume: continue
                if not all(ord(c) < 128 for c in symbol): continue
                
                # Apply Filtering
                if white_list and symbol not in white_list: continue
                if symbol in black_list: continue
                
                candidates.append(t)

            candidates.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
            candidates = candidates[:80]
            
            tasks = [self.analyze_ticker(session, c['symbol'], funding_map.get(c['symbol'], 0.0)) for c in candidates]
            results = await asyncio.gather(*tasks)
            
            ranked_list = [r for r in results if r is not None and r['cycles'] >= 10]
            ranked_list.sort(key=lambda x: x['cycles'], reverse=True)
            
            try:
                with open(CACHE_FILE, "w") as f:
                    json.dump({"timestamp": time.time(), "results": ranked_list}, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save cache: {e}")
            
            return ranked_list

async def main(quiet=False, min_volume=10_000_000):
    threshold = 0.02
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
                threshold = cfg["portfolios"][0].get("rebalance_threshold", 0.02)
    except: pass

    scanner = TickerScanner(concurrent_requests=20, rebalance_threshold=threshold)
    top_tickers = await scanner.get_top_tickers(min_volume=min_volume)
    
    if not quiet:
        print("\n" + "="*125)
        print(f"{'SYMBOL':<15} | {'CYCLES':<8} | {'NET MOVE%':<12} | {'MAX SPURT%':<12} | {'TREND EFF%':<12} | {'FUNDING%'}")
        print("-" * 125)
        for t in top_tickers[:40]:
            print(f"{t['symbol']:<15} | {t['cycles']:<8} | {t['net_change']:<12.2f} | {t['max_spurt']:<12.2f} | {t['trend']:<12.2f} | {t['funding']:.4f}")
        print("="*125)
    
    return top_tickers

if __name__ == "__main__":
    asyncio.run(main())
