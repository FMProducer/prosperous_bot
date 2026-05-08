import numpy as np
import pandas as pd
import numpy.typing as npt
from typing import List, Dict, Optional, Any, Callable
import asyncio
import aiohttp
import time
import os
import json
import logging
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

class TickerRanker:
    def __init__(self, multi_df: pd.DataFrame):
        # Ожидается MultiIndex DataFrame (ticker, datetime) строго с колонками OHLCV
        self.df = multi_df[['open', 'high', 'low', 'close', 'volume']]

    def calculate_metrics(self, threshold: float = 0.02) -> pd.DataFrame:
        # Векторизованный расчет по cross-section
        grouped = self.df.groupby(level='ticker')

        # 1. Net Change %
        first_close = grouped['close'].transform('first')
        last_close = grouped['close'].transform('last')
        net_change = (last_close / first_close - 1) * 100

        # 2. Max Spurt % (1h = 60 min)
        # Использование rolling(60) для поиска максимального всплеска внутри часа
        h_roll = grouped['high'].rolling(60).max().reset_index(level=0, drop=True)
        l_roll = grouped['low'].rolling(60).min().reset_index(level=0, drop=True)
        o_roll = grouped['open'].shift(59)
        # Фильтруем случаи где o_roll относится к другому тикеру (из-за shift)
        # Но в MultiIndex с groupby shift работает корректно внутри групп
        max_spurt = ((h_roll - l_roll) / o_roll * 100).groupby(level='ticker').max()

        # 3. Trend Efficiency %
        abs_diff = grouped['close'].diff().abs()
        total_path = abs_diff.groupby(level='ticker').sum()
        net_move_abs = (last_close - first_close).abs().groupby(level='ticker').last()
        trend = (net_move_abs / total_path * 100)

        # 4. Cycles (Saw Factor) - Векторизованный прокси для количества ребалансировок
        # Сумма абсолютных изменений в единицах порога
        cycles = (abs_diff / (first_close * threshold)).groupby(level='ticker').sum()

        results = pd.DataFrame({
            'net_change': net_change.groupby(level='ticker').last(),
            'max_spurt': max_spurt.fillna(0),
            'trend': trend.fillna(0),
            'cycles': cycles.fillna(0)
        })
        return results

    def rank_by_momentum(self, window: int = 14) -> pd.Series:
        # Векторизованный расчет по cross-section
        returns = self.df.groupby(level='ticker')['close'].pct_change()
        momentum = returns.groupby(level='ticker').rolling(window).mean()
        # Сброс индекса группы rolling для корректного доступа к уровням
        momentum = momentum.reset_index(level=0, drop=True)
        # Возвращаем Series с ранжированием на последний доступный timestamp
        return momentum.groupby(level='ticker').last().sort_values(ascending=False)

from concurrent.futures import ProcessPoolExecutor

def run_ranker_task(multi_df: pd.DataFrame, threshold: float):
    """Helper function to run the ranker in a separate process."""
    ranker = TickerRanker(multi_df)
    metrics_df = ranker.calculate_metrics(threshold)
    return metrics_df

class TickerScanner:
    def __init__(self, concurrent_requests: int = 15, rebalance_threshold: float = 0.02, scanner_period_days: float = 1.0):
        self.base_url = "https://fapi.binance.com"
        self.rebalance_threshold = rebalance_threshold
        self.scanner_period_days = scanner_period_days
        self.semaphore = asyncio.Semaphore(concurrent_requests)
        self.process_executor = ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 8))
        
    @retry_on_network_error(retries=3)
    async def fetch(self, session: aiohttp.ClientSession, endpoint: str, params: dict = None):
        async with self.semaphore:
            try:
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

    async def fetch_klines(self, session: aiohttp.ClientSession, symbol: str, limit: int) -> Optional[pd.DataFrame]:
        now_ms = int(time.time() * 1000)
        params = {"symbol": symbol, "interval": "1m", "limit": limit, "endTime": now_ms}
        klines = await self.fetch(session, "/fapi/v1/klines", params)
        if not klines or len(klines) < 100:
            return None
        
        df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'q_vol', 'trades', 't_base', 't_quote', 'ignore'])
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df['ticker'] = symbol
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        return df.set_index(['ticker', 'time'])

    async def get_top_tickers(self, min_volume: float = 20_000_000) -> List[Dict[str, Any]]:
        logger.info(f"Market Scan (Min Vol: {min_volume/1e6:.0f}M, Threshold: {self.rebalance_threshold*100}%)...")
        
        white_list = set()
        if os.path.exists("tickers.txt"):
            try:
                with open("tickers.txt", "r", encoding="utf-8") as f:
                    white_list = {line.strip() for line in f if line.strip()}
            except Exception as e:
                logger.error(f"Failed to load tickers.txt: {e}")

        black_list = set()
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    black_list = set(cfg.get("black_list", []))
            except Exception as e:
                logger.error(f"Failed to load black_list from config: {e}")

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
                if white_list and symbol not in white_list: continue
                if symbol in black_list: continue
                candidates.append(t)

            candidates.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
            candidates = candidates[:80]
            
            limit = max(1, int(self.scanner_period_days * 1440))
            tasks = [self.fetch_klines(session, c['symbol'], limit) for c in candidates]
            dfs = await asyncio.gather(*tasks)

            valid_dfs = [df for df in dfs if df is not None]
            if not valid_dfs:
                return []

            multi_df = pd.concat(valid_dfs)
            
            # Offload ranking to a separate process
            loop = asyncio.get_running_loop()
            metrics_df = await loop.run_in_executor(
                self.process_executor, run_ranker_task, multi_df, self.rebalance_threshold
            )

            metrics_df['symbol'] = metrics_df.index
            metrics_df['funding'] = metrics_df['symbol'].map(funding_map).fillna(0.0) * 100
            
            # Фильтр по циклам (минимум 10)
            metrics_df = metrics_df[metrics_df['cycles'] >= 10].copy()
            metrics_df.loc[:, 'cycles'] = metrics_df['cycles'].astype(int)
            ranked_list = metrics_df.to_dict('records')
            ranked_list.sort(key=lambda x: x['cycles'], reverse=True)
            
            try:
                with open(CACHE_FILE, "w") as f:
                    json.dump({"timestamp": time.time(), "results": ranked_list}, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save cache: {e}")
            
            return ranked_list

async def main(quiet=False, min_volume=20_000_000):
    threshold = 0.02
    scanner_period_days = 1.0
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
                threshold = cfg["portfolios"][0].get("rebalance_threshold", 0.02)
                scanner_period_days = cfg.get("scanner_period_days", 1.0)
    except: pass

    scanner = TickerScanner(concurrent_requests=20, rebalance_threshold=threshold, scanner_period_days=scanner_period_days)
    top_tickers = await scanner.get_top_tickers(min_volume=min_volume)
    
    if not quiet:
        print("\n" + "="*125)
        print(f"{'SYMBOL':<15} | {'CYCLES':<8} | {'NET MOVE%':<12} | {'MAX SPURT%':<12} | {'TREND EFF%':<12} | {'FUNDING%'}")
        print("-" * 125)
        for t in top_tickers[:50]:
            print(f"{t['symbol']:<15} | {t['cycles']:<8} | {t['net_change']:<12.2f} | {t['max_spurt']:<12.2f} | {t['trend']:<12.2f} | {t['funding']:.4f}")
        print("="*125)
    
    return top_tickers

if __name__ == "__main__":
    asyncio.run(main())
