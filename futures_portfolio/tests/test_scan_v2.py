import asyncio
import aiohttp
import json
import time
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

async def main():
    base_url = "https://fapi.binance.com"
    
    # Load config
    CONFIG_FILE = "config.json"
    CACHE_FILE = "scan_cache.json"
    scanner_period_days = 0.125
    threshold = 0.01
    
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            cfg = json.load(f)
            scanner_period_days = cfg.get("scanner_period_days", 0.125)
    
    # Load whitelist
    white_list = set()
    if os.path.exists("tickers.txt"):
        with open("tickers.txt", "r") as f:
            white_list = {line.strip() for line in f if line.strip()}
    
    # Load blacklist
    black_list = set()
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            cfg = json.load(f)
            black_list = set(cfg.get("black_list", []))
    
    semaphore = asyncio.Semaphore(20)
    
    async def fetch(session, endpoint, params=None):
        async with semaphore:
            try:
                async with session.get(f"{base_url}{endpoint}", params=params, timeout=20, proxy=None) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 5))
                        await asyncio.sleep(retry_after)
                        return await fetch(session, endpoint, params)
                    if response.status != 200:
                        print(f"  Non-200: {response.status} for {endpoint}")
                        return None
                    return await response.json()
            except Exception as e:
                print(f"  Exception: {type(e).__name__}: {e}")
                return None
    
    print("Starting scan...")
    
    async with aiohttp.ClientSession(trust_env=True) as session:
        tickers_24h = await fetch(session, "/fapi/v1/ticker/24hr")
        premium_info = await fetch(session, "/fapi/v1/premiumIndex")
        
        if not tickers_24h or not premium_info:
            print("Failed to fetch market data")
            return []
        
        print(f"Got {len(tickers_24h)} tickers, {len(premium_info)} premium entries")
        
        funding_map = {item['symbol']: float(item['lastFundingRate']) for item in premium_info}
        
        # Filter candidates
        min_volume = 20_000_000
        candidates = []
        for t in tickers_24h:
            symbol = t['symbol']
            if not symbol.endswith("USDT"): continue
            if float(t['quoteVolume']) < min_volume: continue
            if not all(ord(c) < 128 for c in symbol): continue
            if white_list and symbol not in white_list: continue
            if symbol in black_list: continue
            candidates.append(t)
        
        print(f"Candidates after filter: {len(candidates)}")
        candidates.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
        candidates = candidates[:80]
        
        limit = max(1, int(scanner_period_days * 1440))
        print(f"Fetching {limit} x 1m klines for {len(candidates)} symbols...")
        
        now_ms = int(time.time() * 1000)
        valid_dfs = []
        
        for c in candidates:
            symbol = c['symbol']
            params = {"symbol": symbol, "interval": "1m", "limit": limit, "endTime": now_ms}
            df = await fetch_klines(session, fetch, symbol, limit, now_ms)
            if df is not None:
                valid_dfs.append(df)
        
        print(f"Valid klines: {len(valid_dfs)}")
        
        if not valid_dfs:
            return []
        
        multi_df = pd.concat(valid_dfs)
        
        # Calculate metrics inline (no ProcessPoolExecutor for simplicity)
        grouped = multi_df.groupby(level='ticker')
        first_close = grouped['close'].transform('first')
        last_close = grouped['close'].transform('last')
        net_change = (last_close / first_close - 1) * 100
        
        h_roll = grouped['high'].rolling(60).max().reset_index(level=0, drop=True)
        l_roll = grouped['low'].rolling(60).min().reset_index(level=0, drop=True)
        o_roll = grouped['open'].shift(59)
        max_spurt = ((h_roll - l_roll) / o_roll * 100).groupby(level='ticker').max()
        
        abs_diff = grouped['close'].diff().abs()
        total_path = abs_diff.groupby(level='ticker').sum()
        net_move_abs = (last_close - first_close).abs().groupby(level='ticker').last()
        trend = (net_move_abs / total_path * 100)
        cycles = (abs_diff / (first_close * threshold)).groupby(level='ticker').sum()
        
        metrics_df = pd.DataFrame({
            'net_change': net_change.groupby(level='ticker').last(),
            'max_spurt': max_spurt.fillna(0),
            'trend': trend.fillna(0),
            'cycles': cycles.fillna(0)
        })
        
        metrics_df['symbol'] = metrics_df.index
        metrics_df['funding'] = metrics_df['symbol'].map(funding_map).fillna(0.0) * 100
        
        # Filter min cycles >= 10
        metrics_df = metrics_df[metrics_df['cycles'] >= 10].copy()
        metrics_df.loc[:, 'cycles'] = metrics_df['cycles'].astype(int)
        
        ranked_list = metrics_df.to_dict('records')
        ranked_list.sort(key=lambda x: x['cycles'], reverse=True)
        
        # Print results
        print("\n" + "="*125)
        print(f"{'SYMBOL':<15} | {'CYCLES':<8} | {'NET MOVE%':<12} | {'MAX SPURT%':<12} | {'TREND EFF%':<12} | {'FUNDING%'}")
        print("-" * 125)
        for t in ranked_list[:50]:
            print(f"{t['symbol']:<15} | {t['cycles']:<8} | {t['net_change']:<12.2f} | {t['max_spurt']:<12.2f} | {t['trend']:<12.2f} | {t['funding']:.4f}")
        print("="*125)
        
        return ranked_list

async def fetch_klines(session, fetch_func, symbol, limit, now_ms):
    params = {"symbol": symbol, "interval": "1m", "limit": limit, "endTime": now_ms}
    klines = await fetch_func(session, "/fapi/v1/klines", params)
    if not klines or len(klines) < 10:
        return None
    
    df = pd.DataFrame(klines, columns=['time','open','high','low','close','volume','close_time','q_vol','trades','t_base','t_quote','ignore'])
    df = df[['time','open','high','low','close','volume']].copy()
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    df['ticker'] = symbol
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    return df.set_index(['ticker','time'])

asyncio.run(main())
