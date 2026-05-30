import asyncio, aiohttp, json, time
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

CONFIG_FILE = "config.json"
CACHE_FILE = "scan_cache.json"

async def main():
    base_url = "https://fapi.binance.com"
    
    # Load config
    scanner_period_days = 0.125
    threshold = 0.01
    if os_path_exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            cfg = json.load(f)
            scanner_period_days = cfg.get("scanner_period_days", 0.125)
            print(f"Config: scanner_period_days={scanner_period_days}, threshold={threshold}")
    
    # Load whitelist
    white_list = set()
    if os_path_exists("tickers.txt"):
        with open("tickers.txt", "r") as f:
            white_list = {line.strip() for line in f if line.strip()}
        print(f"Whitelist loaded: {len(white_list)} tickers")
    
    # Load blacklist
    black_list = set()
    if os_path_exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            cfg = json.load(f)
            black_list = set(cfg.get("black_list", []))
        print(f"Blacklist: {len(black_list)} tickers")
    
    async with aiohttp.ClientSession(trust_env=True) as session:
        # Fetch 24hr tickers
        print("Fetching 24hr tickers...")
        async with session.get(f"{base_url}/fapi/v1/ticker/24hr", timeout=20, proxy=None) as resp:
            print(f"  Status: {resp.status}")
            tickers_24h = await resp.json() if resp.status == 200 else None
        
        print("Fetching premium index...")
        async with session.get(f"{base_url}/fapi/v1/premiumIndex", timeout=20, proxy=None) as resp:
            print(f"  Status: {resp.status}")
            premium_info = await resp.json() if resp.status == 200 else None
        
        if not tickers_24h or not premium_info:
            print("FAILED to fetch initial data")
            return []
        
        print(f"  Tickers: {len(tickers_24h)}, Premium: {len(premium_info)}")
        
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
        
        print(f"Candidates after filtering (vol>{min_volume/1e6:.0f}M, whitelist): {len(candidates)}")
        candidates.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
        
        for c in candidates[:10]:
            print(f"  {c['symbol']}: vol={float(c['quoteVolume'])/1e6:.1f}M")
        
        if not candidates:
            print("No candidates found!")
            return []
        
        candidates = candidates[:80]
        limit = max(1, int(scanner_period_days * 1440))
        print(f"Fetching {limit} x 1m klines for top {len(candidates)} candidates...")
        
        now_ms = int(time.time() * 1000)
        valid_dfs = []
        failed = []
        for c in candidates:
            symbol = c['symbol']
            params = {"symbol": symbol, "interval": "1m", "limit": limit, "endTime": now_ms}
            try:
                async with session.get(f"{base_url}/fapi/v1/klines", params=params, timeout=20, proxy=None) as resp:
                    if resp.status == 200:
                        klines = await resp.json()
                        if klines and len(klines) >= 10:
                            df = pd.DataFrame(klines, columns=['time','open','high','low','close','volume','close_time','q_vol','trades','t_base','t_quote','ignore'])
                            df = df[['time','open','high','low','close','volume']].copy()
                            for col in ['open','high','low','close','volume']:
                                df[col] = df[col].astype(float)
                            df['ticker'] = symbol
                            df['time'] = pd.to_datetime(df['time'], unit='ms')
                            valid_dfs.append(df.set_index(['ticker','time']))
                        else:
                            failed.append(symbol)
                    else:
                        failed.append(symbol)
            except Exception as e:
                failed.append(symbol)
                print(f"  Error fetching {symbol}: {e}")
        
        print(f"Valid klines: {len(valid_dfs)}, Failed: {len(failed)}")
        if failed:
            print(f"  Failed symbols: {failed[:20]}")
        
        if not valid_dfs:
            print("No valid klines data!")
            return []
        
        multi_df = pd.concat(valid_dfs)
        print(f"Combined dataframe: {multi_df.shape}, tickers: {multi_df.index.get_level_values('ticker').nunique()}")
        
        # Calculate metrics
        from concurrent.futures import ProcessPoolExecutor
        import os
        
        def run_ranker_task(multi_df, threshold):
            import numpy as np
            import pandas as pd
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
            results = pd.DataFrame({
                'net_change': net_change.groupby(level='ticker').last(),
                'max_spurt': max_spurt.fillna(0),
                'trend': trend.fillna(0),
                'cycles': cycles.fillna(0)
            })
            return results
        
        import multiprocessing
        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
            metrics_df = await asyncio.get_running_loop().run_in_executor(
                executor, run_ranker_task, multi_df, threshold
            )
        
        metrics_df['symbol'] = metrics_df.index
        metrics_df['funding'] = metrics_df['symbol'].map(funding_map).fillna(0.0) * 100
        
        # Filter min cycles (already >= 10 in code, but let's also do > 20)
        metrics_df = metrics_df[metrics_df['cycles'] >= 10].copy()
        metrics_df.loc[:, 'cycles'] = metrics_df['cycles'].astype(int)
        
        # Calculate SCORE = cycles (primary ranking)
        # The code sorts by cycles descending
        ranked = metrics_df.sort_values('cycles', ascending=False)
        
        print("\n" + "="*125)
        print(f"{'SYMBOL':<15} | {'CYCLES':<8} | {'NET MOVE%':<12} | {'MAX SPURT%':<12} | {'TREND EFF%':<12} | {'FUNDING%'}")
        print("-" * 125)
        for _, row in ranked.head(50).iterrows():
            print(f"{row['symbol']:<15} | {int(row['cycles']):<8} | {row['net_change']:<12.2f} | {row['max_spurt']:<12.2f} | {row['trend']:<12.2f} | {row['funding']:.4f}")
        print("="*125)
        
        # Filter > 20 cycles
        high_cycle = ranked[ranked['cycles'] > 20]
        print(f"\nTickers with >20 cycles: {len(high_cycle)}")
        for _, row in high_cycle.iterrows():
            print(f"  {row['symbol']}: cycles={int(row['cycles'])}, net={row['net_change']:.2f}%, spurt={row['max_spurt']:.2f}%, trend={row['trend']:.2f}%, funding={row['funding']:.4f}%")
        
        if not high_cycle.empty:
            winner = high_cycle.iloc[0]
            print(f"\n*** WINNER: {winner['symbol']} ***")
            print(f"  Cycles: {int(winner['cycles'])}")
            print(f"  Net Change: {winner['net_change']:.2f}%")
            print(f"  Max Spurt: {winner['max_spurt']:.2f}%")
            print(f"  Trend Efficiency: {winner['trend']:.2f}%")
            print(f"  Funding Rate: {winner['funding']:.4f}%")
        
        return ranked

def os_path_exists(path):
    import os
    return os.path.exists(path)

asyncio.run(main())
