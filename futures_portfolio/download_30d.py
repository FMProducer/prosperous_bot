"""
download_30d.py — Загрузка 30-дневных данных для всех тикеров из tickers.txt
"""
import asyncio
import aiohttp
import time
import os
import math
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TICKERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tickers.txt")
DAYS = 30.0
CONCURRENT = 1   # строго последовательно!
DELAY_BETWEEN_TICKERS = 10.0
RETRY_BASE_DELAY = 60.0
MAX_RETRIES = 3


async def download_one(session, symbol, data_dir, days, retry_count=0):
    endpoint = "https://fapi.binance.com/fapi/v1/klines"
    total_minutes = int(days * 24 * 60)
    limit_per_request = 1500
    num_requests = math.ceil(total_minutes / limit_per_request)

    all_data = []
    end_time_ms = int(time.time() * 1000)

    for i in range(num_requests):
        batch = min(limit_per_request, total_minutes - len(all_data))
        if batch <= 0:
            break
        params = {"symbol": symbol, "interval": "1m", "limit": batch}
        if i > 0:
            params["endTime"] = end_time_ms

        async with session.get(endpoint, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                if not data:
                    break
                all_data = data + all_data
                end_time_ms = data[0][0] - 1
            elif resp.status == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                print(f"  {symbol}: 429, waiting {retry_after}s...")
                await asyncio.sleep(retry_after)
                continue
            elif resp.status == 418:
                if retry_count >= MAX_RETRIES:
                    print(f"  {symbol}: 418 max retries exceeded")
                    return None
                wait = RETRY_BASE_DELAY * (2 ** retry_count)
                print(f"  {symbol}: 418, waiting {wait}s (retry {retry_count+1}/{MAX_RETRIES})...")
                await asyncio.sleep(wait)
                return await download_one(session, symbol, data_dir, days, retry_count + 1)
            else:
                print(f"  {symbol}: error {resp.status}")
                return None

        if i < num_requests - 1:
            await asyncio.sleep(1.0)

    if not all_data:
        return None

    df = pd.DataFrame(all_data, columns=[
        'time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'q_vol', 'trades', 't_base', 't_quote', 'ignore'
    ])
    df = df[['open', 'high', 'low', 'close', 'volume']]
    for col in df.columns:
        df[col] = df[col].astype(float)

    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f"{symbol}_live_30d.feather")
    df.to_feather(file_path)
    return file_path


async def main():
    if not os.path.exists(TICKERS_FILE):
        print("tickers.txt not found.")
        return

    with open(TICKERS_FILE, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]

    pending = []
    for t in tickers:
        path = os.path.join(DATA_DIR, f"{t}_live_30d.feather")
        if os.path.exists(path):
            df = pd.read_feather(path, columns=['close'])
            rows = len(df)
            if rows >= 40000:
                print(f"  SKIP {t:<18s} already exists ({rows} rows)")
                continue
        pending.append(t)

    if not pending:
        print("All tickers already downloaded.")
        return

    # Читаем из аргумента --wait секунд перед стартом (чтобы Binance разбанил)
    wait_before = 0
    import sys
    for i, arg in enumerate(sys.argv):
        if arg == "--wait" and i + 1 < len(sys.argv):
            wait_before = int(sys.argv[i + 1])

    if wait_before > 0:
        print(f"Waiting {wait_before}s before starting...")
        await asyncio.sleep(wait_before)

    print(f"Downloading {DAYS}d for {len(pending)} tickers...")
    print()

    results = {"ok": [], "fail": []}

    async with aiohttp.ClientSession() as session:
        for i, ticker in enumerate(pending):
            try:
                path = await download_one(session, ticker, DATA_DIR, DAYS)
                if path:
                    df = pd.read_feather(path, columns=['close'])
                    rows = len(df)
                    days_actual = rows / 1440
                    print(f"  OK   {ticker:<18s} {rows:>8d} rows ({days_actual:.1f} days)")
                    results["ok"].append(ticker)
                else:
                    results["fail"].append(ticker)
            except Exception as e:
                print(f"  FAIL {ticker:<18s} {e}")
                results["fail"].append(ticker)

            if i < len(pending) - 1:
                await asyncio.sleep(DELAY_BETWEEN_TICKERS)

    print()
    print(f"Done: {len(results['ok'])} OK, {len(results['fail'])} FAIL")
    if results["fail"]:
        print(f"Failed: {', '.join(results['fail'])}")


if __name__ == "__main__":
    asyncio.run(main())
