import asyncio
import aiohttp
import time
import os
import logging
from datetime import datetime, timedelta

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("WhiteListUpdater")

TICKERS_FILE = "tickers.txt"
BASE_URL = "https://fapi.binance.com"

async def fetch_exchange_info():
    """Получает информацию о бирже и всех торговых парах"""
    url = f"{BASE_URL}/fapi/v1/exchangeInfo"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=20) as response:
            if response.status != 200:
                logger.error(f"Failed to fetch exchange info: {response.status}")
                return None
            return await response.json()

async def get_listing_time(session, symbol):
    """
    Получает время листинга тикера через запрос к klines (первая свеча).
    Binance API не отдает дату листинга напрямую в exchangeInfo для фьючерсов.
    """
    # Запрашиваем самую первую свечу (startTime = 0)
    url = f"{BASE_URL}/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": "1M", # Месячные свечи для быстроты
        "limit": 1,
        "startTime": 0
    }
    try:
        async with session.get(url, params=params, timeout=10) as response:
            if response.status != 200:
                return None
            data = await response.json()
            if data and len(data) > 0:
                return data[0][0] # Open time первой свечи в мс
    except Exception as e:
        logger.warning(f"Error fetching listing time for {symbol}: {e}")
    return None

async def main():
    logger.info("Starting White List update based on listing age (> 12 months)...")
    
    info = await fetch_exchange_info()
    if not info:
        return

    # Фильтруем только активные USDT пары
    symbols = [
        s['symbol'] for s in info['symbols'] 
        if s['status'] == 'TRADING' and s['quoteAsset'] == 'USDT'
    ]
    
    logger.info(f"Found {len(symbols)} active USDT futures pairs. Checking listing dates...")
    
    now_ms = int(time.time() * 1000)
    one_year_ms = 365 * 24 * 60 * 60 * 1000
    threshold_ms = now_ms - one_year_ms
    
    qualified_tickers = []
    
    # Используем семафор для ограничения одновременных запросов
    semaphore = asyncio.Semaphore(20)
    
    async def process_symbol(session, symbol):
        async with semaphore:
            listing_time = await get_listing_time(session, symbol)
            if listing_time and listing_time <= threshold_ms:
                listing_date = datetime.fromtimestamp(listing_time / 1000).strftime('%Y-%m-%d')
                logger.info(f"✅ {symbol:<15} | Listed: {listing_date} | Age: >12 months")
                return symbol
            elif listing_time:
                listing_date = datetime.fromtimestamp(listing_time / 1000).strftime('%Y-%m-%d')
                logger.debug(f"❌ {symbol:<15} | Listed: {listing_date} | Too young")
            return None

    async with aiohttp.ClientSession() as session:
        tasks = [process_symbol(session, s) for s in symbols]
        results = await asyncio.gather(*tasks)
        qualified_tickers = [r for r in results if r]

    qualified_tickers.sort()
    
    if qualified_tickers:
        logger.info(f"Update complete. {len(qualified_tickers)} tickers qualified.")
        try:
            with open(TICKERS_FILE, "w", encoding="utf-8") as f:
                for ticker in qualified_tickers:
                    f.write(f"{ticker}\n")
            logger.info(f"Successfully updated {TICKERS_FILE}")
        except Exception as e:
            logger.error(f"Failed to write to {TICKERS_FILE}: {e}")
    else:
        logger.warning("No tickers qualified the age filter. tickers.txt was not updated.")

if __name__ == "__main__":
    asyncio.run(main())
