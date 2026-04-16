import asyncio
import aiohttp
import json
import logging
from typing import List, Dict

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Scanner")

class TickerScanner:
    def __init__(self):
        self.base_url = "https://fapi.binance.com"
        
    async def fetch(self, endpoint: str, params: dict = None):
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}{endpoint}", params=params) as response:
                return await response.json()

    async def get_top_tickers(self, min_volume: float = 50_000_000):
        """Получает топ тикеров по волатильности и фандингу"""
        logger.info("Fetching market data...")
        
        # 1. Получаем 24h статистику
        tickers_24h = await self.fetch("/fapi/v1/ticker/24hr")
        # 2. Получаем текущие ставки фандинга
        funding_rates = await self.fetch("/fapi/v1/premiumIndex")
        
        funding_map = {item['symbol']: float(item['lastFundingRate']) for item in funding_rates}
        
        ranked_list = []
        
        for t in tickers_24h:
            symbol = t['symbol']
            if not symbol.endswith("USDT"): continue
            
            volume = float(t['quoteVolume'])
            if volume < min_volume: continue
            
            high = float(t['highPrice'])
            low = float(t['lowPrice'])
            price = float(t['lastPrice'])
            
            if price <= 0: continue
            
            # Метрики
            volatility = (high - low) / price * 100
            funding = funding_map.get(symbol, 0.0) * 100 # в процентах
            
            # Расчет Score (упрощенно)
            # Приоритет волатильности (70%) + фандинг (30%)
            # Фандинг учитываем как абсолютное значение, так как мы стоим в обе стороны, 
            # но в Market Neutral 2.0 шорт чуть больше, поэтому положительный фандинг выгоднее.
            score = (volatility * 1.5) + (funding * 10)
            
            ranked_list.append({
                "symbol": symbol,
                "volatility": volatility,
                "volume_m": volume / 1_000_000,
                "funding": funding,
                "score": score
            })
            
        # Сортировка по score
        ranked_list.sort(key=lambda x: x['score'], reverse=True)
        return ranked_list[:15]

async def main():
    scanner = TickerScanner()
    top_tickers = await scanner.get_top_tickers()
    
    print("\n" + "="*70)
    print(f"{'SYMBOL':<12} | {'VOLATILITY %':<12} | {'VOLUME (M)':<10} | {'FUNDING %':<10} | {'SCORE':<8}")
    print("-" * 70)
    
    for t in top_tickers:
        print(f"{t['symbol']:<12} | {t['volatility']:<12.2f} | {t['volume_m']:<10.1f} | {t['funding']:<10.4f} | {t['score']:<8.2f}")
    print("="*70)
    print("TIP: Choose tickers with high Volatility and positive Funding for best results.")

if __name__ == "__main__":
    asyncio.run(main())
