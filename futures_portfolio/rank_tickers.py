import asyncio
import aiohttp
import logging
import time
import math
from typing import List, Dict

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Scanner")

class TickerScanner:
    def __init__(self):
        self.base_url = "https://fapi.binance.com"
        self.rebalance_threshold = 0.005 # 0.5% - как в нашем конфиге
        
    async def fetch(self, endpoint: str, params: dict = None):
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}{endpoint}", params=params) as response:
                    if response.status != 200:
                        return None
                    return await response.json()
            except Exception as e:
                logger.debug(f"Fetch error: {e}")
                return None

    async def estimate_cycles_and_trend(self, symbol: str) -> Dict:
        """
        Микро-бэктест за 24 часа на 1-минутных свечах.
        Считает реальное кол-во пересечений порога и силу тренда.
        """
        # 1440 минут = 1 день
        klines = await self.fetch("/fapi/v1/klines", {"symbol": symbol, "interval": "1m", "limit": 1440})
        if not klines or len(klines) < 100:
            return {"cycles": 0, "trend_penalty": 1.0, "net_change": 0}
        
        closes = [float(k[4]) for k in klines]
        
        # 1. Считаем циклы (имитация ребалансировки)
        cycles = 0
        basis = closes[0]
        for price in closes:
            diff = abs(price - basis) / basis
            if diff >= self.rebalance_threshold:
                cycles += 1
                basis = price
                
        # 2. Считаем силу тренда (Trend Efficiency)
        # Отношение чистого движения к общему пройденному пути
        total_path = sum(abs(closes[i] - closes[i-1]) for i in range(1, len(closes)))
        net_move = abs(closes[-1] - closes[0])
        
        # Чем выше trend_ratio, тем более "прямолинейно" движется цена (плохо для нас)
        trend_ratio = net_move / total_path if total_path > 0 else 1.0
        
        # Штраф за тренд: если цена идет палкой, циклы обесцениваются
        # Идеальный тренд-фактор для нас - когда много шума (total_path) при малом net_move
        # Мы хотим, чтобы trend_ratio был низким (например < 0.1)
        trend_penalty = max(0.1, 1.0 - (trend_ratio * 2))
        
        return {
            "cycles": cycles,
            "trend_ratio": trend_ratio,
            "trend_penalty": trend_penalty,
            "net_change_pct": (closes[-1] / closes[0] - 1) * 100
        }

    async def get_top_tickers(self, min_volume: float = 150_000_000):
        """Получает топ тикеров на основе микро-бэктеста и качества волатильности"""
        logger.info("Step 1: Fetching market overview...")
        
        tickers_24h = await self.fetch("/fapi/v1/ticker/24hr")
        funding_rates = await self.fetch("/fapi/v1/premiumIndex")
        
        if not tickers_24h or not funding_rates:
            logger.error("Failed to fetch data from Binance")
            return []

        funding_map = {item['symbol']: float(item['lastFundingRate']) for item in funding_rates}
        now_ms = int(time.time() * 1000)
        one_year_ms = 365 * 24 * 60 * 60 * 1000
        
        # Фильтруем по объему и USDT
        candidates = [t for t in tickers_24h if t['symbol'].endswith("USDT") and float(t['quoteVolume']) >= min_volume]
        # Сортируем по объему и берем топ-40 для детального анализа
        candidates.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
        candidates = candidates[:40]
        
        ranked_list = []
        logger.info(f"Step 2: Micro-backtesting {len(candidates)} candidates (24h history)...")
        
        for t in candidates:
            symbol = t['symbol']
            
            # Проверка возраста (листинг > 1 года)
            old_klines = await self.fetch("/fapi/v1/klines", {"symbol": symbol, "interval": "1M", "startTime": now_ms - one_year_ms, "limit": 1})
            if not old_klines: continue
            
            # Анализ циклов и тренда
            analysis = await self.estimate_cycles_and_trend(symbol)
            
            cycles = analysis['cycles']
            if cycles < 20: continue # Слишком низкая активность
            
            funding = funding_map.get(symbol, 0.0) * 100
            
            # --- ИТОГОВЫЙ SCORE (Реалистичный) ---
            # База = Количество циклов * Штраф за тренд
            # Бонус за фандинг (т.к. мы Net-Short)
            
            score = (cycles * analysis['trend_penalty'])
            
            # Если фандинг положительный - это чистый плюс к доходности шорт-позиции (которая у нас больше)
            if funding > 0:
                score *= (1 + funding * 5) # Небольшой буст за пассивный доход
            else:
                score *= (1 + funding * 2) # Штраф за отрицательный фандинг
                
            # Штраф за экстремальную волатильность (риск вылета по стопу)
            vola_24h = (float(t['highPrice']) - float(t['lowPrice'])) / float(t['lastPrice']) * 100
            if vola_24h > 100:
                score *= 0.5 # Режем вдвое за риск "бешеной монеты"
                
            ranked_list.append({
                "symbol": symbol,
                "cycles": cycles,
                "trend_ratio": analysis['trend_ratio'],
                "vola": vola_24h,
                "funding": funding,
                "score": score,
                "net_change": analysis['net_change_pct']
            })
            
        ranked_list.sort(key=lambda x: x['score'], reverse=True)
        return ranked_list

async def main():
    scanner = TickerScanner()
    top_tickers = await scanner.get_top_tickers()
    
    print("\n" + "="*95)
    print(f"{'SYMBOL':<12} | {'CYCLES(24h)':<12} | {'TREND %':<10} | {'VOLA %':<8} | {'FUND %':<8} | {'SCORE':<8}")
    print("-" * 95)
    
    for t in top_tickers[:15]:
        # TREND % - это насколько прямолинейно шла цена (100% - палка, 5% - пила)
        trend_display = t['trend_ratio'] * 100
        print(f"{t['symbol']:<12} | {t['cycles']:<12} | {trend_display:<10.2f} | {t['vola']:<8.2f} | {t['funding']:<8.4f} | {t['score']:<8.2f}")
    
    print("="*95)
    print("ANALYSIS: CYCLES > 100 with TREND < 10% is the 'Sweet Spot' for Market Neutral.")
    print("SCORING: Cycles * (1 - Trend_Penalty) + Funding_Bias.")

if __name__ == "__main__":
    asyncio.run(main())
