import asyncio
import aiohttp
import logging
import time
from typing import List, Dict

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Scanner")

class TickerScanner:
    def __init__(self):
        self.base_url = "https://fapi.binance.com"
        self.rebalance_threshold = 0.005 # 0.5% - базис для оценки циклов
        
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
        """
        klines = await self.fetch("/fapi/v1/klines", {"symbol": symbol, "interval": "1m", "limit": 1440})
        if not klines or len(klines) < 100:
            return {"cycles": 0, "trend_ratio": 1.0, "vola_24h": 0}
        
        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        
        # 1. Циклы (Saw Factor)
        cycles = 0
        basis = closes[0]
        for price in closes:
            diff = abs(price - basis) / basis
            if diff >= self.rebalance_threshold:
                cycles += 1
                basis = price
                
        # 2. Trend Efficiency (Насколько цена идет по прямой)
        total_path = sum(abs(closes[i] - closes[i-1]) for i in range(1, len(closes)))
        net_move = abs(closes[-1] - closes[0])
        trend_ratio = net_move / total_path if total_path > 0 else 1.0
        
        # 3. Волатильность (Размах за 24ч)
        vola_24h = (max(highs) - min(lows)) / closes[0] * 100
        
        return {
            "cycles": cycles,
            "trend_ratio": trend_ratio,
            "vola_24h": vola_24h,
            "net_change_pct": (closes[-1] / closes[0] - 1) * 100
        }

    async def get_top_tickers(self, min_volume: float = 200_000_000):
        """Получает топ тикеров на основе 'Золотых правил' Market Neutral"""
        logger.info(f"Step 1: Fetching market overview (Min Volume: {min_volume/1e6:.0f}M)...")
        
        tickers_24h = await self.fetch("/fapi/v1/ticker/24hr")
        funding_rates = await self.fetch("/fapi/v1/premiumIndex")
        
        if not tickers_24h or not funding_rates:
            logger.error("Failed to fetch data from Binance")
            return []

        funding_map = {item['symbol']: float(item['lastFundingRate']) for item in funding_rates}
        now_ms = int(time.time() * 1000)
        one_month_ms = 30 * 24 * 60 * 60 * 1000
        
        # Фильтруем по объему и USDT
        candidates = [t for t in tickers_24h if t['symbol'].endswith("USDT") and float(t['quoteVolume']) >= min_volume]
        candidates.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
        candidates = candidates[:50]
        
        ranked_list = []
        logger.info(f"Step 2: Micro-backtesting {len(candidates)} candidates (24h history)...")
        
        for t in candidates:
            symbol = t['symbol']
            
            # Проверка возраста (листинг > 1 месяца, чтобы избежать первичного листинг-пампа)
            old_klines = await self.fetch("/fapi/v1/klines", {"symbol": symbol, "interval": "1M", "startTime": now_ms - one_month_ms, "limit": 1})
            if not old_klines: continue
            
            analysis = await self.estimate_cycles_and_trend(symbol)
            cycles = analysis['cycles']
            trend_pct = analysis['trend_ratio'] * 100
            vola = analysis['vola_24h']
            funding = funding_map.get(symbol, 0.0) * 100
            
            # --- СКОРИНГ ПО "ЗОЛОТЫМ ПРАВИЛАМ" ---
            
            # 1. База - циклы
            score = cycles
            
            # 2. Множитель Тренда (Идеально < 3%)
            if trend_pct < 3.0: trend_factor = 1.5
            elif trend_pct < 7.0: trend_factor = 1.0
            elif trend_pct < 15.0: trend_factor = 0.5
            else: trend_factor = 0.1 # Смерть для нейтральности
            score *= trend_factor
            
            # 3. Множитель Волатильности (Нужна энергия > 30%)
            if vola > 35.0: vola_factor = 1.3
            elif vola > 15.0: vola_factor = 1.0
            else: vola_factor = 0.4 # Слишком вялый актив
            score *= vola_factor
            
            # 4. Множитель Фандинга (Т.к. мы Net-Short)
            if funding > 0:
                funding_factor = 1.0 + (funding * 10) # Бонус за прибыль шорта
            else:
                funding_factor = 1.0 / (1.0 + abs(funding) * 5) # Штраф за расходы шорта
            score *= funding_factor
                
            # Итоговый вердикт
            if score > 500 and trend_pct < 5: tier = "Tier-1 (GOLD)"
            elif score > 250: tier = "Tier-2 (GOOD)"
            else: tier = "Tier-3 (AVOID)"
                
            ranked_list.append({
                "symbol": symbol,
                "cycles": cycles,
                "trend": trend_pct,
                "vola": vola,
                "funding": funding,
                "score": score,
                "tier": tier
            })
            
        ranked_list.sort(key=lambda x: x['score'], reverse=True)
        return ranked_list

async def main():
    scanner = TickerScanner()
    top_tickers = await scanner.get_top_tickers()
    
    print("\n" + "="*110)
    print(f"{'SYMBOL':<12} | {'CYCLES':<8} | {'TREND %':<8} | {'VOLA %':<8} | {'FUND %':<8} | {'SCORE':<8} | {'RECOMMENDATION'}")
    print("-" * 110)
    
    for t in top_tickers[:20]:
        print(f"{t['symbol']:<12} | {t['cycles']:<8} | {t['trend']:<8.2f} | {t['vola']:<8.2f} | {t['funding']:<8.4f} | {t['score']:<8.2f} | {t['tier']}")
    
    print("="*110)
    print("GOLDEN RULES APPLIED:")
    print("1. Trend Efficiency < 3.0% (The 'Saw' Effect)")
    print("2. 24h Volatility > 35.0% (The 'Fuel')")
    print("3. Positive Funding (The 'Passive Rent' for Shorts)")
    print("4. Daily Volume > 200M USDT (The 'Liquidity')")

if __name__ == "__main__":
    asyncio.run(main())
