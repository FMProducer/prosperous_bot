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
        # 1.5% изменения цены примерно соответствуют 0.5% отклонения доли при x5 плече
        self.rebalance_threshold = 0.015 
        
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
        Глубокий анализ за 48 часов (2880 минут) с реалистичными порогами.
        """
        all_data = []
        now = int(time.time() * 1000)
        
        for i in range(2):
            end_time = now - (1 - i) * 1440 * 60 * 1000
            params = {"symbol": symbol, "interval": "1m", "limit": 1440, "endTime": end_time}
            chunk = await self.fetch("/fapi/v1/klines", params)
            if chunk: all_data.extend(chunk)
            
        if not all_data or len(all_data) < 500:
            return {"cycles": 0, "trend_ratio": 1.0, "vola_48h": 0, "net_change_pct": 0, "max_hourly_spurt": 0}
        
        seen_times = set()
        unique_klines = []
        for k in all_data:
            if k[0] not in seen_times:
                unique_klines.append(k)
                seen_times.add(k[0])
        unique_klines.sort(key=lambda x: x[0])
        
        closes = [float(k[4]) for k in unique_klines]
        highs = [float(k[2]) for k in unique_klines]
        lows = [float(k[3]) for k in unique_klines]
        
        # 1. Честные Циклы (с учетом плеча)
        cycles = 0
        basis = closes[0]
        for price in closes:
            diff = abs(price - basis) / basis
            if diff >= self.rebalance_threshold:
                cycles += 1
                basis = price
                
        # 2. Детектор критических всплесков
        max_hourly_spurt = 0.0
        for h in range(0, len(unique_klines), 60):
            window = unique_klines[h:h+60]
            if not window: continue
            w_high = max(float(k[2]) for k in window)
            w_low = min(float(k[3]) for k in window)
            w_open = float(window[0][1])
            spurt = (w_high - w_low) / w_open * 100
            max_hourly_spurt = max(max_hourly_spurt, spurt)

        # 3. Прямолинейность (Trend Efficiency)
        total_path = sum(abs(closes[i] - closes[i-1]) for i in range(1, len(closes)))
        net_move = abs(closes[-1] - closes[0])
        trend_ratio = net_move / total_path if total_path > 0 else 1.0
        
        return {
            "cycles": cycles,
            "trend_ratio": trend_ratio,
            "vola_48h": (max(highs) - min(lows)) / closes[0] * 100,
            "net_change_pct": (closes[-1] / closes[0] - 1) * 100,
            "max_hourly_spurt": max_hourly_spurt
        }

    async def get_top_tickers(self, min_volume: float = 200_000_000):
        """Получает топ тикеров, отфильтрованных по 'профпригодности'"""
        logger.info(f"Step 1: Market Scan (Min Vol: {min_volume/1e6:.0f}M)...")
        
        tickers_24h = await self.fetch("/fapi/v1/ticker/24hr")
        funding_rates = await self.fetch("/fapi/v1/premiumIndex")
        
        if not tickers_24h or not funding_rates:
            logger.error("Failed to fetch data from Binance")
            return []

        funding_map = {item['symbol']: float(item['lastFundingRate']) for item in funding_rates}
        now_ms = int(time.time() * 1000)
        one_month_ms = 30 * 24 * 60 * 60 * 1000
        
        candidates = [t for t in tickers_24h if t['symbol'].endswith("USDT") and float(t['quoteVolume']) >= min_volume]
        candidates.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
        candidates = candidates[:50]
        
        ranked_list = []
        logger.info(f"Step 2: Micro-backtesting {len(candidates)} candidates (48h history)...")
        
        for t in candidates:
            symbol = t['symbol']
            
            # Проверка возраста (листинг > 1 месяца)
            old_klines = await self.fetch("/fapi/v1/klines", {"symbol": symbol, "interval": "1M", "startTime": now_ms - one_month_ms, "limit": 1})
            if not old_klines: continue
            
            analysis = await self.estimate_cycles_and_trend(symbol)
            cycles = analysis['cycles']
            trend_ratio_pct = analysis['trend_ratio'] * 100
            abs_net_change = abs(analysis['net_change_pct'])
            max_spurt = analysis['max_hourly_spurt']
            funding = funding_map.get(symbol, 0.0) * 100
            
            # --- СКОРИНГ 3.0: "REALISTIC" ---
            score = cycles * 10 # Умножаем для наглядности
            
            # 1. ЖЕСТКАЯ ДИСКВАЛИФИКАЦИЯ
            if abs_net_change > 15.0: # Порог обвала/пампа за 48ч
                tier = "Tier-X (NET_TRAP)"
                score = 0
            elif max_spurt > 10.0: # Порог импульса за 1 час
                tier = "Tier-X (SPIKE_TRAP)"
                score = 0
            elif cycles < 15: # Минимум 15 реальных сделок за 48ч
                tier = "Tier-3 (LOW_ENERGY)"
                score = 0
            elif trend_ratio_pct > 7.0:
                tier = "Tier-3 (TRENDING)"
                score *= 0.1
            else:
                # 2. БОНУСЫ
                if abs_net_change < 5.0: score *= 1.5 
                if funding > 0: score *= (1 + funding * 5)
                
                tier = "Tier-1 (GOLD)" if score > 300 else "Tier-2 (GOOD)"

            ranked_list.append({
                "symbol": symbol,
                "cycles": cycles,
                "trend": trend_ratio_pct,
                "net_change": analysis['net_change_pct'],
                "max_spurt": max_spurt,
                "score": score,
                "tier": tier
            })
            
        ranked_list.sort(key=lambda x: x['score'], reverse=True)
        return ranked_list

async def main():
    scanner = TickerScanner()
    top_tickers = await scanner.get_top_tickers()
    
    print("\n" + "="*130)
    print(f"{'SYMBOL':<12} | {'CYCLES':<8} | {'NET MOVE%':<10} | {'MAX SPURT%':<10} | {'TREND EFF%':<10} | {'SCORE':<8} | {'RECOMMENDATION'}")
    print("-" * 130)
    
    for t in top_tickers[:25]:
        print(f"{t['symbol']:<12} | {t['cycles']:<8} | {t['net_change']:<10.2f} | {t['max_spurt']:<10.2f} | {t['trend']:<10.2f} | {t['score']:<8.2f} | {t['tier']}")
    
    print("="*130)
    print("REALISTIC SCORING RULES (48h Basis):")
    print("1. REAL CYCLES: Estimated price moves > 1.5% (approx. triggers for x5 leverage)")
    print("2. SPIKE LIMIT: 1-hour move < 10.0% (Protects against flash crashes)")
    print("3. NET LIMIT: 48-hour total move < 15.0% (Maintains Neutrality)")

if __name__ == "__main__":
    asyncio.run(main())
