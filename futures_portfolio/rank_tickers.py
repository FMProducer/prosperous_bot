import asyncio
import aiohttp
import logging
import time
import json
import os

# ЖЕСТКОЕ ОТКЛЮЧЕНИЕ ПРОКСИ
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['NO_PROXY'] = '*'

from typing import List, Dict, Optional, Callable, Any
from dotenv import load_dotenv

# Загрузка переменных окружения для API ключей (если нужны для лимитов)
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("Scanner")

CACHE_FILE = "scan_cache.json"

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
    def __init__(self, concurrent_requests: int = 10):
        # Используем основной домен, так как он заработал в main.py
        self.base_url = "https://fapi.binance.com"
        self.rebalance_threshold = 0.015 
        self.semaphore = asyncio.Semaphore(concurrent_requests)
        
    @retry_on_network_error(retries=3)
    async def fetch(self, session: aiohttp.ClientSession, endpoint: str, params: dict = None):
        async with self.semaphore:
            # Принудительно отключаем прокси и игнорируем системные настройки
            try:
                async with session.get(f"{self.base_url}{endpoint}", params=params, timeout=20, proxy=None) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 5))
                        logger.warning(f"Rate limited (429). Sleeping for {retry_after}s")
                        await asyncio.sleep(retry_after)
                        return await self.fetch(session, endpoint, params)
                    if response.status != 200:
                        logger.debug(f"Error {response.status} for {endpoint}")
                        return None
                    return await response.json()
            except Exception as e:
                logger.debug(f"Fetch error for {endpoint}: {e}")
                return None

    async def analyze_ticker(self, session: aiohttp.ClientSession, symbol: str, funding_rate: float) -> Optional[Dict]:
        """Параллельный анализ одного тикера."""
        now_ms = int(time.time() * 1000)
        one_month_ms = 30 * 24 * 60 * 60 * 1000

        # 1. Проверка возраста тикера
        age_check = await self.fetch(session, "/fapi/v1/klines", {
            "symbol": symbol, "interval": "1M", "startTime": now_ms - one_month_ms, "limit": 1
        })
        if not age_check or len(age_check) == 0:
            return None

        # 2. Получение данных за 48 часов (2880 минут) двумя чанками
        tasks = []
        for i in range(2):
            end_time = now_ms - (1 - i) * 1440 * 60 * 1000
            params = {"symbol": symbol, "interval": "1m", "limit": 1440, "endTime": end_time}
            tasks.append(self.fetch(session, "/fapi/v1/klines", params))
        
        chunks = await asyncio.gather(*tasks)
        all_data = []
        for c in chunks:
            if c: all_data.extend(c)

        if not all_data or len(all_data) < 2000: # Ожидаем ~2880
            return None

        # Обработка данных
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
        
        # --- Улучшенный подсчет циклов (Z-logic) ---
        cycles = 0
        basis = closes[0]
        for price in closes:
            diff_pct = abs(price - basis) / basis
            if diff_pct >= self.rebalance_threshold:
                cycles += 1
                basis = price

        # --- Детектор всплесков (Spike Trap) ---
        max_hourly_spurt = 0.0
        for h in range(0, len(unique_klines), 60):
            window = unique_klines[h:h+60]
            if not window: continue
            w_high = max(float(k[2]) for k in window)
            w_low = min(float(k[3]) for k in window)
            w_open = float(window[0][1])
            spurt = (w_high - w_low) / w_open * 100
            max_hourly_spurt = max(max_hourly_spurt, spurt)

        # --- Trend Efficiency (Прямолинейность) ---
        total_path = sum(abs(closes[i] - closes[i-1]) for i in range(1, len(closes)))
        net_move = abs(closes[-1] - closes[0])
        trend_ratio = (net_move / total_path) if total_path > 0 else 0
        trend_ratio_pct = trend_ratio * 100 # 1.0 -> 100%

        # --- СКОРИНГ 3.1 ---
        score = cycles * 15 # Увеличили вес циклов
        net_change_pct = (closes[-1] / closes[0] - 1) * 100
        abs_net_change = abs(net_change_pct)
        
        # Дисквалификации
        if abs_net_change > 15.0:
            tier = "Tier-X (NET_TRAP)"
            score = 0
        elif max_hourly_spurt > 10.0:
            tier = "Tier-X (SPIKE_TRAP)"
            score = 0
        elif cycles < 12: # Снизили порог для реалистичности
            tier = "Tier-3 (LOW_ENERGY)"
            score = 0
        elif trend_ratio_pct > 7.0: # Слишком прямолинейно
            tier = "Tier-3 (TRENDING)"
            score *= 0.2
        else:
            # БОНУСЫ
            if abs_net_change < 5.0: score *= 1.3 # Флэт - хорошо
            
            # Funding: Если < 0, то за шорт платят нам (хорошо для нейтральной стратегии)
            if funding_rate < 0:
                score *= (1 + abs(funding_rate) * 50) # funding_rate обычно 0.01% = 0.0001
            
            # Распределение по тирам
            if score >= 250:
                tier = "Tier-1 (GOLD)"
            elif score >= 150:
                tier = "Tier-2 (GOOD)"
            else:
                tier = "Tier-3 (OK)"

        return {
            "symbol": symbol,
            "cycles": cycles,
            "trend": trend_ratio_pct,
            "net_change": net_change_pct,
            "max_spurt": max_hourly_spurt,
            "score": score,
            "tier": tier,
            "funding": funding_rate * 100
        }

    async def get_top_tickers(self, min_volume: float = 200_000_000):
        logger.info(f"Step 1: Market Scan (Min Vol: {min_volume/1e6:.0f}M)...")
        
        # Отключаем trust_env, чтобы aiohttp не лез в системные настройки прокси
        async with aiohttp.ClientSession(trust_env=False) as session:
            tickers_24h = await self.fetch(session, "/fapi/v1/ticker/24hr")
            premium_info = await self.fetch(session, "/fapi/v1/premiumIndex")
            
            if not tickers_24h or not premium_info:
                logger.error("Failed to fetch initial market data. Check your connection.")
                return []

            funding_map = {item['symbol']: float(item['lastFundingRate']) for item in premium_info}
            
            candidates = [t for t in tickers_24h if t['symbol'].endswith("USDT") and float(t['quoteVolume']) >= min_volume]
            candidates.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
            candidates = candidates[:60] # Берем чуть больше для запаса
            
            logger.info(f"Step 2: Parallel Analysis of {len(candidates)} candidates...")
            
            tasks = [self.analyze_ticker(session, c['symbol'], funding_map.get(c['symbol'], 0.0)) for c in candidates]
            results = await asyncio.gather(*tasks)
            
            ranked_list = [r for r in results if r is not None]
            ranked_list.sort(key=lambda x: x['score'], reverse=True)
            
            # Сохранение в кэш
            try:
                with open(CACHE_FILE, "w") as f:
                    json.dump({
                        "timestamp": time.time(),
                        "results": ranked_list
                    }, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save cache: {e}")
                
            return ranked_list

async def main():
    scanner = TickerScanner(concurrent_requests=15)
    start_time = time.time()
    top_tickers = await scanner.get_top_tickers()
    duration = time.time() - start_time
    
    print("\n" + "="*145)
    print(f"{'SYMBOL':<12} | {'CYCLES':<8} | {'NET MOVE%':<10} | {'MAX SPURT%':<10} | {'TREND EFF%':<10} | {'FUNDING%':<10} | {'SCORE':<8} | {'RECOMMENDATION'}")
    print("-" * 145)
    
    for t in top_tickers[:30]:
        print(f"{t['symbol']:<12} | {t['cycles']:<8} | {t['net_change']:<10.2f} | {t['max_spurt']:<10.2f} | {t['trend']:<10.2f} | {t['funding']:<10.4f} | {t['score']:<8.2f} | {t['tier']}")
    
    print("="*145)
    print(f"Scan completed in {duration:.1f} seconds.")
    print("SCORING RULES (48h Basis):")
    print("1. REAL CYCLES: Moves > 1.5%. Score = Cycles * 15.")
    print("2. SPIKE TRAP: 1h spurt > 10% -> DQ.")
    print("3. NET TRAP: 48h net move > 15% -> DQ.")
    print("4. TREND EFF: If > 7.0%, score reduced by 80% (Too linear).")
    print("5. FUNDING: Bonus for negative rates (we get paid for short).")

if __name__ == "__main__":
    asyncio.run(main())
