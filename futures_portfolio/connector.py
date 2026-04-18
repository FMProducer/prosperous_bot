import os
import asyncio
import logging
from typing import Dict, List, Callable, Any
from binance.client import Client
from binance.exceptions import BinanceAPIException
import requests.exceptions

logger = logging.getLogger(__name__)

def retry_on_network_error(retries: int = 3, delay: float = 2.0):
    """Декоратор для повторных попыток при сетевых ошибках."""
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except (requests.exceptions.RequestException, 
                        requests.exceptions.ProxyError,
                        requests.exceptions.ConnectionError,
                        BinanceAPIException) as e:
                    last_err = e
                    if attempt < retries - 1:
                        logger.warning(f"Network error in {func.__name__} (attempt {attempt+1}/{retries}): {e}. Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Max retries reached for {func.__name__}. Last error: {e}")
            raise last_err
        return wrapper
    return decorator

class BinanceConnector:
    def __init__(self, api_key: str, secret_key: str, testnet: bool = True, base_ticker: str = "BTCUSDT"):
        self.testnet = testnet
        self.base_ticker = base_ticker
        self.client = Client(api_key, secret_key, testnet=testnet)
        self.futures_client = self.client

    @retry_on_network_error(retries=5, delay=3.0)
    async def get_positions(self) -> Dict[str, float]:
        """Получение фьючерсных позиций с разделением на LONG и SHORT."""
        positions = await asyncio.to_thread(self.futures_client.futures_position_information)
        result = {}
        for pos in positions:
            qty = float(pos["positionAmt"])
            side = pos["positionSide"] # LONG, SHORT or BOTH
            symbol = pos["symbol"]
            
            if qty != 0:
                key = f"{symbol}_{side}" if side != "BOTH" else symbol
                result[key] = qty
        return result

    @retry_on_network_error(retries=5, delay=3.0)
    async def get_futures_prices(self, tickers: List[str] = None) -> Dict[str, float]:
        """Получение фьючерсных цен для заданных тикеров."""
        # Для фьючерсов используем ticker_price, чтобы иметь живую цену последней сделки
        prices = await asyncio.to_thread(self.futures_client.futures_symbol_ticker)
        if isinstance(prices, dict): # Если вернулся один словарь
            prices = [prices]
        
        # Создаем мапу всех цен
        price_map = {t["symbol"]: float(t["price"]) for t in prices}
        if tickers is None:
            return price_map
        return {sym: price_map.get(sym) for sym in tickers}

    @retry_on_network_error(retries=5, delay=3.0)
    async def get_spot_prices(self, tickers: List[str] = None) -> Dict[str, float]:
        """Получение спот-цены для заданного тикера или всех тикеров."""
        if tickers is None:
            tickers = [self.base_ticker]
        
        prices = await asyncio.to_thread(self.client.get_all_tickers)
        result = {t["symbol"]: float(t["price"]) for t in prices}
        return {sym: result.get(sym.split('_')[0]) for sym in tickers}

    @retry_on_network_error(retries=3, delay=2.0)
    async def get_exchange_info(self) -> Dict:
        return await asyncio.to_thread(self.futures_client.futures_exchange_info)

    @retry_on_network_error(retries=5, delay=3.0)
    async def get_futures_klines(self, symbol: str, interval: str, limit: int = 100) -> List[List]:
        """Получение свечей фьючерсов."""
        return await asyncio.to_thread(self.futures_client.futures_klines, symbol=symbol, interval=interval, limit=limit)

    @retry_on_network_error(retries=3, delay=2.0)
    async def get_margin_ratio(self) -> Dict[str, float]:
        """
        Получение информации о марже и уровне риска.
        Возвращает словарь с marginRatio, availableBalance, totalMaintMargin
        """
        account_info = await asyncio.to_thread(self.futures_client.futures_account)
        return {
            "margin_ratio": float(account_info.get("totalMarginBalance", 0)) / float(account_info.get("totalMaintMargin", 1)) if float(account_info.get("totalMaintMargin", 0)) > 0 else float('inf'),
            "available_balance": float(account_info.get("availableBalance", 0)),
            "total_maint_margin": float(account_info.get("totalMaintMargin", 0)),
            "total_margin_balance": float(account_info.get("totalMarginBalance", 0)),
            "liquidation_price": float(account_info.get("liquidationPrice", 0)) if account_info.get("liquidationPrice") else None
        }

    @retry_on_network_error(retries=3, delay=2.0)
    async def get_free_balance(self) -> float:
        if not self.client.api_key or self.client.api_key == "YOUR_API_KEY":
            return 10000.0
        balances = await asyncio.to_thread(self.futures_client.futures_account_balance)
        usdt_balance = next((b["balance"] for b in balances if b["asset"] == "USDT"), 0.0)
        return float(usdt_balance)
