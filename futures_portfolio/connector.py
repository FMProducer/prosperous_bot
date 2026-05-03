import os
import asyncio
import logging
from typing import Dict, List, Callable, Any

from binance.client import Client
from binance.exceptions import BinanceAPIException
import requests.exceptions

logger = logging.getLogger(__name__)

# Подмена URL на уровне класса для обхода блокировок в РФ (до инициализации)
Client.API_URL = 'https://api1.binance.com/api'
Client.FUTURES_URL = 'https://fapi.binance.com/fapi'

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
        self.api_key = api_key
        
        # Настройка сессии: отключаем доверие к системному окружению (прокси)
        requests_params = {
            'proxies': {'http': None, 'https': None},
            'timeout': 15
        }
        
        if testnet:
            # Для тестнета зеркала обычно не нужны или не работают, но прокси отключаем
            self.client = Client(api_key, secret_key, testnet=True, requests_params=requests_params)
        else:
            self.client = Client(api_key, secret_key, testnet=False, requests_params=requests_params)
            # Дополнительная проверка, что URL подменились
            self.client.API_URL = 'https://api1.binance.com/api'
            self.client.FUTURES_URL = 'https://fapi.binance.com/fapi'
            
        self.futures_client = self.client
        # Отключаем использование системных переменных в сессии requests
        self.client.session.trust_env = False

    @retry_on_network_error(retries=5, delay=3.0)
    async def get_positions(self) -> Dict[str, Dict]:
        """Получение фьючерсных позиций с разделением на LONG и SHORT, включая цену входа."""
        positions = await asyncio.to_thread(self.futures_client.futures_position_information)
        result = {}
        for pos in positions:
            qty = float(pos["positionAmt"])
            entry_price = float(pos.get("entryPrice", 0.0))
            side = pos["positionSide"] # LONG, SHORT or BOTH
            symbol = pos["symbol"]
            
            if qty != 0:
                key = f"{symbol}_{side}" if side != "BOTH" else symbol
                result[key] = {
                    "qty": qty,
                    "entry_price": entry_price
                }
        return result

    @retry_on_network_error(retries=5, delay=3.0)
    async def get_futures_prices(self, tickers: List[str] = None) -> Dict[str, float]:
        """Получение фьючерсных цен (Last Price) для заданных тикеров."""
        prices = await asyncio.to_thread(self.futures_client.futures_symbol_ticker)
        if isinstance(prices, dict):
            prices = [prices]
        
        price_map = {t["symbol"]: float(t["price"]) for t in prices}
        if tickers is None:
            return price_map
        return {sym: price_map.get(sym) for sym in tickers}

    @retry_on_network_error(retries=5, delay=3.0)
    async def get_mark_prices(self, tickers: List[str] = None) -> Dict[str, float]:
        """Получение цен маркировки (Mark Price) для заданных тикеров."""
        prices = await asyncio.to_thread(self.futures_client.futures_mark_price)
        if isinstance(prices, dict):
            prices = [prices]
        
        price_map = {t["symbol"]: float(t["markPrice"]) for t in prices}
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
        return await asyncio.to_thread(self.futures_client.futures_klines, symbol=symbol, interval=interval, limit=limit)

    @retry_on_network_error(retries=3, delay=2.0)
    async def get_margin_ratio(self) -> Dict[str, float]:
        account_info = await asyncio.to_thread(self.futures_client.futures_account)
        return {
            "margin_ratio": float(account_info.get("totalMarginBalance", 0)) / float(account_info.get("totalMaintMargin", 1)) if float(account_info.get("totalMaintMargin", 0)) > 0 else float('inf'),
            "available_balance": float(account_info.get("availableBalance", 0)),
            "total_maint_margin": float(account_info.get("totalMaintMargin", 0)),
            "total_margin_balance": float(account_info.get("totalMarginBalance", 0)),
            "liquidation_price": float(account_info.get("liquidationPrice", 0)) if account_info.get("liquidationPrice") else None
        }

    @retry_on_network_error(retries=3, delay=2.0)
    async def get_hedge_mode(self) -> bool:
        """Проверка, включен ли Hedge Mode (True - включен, False - One-Way)."""
        mode_info = await asyncio.to_thread(self.futures_client.futures_get_position_mode)
        return mode_info.get("dualSidePosition", False)

    @retry_on_network_error(retries=3, delay=2.0)
    async def get_free_balance(self) -> float:
        if not self.api_key or self.api_key == "YOUR_API_KEY":
            return 10000.0
        balances = await asyncio.to_thread(self.futures_client.futures_account_balance)
        usdt_balance = next((b["balance"] for b in balances if b["asset"] == "USDT"), 0.0)
        return float(usdt_balance)

    @retry_on_network_error(retries=3, delay=2.0)
    async def get_bnb_balance(self) -> float:
        """Получение баланса BNB на фьючерсном аккаунте."""
        if not self.api_key or self.api_key == "YOUR_API_KEY":
            return 0.0
        balances = await asyncio.to_thread(self.futures_client.futures_account_balance)
        bnb_balance = next((b["balance"] for b in balances if b["asset"] == "BNB"), 0.0)
        return float(bnb_balance)

    @retry_on_network_error(retries=3, delay=2.0)
    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """Получение стакана ордеров (глубина 5-1000 уровней)."""
        return await asyncio.to_thread(
            self.futures_client.futures_depth,
            symbol=symbol,
            limit=limit
        )

    @retry_on_network_error(retries=3, delay=2.0)
    async def place_limit_order(self, symbol: str, side: str, qty: float, price: float,
                                position_side: str = "BOTH", reduce_only: bool = False,
                                time_in_force: str = "GTC") -> Dict:
        """
        Выставление лимитного ордера (Hedge Mode).
        """
        params = {
            "symbol": symbol,
            "side": side,
            "type": "LIMIT",
            "timeInForce": time_in_force,
            "quantity": abs(qty),
            "price": price,
            "positionSide": position_side
        }
        # Исключаем reduceOnly из параметров, так как он вызывает ошибку -1106
        if reduce_only:
            params["reduceOnly"] = True
            
        result = await asyncio.to_thread(
            self.futures_client.futures_create_order,
            **params
        )
        return result

    @retry_on_network_error(retries=3, delay=2.0)
    async def get_order_status(self, symbol: str, order_id: int) -> Dict:
        """Проверка статуса ордера."""
        return await asyncio.to_thread(
            self.futures_client.futures_get_order,
            symbol=symbol,
            orderId=order_id
        )

    @retry_on_network_error(retries=3, delay=2.0)
    async def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """Отмена ордера."""
        return await asyncio.to_thread(
            self.futures_client.futures_cancel_order,
            symbol=symbol,
            orderId=order_id
        )

    @retry_on_network_error(retries=3, delay=2.0)
    async def place_limit_maker_order(self, symbol: str, side: str, qty: float, price: float,
                                       position_side: str = "BOTH", reduce_only: bool = False) -> Dict:
        """
        Выставление POST-ONLY лимитного ордера (гарантия maker-комиссии).
        Если ордер не может быть maker — будет отклонён биржей.
        """
        result = await asyncio.to_thread(
            self.futures_client.futures_create_order,
            symbol=symbol,
            side=side,
            type="LIMIT_MAKER",
            quantity=abs(qty),
            price=price,
            positionSide=position_side
        )
        return result
