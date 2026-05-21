import aiohttp
import os
import asyncio
import logging
from typing import Dict, List, Callable, Any

from binance import AsyncClient
from binance.exceptions import BinanceAPIException
import requests.exceptions

logger = logging.getLogger(__name__)

# Подмена URL на уровне класса для обхода блокировок в РФ (до инициализации)
# Для AsyncClient это может работать иначе, но пока оставляем как в инструкции.

class BinanceConnectorMock:
    def __init__(self, *args, **kwargs):
        from unittest.mock import MagicMock
        self.futures_client = MagicMock()
    async def get_exchange_info(self): return {"symbols": []}
    async def get_hedge_mode(self): return True
    async def set_leverage(self, *args): pass
    async def set_margin_type(self, *args): pass
    async def get_mark_prices(self, tickers): return {t: 60000.0 for t in tickers}
    async def get_positions(self): return {}
    async def get_margin_ratio(self): return {"margin_ratio": 10.0}

def retry_on_network_error(retries: int = 3, delay: float = 2.0):
    """Декоратор для повторных попыток при сетевых ошибках."""
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            # Lazy initialization of the client strictly inside the active Event Loop
            self_obj = args[0] if args else None
            if self_obj and hasattr(self_obj, '_ensure_client'):
                await self_obj._ensure_client()

            last_err = None
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except (aiohttp.ClientError, requests.exceptions.RequestException,
                        requests.exceptions.ProxyError,
                        requests.exceptions.ConnectionError) as e:
                    last_err = e
                    if attempt < retries - 1:
                        logger.warning(f"Network error in {func.__name__} (attempt {attempt+1}/{retries}): {e}. Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Max retries reached for {func.__name__}. Last error: {e}")
                except BinanceAPIException as e:
                    last_err = e
                    # Do not retry on client-side errors (except 429 Rate Limit)
                    if e.status_code and 400 <= e.status_code < 500 and e.status_code != 429:
                        logger.error(f"Fatal Binance API Error in {func.__name__}: {e}. Aborting retry.")
                        raise e

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
        self.secret_key = secret_key

        requests_params = {'timeout': 15}
        self.client = AsyncClient(
            self.api_key,
            self.secret_key,
            testnet=self.testnet,
            requests_params=requests_params
        )
        if not self.testnet:
            self.client.API_URL = 'https://api1.binance.com/api'
            self.client.FUTURES_URL = 'https://fapi.binance.com/fapi'
        self.futures_client = self.client

    async def _ensure_client(self):
        """No-op for compatibility with decorator if needed, but client is already init in __init__"""
        pass

    @retry_on_network_error(retries=5, delay=3.0)
    async def get_positions(self) -> Dict[str, Dict]:
        """Получение фьючерсных позиций с разделением на LONG и SHORT, включая цену входа."""
        positions = await self.futures_client.futures_position_information()
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
        prices = await self.futures_client.futures_symbol_ticker()
        if isinstance(prices, dict):
            prices = [prices]
        
        price_map = {t["symbol"]: float(t["price"]) for t in prices}
        if tickers is None:
            return price_map
        return {sym: price_map.get(sym) for sym in tickers}

    @retry_on_network_error(retries=5, delay=3.0)
    async def get_mark_prices(self, tickers: List[str] = None) -> Dict[str, float]:
        """Получение цен маркировки (Mark Price) для заданных тикеров."""
        prices = await self.futures_client.futures_mark_price()
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
        
        prices = await self.client.get_all_tickers()
        result = {t["symbol"]: float(t["price"]) for t in prices}
        return {sym: result.get(sym.split('_')[0]) for sym in tickers}

    @retry_on_network_error(retries=3, delay=2.0)
    async def get_exchange_info(self) -> Dict:
        return await self.futures_client.futures_exchange_info()

    @retry_on_network_error(retries=5, delay=3.0)
    async def get_futures_klines(self, symbol: str, interval: str, limit: int = 100) -> List[List]:
        return await self.futures_client.futures_klines(symbol=symbol, interval=interval, limit=limit)

    @retry_on_network_error(retries=3, delay=2.0)
    async def get_margin_ratio(self) -> Dict[str, float]:
        account_info = await self.futures_client.futures_account()
        return {
            "margin_ratio": float(account_info.get("totalMarginBalance", 0)) / float(account_info.get("totalMaintMargin", 1)) if float(account_info.get("totalMaintMargin", 0)) > 0 else float('inf'),
            "available_balance": float(account_info.get("availableBalance", 0)),
            "total_maint_margin": float(account_info.get("totalMaintMargin", 0)),
            "total_margin_balance": float(account_info.get("totalMarginBalance", 0)),
            "total_wallet_balance": float(account_info.get("totalWalletBalance", 0)),
            "liquidation_price": float(account_info.get("liquidationPrice", 0)) if account_info.get("liquidationPrice") else None
        }

    @retry_on_network_error(retries=3, delay=2.0)
    async def get_hedge_mode(self) -> bool:
        """Проверка, включен ли Hedge Mode (True - включен, False - One-Way)."""
        mode_info = await self.futures_client.futures_get_position_mode()
        return mode_info.get("dualSidePosition", False)

    @retry_on_network_error(retries=3, delay=2.0)
    async def get_free_balance(self) -> float:
        if not self.api_key or self.api_key == "YOUR_API_KEY":
            return 10000.0
        balances = await self.futures_client.futures_account_balance()
        usdt_balance = next((b["balance"] for b in balances if b["asset"] == "USDT"), 0.0)
        return float(usdt_balance)

    @retry_on_network_error(retries=3, delay=2.0)
    async def get_bnb_balance(self) -> float:
        """Получение баланса BNB на фьючерсном аккаунте."""
        if not self.api_key or self.api_key == "YOUR_API_KEY":
            return 0.0
        balances = await self.futures_client.futures_account_balance()
        bnb_balance = next((b["balance"] for b in balances if b["asset"] == "BNB"), 0.0)
        return float(bnb_balance)

    @retry_on_network_error(retries=3, delay=2.0)
    async def set_leverage(self, symbol: str, leverage: int):
        """Установка плеча для символа."""
        return await self.futures_client.futures_change_leverage(
            symbol=symbol,
            leverage=leverage
        )

    @retry_on_network_error(retries=3, delay=2.0)
    async def set_margin_type(self, symbol: str, margin_type: str):
        """Установка типа маржи (ISOLATED или CROSS)."""
        try:
            return await self.futures_client.futures_change_margin_type(
                symbol=symbol,
                marginType=margin_type
            )
        except BinanceAPIException as e:
            # Если тип маржи уже установлен, Бинанс вернет ошибку -4046 "No need to change margin type"
            if "No need to change margin type" in str(e):
                return None
            raise e

    @retry_on_network_error(retries=3, delay=2.0)
    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """Получение стакана ордеров (глубина 5-1000 уровней)."""
        return await self.futures_client.futures_order_book(
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
        # Исключаем reduceOnly из параметров, так как в Hedge Mode он вызывает ошибку -1106
        # В режиме хеджирования достаточно указать side и positionSide
        
        result = await self.futures_client.futures_create_order(
            **params
        )
        return result

    @retry_on_network_error(retries=3, delay=2.0)
    async def get_order_status(self, symbol: str, order_id: int) -> Dict:
        """Проверка статуса ордера."""
        return await self.futures_client.futures_get_order(
            symbol=symbol,
            orderId=order_id
        )

    @retry_on_network_error(retries=3, delay=2.0)
    async def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """Отмена ордера."""
        return await self.futures_client.futures_cancel_order(
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
        result = await self.futures_client.futures_create_order(
            symbol=symbol,
            side=side,
            type="LIMIT_MAKER",
            quantity=abs(qty),
            price=price,
            positionSide=position_side
        )
        return result

    @retry_on_network_error(retries=3, delay=2.0)
    async def get_order_trades(self, symbol: str, order_id: int) -> List[Dict]:
        """Получение списка сделок по конкретному ID ордера."""
        return await asyncio.to_thread(
            self.futures_client.futures_account_trades,
            symbol=symbol,
            orderId=order_id
        )
