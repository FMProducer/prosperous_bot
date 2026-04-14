import os
from binance.client import Client
from binance.exceptions import BinanceAPIException
import asyncio
from typing import Dict, List

class BinanceConnector:
    def __init__(self, api_key: str, secret_key: str, testnet: bool = True):
        self.testnet = testnet
        self.client = Client(api_key, secret_key, testnet=testnet)
        # In python-binance, all futures methods are on the main Client
        self.futures_client = self.client

    async def get_positions(self) -> Dict[str, float]:
        """Получение текущих фьючерсных позиций (symbol: qty)."""
        try:
            positions = await asyncio.to_thread(self.futures_client.futures_position_information)
            # Пример: [{'symbol': 'BTCUSDT', 'positionAmt': '0.5', ...}, ...]
            result = {}
            for pos in positions:
                symbol = pos["symbol"]
                qty = float(pos["positionAmt"])
                if qty != 0:
                    result[symbol] = qty
            return result
        except BinanceAPIException as e:
            raise Exception(f"Binance API error in get_positions: {e}")

    async def get_spot_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Получение текущих спот-цен (symbol: price)."""
        try:
            # Binance Spot API: /api/v3/ticker/price
            prices = await asyncio.to_thread(self.client.get_all_tickers)
            result = {t["symbol"]: float(t["price"]) for t in prices}
            return {sym: result.get(sym) for sym in tickers}
        except BinanceAPIException as e:
            raise Exception(f"Binance API error in get_spot_prices: {e}")

    async def get_exchange_info(self) -> Dict:
        """Получение информации о бирже (фильтры, точность)."""
        try:
            return await asyncio.to_thread(self.futures_client.futures_exchange_info)
        except BinanceAPIException as e:
            raise Exception(f"Binance API error in get_exchange_info: {e}")

    async def get_free_balance(self) -> float:
        """Получение свободного баланса USDT."""
        try:
            # get_asset_balance(asset='USDT') returns {'asset': 'USDT', 'free': '...', 'locked': '...'}
            balance = await asyncio.to_thread(self.client.get_asset_balance, asset='USDT')
            if balance:
                return float(balance.get("free", 0.0))
            return 0.0
        except BinanceAPIException as e:
            raise Exception(f"Binance API error in get_free_balance: {e}")
