import os
from binance.client import Client
from binance.exceptions import BinanceAPIException
import asyncio
from typing import Dict, List

class BinanceConnector:
    def __init__(self, api_key: str, secret_key: str, testnet: bool = True):
        self.testnet = testnet
        self.client = Client(api_key, secret_key, testnet=testnet)
        self.futures_client = self.client

    async def get_positions(self) -> Dict[str, float]:
        """Получение фьючерсных позиций с разделением на LONG и SHORT."""
        try:
            positions = await asyncio.to_thread(self.futures_client.futures_position_information)
            result = {}
            for pos in positions:
                qty = float(pos["positionAmt"])
                side = pos["positionSide"] # LONG, SHORT or BOTH
                symbol = pos["symbol"]
                
                if qty != 0:
                    # Создаем уникальный ключ для каждой стороны
                    key = f"{symbol}_{side}" if side != "BOTH" else symbol
                    result[key] = qty
            return result
        except BinanceAPIException as e:
            raise Exception(f"Binance API error in get_positions: {e}")

    async def get_spot_prices(self, tickers: List[str]) -> Dict[str, float]:
        try:
            prices = await asyncio.to_thread(self.client.get_all_tickers)
            result = {t["symbol"]: float(t["price"]) for t in prices}
            # Для цен нам не важна сторона, используем базовый тикер
            return {sym: result.get(sym.split('_')[0]) for sym in tickers}
        except BinanceAPIException as e:
            raise Exception(f"Binance API error in get_spot_prices: {e}")

    async def get_exchange_info(self) -> Dict:
        try:
            return await asyncio.to_thread(self.futures_client.futures_exchange_info)
        except BinanceAPIException as e:
            raise Exception(f"Binance API error in get_exchange_info: {e}")

    async def get_free_balance(self) -> float:
        try:
            # Для фьючерсов лучше использовать futures_account_balance
            balances = await asyncio.to_thread(self.futures_client.futures_account_balance)
            usdt_balance = next((b["balance"] for b in balances if b["asset"] == "USDT"), 0.0)
            return float(usdt_balance)
        except BinanceAPIException as e:
            raise Exception(f"Binance API error in get_free_balance: {e}")
