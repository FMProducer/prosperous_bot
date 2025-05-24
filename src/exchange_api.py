import time
import logging
import gate_api
from gate_api.exceptions import ApiException, GateApiException
from typing import Dict, Optional

class ExchangeAPI:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        
        configuration = gate_api.Configuration(
            key = api_key,
            secret = api_secret
        )
        self.api_client = gate_api.ApiClient(configuration)
        self.spot_api = gate_api.SpotApi(self.api_client)
        self.time_offset = 0

    async def update_time_offset(self):
        try:
            server_time = await self.get_system_time()
            self.time_offset = server_time - int(time.time() * 1000)
            logging.info(f"Time offset updated: {self.time_offset} ms")
        except Exception as e:
            logging.error(f"Error updating time offset: {e}")

    async def make_request(self, api_method, *args, **kwargs):
        try:
            logging.info(f"Making request: method={api_method}, args={args}, kwargs={kwargs}")
            result = api_method(*args, **kwargs)
            logging.info(f"Request result: {result}")
            return result
        except GateApiException as ex:
            logging.error(f"Gate API Exception, label: {ex.label}, message: {ex.message}")
            raise
        except ApiException as e:
            logging.error(f"API Exception: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error in make_request: {e}", exc_info=True)
            raise
            
    async def get_system_time(self):
        try:
            result = await self.make_request(self.spot_api.get_system_time)
            return result.server_time  # Используем атрибут server_time
        except Exception as e:
            logging.error(f"Error getting system time: {e}", exc_info=True)
            raise

    async def get_current_prices(self):
        try:
            tickers = await self.make_request(self.spot_api.list_tickers)
            return {ticker.currency_pair: float(ticker.last) for ticker in tickers if ticker.currency_pair.endswith('_USDT')}
        except Exception as e:
            logging.error(f"Error fetching current prices: {e}", exc_info=True)
            raise

    async def get_wallet_balance(self, currency=None):
        try:
            balances = await self.make_request(self.spot_api.list_spot_accounts, currency=currency)
            if currency:
                return balances[0] if balances else None
            return {balance.currency: balance for balance in balances}
        except Exception as e:
            logging.error(f"Error fetching wallet balance: {e}", exc_info=True)
            raise

    async def get_current_price(self, symbol):
        try:
            tickers = await self.make_request(self.spot_api.list_tickers, currency_pair=symbol)
            if tickers and len(tickers) > 0:
                return float(tickers[0].last)
            else:
                logging.warning(f"No ticker data found for {symbol}")
                return None
        except Exception as e:
            logging.error(f"Error getting current price for {symbol}: {e}", exc_info=True)
            return None

    async def check_open_orders(self, symbol, side):
        try:
            open_orders = await self.make_request(self.spot_api.list_orders, currency_pair=symbol, status='open')
            return any(order.side.lower() == side.lower() for order in open_orders)
        except Exception as e:
            logging.error(f"Error checking open orders for {symbol}: {e}")
            raise

    async def cancel_all_open_orders(self, symbols):
        try:
            for symbol in symbols:
                open_orders = await self.make_request(self.spot_api.list_orders, currency_pair=symbol, status='open')
                for order in open_orders:
                    await self.make_request(self.spot_api.cancel_order, order_id=order.id, currency_pair=symbol)
                logging.info(f"Cancelled all open orders for {symbol}")
        except Exception as e:
            logging.error(f"Error cancelling orders: {e}")
            raise

    async def place_order(self, symbol, side, amount, price):
        try:
            order = gate_api.Order(amount=str(amount), price=str(price), side=side, currency_pair=symbol, time_in_force='ioc')
            result = await self.make_request(self.spot_api.create_order, order=order)
            logging.info(f"Order placed for {symbol}: {result}")
            return result
        except Exception as e:
            logging.error(f"Error placing order for {symbol}: {e}", exc_info=True)
            raise

    async def place_market_order(self, symbol, side, amount, is_value=False):
        try:
            if side == 'buy' and is_value:
                order = gate_api.Order(
                    currency_pair=symbol,
                    side=side,
                    amount=str(amount),
                    type='market',
                    time_in_force='ioc',
                    auto_borrow=False,
                    text='t-' + str(int(time.time() * 1000))
                )
            else:
                order = gate_api.Order(
                    currency_pair=symbol,
                    side=side,
                    amount=str(amount),
                    type='market',
                    time_in_force='ioc',
                    auto_borrow=False
                )
            
            logging.info(f"Placing market order: {order}")
            result = await self.make_request(self.spot_api.create_order, order)
            logging.info(f"Market order placed for {symbol}: {result}")
            return result.status == 'closed'
        except Exception as e:
            logging.error(f"Error placing market order for {symbol}: {e}", exc_info=True)
            return False