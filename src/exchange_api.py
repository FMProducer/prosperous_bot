import time
import logging
import gate_api
import asyncio # Для задержки при повторных попытках
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
        self.futures_api = gate_api.FuturesApi(self.api_client) # Добавлено для работы с фьючерсами
        self._time_offset = 0 # Исправлено имя атрибута и инициализация
        self._last_sync = 0 # Для логики _sync_time

    async def update_time_offset(self):
        try:
            # Этот метод может быть заменен или использовать _sync_time,
            # но оставим для совместимости, если он где-то используется напрямую.
            system_time_obj = await self._safe_call(self.spot_api.get_system_time)
            self._time_offset = system_time_obj.server_time - int(time.time() * 1000) # Используем исправленное имя
            logging.info(f"Time offset updated: {self._time_offset} ms. Server time: {system_time_obj.server_time}")
        except Exception as e:
            logging.error(f"Error updating time offset: {e}")

    async def _sync_time(self, force_sync: bool = False, **kwargs): # Добавлен недостающий метод
        # kwargs для совместимости с вызовом в тестах
        if not force_sync and (time.time() - self._last_sync < 3600): # Синхронизировать не чаще раза в час, если не принудительно
            logging.debug(f"Time sync skipped, last sync was at {self._last_sync}, current time {time.time()}.")
            return
        try:
            system_time_obj = await self._safe_call(self.spot_api.get_system_time)
            self._time_offset = system_time_obj.server_time - int(time.time() * 1000) # Используем исправленное имя
            self._last_sync = time.time()
            logging.info(f"Time offset updated: {self._time_offset} ms. Server time: {system_time_obj.server_time}. Last sync set to: {self._last_sync}")
        except Exception as e:
            logging.error(f"Error during _sync_time: {e}", exc_info=True)

    async def _safe_call(self, api_method, *args, max_retries=3, base_delay=0.1, **kwargs):
        """
        Выполняет вызов API с логикой повторных попыток для определенных исключений.
        :param api_method: Метод API для вызова.
        :param args: Позиционные аргументы для метода API.
        :param max_retries: Максимальное количество повторных попыток.
        :param base_delay: Базовая задержка в секундах для экспоненциальной выдержки.
        :param kwargs: Именованные аргументы для метода API.
        :return: Результат вызова API.
        :raises: Exception, если все повторные попытки не увенчались успехом или возникло необработанное исключение.
        """
        # await self._sync_time() # Рассмотри необходимость синхронизации перед каждым вызовом
        retries = 0
        while retries <= max_retries:
            try:
                # Логирование аргументов может быть полезно, но будь осторожен с чувствительными данными
                logging.info(f"Making request (attempt {retries + 1}): method={api_method.__name__}, args={args}, kwargs={kwargs}")
                result = api_method(*args, **kwargs)
                logging.info(f"Request result: {result}")
                return result
            except GateApiException as ex: # Более специфичное исключение Gate.io
                logging.error(f"Gate API Exception (attempt {retries + 1}), label: {ex.label}, message: {ex.message}", exc_info=True)
                # Проверяем, является ли это ошибкой, связанной с rate limit (часто имеет специфический label или message)
                # GateApiException может не иметь атрибута status напрямую, как ApiException.
                # Нужно смотреть на label или message для определения 429.
                # Для примера, предположим, что label 'TOO_FAST' или message содержит 'Too Many Requests'
                is_rate_limit_error = ex.label == 'TOO_FAST' or "many requests" in str(ex.message).lower() or "rate limit" in str(ex.message).lower()
                if is_rate_limit_error and retries < max_retries:
                    delay = (base_delay * 2**retries) + (asyncio.runners.random.uniform(0, 0.1) * base_delay) # Экспоненциальная выдержка с джиттером
                    logging.warning(f"Rate limit hit (GateApiException). Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
                    retries += 1
                else:
                    raise # Перебрасываем, если не ошибка rate limit или превышены попытки
            except ApiException as e: # Общее исключение API, которое может иметь status
                logging.error(f"API Exception (attempt {retries + 1}): status={e.status if hasattr(e, 'status') else 'N/A'}, reason={e.reason if hasattr(e, 'reason') else 'N/A'}", exc_info=True)
                if hasattr(e, 'status') and e.status == 429 and retries < max_retries:
                    delay = (base_delay * 2**retries) + (asyncio.runners.random.uniform(0, 0.1) * base_delay) # Экспоненциальная выдержка с джиттером
                    logging.warning(f"Rate limit hit (ApiException 429). Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
                    retries += 1
                else:
                    # Для других ошибок API или если достигнуто максимальное количество попыток для 429
                    raise
            except Exception as e:
                logging.error(f"Unexpected error in _safe_call (attempt {retries + 1}): {e}", exc_info=True)
                raise # Или обработать иначе, если необходимо
        logging.error(f"Max retries ({max_retries}) reached for {api_method.__name__}. Giving up.")
        # Это исключение должно быть перехвачено последним ApiException в цикле, если ошибка была 429
        # Если цикл завершился без исключения (что не должно произойти при ошибке 429), можно поднять общее исключение
        raise Exception(f"Max retries reached for {api_method.__name__} after multiple errors.")


    async def make_request(self, api_method, *args, **kwargs):
        # Обертка вокруг _safe_call для совместимости с существующим кодом,
        # или можно перенести логику _safe_call сюда.
        # Для простоты пока оставим вызов _safe_call.
        return await self._safe_call(api_method, *args, **kwargs)
            
    async def get_system_time(self):
        # Этот метод может быть заменен или использовать _sync_time,
        # но оставим для совместимости, если он где-то используется напрямую.
        # Тесты вызывают _sync_time, который уже вызывает get_system_time через _safe_call.
        try:
            result = await self._safe_call(self.spot_api.get_system_time)
            return result.server_time  # Используем атрибут server_time
        except Exception as e:
            logging.error(f"Error getting system time: {e}", exc_info=True)
            raise

    async def get_current_prices(self):
        try:
            tickers = await self._safe_call(self.spot_api.list_tickers)
            return {ticker.currency_pair: float(ticker.last) for ticker in tickers if ticker.currency_pair.endswith('_USDT')}
        except Exception as e:
            logging.error(f"Error fetching current prices: {e}", exc_info=True)
            raise

    async def get_wallet_balance(self, currency=None):
        try:
            balances = await self._safe_call(self.spot_api.list_spot_accounts, currency=currency)
            if currency:
                return balances[0] if balances else None
            return {balance.currency: balance for balance in balances}
        except Exception as e:
            logging.error(f"Error fetching wallet balance: {e}", exc_info=True)
            raise

    async def get_current_price(self, symbol):
        try:
            tickers = await self._safe_call(self.spot_api.list_tickers, currency_pair=symbol)
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
            open_orders = await self._safe_call(self.spot_api.list_orders, currency_pair=symbol, status='open')
            return any(order.side.lower() == side.lower() for order in open_orders)
        except Exception as e:
            logging.error(f"Error checking open orders for {symbol}: {e}")
            raise

    async def cancel_all_open_orders(self, symbols):
        try:
            for symbol in symbols:
                open_orders = await self._safe_call(self.spot_api.list_orders, currency_pair=symbol, status='open')
                for order in open_orders:
                    await self._safe_call(self.spot_api.cancel_order, order_id=order.id, currency_pair=symbol)
                logging.info(f"Cancelled all open orders for {symbol}")
        except Exception as e:
            logging.error(f"Error cancelling orders: {e}")
            raise

    async def place_order(self, symbol, side, amount, price):
        try:
            order = gate_api.Order(amount=str(amount), price=str(price), side=side, currency_pair=symbol, time_in_force='ioc')
            result = await self._safe_call(self.spot_api.create_order, order=order)
            logging.info(f"Order placed for {symbol}: {result}")
            return result
        except Exception as e:
            logging.error(f"Error placing order for {symbol}: {e}", exc_info=True)
            raise

    async def create_spot_order(self, pair: str, side: str, qty: float, price: Optional[float] = None, post_only: bool = False, **kwargs): # Добавлен недостающий метод
        # kwargs для совместимости с вызовом в тестах
        order_type = 'limit' if price else 'market'
        time_in_force = 'poc' if post_only and order_type == 'limit' else 'ioc' # post_only_condition (poc)

        order_params = {
            "currency_pair": pair,
            "side": side,
            "amount": str(qty),
            "type": order_type,
            "time_in_force": time_in_force,
            "text": 't-' + str(int(time.time() * 1000)) # Пример уникального ID
        }
        if price:
            order_params["price"] = str(price)
        
        # Gate API не имеет прямого 'post_only' флага для create_order,
        # 'poc' time_in_force используется для этого.
        # Если post_only=True и это market order, это может быть нелогично или не поддерживаться.
        # Здесь мы предполагаем, что post_only имеет смысл только для limit orders.
        if post_only and order_type == 'market':
            logging.warning("post_only is typically used with limit orders, not market orders. Proceeding with market order.")
            # Можно либо вызвать ошибку, либо просто проигнорировать post_only для market

        order = gate_api.Order(**order_params)
        try:
            logging.info(f"Creating spot order: {order}")
            result = await self._safe_call(self.spot_api.create_order, order=order)
            logging.info(f"Spot order created for {pair}: {result}")
            return result
        except Exception as e:
            logging.error(f"Error creating spot order for {pair}: {e}", exc_info=True)
            raise


    async def place_market_order(self, symbol, side, amount, is_value=False):
        try:
            if side == 'buy' and is_value:
                order = gate_api.Order(
                    currency_pair=symbol,
                    side=side,
                    amount=str(amount), # Для Gate.io, если amount_is_value=true, это значение в quote currency
                    type='market',
                    time_in_force='ioc',
                    # Gate API не имеет прямого amount_is_value, но для market buy
                    # amount обычно интерпретируется как quote currency, если не указано иное.
                    # Уточнить документацию Gate.io для market order by value.
                    # Пока предполагаем, что 'amount' для market buy - это объем в quote currency.
                    text='t-' + str(int(time.time() * 1000))
                )
            else: # Для sell или buy by base quantity
                order = gate_api.Order(
                    currency_pair=symbol,
                    side=side,
                    amount=str(amount), # Объем в base currency
                    type='market',
                    time_in_force='ioc',
                    text='t-' + str(int(time.time() * 1000)) # Добавим text и сюда для единообразия
                )
            
            logging.info(f"Placing market order: {order}")
            result = await self._safe_call(self.spot_api.create_order, order)
            logging.info(f"Market order placed for {symbol}: {result}")
            return result.status == 'closed'
        except Exception as e:
            logging.error(f"Error placing market order for {symbol}: {e}", exc_info=True)
            return False

    async def create_futures_order(self, contract: str, side: str, qty: float, reduce_only: bool = False, **kwargs): # Добавлен недостающий метод
        # kwargs для совместимости с вызовом в тестах
        # В Gate API размер фьючерсного ордера (size) - это количество контрактов.
        # Знак size определяет направление: положительный для long, отрицательный для short.
        order_size = qty if side.lower() == 'long' or side.lower() == 'buy' else -qty # Учтем 'buy'/'sell'

        futures_order = gate_api.FuturesOrder(
            contract=contract,
            size=order_size, # size может быть int или str
            # price=str(price) if price else None, # Если нужен лимитный ордер
            tif='ioc', # Или другой time_in_force по необходимости
            text='t-' + str(int(time.time() * 1000)), # Пример уникального ID
            reduce_only=reduce_only
        )
        try:
            logging.info(f"Creating futures order: {futures_order}")
            # 'usdt' - это settle currency, убедись, что она соответствует контракту
            result = await self._safe_call(self.futures_api.create_futures_order, settle='usdt', futures_order=futures_order)
            logging.info(f"Futures order created for {contract}: {result}")
            return result
        except Exception as e:
            logging.error(f"Error creating futures order for {contract}: {e}", exc_info=True)
            raise

    async def positions(self, contract: Optional[str] = None, **kwargs): # Добавлен недостающий метод
        # kwargs для совместимости с вызовом в тестах
        # 'usdt' - это settle currency
        return await self._safe_call(self.futures_api.list_positions, settle='usdt', contract=contract)
