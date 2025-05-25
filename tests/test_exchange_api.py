import unittest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from prosperous_bot.exchange_gate import ExchangeAPI
import gate_api
import time

def create_mock_balance(currency, available, locked):
    balance = gate_api.SpotAccount(currency=currency, available=available, locked=locked)
    return balance

def create_mock_ticker(currency_pair, last_price):
    ticker = gate_api.Ticker(currency_pair=currency_pair, last=last_price)
    return ticker

def create_mock_order(order_id, side, currency_pair, amount):
    order = gate_api.Order(id=order_id, side=side, currency_pair=currency_pair, amount=str(amount))
    return order

class TestExchangeAPI(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.exchange_api = ExchangeAPI("test_key", "test_secret")

    @patch('gate_api.SpotApi.get_system_time')
    async def test_get_system_time(self, mock_get_system_time):
        mock_result = Mock()
        mock_result.server_time = 1000000000000
        mock_get_system_time.return_value = mock_result
        server_time = await self.exchange_api.get_system_time()
        self.assertEqual(server_time, 1000000000000)

    @patch('gate_api.SpotApi.get_system_time')
    async def test_update_time_offset(self, mock_get_system_time):
        mock_result = Mock()
        mock_result.server_time = int(time.time() * 1000) + 5000  # Simulate server time 5 seconds ahead
        mock_get_system_time.return_value = mock_result
        await self.exchange_api.update_time_offset()
        self.assertNotEqual(self.exchange_api.time_offset, 0)
        self.assertGreater(self.exchange_api.time_offset, 0)

    @patch('gate_api.SpotApi.list_spot_accounts')
    async def test_get_wallet_balance(self, mock_list_spot_accounts):
        mock_balance = create_mock_balance('BTC', '1.0', '0.5')
        mock_list_spot_accounts.return_value = [mock_balance]
        balance = await self.exchange_api.get_wallet_balance('BTC')
        self.assertEqual(balance.currency, 'BTC')
        self.assertEqual(balance.available, '1.0')
        self.assertEqual(balance.locked, '0.5')

    @patch('gate_api.SpotApi.list_tickers')
    async def test_get_current_price(self, mock_list_tickers):
        mock_ticker = create_mock_ticker('BTC_USDT', '50000')
        mock_list_tickers.return_value = [mock_ticker]
        price = await self.exchange_api.get_current_price('BTC_USDT')
        self.assertEqual(price, 50000.0)

        mock_list_tickers.return_value = []  # Simulate no ticker data
        price = await self.exchange_api.get_current_price('BTC_USDT')
        self.assertIsNone(price)

    @patch('gate_api.SpotApi.list_orders')
    async def test_check_open_orders(self, mock_list_orders):
        mock_order1 = create_mock_order('1', 'buy', 'BTC_USDT', 1.0)
        mock_order2 = create_mock_order('2', 'sell', 'BTC_USDT', 2.0)
        mock_list_orders.return_value = [mock_order1, mock_order2]
        has_open_orders = await self.exchange_api.check_open_orders('BTC_USDT', 'buy')
        self.assertTrue(has_open_orders)

    @patch('gate_api.SpotApi.list_orders')
    @patch('gate_api.SpotApi.cancel_order')
    async def test_cancel_all_open_orders(self, mock_cancel_order, mock_list_orders):
        mock_order1 = create_mock_order('1', 'buy', 'BTC_USDT', 1.0)
        mock_order2 = create_mock_order('2', 'sell', 'BTC_USDT', 2.0)
        mock_list_orders.return_value = [mock_order1, mock_order2]
        mock_cancel_order.return_value = None
        await self.exchange_api.cancel_all_open_orders(['BTC_USDT'])
        self.assertEqual(mock_cancel_order.call_count, 2)

    @patch('gate_api.SpotApi.create_order')
    async def test_place_order(self, mock_create_order):
        mock_result = create_mock_order('1', 'buy', 'BTC_USDT', 1.0)
        mock_result.status = 'open'
        mock_create_order.return_value = mock_result
        result = await self.exchange_api.place_order('BTC_USDT', 'buy', 1, 50000)
        self.assertEqual(result.id, '1')
        self.assertEqual(result.status, 'open')

    @patch('gate_api.SpotApi.create_order')
    async def test_place_market_order(self, mock_create_order):
        mock_result = create_mock_order('1', 'buy', 'BTC_USDT', 1.0)
        mock_result.status = 'closed'
        mock_create_order.return_value = mock_result
        result = await self.exchange_api.place_market_order('BTC_USDT', 'buy', 1)
        self.assertTrue(result)

        mock_result.status = 'open'  # Simulate order not filled immediately
        mock_create_order.return_value = mock_result
        result = await self.exchange_api.place_market_order('BTC_USDT', 'buy', 1)
        self.assertFalse(result)

    @patch('gate_api.SpotApi.create_order')
    async def test_place_market_order_with_value(self, mock_create_order):
        mock_result = create_mock_order('1', 'buy', 'BTC_USDT', 100)
        mock_result.status = 'closed'
        mock_create_order.return_value = mock_result
        result = await self.exchange_api.place_market_order('BTC_USDT', 'buy', 100, is_value=True)
        self.assertTrue(result)
        args, kwargs = mock_create_order.call_args
        self.assertEqual(args[0].amount, '100')
        self.assertEqual(args[0].type, 'market')
        self.assertTrue(args[0].text.startswith('t-'))

if __name__ == '__main__':
    asyncio.run(unittest.main())