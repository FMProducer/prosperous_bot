import unittest
import asyncio
import time
from unittest.mock import MagicMock, patch, Mock
import gate_api
from prosperous_bot.exchange_gate import ExchangeAPI

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
        mock_result.server_time = int(time.time() * 1000) + 5000
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

    @patch('gate_api.SpotApi.list_spot_accounts')
    async def test_get_wallet_balance_not_found(self, mock_list_spot_accounts):
        mock_list_spot_accounts.return_value = []
        balance = await self.exchange_api.get_wallet_balance('NONEXISTENT')
        self.assertEqual(balance.currency, 'NONEXISTENT')
        self.assertEqual(balance.available, "0")

    @patch('gate_api.FuturesApi.list_positions')
    async def test_positions(self, mock_list_positions):
        mock_position = MagicMock()
        mock_list_positions.return_value = [mock_position]
        res = await self.exchange_api.positions()
        self.assertEqual(len(res), 1)
        mock_position.contract = "BTC_USDT"
        res = await self.exchange_api.positions(contract="BTC_USDT")
        self.assertEqual(len(res), 1)
        res = await self.exchange_api.positions(contract="ETH_USDT")
        self.assertEqual(len(res), 0)

    @patch('gate_api.SpotApi.list_tickers')
    async def test_get_current_price(self, mock_list_tickers):
        mock_ticker = create_mock_ticker('BTC_USDT', '20000.0')
        mock_list_tickers.return_value = [mock_ticker]
        price = await self.exchange_api.get_current_price('BTC_USDT')
        self.assertEqual(price, 20000.0)

    @patch('gate_api.SpotApi.create_order')
    async def test_create_spot_order(self, mock_create_order):
        mock_res = MagicMock()
        mock_create_order.return_value = mock_res
        await self.exchange_api.create_spot_order('BTC_USDT', 'buy', 1, 20000, post_only=True)
        args = mock_create_order.call_args[0][0]
        self.assertEqual(args.type, 'limit')
        self.assertEqual(args.price, '20000')
        self.assertEqual(args.time_in_force, 'poc')
        await self.exchange_api.create_spot_order('BTC_USDT', 'buy', 1)
        args = mock_create_order.call_args[0][0]
        self.assertEqual(args.type, 'market')
        self.assertIsNone(args.price)

    @patch('gate_api.FuturesApi.create_futures_order')
    async def test_create_futures_order(self, mock_create_futures_order):
        mock_res = MagicMock()
        mock_create_futures_order.return_value = mock_res
        await self.exchange_api.create_futures_order('BTC_USDT', 'OPEN_LONG', 1)
        args = mock_create_futures_order.call_args[0][1]
        self.assertEqual(args.size, 1)
        self.assertFalse(args.reduce_only)
        await self.exchange_api.create_futures_order('BTC_USDT', 'CLOSE_SHORT', 1)
        args = mock_create_futures_order.call_args[0][1]
        self.assertEqual(args.size, -1)
        self.assertTrue(args.reduce_only)
        with self.assertRaises(ValueError):
            await self.exchange_api.create_futures_order('BTC_USDT', 'INVALID', 1)

    @patch('gate_api.SpotApi.list_orders')
    @patch('gate_api.SpotApi.cancel_order')
    async def test_cancel_all_open_orders(self, mock_cancel, mock_list):
        mock_order = MagicMock(id='123')
        mock_list.return_value = [mock_order]
        await self.exchange_api.cancel_all_open_orders(['BTC_USDT'])
        mock_cancel.assert_called_once()

    @patch('gate_api.SpotApi.list_orders')
    async def test_check_open_orders(self, mock_list):
        mock_order = MagicMock(side='buy')
        mock_list.return_value = [mock_order]
        res = await self.exchange_api.check_open_orders('BTC_USDT', 'buy')
        self.assertTrue(res)
        res = await self.exchange_api.check_open_orders('BTC_USDT', 'sell')
        self.assertFalse(res)

    @patch('gate_api.SpotApi.list_tickers')
    @patch('gate_api.SpotApi.create_order')
    async def test_place_market_order(self, mock_create, mock_list):
        mock_list.return_value = [create_mock_ticker('BTC_USDT', '20000.0')]
        mock_res = MagicMock(status='closed')
        mock_create.return_value = mock_res
        res = await self.exchange_api.place_market_order('BTC_USDT', 'buy', 20000)
        self.assertTrue(res)
        args = mock_create.call_args[0][0]
        self.assertEqual(args.amount, '1')

    async def test_safe_call_async_retry(self):
        async def mock_fn():
            raise gate_api.exceptions.ApiException(status=429)

        with patch('asyncio.sleep', return_value=None):
            with self.assertRaises(RuntimeError):
                await self.exchange_api._safe_call(mock_fn)

    async def test_safe_call_generic_retry(self):
        async def mock_fn():
            raise Exception("Generic error")

        with patch('asyncio.sleep', return_value=None):
            with self.assertRaises(Exception):
                await self.exchange_api._safe_call(mock_fn)

