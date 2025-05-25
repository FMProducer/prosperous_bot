import pytest
import pytest_asyncio
import os
from prosperous_bot.exchange_gate import ExchangeAPI
# Assuming your PortfolioManager is in a file named portfolio_manager.py in src
# from prosperous_bot.portfolio_manager import PortfolioManager # Uncomment if you have this and need it in conftest

# Фикстура для PortfolioManager, если она нужна глобально
# @pytest_asyncio.fixture
# async def portfolio_manager_fixture(mocker):
#     # Настройте моки для зависимостей PortfolioManager, если необходимо
#     spot_api_mock = mocker.AsyncMock()
#     futures_api_mock = mocker.AsyncMock()
#     pm = PortfolioManager(spot_api_mock, futures_api_mock)
#     return pm

@pytest_asyncio.fixture
async def exch(monkeypatch, mocker):
    monkeypatch.setenv("GATE_KEY", "stub")
    monkeypatch.setenv("GATE_SECRET", "stub")

    # Подменяем __init__ SDK, чтобы не открывал соединение
    mocker.patch("gate_api.SpotApi.__init__", return_value=None)
    mocker.patch("gate_api.FuturesApi.__init__", return_value=None)
    
    # Используем установленные переменные окружения
    api_key = os.getenv("GATE_KEY")
    api_secret = os.getenv("GATE_SECRET")
    api = ExchangeAPI(api_key=api_key, api_secret=api_secret)
    yield api
    # Код для очистки, если нужен, после yield
