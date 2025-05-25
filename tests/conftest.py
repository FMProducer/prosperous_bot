# tests/conftest.py
import pytest_asyncio, pytest, os, asyncio, contextlib
from exchange_api import ExchangeAPI
from unittest import mock

# фикстура теперь async: создаём объект внутри работающего цикла
@pytest_asyncio.fixture
async def exch(monkeypatch, mocker):
    monkeypatch.setenv("GATE_KEY", "stub")
    monkeypatch.setenv("GATE_SECRET", "stub")

    # Подменяем __init__ SDK, чтобы не открывал соединение
    mocker.patch("gate_api.SpotApi.__init__", return_value=None)
    mocker.patch("gate_api.FuturesApi.__init__", return_value=None)

    api = ExchangeAPI()
    yield api

    # корректно закрываем сессию, если она была создана
    if hasattr(api, "session"):
        with contextlib.suppress(Exception):
            await api.session.close()