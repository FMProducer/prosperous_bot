import pytest
from unittest.mock import Mock

from futures_portfolio.calculator import PortfolioCalculator


@pytest.fixture
def sample_data():
    # Позиции: 0.5 BTC, 0.2 ETH
    positions = {
        "BTCUSDT": 0.5,
        "ETHUSDT": 0.2,
        "SOLUSDT": 0.0,
    }
    # Спот-цены
    spot_prices = {
        "BTCUSDT": 60000.0,
        "ETHUSDT": 3000.0,
        "SOLUSDT": 150.0,
    }
    # Свободный баланс USDT
    free_balance = 1000.0
    return positions, spot_prices, free_balance


@pytest.fixture
def base_ticker_config():
    return "BTCUSDT"


def test_calculate_position_value(sample_data, base_ticker_config):
    positions, spot_prices, free_balance = sample_data
    calc = PortfolioCalculator(positions, spot_prices, free_balance, base_ticker=base_ticker_config)

    # Проверяем стоимость позиции
    value = calc.calculate_position_value(f"{base_ticker_config}USDT")
    assert value == pytest.approx(0.5 * 60000.0)  # 30000.0


def test_total_portfolio_value(sample_data, base_ticker_config):
    positions, spot_prices, free_balance = sample_data
    calc = PortfolioCalculator(positions, spot_prices, free_balance, base_ticker=base_ticker_config)

    total = calc.total_portfolio_value()
    # Позиции: 0.5*60000 + 0.2*3000 = 30000 + 600 = 30600
    # + free_balance 1000 = 31600
    assert total == pytest.approx(31600.0)


def test_current_shares(sample_data, base_ticker_config):
    positions, spot_prices, free_balance = sample_data
    calc = PortfolioCalculator(positions, spot_prices, free_balance, base_ticker=base_ticker_config)

    shares = calc.current_shares()
    # Общая стоимость 31600, доля BTC = 30000/31600, ETH = 600/31600
    assert pytest.approx(shares[f"{base_ticker_config}USDT"], 0.0001) == 30000.0 / 31600.0
    assert pytest.approx(shares["ETHUSDT"], 0.0001) == 600.0 / 31600.0
    # SOL отсутствует в позициях, доля должна быть 0
    assert shares.get("SOLUSDT", 0.0) == 0.0


def test_calculate_deviations(sample_data, base_ticker_config):
    positions, spot_prices, free_balance = sample_data
    calc = PortfolioCalculator(positions, spot_prices, free_balance, base_ticker=base_ticker_config)

    # Целевые доли
    targets = {
        f"{base_ticker_config}_LONG": 0.4,
        f"{base_ticker_config}_SHORT": 0.4,
        "VIRTUAL": 0.2
    }
    threshold = 0.02

    deviations = calc.calculate_deviations(targets, threshold)

    # Проверяем, что возвращается список и есть нужные поля
    assert isinstance(deviations, list)
    for dev in deviations:
        assert "symbol" in dev
        assert "deviation" in dev
        assert "direction" in dev
        assert "current_value" in dev
        assert "target_value" in dev
        assert "current_share" in dev
        assert "target_share" in dev

    # Должны быть отклонения для BTC и ETH (предположим, что текущие доли отличаются от целей более чем на 2%)
    # В этом простом примере проверяем, что для BTC и ETH есть записи
    symbols = [dev["symbol"] for dev in deviations]
    assert f"{base_ticker_config}_LONG" in symbols
    assert f"{base_ticker_config}_SHORT" in symbols


def test_calculate_position_value_eth(sample_data, base_ticker_config):
    """Тест для проверки расчета с ETH как базовым тикером."""
    positions, spot_prices, free_balance = sample_data
    # Используем ETH как базовый тикер
    eth_calc = PortfolioCalculator(positions, spot_prices, free_balance, base_ticker="ETHUSDT")
    
    # Проверяем стоимость ETH позиции
    value = eth_calc.calculate_position_value("ETHUSDT")
    assert value == pytest.approx(0.2 * 3000.0)  # 600.0