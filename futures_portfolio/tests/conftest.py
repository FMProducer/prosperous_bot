import sys
import os
import json
import subprocess
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

# Добавляем родительскую директорию (futures_portfolio/) в sys.path,
# чтобы тесты могли импортировать main, supervisor и т.д.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))


# ═══════════════════════════════════════════════════════════════════════
# SAFETY NET: обнаружение живой торговой системы
# ═══════════════════════════════════════════════════════════════════════

def _get_running_bot_names() -> list[str]:
    """Возвращает имена всех PM2-процессов типа real-*/paper-*."""
    try:
        # На Windows pm2 — это .cmd файл; shell=True нужен для его нахождения
        result = subprocess.run(
            ["pm2", "jlist"],
            capture_output=True,
            text=True,
            timeout=5,
            shell=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0:
            return []
        procs = json.loads(result.stdout)
        bots = [
            p["name"]
            for p in procs
            if p.get("pm2_env", {}).get("status") == "online"
            and p.get("name", "").startswith(("real-", "paper-", "supervisor"))
        ]
        return bots
    except Exception:
        # PM2 недоступен — считаем что система не запущена
        return []


def _detect_live_system() -> bool:
    """Проверяет, запущена ли торговая система (PM2 боты)."""
    bots = _get_running_bot_names()
    return len(bots) > 0


def pytest_addoption(parser):
    parser.addoption(
        "--ignore-safety",
        action="store_true",
        default=False,
        help="Пропустить проверку безопасности (тесты при живой системе)",
    )


def pytest_configure(config):
    """Проверка при конфигурации — ещё до сбора тестов."""
    if config.getoption("--ignore-safety"):
        return

    live = _detect_live_system()
    if live:
        bots = _get_running_bot_names()
        msg = (
            "\n"
            "═══════════════════════════════════════════════════════════════\n"
            "⚠️  ОБНАРУЖЕНА ЖИВАЯ ТОРГОВАЯ СИСТЕМА (PM2 активен)\n"
            f"   Активные процессы: {', '.join(bots)}\n"
            "\n"
            "   Тесты НЕ запущены для защиты реальных ботов.\n"
            "\n"
            "   Варианты:\n"
            "   1. Остановить систему:  pm2 stop all\n"
            "   2. Принудительно:       pytest --ignore-safety\n"
            "═══════════════════════════════════════════════════════════════\n"
        )
        # Используем pytest.exit чтобы остановить весь прогон
        pytest.exit(msg, returncode=1)


# ═══════════════════════════════════════════════════════════════════════
# ФИКСТУРЫ (mock-слой — каждый тест изолирован от реальности)
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_config():
    return {
        "binance_api_key": "test_key",
        "binance_api_secret": "test_secret",
        "telegram_token": "test_token",
        "telegram_chat_id": "test_chat_id",
        "initial_capital": 1000.0,
        "leverage": 1,
        "rebalance_threshold": 0.01,
        "rebalance_threshold_surplus": 0.01,
        "rebalance_threshold_deficit": 0.01,
        "tickers": ["BTCUSDT", "ETHUSDT"],
        "share_long": 40.0,
        "share_short": 40.0,
        "share_virt": 20.0,
        "min_cycles_for_rank": 10,
        "max_replace_per_cycle": 2,
        "tpv_safety_floor": 10.0,
        "limit_order_enabled": False,
        "limit_offset_pct": 0.01,
        "limit_timeout_sec": 60,
        "ticker_thresholds": {}
    }

@pytest.fixture
def mock_targets():
    return {
        "long": 40.0,
        "short": 40.0,
        "virt": 20.0
    }

@pytest.fixture
def mock_connector():
    connector = MagicMock()
    connector.client = MagicMock()
    connector.verify_connection = AsyncMock(return_value=True)
    connector.get_mark_prices = AsyncMock(return_value={"BTCUSDT": 50000.0, "ETHUSDT": 3000.0})
    connector.get_futures_balance = AsyncMock(return_value=1000.0)
    connector.get_futures_positions = AsyncMock(return_value=[])
    return connector

@pytest.fixture(autouse=True)
def mock_binance_client():
    """ГЛОБАЛЬНЫЙ мок: предотвращает любые реальные Binance API-вызовы."""
    with patch("binance.client.Client") as mock:
        yield mock
