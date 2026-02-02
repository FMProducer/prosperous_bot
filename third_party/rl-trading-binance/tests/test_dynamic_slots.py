# test_dynamic_slots.py
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path

# Add project root to sys.path for imports to work
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Mock freqtrade before importing strategy
class MockIStrategy:
    def __init__(self, config, **kwargs):
        self.config = config

mock_strategy = MagicMock()
mock_strategy.IStrategy = MockIStrategy
mock_strategy.DecimalParameter = MagicMock
mock_strategy.IntParameter = MagicMock

mock_persistence = MagicMock()
mock_persistence.Trade = MagicMock()

sys.modules['freqtrade'] = MagicMock()
sys.modules['freqtrade.strategy'] = mock_strategy
sys.modules['freqtrade.persistence'] = mock_persistence

@pytest.fixture
def strategy_config():
    return {
        'max_open_trades': 100,  # ИСПОЛЬЗУЕМ ГЛОБАЛЬНЫЙ ПАРАМЕТР
        'cpu_threads': 4,
        'rl_enable_long_1': True,
        'rl_enable_long_2': True,
        'rl_enable_short_1': True,
        'rl_enable_short_2': True,
        'dynamic_slots': {
            'enabled': True,
            'min_slots_per_side': 10,
            'update_interval_sec': 300
            # БЕЗ total_slots — берётся из max_open_trades
        },
        'rl_ensemble': {
            'epsilon_threshold': 0.15,
            'q_normalization': {}
        }
    }

@pytest.fixture
def strategy(strategy_config):
    from user_data.strategies.CustomD3QNStrategy4z import CustomD3QNStrategy4z

    with patch('user_data.strategies.CustomD3QNStrategy4z.D3QN_PER_Agent'), \
         patch.object(CustomD3QNStrategy4z, '_find_config_file', return_value=Path("dummy_cfg.py")), \
         patch.object(CustomD3QNStrategy4z, '_load_py_config', return_value=MagicMock()), \
         patch.object(CustomD3QNStrategy4z, '_load_weights'):
        strat = CustomD3QNStrategy4z(strategy_config)
        return strat

def test_total_slots_from_config(strategy):
    """total_slots должен браться из max_open_trades"""
    assert strategy.total_slots == 100
    assert strategy.max_long_slots + strategy.max_short_slots == 100

def test_pnl_calculation_via_freqtrade(strategy):
    """Проверка получения PnL через calc_profit (USDT)"""
    mock_trades = [
        Mock(is_short=False, close_rate_requested=105, calc_profit=lambda rate: 50.0),  # +50 USDT
        Mock(is_short=True, close_rate_requested=95, calc_profit=lambda rate: -20.0),   # -20 USDT
    ]

    with patch('freqtrade.persistence.Trade.get_open_trades', return_value=mock_trades):
        pnl_long, pnl_short = strategy._get_pnl_from_freqtrade()

    assert pnl_long == 50.0
    assert pnl_short == -20.0

def test_slot_allocation_long_profitable(strategy):
    """При прибыльных лонгах и убыточных шортах: 70% слотов лонгам"""
    with patch.object(strategy, '_get_pnl_from_freqtrade', return_value=(250.0, -100.0)):
        strategy._update_slot_allocation(datetime.now())

    assert strategy.max_long_slots > strategy.max_short_slots
    assert strategy.max_long_slots >= 60  # ~70% от 100 минус минимум

def test_slot_allocation_short_profitable(strategy):
    """При прибыльных шортах и убыточных лонгах: 70% слотов шортам"""
    with patch.object(strategy, '_get_pnl_from_freqtrade', return_value=(-150.0, 300.0)):
        strategy._update_slot_allocation(datetime.now())

    assert strategy.max_short_slots > strategy.max_long_slots
    assert strategy.max_short_slots >= 60

def test_slot_allocation_both_negative(strategy):
    """При убытках по обеим сторонам: равное распределение"""
    with patch.object(strategy, '_get_pnl_from_freqtrade', return_value=(-50.0, -80.0)):
        strategy._update_slot_allocation(datetime.now())

    assert abs(strategy.max_long_slots - 50) <= 1
    assert abs(strategy.max_short_slots - 50) <= 1

def test_slot_allocation_both_negative_extreme(strategy):
    """При сильно разных убытках всё равно должно быть 50/50"""
    with patch.object(strategy, '_get_pnl_from_freqtrade', return_value=(-500.0, -50.0)):
        strategy._update_slot_allocation(datetime.now())

    # Оба убыточны → равное распределение, независимо от размера убытков
    assert abs(strategy.max_long_slots - 50) <= 1
    assert abs(strategy.max_short_slots - 50) <= 1

def test_min_slots_guarantee(strategy):
    """Минимальная гарантия соблюдается даже при экстремальном PnL"""
    with patch.object(strategy, '_get_pnl_from_freqtrade', return_value=(1000.0, -500.0)):
        strategy._update_slot_allocation(datetime.now())

    assert strategy.max_long_slots >= strategy.min_slots_per_side
    assert strategy.max_short_slots >= strategy.min_slots_per_side

def test_disabled_dynamic_slots_uses_legacy(strategy):
    """При отключённой системе используются фиксированные лимиты"""
    strategy.dynamic_slots_enabled = False

    # В confirm_trade_entry должно установиться 50/50
    with patch('freqtrade.persistence.Trade.get_trades') as mock:
        mock.return_value.all.return_value = []

        # Симуляция части confirm_trade_entry
        current_time = datetime.now()
        with patch.object(strategy, 'dp', create=True):
            strategy.confirm_trade_entry("BTC/USDT", "limit", 1.0, 50000.0, "gtc", current_time, "tag", "long")

    assert strategy.max_long_slots == 50
    assert strategy.max_short_slots == 50

def test_slot_history_saved(strategy):
    """История изменений должна сохраняться"""
    now = datetime.now()
    with patch.object(strategy, '_get_pnl_from_freqtrade', return_value=(100.0, 50.0)):
        strategy._update_slot_allocation(now)

    assert len(strategy.slot_history) == 1
    entry = strategy.slot_history[0]
    assert len(entry) == 5  # (timestamp, long_slots, short_slots, pnl_long, pnl_short)
    assert entry[3] == 100.0  # pnl_long
    assert entry[4] == 50.0   # pnl_short
