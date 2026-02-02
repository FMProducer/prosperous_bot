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
        'runmode': 'dry_run',
        'max_open_trades': 100,  # ИСПОЛЬЗУЕМ ГЛОБАЛЬНЫЙ ПАРАМЕТР
        'cpu_threads': 4,
        'rl_enable_long_1': True,
        'rl_enable_long_2': True,
        'rl_enable_short_1': True,
        'rl_enable_short_2': True,
        'dynamic_slots': {
            'enabled': True,
            'min_slots_per_side': 5,
            'update_interval_sec': 300,
            'aggression_factor': 1.5
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
        Mock(is_short=False, close_rate_requested=105, calc_profit=lambda rate: 50.0, stake_amount=1.0, pair="BTC/USDT"),  # +50 USDT
        Mock(is_short=True, close_rate_requested=95, calc_profit=lambda rate: -20.0, stake_amount=1.0, pair="ETH/USDT"),   # -20 USDT
    ]

    with patch('freqtrade.persistence.Trade.get_open_trades', return_value=mock_trades):
        pnl_long, pnl_short = strategy._get_pnl_from_freqtrade()

    assert pnl_long == 50.0
    assert pnl_short == -20.0

def test_slot_allocation_long_profitable(strategy):
    """При прибыльных лонгах и убыточных шортах: V2 logic с aggression_factor"""
    # pnl_long = 250, pnl_short = -100.
    # profit_long = 250, loss_short = 100, total = 350
    # penalty_ratio = (100/350)**1.5 ≈ 0.1527
    # long_ratio = 0.8 + 0.15 * 0.1527 ≈ 0.8229
    # available = 100 - 2*5 = 90
    # Expected long = 5 + int(90 * 0.8229) = 5 + 74 = 79
    with patch.object(strategy, '_get_pnl_from_freqtrade', return_value=(250.0, -100.0)):
        strategy._update_slot_allocation(datetime.now())

    assert strategy.max_long_slots == 79
    assert strategy.max_short_slots == 21

def test_slot_allocation_short_profitable(strategy):
    """При прибыльных шортах и убыточных лонгах: V2 logic с aggression_factor"""
    # pnl_long = -150, pnl_short = 300.
    # profit_short = 300, loss_long = 150, total = 450
    # penalty_ratio = (150/450)**1.5 ≈ 0.1924
    # long_ratio = 0.2 - 0.15 * 0.1924 ≈ 0.1711
    # Expected long = 5 + int(90 * 0.1711) = 5 + 15 = 20
    with patch.object(strategy, '_get_pnl_from_freqtrade', return_value=(-150.0, 300.0)):
        strategy._update_slot_allocation(datetime.now())

    assert strategy.max_long_slots == 20
    assert strategy.max_short_slots == 80

def test_slot_allocation_both_negative(strategy):
    """При убытках по обеим сторонам: распределение обратно пропорционально убытку"""
    # Long -50, Short -80. Total loss 130.
    # Long ratio = 80 / 130 = 0.6153
    # Available slots = 100 - 2*5 = 90
    # Expected long = 5 + int(90 * 0.6153) = 5 + 55 = 60
    with patch.object(strategy, '_get_pnl_from_freqtrade', return_value=(-50.0, -80.0)):
        strategy._update_slot_allocation(datetime.now())

    assert strategy.max_long_slots == 60
    assert strategy.max_short_slots == 40

def test_slot_allocation_both_negative_extreme(strategy):
    """При сильно разных убытках: значительное преимущество менее убыточному"""
    # Long -500, Short -50. Total loss 550.
    # Long ratio = 50 / 550 = 0.0909
    # Expected long = 5 + int(90 * 0.0909) = 5 + 8 = 13
    with patch.object(strategy, '_get_pnl_from_freqtrade', return_value=(-500.0, -50.0)):
        strategy._update_slot_allocation(datetime.now())

    assert strategy.max_long_slots == 13
    assert strategy.max_short_slots == 87

def test_slot_allocation_both_negative_inverse(strategy):
    """При разных убытках - меньше слотов направлению с большим убытком"""
    # Long теряет меньше (-47), Short теряет больше (-119)
    # Total loss = 166. Long ratio = 119 / 166 = 0.7168
    # Expected long = 5 + int(90 * 0.7168) = 5 + 64 = 69
    with patch.object(strategy, '_get_pnl_from_freqtrade', return_value=(-47.0, -119.0)):
        strategy._update_slot_allocation(datetime.now())

    # Лонгам должно достаться БОЛЬШЕ слотов (т.к. они теряют меньше)
    assert strategy.max_long_slots > strategy.max_short_slots
    assert strategy.max_long_slots == 69

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

def test_on_demand_mode(strategy):
    """При update_interval_sec == 0 (On-Demand) обновление происходит при каждом вызове"""
    strategy.slot_update_interval = 0
    strategy.last_slot_update = datetime.now() - timedelta(seconds=1) # Прошлый апдейт 1 сек назад

    with patch.object(strategy, '_update_slot_allocation') as mock_update:
        # Симулируем confirm_trade_entry
        with patch('freqtrade.persistence.Trade.get_trades') as mock_trades:
            # Mock the query object returned by get_trades
            mock_query = MagicMock()
            mock_query.all.return_value = []
            # Make sure directional timeout is skipped by returning None for last_trade
            mock_query.order_by.return_value.first.return_value = None
            mock_trades.return_value = mock_query

            # Первый вызов
            strategy.confirm_trade_entry("BTC/USDT", "limit", 1.0, 50000.0, "gtc", datetime.now(), "tag", "long")
            # Второй вызов (через мгновение)
            strategy.confirm_trade_entry("ETH/USDT", "limit", 1.0, 2500.0, "gtc", datetime.now(), "tag", "long")

    assert mock_update.call_count == 2
