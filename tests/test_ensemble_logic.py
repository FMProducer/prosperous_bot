import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import torch

# --- МОДУЛЬНОЕ МОКИРОВАНИЕ ---
class MockIStrategy:
    def __init__(self, config):
        self.config = config
        self.dp = MagicMock()
        self.wallets = MagicMock()

mock_ft_strategy_mod = MagicMock()
mock_ft_strategy_mod.IStrategy = MockIStrategy
mock_ft_strategy_mod.DecimalParameter = MagicMock(return_value=0.0)
mock_ft_strategy_mod.IntParameter = MagicMock(return_value=0)

mock_ft_persistence_mod = MagicMock()
mock_ft_persistence_mod.Trade = MagicMock()

sys.modules['freqtrade'] = MagicMock()
sys.modules['freqtrade.strategy'] = mock_ft_strategy_mod
sys.modules['freqtrade.persistence'] = mock_ft_persistence_mod

# --- НАСТРОЙКА ПУТЕЙ И ИМПОРТ ---
project_root = Path(__file__).parent.parent
strategy_dir = project_root / 'third_party/rl-trading-binance/user_data/strategies'
third_party_root = project_root / 'third_party/rl-trading-binance'

if str(strategy_dir) not in sys.path:
    sys.path.insert(0, str(strategy_dir))
if str(third_party_root) not in sys.path:
    sys.path.insert(0, str(third_party_root))

try:
    from CustomD3QNStrategy4 import CustomD3QNStrategy4
except (ImportError, ModuleNotFoundError) as e:
    pytest.fail(f"Не удалось импортировать CustomD3QNStrategy4. Ошибка: {e}", pytrace=False)

# --- ФИКСТУРА PYTEST ---

@pytest.fixture
def strategy():
    mock_cfg = MagicMock()
    mock_cfg.seq.state_shape = (5, 90, 1)
    mock_cfg.market.num_actions = 3
    mock_cfg.rl.lr = 0.001
    mock_cfg.rl.gamma = 0.99
    mock_cfg.per.buffer_size = 1000
    mock_cfg.per.per_alpha = 0.6
    mock_cfg.per.per_beta_start = 0.4
    mock_cfg.per.per_beta_frames = 1000

    mock_parameter = torch.tensor([1.0], requires_grad=True)
    mock_model_instance = MagicMock()
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.parameters.return_value = [mock_parameter]

    with patch('agent.DuelingQNetwork', return_value=mock_model_instance), \
         patch('torch.load', MagicMock()), \
         patch('torch.set_num_threads', MagicMock()), \
         patch('torch.set_num_interop_threads', MagicMock()), \
         patch('importlib.util.spec_from_file_location', MagicMock()), \
         patch.object(CustomD3QNStrategy4, '_find_config_file', return_value=Path("mock_config.py")), \
         patch.object(CustomD3QNStrategy4, '_load_py_config', return_value=mock_cfg), \
         patch.object(CustomD3QNStrategy4, '_load_norm_stats', return_value={'mean': [10], 'std': [5]}), \
         patch.object(CustomD3QNStrategy4, '_load_weights', return_value=None):

        config = {'exchange': {}}
        strat = CustomD3QNStrategy4(config=config)

        strat.long_1_agent, strat.long_2_agent = MagicMock(), MagicMock()
        strat.short_1_agent, strat.short_2_agent = MagicMock(), MagicMock()

        strat.norm_stats_long_1 = {'mean': [20.0], 'std': [10.0]}

        yield strat

# --- ЮНИТ-ТЕСТЫ ---

def test_map_action_for_model(strategy):
    """Тестирует РЕАЛЬНЫЙ метод map_action_for_model в стратегии."""
    assert strategy.map_action_for_model(0, 'LONG') == 0
    assert strategy.map_action_for_model(1, 'LONG') == 1
    assert strategy.map_action_for_model(2, 'LONG') == 2
    assert strategy.map_action_for_model(0, 'SHORT') == 0
    assert strategy.map_action_for_model(1, 'SHORT') == -1
    assert strategy.map_action_for_model(2, 'SHORT') == 2
    assert strategy.map_action_for_model(99, 'LONG') == 0

def test_apply_soft_voting_logic(strategy):
    """Тестирует РЕАЛЬНЫЙ метод _apply_soft_voting в стратегии."""
    res = strategy._apply_soft_voting(long_actions=[1, 0], short_actions=[0, 0])
    assert res['enter_long'] == 1 and res['enter_short'] == 0 and res['exit_position'] == 0

    res = strategy._apply_soft_voting(long_actions=[0, 0], short_actions=[-1, -1])
    assert res['enter_long'] == 0 and res['enter_short'] == 1 and res['exit_position'] == 0

    res = strategy._apply_soft_voting(long_actions=[1, 0], short_actions=[-1, 0])
    assert res['enter_long'] == 0 and res['enter_short'] == 0 and 'conflict' in res['reason']

    res = strategy._apply_soft_voting(long_actions=[1, 1], short_actions=[-1, 2])
    assert res['exit_position'] == 1 and res['enter_long'] == 0 and res['enter_short'] == 0

    res = strategy._apply_soft_voting(long_actions=[0, 0], short_actions=[0, 0])
    assert res['enter_long'] == 0 and res['enter_short'] == 0 and res['reason'] == 'no_signal'

def test_normalization_applied_correctly(strategy):
    """
    Проверяет, что РЕАЛЬНЫЙ метод get_model_input корректно применяет нормализацию.
    """
    data = {
        'open': np.full(91, 1.0), 'high': np.full(91, 2.0), 'low': np.full(91, 3.0),
        'close': np.linspace(10, 30, 91), 'volume': np.full(91, 4.0)
    }
    dataframe = pd.DataFrame(data)

    # Мокаем sliding_window_view, чтобы он возвращал корректный shape для downstream-кода
    mock_window_view_return = np.random.rand(5, 1, 90)  # (channels, num_windows, window_size)
    with patch('CustomD3QNStrategy4.sliding_window_view', return_value=mock_window_view_return) as mock_sliding_window:
        # Вызываем реальный метод
        strategy.get_model_input(dataframe, pair="BTC/USDT", side="LONG", model_num=1)

        # Проверяем, ЧТО было передано в наш мок
        mock_sliding_window.assert_called_once()
        normalized_data = mock_sliding_window.call_args[0][0]

        assert normalized_data.shape == (5, 90)

        # Канал 'open': log_returns ~0. Нормализация: (0 - 20) / 10 = -2.0
        open_channel_normalized = normalized_data[0, :]
        assert np.allclose(open_channel_normalized, -2.0)

        # Канал 'volume': log(1+4)=1.609. Нормализация: (1.609 - 20) / 10 = -1.839
        volume_channel_normalized = normalized_data[4, :]
        assert np.allclose(volume_channel_normalized, (np.log(1.0 + 4.0) - 20.0) / 10.0)
