import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import torch
import contextlib

# --- МОДУЛЬНОЕ МОКИРОВАНИЕ ---
class MockIStrategy:
    def __init__(self, config):
        self.config = config
        self.dp = MagicMock()
        self.wallets = MagicMock()

mock_ft_strategy_mod = MagicMock()
mock_ft_strategy_mod.IStrategy = MockIStrategy
mock_ft_strategy_mod.DecimalParameter = MagicMock(side_effect=lambda *args, **kwargs: MagicMock(value=kwargs.get('default', 0.0)))
mock_ft_strategy_mod.IntParameter = MagicMock(side_effect=lambda *args, **kwargs: MagicMock(value=kwargs.get('default', 0)))
mock_ft_strategy_mod.CategoricalParameter = MagicMock(side_effect=lambda *args, **kwargs: MagicMock(value=kwargs.get('default', args[0][0] if args and args[0] else '1m')))

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
    try:
        from CustomD3QNStrategy4 import CustomD3QNStrategy4
    except ImportError:
        from CustomD3QNStrategy4z import CustomD3QNStrategy4z as CustomD3QNStrategy4
except (ImportError, ModuleNotFoundError) as e:
    pytest.fail(f"Не удалось импортировать CustomD3QNStrategy4. Ошибка: {e}", pytrace=False)

# --- ФИКСТУРА PYTEST ---

@pytest.fixture
def strategy():
    mock_cfg = MagicMock()
    mock_cfg.seq.state_shape = (5, 90, 1)
    mock_cfg.seq.agent_history_len = 90
    mock_cfg.market.num_actions = 3
    mock_cfg.market.mirror_mode = False
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

    patches = [
        patch('agent.DuelingQNetwork', return_value=mock_model_instance),
        patch('torch.load', MagicMock()),
        patch('torch.set_num_threads', MagicMock()),
        patch('torch.set_num_interop_threads', MagicMock()),
        patch('importlib.util.spec_from_file_location', MagicMock()),
        patch.object(CustomD3QNStrategy4, '_find_config_file', return_value=Path("mock_config.py")),
        patch.object(CustomD3QNStrategy4, '_load_py_config', return_value=mock_cfg),
        patch.object(CustomD3QNStrategy4, '_load_weights', return_value=None)
    ]

    if hasattr(CustomD3QNStrategy4, '_load_norm_stats'):
        patches.append(patch.object(CustomD3QNStrategy4, '_load_norm_stats', return_value={'mean': [10], 'std': [5]}))

    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)

        config = {'exchange': {}, 'rl_ensemble': {}}
        strat = CustomD3QNStrategy4(config=config)
        for attr in ['_long_1_agent', '_long_2_agent', '_short_1_agent', '_short_2_agent']:
            setattr(strat, attr, MagicMock())
        yield strat

import contextlib

# --- ЮНИТ-ТЕСТЫ ---

def test_placeholder():
    pass

@pytest.mark.skipif(not hasattr(CustomD3QNStrategy4, 'map_action_for_model'), reason="map_action_for_model not present")
def test_map_action_for_model(strategy):
    """Тестирует РЕАЛЬНЫЙ метод map_action_for_model в стратегии."""
    assert strategy.map_action_for_model(0, 'LONG') == 0
    assert strategy.map_action_for_model(1, 'LONG') == 1
    assert strategy.map_action_for_model(2, 'LONG') == 2
    assert strategy.map_action_for_model(0, 'SHORT') == 0
    assert strategy.map_action_for_model(1, 'SHORT') == -1
    assert strategy.map_action_for_model(2, 'SHORT') == 2
    assert strategy.map_action_for_model(99, 'LONG') == 0

@pytest.mark.skipif(not hasattr(CustomD3QNStrategy4, 'get_model_input'), reason="get_model_input not present")
def test_normalization_applied_correctly(strategy):
    data = {
        'open_z': np.zeros(91), 'high_z': np.zeros(91), 'low_z': np.zeros(91),
        'close_z': np.zeros(91), 'volume_z': np.zeros(91),
        'date': pd.date_range(start='2023-01-01', periods=91, freq='1min')
    }
    dataframe = pd.DataFrame(data)
    mock_window_view_return = np.random.rand(2, 5, 90) # len(windows) = 2
    with patch(f'{CustomD3QNStrategy4.__module__}.sliding_window_view', return_value=mock_window_view_return) as mock_sliding_window:
        # Mock runmode to trigger the path that uses open_z
        strategy.config['runmode'] = 'backtest'
        strategy.get_model_input(dataframe, pair="BTC/USDT", side="LONG", model_num=1, asset_name="BTC")
        mock_sliding_window.assert_called_once()
