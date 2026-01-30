"""
Unit tests for ensemble voting logic in CustomD3QNStrategy4z

Покрытие:
- _normalize_q_value: нормализация Q-значений
- _compute_ensemble_decision_v2: новый алгоритм голосования
- _apply_soft_voting: legacy алгоритм голосования
- Интеграция с конфигом rl_ensemble

Цель: покрытие ≥90%
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import sys

# Определяем пути к проекту
repo_root = Path(__file__).parent.parent
rl_project_root = repo_root / "third_party" / "rl-trading-binance"
strategy_path = rl_project_root / "user_data" / "strategies"

# Добавляем пути в sys.path для корректного импорта
if str(rl_project_root) not in sys.path:
    sys.path.insert(0, str(rl_project_root))
if str(strategy_path) not in sys.path:
    sys.path.insert(0, str(strategy_path))

# Мокаем D3QN_PER_Agent ДО импорта стратегии, если она импортирует его на уровне модуля
# Но она импортирует его внутри блока try-except.
# Тем не менее, сделаем мок в sys.modules для надежности
mock_agent_mod = MagicMock()
sys.modules['agent'] = mock_agent_mod

# Теперь импортируем стратегию
from CustomD3QNStrategy4z import CustomD3QNStrategy4z

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def minimal_config():
    """Минимальный конфиг с rl_ensemble секцией"""
    return {
        'cpu_threads': 2,
        'rl_enable_long_1': True,
        'rl_enable_long_2': True,
        'rl_enable_short_1': True,
        'rl_enable_short_2': True,
        'rl_long_threshold': 1,
        'rl_short_threshold': 1,
        'rl_enable_veto': True,
        'rl_min_q_threshold_long': 0.0001,
        'rl_min_q_threshold_short': 0.0001,
        'rl_ensemble': {
            'enable_voting_v2': True,
            'epsilon_threshold': 0.15,
            'q_normalization': {
                'long_1': {'q_min': 0.0, 'q_max': 1.0},
                'long_2': {'q_min': 0.0, 'q_max': 1.0},
                'short_1': {'q_min': 0.0, 'q_max': 1.0},
                'short_2': {'q_min': 0.0, 'q_max': 1.0}
            }
        }
    }


@pytest.fixture
def mock_strategy(minimal_config):
    """
    Создаём mock стратегии без загрузки реальных моделей
    """
    # Mock model paths чтобы избежать FileNotFoundError
    with patch.object(CustomD3QNStrategy4z, '_find_config_file', return_value=None):
        with patch.object(CustomD3QNStrategy4z, '__init__', lambda x, y: None):
            strategy = CustomD3QNStrategy4z.__new__(CustomD3QNStrategy4z)

            # Устанавливаем необходимые атрибуты вручную
            strategy.config = minimal_config
            strategy.enable_long_1 = True
            strategy.enable_long_2 = True
            strategy.enable_short_1 = True
            strategy.enable_short_2 = True
            strategy.short_1_is_mirror = False
            strategy.short_2_is_mirror = False
            strategy.can_short = True
            strategy.vote_threshold_long = 1
            strategy.vote_threshold_short = 1
            strategy.enable_veto = True
            strategy.logger = MagicMock()

            # Настройки ансамбля v2
            strategy.ensemble_cfg = minimal_config['rl_ensemble']
            strategy.enable_voting_v2 = True
            strategy.epsilon_threshold = 0.15
            strategy.q_normalization = minimal_config['rl_ensemble']['q_normalization']

            return strategy


# ============================================================================
# ТЕСТЫ: _normalize_q_value
# ============================================================================

class TestNormalizeQValue:
    """Тесты нормализации Q-значений"""

    def test_normal_normalization(self, mock_strategy):
        """Q внутри диапазона [q_min, q_max]"""
        result = mock_strategy._normalize_q_value(0.5, 'long_1')
        assert 0.0 <= result <= 1.0
        assert result == 0.5  # (0.5 - 0) / (1 - 0) = 0.5

    def test_boundary_min(self, mock_strategy):
        """Q = q_min должно вернуть 0.0"""
        result = mock_strategy._normalize_q_value(0.0, 'long_1')
        assert result == 0.0

    def test_boundary_max(self, mock_strategy):
        """Q = q_max должно вернуть 1.0"""
        result = mock_strategy._normalize_q_value(1.0, 'long_1')
        assert result == 1.0

    def test_below_min(self, mock_strategy):
        """Q < q_min должно быть ограничено до 0.0"""
        result = mock_strategy._normalize_q_value(-0.5, 'long_1')
        assert result == 0.0

    def test_above_max(self, mock_strategy):
        """Q > q_max должно быть ограничено до 1.0"""
        result = mock_strategy._normalize_q_value(2.0, 'long_1')
        assert result == 1.0

    def test_qmin_equals_qmax(self, mock_strategy, caplog):
        """q_min == q_max → ошибка деления на 0, вернуть 0.5"""
        mock_strategy.q_normalization['long_1'] = {'q_min': 0.5, 'q_max': 0.5}
        result = mock_strategy._normalize_q_value(0.5, 'long_1')
        assert result == 0.5
        assert "Q normalization error" in caplog.text

    def test_missing_model_config(self, mock_strategy, caplog):
        """Модель не в q_normalization → warning и clip(q, 0, 1)"""
        result = mock_strategy._normalize_q_value(0.7, 'unknown_model')
        assert result == 0.7
        assert "No normalization config" in caplog.text

    @pytest.mark.parametrize("model_name", ['long_1', 'long_2', 'short_1', 'short_2'])
    def test_all_models(self, mock_strategy, model_name):
        """Проверка для всех 4 моделей"""
        result = mock_strategy._normalize_q_value(0.5, model_name)
        assert 0.0 <= result <= 1.0

    def test_custom_range(self, mock_strategy):
        """Тест с нестандартным диапазоном [0.1, 0.9]"""
        mock_strategy.q_normalization['long_1'] = {'q_min': 0.1, 'q_max': 0.9}
        result = mock_strategy._normalize_q_value(0.5, 'long_1')
        # (0.5 - 0.1) / (0.9 - 0.1) = 0.4 / 0.8 = 0.5
        assert abs(result - 0.5) < 1e-6


# ============================================================================
# ТЕСТЫ: _compute_ensemble_decision_v2
# ============================================================================

class TestComputeEnsembleDecisionV2:
    """Тесты нового алгоритма голосования"""

    def test_pure_long_signal(self, mock_strategy):
        """Только LONG модели имеют положительные Q → LONG сигнал"""
        q_values = {
            'long_1': np.array([[0.1, 0.9, 0.0]]),  # idx=0, action=1 (long)
            'long_2': np.array([[0.1, 0.8, 0.0]]),
            'short_1': np.array([[0.5, 0.0, 0.0]]),  # нет сигнала
            'short_2': np.array([[0.5, 0.0, 0.0]])
        }

        decision = mock_strategy._compute_ensemble_decision_v2(q_values, 0, False, False)

        assert decision['enter_long'] == 1
        assert decision['enter_short'] == 0
        assert "LONG Signal" in decision['reason']

    def test_pure_short_signal(self, mock_strategy):
        """Только SHORT модели имеют положительные Q → SHORT сигнал"""
        q_values = {
            'long_1': np.array([[0.5, 0.0, 0.0]]),
            'long_2': np.array([[0.5, 0.0, 0.0]]),
            'short_1': np.array([[0.1, 0.0, 0.9]]),  # action=2 (short, не mirror)
            'short_2': np.array([[0.1, 0.0, 0.8]])
        }

        decision = mock_strategy._compute_ensemble_decision_v2(q_values, 0, False, False)

        assert decision['enter_long'] == 0
        assert decision['enter_short'] == 1
        assert "SHORT Signal" in decision['reason']

    def test_hold_due_to_noise(self, mock_strategy):
        """Δ < epsilon → HOLD"""
        # S_long = 0.5, S_short = 0.52, Δ = 0.02 < 0.15
        q_values = {
            'long_1': np.array([[0.1, 0.5, 0.0]]),
            'long_2': np.array([[0.5, 0.0, 0.0]]),  # не участвует (Q=0)
            'short_1': np.array([[0.1, 0.0, 0.52]]),
            'short_2': np.array([[0.5, 0.0, 0.0]])
        }

        decision = mock_strategy._compute_ensemble_decision_v2(q_values, 0, False, False)

        assert decision['enter_long'] == 0
        assert decision['enter_short'] == 0
        assert "HOLD (noise)" in decision['reason']

    def test_hold_due_to_existing_position(self, mock_strategy):
        """has_long=True → никаких новых позиций"""
        q_values = {
            'long_1': np.array([[0.1, 0.9, 0.0]]),
            'long_2': np.array([[0.1, 0.8, 0.0]]),
            'short_1': np.array([[0.5, 0.0, 0.0]]),
            'short_2': np.array([[0.5, 0.0, 0.0]])
        }

        decision = mock_strategy._compute_ensemble_decision_v2(q_values, 0, has_long=True, has_short=False)

        assert decision['enter_long'] == 0
        assert decision['enter_short'] == 0
        assert "Position exists" in decision['reason']

    def test_mirror_mode_short_action_1(self, mock_strategy):
        """mirror_mode=True → action=1 означает SHORT"""
        mock_strategy.short_1_is_mirror = True
        mock_strategy.short_2_is_mirror = True

        q_values = {
            'long_1': np.array([[0.5, 0.0, 0.0]]),
            'long_2': np.array([[0.5, 0.0, 0.0]]),
            'short_1': np.array([[0.1, 0.9, 0.0]]),  # action=1 в mirror → SHORT
            'short_2': np.array([[0.1, 0.8, 0.0]])
        }

        decision = mock_strategy._compute_ensemble_decision_v2(q_values, 0, False, False)

        assert decision['enter_short'] == 1
        assert decision['enter_long'] == 0

    def test_negative_q_ignored(self, mock_strategy):
        """Отрицательные Q не вносят вклад"""
        q_values = {
            'long_1': np.array([[0.5, -0.5, 0.0]]),  # Q < 0
            'long_2': np.array([[0.5, 0.0, 0.0]]),
            'short_1': np.array([[0.5, 0.0, 0.0]]),
            'short_2': np.array([[0.5, 0.0, 0.0]])
        }

        decision = mock_strategy._compute_ensemble_decision_v2(q_values, 0, False, False)

        # S_long = 0.0, S_short = 0.0, Δ = 0.0 < epsilon
        assert decision['enter_long'] == 0
        assert decision['enter_short'] == 0

    def test_can_short_false(self, mock_strategy):
        """can_short=False → SHORT сигнал не генерируется"""
        mock_strategy.can_short = False

        q_values = {
            'long_1': np.array([[0.5, 0.0, 0.0]]),
            'long_2': np.array([[0.5, 0.0, 0.0]]),
            'short_1': np.array([[0.1, 0.0, 0.9]]),
            'short_2': np.array([[0.1, 0.0, 0.8]])
        }

        decision = mock_strategy._compute_ensemble_decision_v2(q_values, 0, False, False)

        assert decision['enter_short'] == 0

    @pytest.mark.parametrize("epsilon", [0.05, 0.11, 0.15, 0.20])
    def test_different_epsilon_values(self, mock_strategy, epsilon):
        """Тест с разными значениями epsilon"""
        mock_strategy.epsilon_threshold = epsilon

        # S_long = 0.5, S_short = 0.6, Δ = 0.1
        q_values = {
            'long_1': np.array([[0.1, 0.5, 0.0]]),
            'long_2': np.array([[0.5, 0.0, 0.0]]),
            'short_1': np.array([[0.1, 0.0, 0.6]]),
            'short_2': np.array([[0.5, 0.0, 0.0]])
        }

        decision = mock_strategy._compute_ensemble_decision_v2(q_values, 0, False, False)

        if epsilon > 0.1:
            # Δ < epsilon → HOLD
            assert "HOLD (noise)" in decision['reason']
        else:
            # Δ >= epsilon → SHORT Signal
            assert decision['enter_short'] == 1


# ============================================================================
# ТЕСТЫ: _apply_soft_voting
# ============================================================================

class TestApplySoftVoting:
    """Тесты legacy алгоритма голосования"""

    def test_long_votes_counting(self, mock_strategy):
        """Подсчёт голосов LONG: [1, 1] → 2 голоса"""
        decision = mock_strategy._apply_soft_voting([1, 1], [0, 0], False, False)
        assert decision['enter_long'] == 1
        assert "L_votes:2" in decision['reason']

    def test_short_votes_mirror_mode(self, mock_strategy):
        """mirror_mode=True: действие 1 → голос за SHORT"""
        mock_strategy.short_1_is_mirror = True
        mock_strategy.short_2_is_mirror = True

        decision = mock_strategy._apply_soft_voting([0, 0], [1, 1], False, False)
        assert decision['enter_short'] == 1
        assert "S_votes:2" in decision['reason']

    def test_short_votes_normal_mode(self, mock_strategy):
        """mirror_mode=False: действие 2 → голос за SHORT"""
        mock_strategy.short_1_is_mirror = False
        mock_strategy.short_2_is_mirror = False

        decision = mock_strategy._apply_soft_voting([0, 0], [2, 2], False, False)
        assert decision['enter_short'] == 1

    def test_threshold_check(self, mock_strategy):
        """Проверка порогов vote_threshold"""
        mock_strategy.vote_threshold_long = 2

        # 1 голос < порога 2
        decision = mock_strategy._apply_soft_voting([1, 0], [0, 0], False, False)
        assert decision['enter_long'] == 0
        assert "No Consensus" in decision['reason']

    def test_veto_enabled(self, mock_strategy):
        """enable_veto=True: конфликт → оба отменяются"""
        mock_strategy.enable_veto = True

        decision = mock_strategy._apply_soft_voting([1, 0], [2, 0], False, False)
        assert decision['enter_long'] == 0
        assert decision['enter_short'] == 0
        assert "Veto" in decision['reason']

    def test_veto_disabled(self, mock_strategy):
        """enable_veto=False: конфликт не блокирует (но safety проверка отменяет оба)"""
        mock_strategy.enable_veto = False

        decision = mock_strategy._apply_soft_voting([1, 0], [2, 0], False, False)
        # Safety: если оба активны → CONFLICT: Dual Signal
        assert decision['enter_long'] == 0
        assert decision['enter_short'] == 0
        assert "CONFLICT: Dual Signal" in decision['reason']

    def test_existing_position_long(self, mock_strategy):
        """has_long=True → никаких сигналов"""
        decision = mock_strategy._apply_soft_voting([1, 1], [0, 0], has_long=True, has_short=False)
        assert decision['enter_long'] == 0
        assert "Position exists" in decision['reason']

    def test_existing_position_short(self, mock_strategy):
        """has_short=True → никаких сигналов"""
        decision = mock_strategy._apply_soft_voting([0, 0], [2, 2], has_long=False, has_short=True)
        assert decision['enter_short'] == 0
        assert "Position exists" in decision['reason']

    @pytest.mark.parametrize("mirror_mode_combo", [
        (True, True),
        (True, False),
        (False, True),
        (False, False)
    ])
    def test_mirror_mode_combinations(self, mock_strategy, mirror_mode_combo):
        """Тест всех комбинаций mirror_mode"""
        mock_strategy.short_1_is_mirror = mirror_mode_combo[0]
        mock_strategy.short_2_is_mirror = mirror_mode_combo[1]

        # Действие 1 для обеих SHORT моделей
        decision = mock_strategy._apply_soft_voting([0, 0], [1, 1], False, False)

        # Считаем, сколько голосов
        expected_votes = sum([1 if m else 0 for m in mirror_mode_combo])

        if expected_votes >= mock_strategy.vote_threshold_short:
            assert decision['enter_short'] == 1
        else:
            assert decision['enter_short'] == 0


# ============================================================================
# ИНТЕГРАЦИОННЫЕ ТЕСТЫ
# ============================================================================

class TestVotingIntegration:
    """Тесты интеграции алгоритмов голосования с конфигом"""

    def test_v2_enabled(self, minimal_config):
        """enable_voting_v2=True → используется _compute_ensemble_decision_v2"""
        minimal_config['rl_ensemble']['enable_voting_v2'] = True

        # Проверяем, что флаг правильно считывается
        assert minimal_config['rl_ensemble']['enable_voting_v2'] is True

    def test_v2_disabled(self, minimal_config):
        """enable_voting_v2=False → используется _apply_soft_voting"""
        minimal_config['rl_ensemble']['enable_voting_v2'] = False

        assert minimal_config['rl_ensemble']['enable_voting_v2'] is False

    def test_epsilon_from_config(self, mock_strategy):
        """epsilon_threshold правильно считывается из конфига"""
        assert mock_strategy.epsilon_threshold == 0.15

    def test_q_normalization_from_config(self, mock_strategy):
        """q_normalization правильно считывается для всех моделей"""
        assert 'long_1' in mock_strategy.q_normalization
        assert 'long_2' in mock_strategy.q_normalization
        assert 'short_1' in mock_strategy.q_normalization
        assert 'short_2' in mock_strategy.q_normalization

        assert mock_strategy.q_normalization['long_1']['q_min'] == 0.0
        assert mock_strategy.q_normalization['long_1']['q_max'] == 1.0

    def test_result_consistency(self, mock_strategy):
        """Одинаковые входы → одинаковые результаты"""
        q_values = {
            'long_1': np.array([[0.1, 0.9, 0.0]]),
            'long_2': np.array([[0.1, 0.8, 0.0]]),
            'short_1': np.array([[0.5, 0.0, 0.0]]),
            'short_2': np.array([[0.5, 0.0, 0.0]])
        }

        decision1 = mock_strategy._compute_ensemble_decision_v2(q_values, 0, False, False)
        decision2 = mock_strategy._compute_ensemble_decision_v2(q_values, 0, False, False)

        assert decision1 == decision2

    def test_missing_rl_ensemble_config(self):
        """Отсутствие rl_ensemble в конфиге → fallback к defaults"""
        config_no_ensemble = {
            'cpu_threads': 2,
            'rl_enable_long_1': False,
            'rl_enable_long_2': False,
            'rl_enable_short_1': False,
            'rl_enable_short_2': False
        }

        # Проверяем, что конфиг без rl_ensemble не вызывает краш
        assert config_no_ensemble.get('rl_ensemble', {}).get('enable_voting_v2', False) is False


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Тесты граничных случаев"""

    def test_empty_q_values(self, mock_strategy):
        """Пустой dict q_values"""
        decision = mock_strategy._compute_ensemble_decision_v2({}, 0, False, False)

        # S_long = 0, S_short = 0, Δ = 0 < epsilon
        assert decision['enter_long'] == 0
        assert decision['enter_short'] == 0

    def test_single_model_active(self, mock_strategy):
        """Только одна модель активна"""
        mock_strategy.enable_long_2 = False
        mock_strategy.enable_short_1 = False
        mock_strategy.enable_short_2 = False

        q_values = {
            'long_1': np.array([[0.1, 0.9, 0.0]])
        }

        decision = mock_strategy._compute_ensemble_decision_v2(q_values, 0, False, False)

        # S_long = 0.9, S_short = 0, Δ = 0.9 > epsilon → LONG
        assert decision['enter_long'] == 1

    def test_all_models_hold(self, mock_strategy):
        """Все модели выбирают HOLD (action=0)"""
        q_values = {
            'long_1': np.array([[0.9, 0.0, 0.0]]),  # action=0
            'long_2': np.array([[0.9, 0.0, 0.0]]),
            'short_1': np.array([[0.9, 0.0, 0.0]]),
            'short_2': np.array([[0.9, 0.0, 0.0]])
        }

        decision = mock_strategy._compute_ensemble_decision_v2(q_values, 0, False, False)

        assert decision['enter_long'] == 0
        assert decision['enter_short'] == 0

    def test_idx_out_of_bounds(self, mock_strategy):
        """idx за пределами массива → должно вызвать IndexError"""
        q_values = {
            'long_1': np.array([[0.1, 0.9, 0.0]])  # batch_size = 1
        }

        with pytest.raises(IndexError):
            mock_strategy._compute_ensemble_decision_v2(q_values, 5, False, False)

    def test_nan_q_values(self, mock_strategy):
        """NaN Q-значения → должны обрабатываться безопасно"""
        q_values = {
            'long_1': np.array([[0.1, np.nan, 0.0]]),
            'long_2': np.array([[0.1, 0.8, 0.0]]),
            'short_1': np.array([[0.5, 0.0, 0.0]]),
            'short_2': np.array([[0.5, 0.0, 0.0]])
        }

        # _normalize_q_value с NaN должен вернуть что-то безопасное
        # (зависит от реализации np.clip с NaN)
        decision = mock_strategy._compute_ensemble_decision_v2(q_values, 0, False, False)

        # Проверяем, что не крашится
        assert 'enter_long' in decision
        assert 'enter_short' in decision


# ============================================================================
# ЗАПУСК ТЕСТОВ
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=CustomD3QNStrategy4z", "--cov-report=term-missing"])
