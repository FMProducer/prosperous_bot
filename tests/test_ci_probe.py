"""
CI Probe: минимальный тест, чтобы триггерить основной workflow и убедиться,
что он запускается на windows-latest и выполняет базовые шаги.
Не проверяет бизнес-логику, не меняет состояние проекта.
"""

import sys
import platform


def test_ci_probe_runs_on_windows_runner():
    # Сам тест всегда успешен; цель — инициировать прогон workflow.
    # Печатаем базовую диагностику в лог pytest.
    print("PY", sys.version)
    print("PLATFORM", platform.platform())
    assert True


def test_ci_probe_sanity():
    assert 1 + 1 == 2
