# --- Repo-State Header (prosperous_bot @ 47ee3d99cd6af878dd8b19c59d217709a60a2a19) ---
# Ветка: prosperous_bot | SHA-1: 47ee3d99cd6af878dd8b19c59d217709a60a2a19
# Коммит: "docs: config_rl4z CustomD3QNStrategy4z modified"
# Ссылка: https://github.com/FMProducer/prosperous_bot/commit/47ee3d99cd6af878dd8b19c59d217709a60a2a19
# ---

import pytest
import time
from agent_router import AgentRouter

@pytest.fixture
def router():
    return AgentRouter()

def test_route_task_types(router):
    """Проверка корректного роутинга по типам задач"""
    assert router.route({"id": "1", "type": "complex_math"}) == "qwen_235b"
    assert router.route({"id": "2", "type": "code_writing"}) == "nemotron_30b"
    assert router.route({"id": "3", "type": "quick_fixes"}) == "step_flash"
    assert router.route({"id": "4", "type": "unknown_task"}) == "nemotron_30b"

def test_fallback_priority(router):
    """Проверка fallback-цепочки"""
    # Сценарий 1: без ошибок (ожидаем успех)
    result = router.execute_with_fallback(
        {"id": "5", "type": "complex_math"},
        max_retries=0
    )
    assert result["model"] == "qwen_235b"
    
    # Сценарий 2: с принудительной ошибкой (ожидаем fallback)
    with pytest.raises(RuntimeError):
        router.execute_with_fallback(
            {"id": "6", "type": "complex_math"},
            max_retries=0,
            force_fail=True
        )

def test_log_uniqueness(router):
    """Проверка отсутствия конфликтов имен логов"""
    log1 = router._get_log_filename()
    time.sleep(1)
    log2 = router._get_log_filename()
    assert log1 != log2