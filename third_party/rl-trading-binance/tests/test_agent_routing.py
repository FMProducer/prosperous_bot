# --- Repo-State Header (prosperous_bot @ 47ee3d99cd6af878dd8b19c59d217709a60a2a19) ---
# Ветка: prosperous_bot | SHA-1: 47ee3d99cd6af878dd8b19c59d217709a60a2a19
# Коммит: "docs: config_rl4z CustomD3QNStrategy4z modified"
# Ссылка: https://github.com/FMProducer/prosperous_bot/commit/47ee3d99cd6af878dd8b19c59d217709a60a2a19
# ---

import pytest
from unittest.mock import patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

def test_execute_successful_primary(router):
    """Проверка успешного выполнения на основной модели без fallback."""
    task = {"id": "task_primary_ok", "type": "code_writing"}
    # Мокаем _execute_task_on_model, чтобы он всегда возвращал успех
    with patch.object(router, '_execute_task_on_model', return_value={"success": True, "model": "nemotron_30b"}) as mock_execute:
        result = router.execute_with_fallback(task)
        mock_execute.assert_called_once_with(task, "nemotron_30b")
        assert result["model"] == "nemotron_30b"

def test_execute_fallback_on_failure(router):
    """Проверка перехода к fallback-модели при ошибке основной."""
    task = {"id": "task_fallback", "type": "design"} # primary is qwen_235b

    # Мокаем _execute_task_on_model, чтобы он сначала возвращал ошибку, а потом успех
    mock_responses = [
        {"success": False, "model": "qwen_235b", "error": "Simulated failure"},
        {"success": True, "model": "nemotron_30b"},
    ]
    with patch.object(router, '_execute_task_on_model', side_effect=mock_responses) as mock_execute:
        result = router.execute_with_fallback(task)
        # Проверяем, что были вызваны обе модели
        assert mock_execute.call_count == 2
        mock_execute.assert_any_call(task, "qwen_235b")
        mock_execute.assert_any_call(task, "nemotron_30b")
        # Проверяем, что итоговый результат от fallback-модели
        assert result["model"] == "nemotron_30b"

def test_execute_all_models_fail(router):
    """Проверка, что при ошибке всех моделей выбрасывается исключение."""
    task = {"id": "task_all_fail", "type": "design"} # primary: qwen, fallbacks: nemotron, step_flash

    # Мокаем _execute_task_on_model, чтобы он всегда возвращал ошибку
    with patch.object(router, '_execute_task_on_model', return_value={"success": False}) as mock_execute:
        with pytest.raises(RuntimeError, match="All models failed"):
            router.execute_with_fallback(task)
        # Проверяем, что были вызваны все 3 модели из цепочки
        assert mock_execute.call_count == 3
        mock_execute.assert_any_call(task, "qwen_235b")
        mock_execute.assert_any_call(task, "nemotron_30b")
        mock_execute.assert_any_call(task, "step_flash")