# Changelog

## [2026-07-14] — Test Safety Net: автоблокировка тестов при живой системе

**Проблема:** Тесты теоретически могли навредить реальным ботам — human factor (случайный сброс мока, тест без изоляции).

**Решение:** Safety Net в `conftest.py`:
- `pytest_configure()` хук проверяет PM2 через `pm2 jlist` (shell=True для Windows .cmd)
- Если обнаружены `real-*` / `paper-*` / `supervisor` процессы → `pytest.exit()` с блокировкой
- `--ignore-safety` — принудительный запуск (только для осознанного использования)
- Пространство имён: `--ignore-safety` добавлен через `pytest_addoption`

**Слои защиты тестов (итого 5):**
1. `conftest.py:autouse=True` — глобальный mock `binance.client.Client`
2. Все файловые чтения — `mock_open()` / `patch("safe_load_json_sync")`
3. Все сетевые вызовы — `patch("aiohttp.ClientSession")` / `patch("asyncio.to_thread")`
4. Все PM2-вызовы — `patch("create_subprocess_shell")` / `patch("start_bot")`
5. **[NEW]** Safety Net — PM2-детекция при старте pytest

**Команда:** `pytest` (с автоблокировкой) / `pytest --ignore-safety` (принудительно)
**Тесты:** 327/327 passed.

---

## [YYYY-MM-DD] - Refactor and Enhance RL Trading System

### Architectural Changes

- **Vectorized Replay Buffer**: Refactored `replay_buffer.py` to use NumPy vectorized operations for priority updates. This eliminates Python loops, improving performance and ensuring the atomicity of tree updates, which is critical for the stability of the Prioritized Experience Replay algorithm.

- **Robust Model Input**: Enhanced `model.py` by adding defensive assertions in the `forward()` method to validate input tensor shapes. This provides a more robust way to handle dynamic input shapes and prevents potential runtime errors due to mismatched tensor dimensions.

- **Secure Configuration Loading**: Fixed `validate_ensemble_prod_q.py` by replacing the insecure `SourceFileLoader` with a proper configuration injection pattern using `importlib.util`. This is a safer and more standard approach for loading Python-based configuration files, reducing the risk of arbitrary code execution.

- **Standardized Documentation**: Added Google-style docstrings to all public methods in `agent.py` and `trading_environment.py`. The docstrings focus on the mathematical meaning of 'reward' and 'state' transitions, improving code clarity and maintainability for future development.
