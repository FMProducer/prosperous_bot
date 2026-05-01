import os
import json
import time
import asyncio
import random
import tempfile
from typing import Dict, Any

async def safe_load_json(path: str, default: Dict[str, Any], retries: int = 15) -> Dict[str, Any]:
    """
    Безопасное чтение JSON без блокировок.
    Использует ретраи при ошибках чтения или декодирования.
    """
    def _read_json() -> Dict[str, Any]:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            if not content:
                return default
            return json.loads(content)

    for attempt in range(retries):
        try:
            return await asyncio.to_thread(_read_json)
        except (json.JSONDecodeError, PermissionError, OSError) as e:
            if attempt == retries - 1:
                return default
            await asyncio.sleep(0.05 + random.random() * 0.1)
    return default

async def safe_save_json(path: str, data: Dict[str, Any]) -> None:
    """
    Атомарная запись JSON через временный файл (без filelock).
    Гарантирует целостность данных при сбоях.
    """
    def _atomic_write() -> None:
        dir_name = os.path.dirname(os.path.abspath(path))
        os.makedirs(dir_name, exist_ok=True)

        fd, temp_path = tempfile.mkstemp(dir=dir_name, prefix=".tmp_state_", suffix=".json")
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            # os.replace атомарен на большинстве систем (POSIX и Windows)
            # В Windows может бросить PermissionError, если файл открыт другим процессом
            # Но в нашей архитектуре мы минимизируем время открытия файлов на чтение
            max_retries = 5
            for i in range(max_retries):
                try:
                    os.replace(temp_path, path)
                    break
                except PermissionError:
                    if i == max_retries - 1:
                        raise
                    time.sleep(0.05 * (i + 1))
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    await asyncio.to_thread(_atomic_write)
