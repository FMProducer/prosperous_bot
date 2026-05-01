import os
import json
import time
import asyncio
import random
import aiofiles
from filelock import FileLock, Timeout
from typing import Dict, Any

async def safe_load_json(path: str, default: Dict[str, Any], retries: int = 15) -> Dict[str, Any]:
    """Безопасное чтение JSON без блокировок (но с ретраями при ошибках декодирования)."""
    for attempt in range(retries):
        try:
            if not os.path.exists(path): return default
            async with aiofiles.open(path, "r", encoding="utf-8") as f:
                content = await f.read()
                if not content: return default
                return json.loads(content)
        except (json.JSONDecodeError, PermissionError) as e:
            if attempt == retries - 1:
                # Если последний шанс и файл пустой или битый, возвращаем дефолт вместо падения
                return default
            await asyncio.sleep(0.1 + random.random() * 0.2)
    return default

async def safe_save_json(path: str, data: Dict[str, Any]) -> None:
    lock_path = f"{path}.lock"
    lock = FileLock(lock_path, timeout=10)
    await asyncio.to_thread(lock.acquire)
    try:
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, indent=2))
    finally:
        await asyncio.to_thread(lock.release)
