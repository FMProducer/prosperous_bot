import os
import json
import time
import asyncio
import random
import aiofiles
from filelock import FileLock, Timeout
from typing import Dict, Any

async def safe_load_json(path: str, default: Dict[str, Any], retries: int = 15) -> Dict[str, Any]:
    lock_path = f"{path}.lock"
    try:
        if os.path.exists(lock_path) and time.time() - os.path.getmtime(lock_path) > 60:
            os.remove(lock_path)
    except FileNotFoundError:
        pass

    lock = FileLock(lock_path, timeout=30)
    for attempt in range(retries):
        try:
            if not os.path.exists(path): return default
            await asyncio.to_thread(lock.acquire)
            try:
                async with aiofiles.open(path, "r", encoding="utf-8") as f:
                    content = await f.read()
                    return json.loads(content)
            finally:
                await asyncio.to_thread(lock.release)
        except (Timeout, json.JSONDecodeError, PermissionError) as e:
            if attempt == retries - 1: raise e
            await asyncio.sleep(0.5 + random.random())
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
