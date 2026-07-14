import os
import time
import json
import asyncio
import aiohttp
from aiohttp_socks import ProxyConnector
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("TelegramWorker")

QUEUE_DIR = Path(__file__).resolve().parent.parent / "core" / "signals" / "telegram_queue"
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
API_BASE = os.environ.get("TELEGRAM_API_BASE", "https://api.telegram.org")

async def send_to_telegram(session, message_data):
    url = f"{API_BASE}/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message_data.get("text"),
        "parse_mode": "HTML"
    }
    
    # Использование локального SOCKS5 прокси
    proxy = "socks5://127.0.0.1:10808"
    
    try:
        # Увеличиваем таймаут до 30 секунд для нестабильной сети
        async with session.post(url, json=payload, proxy=proxy, timeout=30) as response:
            if response.status == 200:
                return True
            
            err_text = await response.text()
            if response.status == 429:
                try:
                    data = json.loads(err_text)
                    retry_after = data.get('parameters', {}).get('retry_after', 30)
                except:
                    retry_after = 30
                logger.warning(f"Telegram 429: Rate limit hit. Sleeping for {retry_after}s")
                await asyncio.sleep(retry_after)
                return False
            
            logger.error(f"Telegram API Error: {err_text}")
            return False
    except asyncio.TimeoutError:
        logger.warning("Telegram connection timed out. Will retry.")
        return False
    except Exception as e:
        logger.error(f"Connection error: {type(e).__name__} - {e}")
        return False

async def worker():
    logger.info("Telegram Sender Worker started.")
    if not TOKEN or not CHAT_ID:
        logger.error("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env")
        return

    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Используем SOCKS5 прокси для обхода блокировки
    connector = ProxyConnector.from_url("socks5://127.0.0.1:10808", ssl=False)
    async with aiohttp.ClientSession(connector=connector, trust_env=False) as session:
        while True:
            files = sorted(list(QUEUE_DIR.glob("msg_*.json")))
            
            if not files:
                await asyncio.sleep(1)
                continue
                
            file_path = files[0]
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                success = await send_to_telegram(session, data)
                
                if success:
                    os.remove(file_path)
                    logger.debug(f"Sent and removed: {file_path.name}")
                    # Базовая задержка между сообщениями, чтобы не спамить
                    await asyncio.sleep(1.2)
                else:
                    # Если ошибка, ждем немного перед следующей попыткой
                    await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Worker error: {e}")
                # Переименуем файл, чтобы он не блокировал очередь, если он битый
                file_path.rename(file_path.with_suffix('.error'))
                await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(worker())
