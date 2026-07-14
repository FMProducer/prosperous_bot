import os
import asyncio
import aiohttp
from aiohttp_socks import ProxyConnector
import json
import logging
import time
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("Notifier")

class TelegramNotifier:
    def __init__(self):
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        self.api_base = os.environ.get("TELEGRAM_API_BASE", "https://api.telegram.org")
        
        # Загружаем настройки из config.json
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        self.config = self._load_config()
        self.enabled = self._is_enabled()
        
        # Очередь сообщений для предотвращения 429
        # Используем значение из конфига или True по умолчанию
        self.use_queue = self.config.get("telegram_use_queue", True)
        self.queue_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signals", "telegram_queue")
        
        if self.use_queue:
            os.makedirs(self.queue_dir, exist_ok=True)
            logger.info(f"Telegram Queue ENABLED: {self.queue_dir}")
        else:
            logger.info("Telegram Queue DISABLED")

        self._session = None
        # Максимальное время ожидания при 429, чтобы не блокировать логику бота
        self.max_retry_wait = 30 
        
        if not self.enabled:
            if not self.token or not self.chat_id:
                logger.warning("Telegram Notifier disabled: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not found in .env")
            else:
                logger.info("Telegram Notifier disabled via config.json")

    def _load_config(self) -> dict:
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except:
            pass
        return {}

    def _is_enabled(self) -> bool:
        """Проверка включен ли Telegram в .env и config.json"""
        env_ok = all([self.token, self.chat_id])
        if not env_ok: return False
        return self.config.get("telegram_enabled", True)

    async def _get_session(self):
        if self._session is None or self._session.closed:
            connector = ProxyConnector.from_url(self._proxy, ssl=False)
            self._session = aiohttp.ClientSession(connector=connector, trust_env=False)
        return self._session

    @property
    def _proxy(self):
        return os.environ.get("TG_PROXY", "socks5://127.0.0.1:10808")

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def send_message(self, text: str, max_retries: int = 2, force_direct: bool = False):
        if not self.enabled:
            return False

        if self.use_queue and not force_direct:
            return await self._queue_message({"type": "text", "text": text})

        url = f"{self.api_base}/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML"
        }
        # ... rest of the existing send_message logic ...

        for attempt in range(max_retries):
            try:
                session = await self._get_session()
                headers = {'User-Agent': 'ProsperousBot/1.0'}
                async with session.post(url, json=payload, headers=headers, timeout=10, proxy=self._proxy) as response:
                    if response.status == 200:
                        return True
                    
                    err_text = await response.text()
                    if response.status == 429:
                        try:
                            data = json.loads(err_text)
                            retry_after = data.get('parameters', {}).get('retry_after', 5)
                        except:
                            retry_after = 5
                        
                        if retry_after > self.max_retry_wait:
                            logger.error(f"Telegram 429: Retry-After ({retry_after}s) too long. Skipping message.")
                            return False
                        
                        logger.warning(f"Telegram 429: Too Many Requests. Retrying in {retry_after}s... (Attempt {attempt+1}/{max_retries})")
                        await asyncio.sleep(retry_after + 0.5)
                        continue
                    
                    logger.error(f"Telegram API Error (Status {response.status}): {err_text}")
                    return False
            except Exception as e:
                logger.error(f"Telegram Connection Error: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                return False
        return False

    async def send_photo(self, photo_path: str, caption: str = "", max_retries: int = 2, force_direct: bool = False):
        if not self.enabled or not os.path.exists(photo_path):
            return False

        if self.use_queue and not force_direct:
            return await self._queue_message({"type": "photo", "path": photo_path, "caption": caption})

        url = f"{self.api_base}/bot{self.token}/sendPhoto"
        
        for attempt in range(max_retries):
            try:
                session = await self._get_session()
                data = aiohttp.FormData()
                data.add_field('chat_id', self.chat_id)
                data.add_field('caption', caption)
                data.add_field('parse_mode', 'HTML')
                
                with open(photo_path, 'rb') as f:
                    data.add_field('photo', f.read(), filename=os.path.basename(photo_path))
                
                headers = {'User-Agent': 'ProsperousBot/1.0'}
                async with session.post(url, data=data, headers=headers, timeout=20, proxy=self._proxy) as response:
                    if response.status == 200:
                        return True
                    
                    err_text = await response.text()
                    if response.status == 429:
                        try:
                            data_json = json.loads(err_text)
                            retry_after = data_json.get('parameters', {}).get('retry_after', 5)
                        except:
                            retry_after = 5
                        
                        if retry_after > self.max_retry_wait:
                            logger.error(f"Telegram 429 (Photo): Retry-After ({retry_after}s) too long. Skipping.")
                            return False

                        logger.warning(f"Telegram 429 (Photo): Too Many Requests. Retrying in {retry_after}s... (Attempt {attempt+1}/{max_retries})")
                        await asyncio.sleep(retry_after + 0.5)
                        continue
                    
                    logger.error(f"Telegram API Error (Status {response.status}): {err_text}")
                    return False
            except Exception as e:
                logger.error(f"Telegram Connection Error (Photo): {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                return False
        return False

    async def _queue_message(self, data: dict) -> bool:
        """Сохраняет сообщение в файл для последующей отправки централизованным сервисом"""
        try:
            timestamp = time.time()
            filename = f"msg_{int(timestamp * 1000)}_{os.getpid()}.json"
            filepath = os.path.join(self.queue_dir, filename)
            
            async with asyncio.Lock(): # Локальный лок для безопасности в рамках одного процесса
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to queue telegram message: {e}")
            return False

    async def send_alert(self, title: str, message: str, force_direct: bool = True):
        formatted_text = f"<b>⚠️ {title}</b>\n\n{message}"
        await self.send_message(formatted_text, force_direct=force_direct)

    async def send_status(self, bot_name: str, tpv: float, profit: float, cycles: int, safe_reserve: float, total_balance: float = None, bnb_balance: float = None):
        active_balance = tpv - safe_reserve
        balance_str = f"🏦 Account: <code>{total_balance:.2f} USDT</code>\n" if total_balance is not None else ""
        bnb_str = f"🪙 BNB Fee: <code>{bnb_balance:.4f} BNB</code>\n" if bnb_balance is not None else ""
        formatted_text = (
            f"<b>🤖 Bot Status: {bot_name}</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 Balance: <code>{active_balance:.2f} USDT</code>\n"
            f"📈 Net Profit: <code>{profit:+.2f} USDT</code>\n"
            f"🔄 Cycles: <code>{cycles}</code>\n"
            f"🛡️ SAFE: <code>{safe_reserve:.2f} USDT</code>\n"
            f"{balance_str}"
            f"{bnb_str}"
            f"━━━━━━━━━━━━━━━━━━"
        )
        return await self.send_message(formatted_text)
