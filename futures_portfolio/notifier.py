import os
import aiohttp
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("Notifier")

class TelegramNotifier:
    def __init__(self):
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        self.enabled = all([self.token, self.chat_id])
        
        if not self.enabled:
            logger.warning("Telegram Notifier disabled: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not found in .env")

    async def send_message(self, text: str):
        if not self.enabled:
            return

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as response:
                    if response.status != 200:
                        err_text = await response.text()
                        logger.error(f"Telegram API Error: {err_text}")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")

    async def send_alert(self, title: str, message: str):
        """Отправка важного уведомления (например, срабатывание стопа)."""
        formatted_text = f"<b>⚠️ {title}</b>\n\n{message}"
        await self.send_message(formatted_text)

    async def send_status(self, bot_name: str, tpv: float, profit: float, cycles: int):
        """Отправка регулярного статуса."""
        formatted_text = (
            f"<b>🤖 Bot Status: {bot_name}</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 TPV: <code>{tpv:.2f} USDT</code>\n"
            f"📈 Net Profit: <code>{profit:+.2f} USDT</code>\n"
            f"🔄 Cycles: <code>{cycles}</code>\n"
            f"━━━━━━━━━━━━━━━━━━"
        )
        await self.send_message(formatted_text)
