import os
import asyncio
import aiohttp
import json
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("Notifier")

class TelegramNotifier:
    def __init__(self):
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        # По умолчанию используем официальный API, но позволяем сменить на зеркало через .env
        self.api_base = os.environ.get("TELEGRAM_API_BASE", "https://api.telegram.org")
        self.enabled = all([self.token, self.chat_id])
        
        if not self.enabled:
            logger.warning("Telegram Notifier disabled: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not found in .env")

    async def send_message(self, text: str):
        if not self.enabled:
            return

        url = f"{self.api_base}/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML"
        }

        try:
            # trust_env=True позволяет использовать системные прокси (важно для VPN на Windows)
            async with aiohttp.ClientSession(trust_env=True) as session:
                headers = {'User-Agent': 'ProsperousBot/1.0'}
                async with session.post(url, json=payload, headers=headers, timeout=15) as response:
                    if response.status != 200:
                        err_text = await response.text()
                        logger.error(f"Telegram API Error (Status {response.status}): {err_text}")
                        return False
                    return True
        except Exception as e:
            logger.error(f"Telegram Connection Error: {type(e).__name__}: {e}")
            return False

    async def send_alert(self, title: str, message: str):
        """Отправка важного уведомления (например, срабатывание стопа)."""
        formatted_text = f"<b>⚠️ {title}</b>\n\n{message}"
        await self.send_message(formatted_text)

    async def send_status(self, bot_name: str, tpv: float, profit: float, cycles: int, safe_reserve: float, total_balance: float = None, bnb_balance: float = None):
        """Отправка регулярного статуса."""
        # Отображаем Balance как активный капитал (TPV за вычетом сейфа)
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
        await self.send_message(formatted_text)