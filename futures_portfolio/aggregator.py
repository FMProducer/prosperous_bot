import asyncio
import json
import os
import logging
import time
import glob
from datetime import datetime
from dotenv import load_dotenv
from notifier import TelegramNotifier

load_dotenv()

# Настройка логирования
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "aggregator.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Aggregator")

class StatusAggregator:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.notifier = TelegramNotifier()
        
    def load_config(self):
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    async def collect_and_send(self):
        config = self.load_config()
        if not config.get("telegram_enabled", True):
            logger.info("Telegram is disabled in config. Skipping summary.")
            return

        state_files = glob.glob("state_*.json")
        summary_lines = []
        total_profit = 0.0
        active_bots = 0

        for f_path in state_files:
            try:
                with open(f_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                
                ticker = state.get("base_ticker", "UNKNOWN")
                if ticker == "UNKNOWN": continue
                
                last_tpv = state.get("last_tpv", 0)
                initial_tpv = state.get("initial_tpv", 0)
                profit = state.get("last_profit", last_tpv - initial_tpv if initial_tpv > 0 else 0)
                siphoned = state.get("siphoning_reserve", 0.0)
                cycles = state.get("rebalance_cycles", 0)
                last_update = state.get("last_update", 0)
                
                # Проверка на "протухание" данных (5 минут)
                is_active = (time.time() - last_update) < 300 if last_update > 0 else False
                
                if is_active:
                    active_bots += 1
                    total_profit += profit
                    status_icon = "🟢"
                else:
                    status_icon = "🔴"
                
                line = f"{status_icon} <b>{ticker}</b>: <code>{profit:+.2f}</code> USDT ({cycles} cyc)"
                if siphoned > 0:
                    line += f" 🛡️<code>{siphoned:.2f}</code>"
                summary_lines.append(line)
            except Exception as e:
                logger.error(f"Error reading {f_path}: {e}")

        if not summary_lines:
            logger.info("No bot states found to aggregate.")
            return

        header = f"📊 <b>Swarm Summary</b> ({datetime.now().strftime('%H:%M')})\n"
        header += f"Bots: {active_bots} | Total PnL: <code>{total_profit:+.2f} USDT</code>\n"
        header += "━━━━━━━━━━━━━━━━━━\n"
        
        message = header + "\n".join(summary_lines)
        
        # Отправляем в Telegram
        success = await self.notifier.send_message(message)
        if success:
            logger.info(f"Summary sent for {active_bots} bots. Total PnL: {total_profit:+.2f}")
        else:
            logger.warning("Failed to send summary to Telegram (throttled or disabled).")

    async def run(self):
        logger.info("Status Aggregator started.")
        while True:
            config = self.load_config()
            interval_min = config.get("telegram_summary_interval_min", 1)
            
            try:
                await self.collect_and_send()
            except Exception as e:
                logger.error(f"Error in aggregator loop: {e}")
            
            await asyncio.sleep(interval_min * 60)

if __name__ == "__main__":
    aggregator = StatusAggregator()
    asyncio.run(aggregator.run())
