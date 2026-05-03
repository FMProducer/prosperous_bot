import asyncio
import json
import os
import logging
import time
import glob
from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from notifier import TelegramNotifier
from storage import safe_load_json

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
        self.notifier = None

    def load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read config {self.config_path}: {e}")
            return {}

    async def collect_and_send(self) -> None:
        config = self.load_config()
        if not config.get("telegram_enabled", True):
            return
            
        if self.notifier is None:
            self.notifier = TelegramNotifier()

        # Параметры для ROI (из swarm_analyzer logic)
        try:
            initial_per_bot = config['portfolios'][0].get('initial_capital', 39.0)
            max_bots = config.get('max_bots', 10)
            working_capital = initial_per_bot * max_bots
            active_tickers = config.get('tickers', [])
            live_swarm = config.get('live_swarm', [])
        except:
            initial_per_bot = 39.0
            working_capital = 390.0
            active_tickers = []
            live_swarm = []

        state_files = await asyncio.to_thread(glob.glob, "state_*.json")
        summary_lines = []
        total_profit = 0.0
        total_safe = 0.0
        active_bots_count = 0
        active_pnl = 0.0
        removed_pnl = 0.0

        # Для PAPER ботов нам нужно читать paper_state_*.json чтобы получить реальный баланс (как в swarm_analyzer)
        for f_path in state_files:
            try:
                state = await safe_load_json(f_path, {})
                
                ticker = state.get("base_ticker", "UNKNOWN")
                if ticker == "UNKNOWN": continue
                
                active_bots_count += 1
                
                # Читаем баланс из paper_state если он есть
                paper_state_path = f"paper_state_{ticker}.json"
                ps = await safe_load_json(paper_state_path, {})
                paper_balance = ps.get("balance", initial_per_bot)

                siphoned = state.get("siphoning_reserve", 0.0)
                # Расчет профита: (Баланс - Начальный) + SAFE (как в swarm_analyzer)
                profit = (paper_balance - initial_per_bot) + siphoned
                
                cycles = state.get("rebalance_cycles", 0)
                last_update = state.get("last_update", 0)
                
                # Общие итоги
                total_profit += profit
                total_safe += siphoned
                
                # Проверка активности процесса (5 мин)
                is_active_process = (time.time() - last_update) < 300 if last_update > 0 else False
                
                # Логика "Active" vs "Removed" для шапки
                if ticker in active_tickers:
                    active_pnl += profit
                else:
                    removed_pnl += profit

                # НОВАЯ ЛОГИКА ЦВЕТОВ: 
                # 🟢 - Активен в REAL (в списке live_swarm)
                # 🟡 - Активен в PAPER (не в live_swarm, но в активных)
                # 🔴 - Неактивен (процесс давно не обновлялся)
                if is_active_process:
                    status_icon = "🟢" if ticker in live_swarm else "🟡"
                else:
                    status_icon = "🔴"
                
                line = f"{status_icon} <b>{ticker}</b>: <code>{profit:+.2f}</code> USDT ({cycles} cyc)"
                if siphoned > 0:
                    line += f" 🛡️<code>{siphoned:.2f}</code>"
                summary_lines.append((profit, line)) # Сохраняем с профитом для сортировки
            except Exception as e:
                logger.error(f"Error reading {f_path}: {e}")

        if not summary_lines:
            logger.info("No bot states found to aggregate.")
            return

        # Сортировка по профиту (как в swarm_analyzer)
        summary_lines.sort(key=lambda x: x[0], reverse=True)
        lines_text = [x[1] for x in summary_lines]

        # Формируем шапку как в swarm_analyzer
        roi = (total_profit / working_capital) * 100 if working_capital > 0 else 0
        
        header = (
            f"📊 <b>Swarm Summary</b> ({datetime.now().strftime('%H:%M')})\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 Total Net Profit: <code>{total_profit:+.2f} USDT</code>\n"
            f"🛡️ Total SAFE Reserve: <code>{total_safe:+.2f} USDT</code>\n"
            f"📈 Overall ROI: <code>{roi:.2f}%</code> (of {working_capital:.1f})\n"
            f"✅ Active Bots PnL: <code>{active_pnl:+.2f} USDT</code>\n"
            f"🗑️ Removed Bots PnL: <code>{removed_pnl:+.2f} USDT</code>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
        )
        
        message = header + "\n".join(lines_text)
        
        # Отправляем в Telegram
        success = await self.notifier.send_message(message)
        if success:
            logger.info(f"Summary sent for {active_bots_count} bots. Total PnL: {total_profit:+.2f}")
        else:
            logger.warning("Failed to send summary to Telegram (throttled or disabled).")

    async def run(self) -> None:
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
