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
from connector import BinanceConnector

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
        self.queue_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signals", "telegram_queue")
        os.makedirs(self.queue_dir, exist_ok=True)

    def load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read config {self.config_path}: {e}")
            return {}

    async def process_telegram_queue(self):
        """Обработка очереди сообщений от других ботов с соблюдением лимитов Telegram"""
        if self.notifier is None:
            self.notifier = TelegramNotifier()
            
        logger.info("Telegram Queue Processor started.")
        while True:
            try:
                msg_files = sorted(glob.glob(os.path.join(self.queue_dir, "*.json")))
                if not msg_files:
                    await asyncio.sleep(1)
                    continue

                # Обрабатываем по одному, чтобы не спамить
                f_path = msg_files[0]
                try:
                    with open(f_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    success = False
                    if data.get("type") == "text":
                        success = await self.notifier.send_message(data["text"], force_direct=True)
                    elif data.get("type") == "photo":
                        success = await self.notifier.send_photo(data["path"], data.get("caption", ""), force_direct=True)
                    
                    if success:
                        if os.path.exists(f_path): os.remove(f_path)
                        # Пауза между сообщениями для соблюдения лимитов
                        await asyncio.sleep(1.2)
                    else:
                        # Если не удалось отправить (например 429), ждем подольше
                        logger.warning(f"Failed to send queued message {f_path}, retrying later...")
                        await asyncio.sleep(5)
                except Exception as e:
                    logger.error(f"Error processing queued message {f_path}: {e}")
                    # Если файл битый, удаляем его
                    if os.path.exists(f_path): os.remove(f_path)

            except Exception as e:
                logger.error(f"Error in telegram queue processor: {e}")
                await asyncio.sleep(5)

    async def collect_and_send(self) -> None:
        config = self.load_config()
        if not config.get("telegram_enabled", True):
            return
            
        if self.notifier is None:
            self.notifier = TelegramNotifier()

        # Получаем реальные балансы с биржи для "Reality Check"
        api_key = os.environ.get("BINANCE_API_KEY", config.get("api_key", ""))
        secret_key = os.environ.get("BINANCE_SECRET_KEY", config.get("secret_key", ""))
        testnet = config.get("testnet", False)
        
        wallet_usdt = 0.0
        wallet_bnb = 0.0
        
        try:
            connector = BinanceConnector(api_key, secret_key, testnet=testnet)
            wallet_usdt = await connector.get_free_balance()
            wallet_bnb = await connector.get_bnb_balance()
        except Exception as e:
            logger.error(f"Failed to fetch real balances: {e}")

        # Параметры для ROI (динамически из конфига)
        try:
            initial_per_bot = config['portfolios'][0].get('initial_capital', 60.0)
            max_bots = config.get('max_bots', 10)
            working_capital = initial_per_bot * max_bots
            active_tickers = config.get('tickers', [])
            live_swarm = config.get('live_swarm', [])
        except:
            initial_per_bot = 60.0
            working_capital = 600.0
            active_tickers = []
            live_swarm = []

        state_files = await asyncio.to_thread(glob.glob, "state_*.json")
        summary_lines = []
        total_profit = 0.0
        total_safe = 0.0
        active_bots_count = 0
        active_pnl = 0.0
        removed_pnl = 0.0

        for f_path in state_files:
            try:
                state = await safe_load_json(f_path, {})
                ticker = state.get("base_ticker", "UNKNOWN")
                if ticker == "UNKNOWN": continue
                
                siphoned = state.get("siphoning_reserve", 0.0)
                
                is_fired = f_path.endswith(".fired")
                if is_fired and "final_profit" in state:
                    profit = state["final_profit"]
                else:
                    profit = state.get("last_profit", 0.0)
                
                cycles = state.get("rebalance_cycles", 0)
                last_update = state.get("last_update", 0)
                
                if ticker in active_tickers and not is_fired:
                    active_bots_count += 1
                    active_pnl += profit
                    total_profit += profit
                    total_safe += siphoned
                    status_icon = "🟢" if ticker in live_swarm else "🟡"
                else:
                    if abs(profit) < 0.01:
                        continue
                    removed_pnl += profit
                    status_icon = "🔴"
                
                line = f"{status_icon} <b>{ticker}</b>: <code>{profit:+.2f}</code> USDT ({cycles} cyc)"
                if siphoned > 0:
                    line += f" 🛡️<code>{siphoned:.2f}</code>"
                summary_lines.append((profit, line))
            except Exception as e:
                logger.error(f"Error reading {f_path}: {e}")

        if not summary_lines:
            logger.info("No bot states found to aggregate.")
            return

        summary_lines.sort(key=lambda x: x[0], reverse=True)
        lines_text = [x[1] for x in summary_lines]

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
            f"💳 Wallet USDT: <code>{wallet_usdt:.2f}</code>\n"
            f"🪙 Wallet BNB: <code>{wallet_bnb:.4f}</code>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
        )
        
        message = header + "\n".join(lines_text)
        
        success = await self.notifier.send_message(message, force_direct=True)
        if success:
            logger.info(f"Summary sent for {active_bots_count} bots. Total PnL: {total_profit:+.2f}")
        else:
            logger.warning("Failed to send summary to Telegram (throttled or disabled).")

    async def run(self) -> None:
        logger.info("Status Aggregator started.")
        # Запускаем обработчик очереди как фоновую задачу
        asyncio.create_task(self.process_telegram_queue())
        
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
