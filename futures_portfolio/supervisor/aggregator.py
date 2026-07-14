import asyncio
import json
import os
import logging
import time
import glob
import sys
import io
from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from core.notifier import TelegramNotifier
from core.storage import safe_load_json
from core.connector import BinanceConnector

load_dotenv()

# Настройка логирования
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "aggregator.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Aggregator")

class StatusAggregator:
    # CORE_DIR: корневая папка пакета, где лежат core/, supervisor/, monitoring/
    CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # futures_portfolio/
    STATE_DIR = os.path.join(CORE_DIR, "core")
    QUEUE_DIR = os.path.join(STATE_DIR, "signals", "telegram_queue")

    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.notifier = None
        self.queue_dir = self.QUEUE_DIR
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

    async def _generate_swarm_section(self, files, live_swarm, active_tickers, label):
        total_profit = 0.0
        total_safe = 0.0
        summary_lines = []
        
        # Получаем конфиг для расчета ROI в этой секции если нужно
        config = self.load_config()

        for f_path in files:
            try:
                # В асинхронном контексте читаем через safe_load_json
                state = await safe_load_json(f_path, {})
                if not state:
                    continue
                
                ticker = state.get("base_ticker", "UNKNOWN")
                profit = float(state.get("last_profit", 0.0))
                cycles = int(state.get("rebalance_cycles", 0))
                siphoned = float(state.get("siphoning_reserve", 0.0))
                
                # РАСЧЕТ ЭФФЕКТИВНОСТИ (Profit per Cycle)
                # Используем min_cycles=20 для стабилизации рейтинга новичков
                min_cycles = config.get("min_cycles_for_rank", 20)
                efficiency = profit / max(cycles, min_cycles)
                
                total_profit += profit
                total_safe += siphoned
                
                # РЕЖИМ ОТОБРАЖЕНИЯ: 🟢 - работает, 💤 - остановлен (хранит историю)
                if label == "INCUBATOR":
                    is_active = ticker in active_tickers
                else:
                    is_active = ticker in live_swarm
                
                status_icon = "🟢" if is_active else "💤"
                
                # Показываем щит если тикер в белом списке (доверенный для реала)
                real_whitelist = config.get("real_whitelist", [])
                vetted_icon = " 🛡️" if ticker in real_whitelist else ""
                
                # Чистый формат без скобок и лишних слов
                line = f"{status_icon} <b>{ticker}</b>{vetted_icon}: <code>{profit:+.2f}</code> USDT {cycles} cyc"
                if siphoned > 0:
                    line += f" 🛡️<code>{siphoned:.2f}</code>"
                summary_lines.append((efficiency, line))
            except Exception as e:
                logger.error(f"Error reading {f_path}: {e}")

        if not summary_lines:
            return "", 0.0, 0.0

        # Сортировка по ЭФФЕКТИВНОСТИ (x[0] теперь содержит efficiency)
        summary_lines.sort(key=lambda x: x[0], reverse=True)
        section_text = f"<b>{label} SWARM</b>\n" + "\n".join([x[1] for x in summary_lines]) + "\n"
        return section_text, total_profit, total_safe

    async def collect_and_send(self):
        config = self.load_config()
        if not config: return
        if not config.get("telegram_enabled", True): return
        
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
            await connector.verify_connection()
            wallet_usdt = await connector.get_free_balance()
            wallet_bnb = await connector.get_bnb_balance()
        except Exception as e:
            logger.error(f"Failed to fetch real balances: {e}")

        # Параметры
        initial_per_bot = config['portfolios'][0].get('initial_capital', 60.0)
        live_swarm = config.get("live_swarm", [])
        active_tickers = config.get("tickers", [])
        
        # Разделяем стейты — ищем в core/ где боты реально пишут
        paper_files = glob.glob(os.path.join(self.STATE_DIR, "paper_state_*.json"))
        real_files = glob.glob(os.path.join(self.STATE_DIR, "real_state_*.json"))
        
        combat_text, c_profit, c_safe = await self._generate_swarm_section(real_files, live_swarm, active_tickers, "COMBAT")
        incubator_text, i_profit, i_safe = await self._generate_swarm_section(paper_files, live_swarm, active_tickers, "INCUBATOR")

        total_profit = c_profit # ROI считаем только по реальным деньгам
        working_capital = initial_per_bot * len(live_swarm) if live_swarm else initial_per_bot
        roi = (total_profit / working_capital) * 100 if working_capital > 0 else 0
        
        header = (
            f"📊 <b>Swarm Summary</b> ({datetime.now().strftime('%H:%M')})\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"⚔️ Combat PnL: <code>{c_profit:+.2f} USDT</code>\n"
            f"🧪 Incubator PnL: <code>{i_profit:+.2f} USDT</code>\n"
            f"📈 Combat ROI: <code>{roi:.2f}%</code>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💳 Wallet USDT: <code>{wallet_usdt:.2f}</code>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
        )
        
        message = header + combat_text + "\n" + incubator_text
        
        await self.notifier.send_message(message, force_direct=True)

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
    # Force UTF-8 for Windows streams (only when running directly)
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    aggregator = StatusAggregator()
    asyncio.run(aggregator.run())
