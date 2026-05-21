import asyncio
import json
import os
import logging
import time
import glob
import sys
import io
from typing import Dict, Any, List, Set
from datetime import datetime
from dotenv import load_dotenv
from storage import safe_load_json, safe_load_json_sync

# Force UTF-8 for Windows streams
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

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
    def __init__(self, config_path="config.json"):
        # If config_path is just a filename, assume it's in the same directory as this script
        if not os.path.isabs(config_path) and not os.path.exists(config_path):
            potential_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
            if os.path.exists(potential_path):
                config_path = potential_path

        self.config_path = config_path
        self.notifier = None
        self.connector = None  # Инициализируем один раз для повторного использования
        self.queue_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signals", "telegram_queue")
        os.makedirs(self.queue_dir, exist_ok=True)

    def load_config(self) -> Dict[str, Any]:
        """Синхронная загрузка конфига для совместимости."""
        return safe_load_json_sync(self.config_path, {})

    async def load_config_async(self) -> Dict[str, Any]:
        return await safe_load_json(self.config_path, {})

    async def init_services(self) -> None:
        """Делегированная асинхронная инициализация долгоживущих клиентов."""
        config = await self.load_config_async()

        # Инициализируем TelegramNotifier, если еще не создан
        if not self.notifier:
            from notifier import TelegramNotifier
            self.notifier = TelegramNotifier(config)

        # Инициализируем BinanceConnector один раз (Keep-Alive)
        if not self.connector:
            from connector import BinanceConnector
            api_key = config.get("api_key", "")
            secret_key = config.get("secret_key", "")
            testnet = config.get("testnet", True)
            self.connector = BinanceConnector(
                api_key=api_key,
                secret_key=secret_key,
                testnet=testnet
            )

    async def close_services(self) -> None:
        """Детерминированное освобождение ресурсов при остановке агрегатора."""
        if self.connector and hasattr(self.connector, 'futures_client') and self.connector.futures_client:
            logger.info("Closing Binance AsyncClient session...")
            try:
                # Явное закрытие сессии aiohttp внутри python-binance AsyncClient
                await self.connector.futures_client.close_connection()
            except Exception as e:
                logger.error(f"Failed to close futures_client connection cleanly: {e}")

        if self.notifier:
            logger.info("Closing TelegramNotifier session...")
            await self.notifier.close()

    async def process_telegram_queue(self):
        """Обработка очереди сообщений от других ботов с соблюдением лимитов Telegram"""
        logger.info("Telegram Queue Processor started.")
        while True:
            try:
                if self.notifier is None:
                    await asyncio.sleep(1)
                    continue

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

    async def _generate_swarm_section(self, files: List[str], live_swarm: List[str], active_tickers: List[str], label: str, config: Dict[str, Any]):
        total_profit = 0.0
        total_safe = 0.0
        summary_lines = []
        
        for f_path in files:
            try:
                # В асинхронном контексте читаем через safe_load_json
                state = await safe_load_json(f_path, {})
                if not state:
                    continue
                
                ticker = state.get("base_ticker", "UNKNOWN")
                profit = state.get("last_profit", 0.0)
                cycles = state.get("rebalance_cycles", 0)
                siphoned = state.get("siphoning_reserve", 0.0)
                
                # РАСЧЕТ ЭФФЕКТИВНОСТИ (Profit per Cycle)
                min_cycles = config.get("min_cycles_for_rank", 20)
                efficiency = profit / max(cycles, min_cycles)
                
                total_profit += profit
                total_safe += siphoned
                
                if label == "INCUBATOR":
                    is_active = ticker in active_tickers
                else:
                    is_active = ticker in live_swarm
                
                status_icon = "🟢" if is_active else "💤"
                real_whitelist = config.get("real_whitelist", [])
                vetted_icon = " 🛡️" if ticker in real_whitelist else ""
                
                line = f"{status_icon} <b>{ticker}</b>{vetted_icon}: <code>{profit:+.2f}</code> USDT {cycles} cyc"
                if siphoned > 0:
                    line += f" 🛡️<code>{siphoned:.2f}</code>"
                summary_lines.append((efficiency, line))
            except Exception as e:
                logger.error(f"Error reading {f_path}: {e}")

        if not summary_lines:
            return "", 0.0, 0.0

        summary_lines.sort(key=lambda x: x[0], reverse=True)
        section_text = f"<b>{label} SWARM</b>\n" + "\n".join([x[1] for x in summary_lines]) + "\n"
        return section_text, total_profit, total_safe

    async def collect_and_send(self):
        config = await self.load_config_async()
        if not config: return
        if not config.get("telegram_enabled", True): return
        
        await self.init_services()

        wallet_usdt = 0.0
        wallet_bnb = 0.0
        
        try:
            # Переиспользуем существующий self.connector
            wallet_usdt = await self.connector.get_free_balance()
            wallet_bnb = await self.connector.get_bnb_balance()
        except Exception as e:
            logger.error(f"Failed to fetch real balances: {e}")

        initial_per_bot = config.get('portfolios', [{}])[0].get('initial_capital', 60.0)
        live_swarm = config.get("live_swarm", [])
        active_tickers = config.get("tickers", [])
        
        # Look for state files in the same directory as config if possible
        state_dir = os.path.dirname(self.config_path) if self.config_path else "."
        paper_files = glob.glob(os.path.join(state_dir, "paper_state_*.json"))
        real_files = glob.glob(os.path.join(state_dir, "real_state_*.json"))
        
        combat_text, c_profit, c_safe = await self._generate_swarm_section(real_files, live_swarm, active_tickers, "COMBAT", config)
        incubator_text, i_profit, i_safe = await self._generate_swarm_section(paper_files, live_swarm, active_tickers, "INCUBATOR", config)

        working_capital = initial_per_bot * len(live_swarm) if live_swarm else initial_per_bot
        roi = (c_profit / working_capital) * 100 if working_capital > 0 else 0
        
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

    def execute_aggregation_cycle(self) -> None:
        """
        Прокси-метод для обратной совместимости с внешним диспетчером PM2/Supervisor.
        Направляет вызов на актуальную внутреннюю логику.
        """
        try:
            asyncio.run(self.collect_and_send())
        except Exception as e:
            logger.error(f"Critical error inside explicit aggregation cycle: {e}")

    async def run(self) -> None:
        logger.info("Status Aggregator started.")
        
        # Запускаем обработчик очереди Telegram в фоновом режиме
        asyncio.create_task(self.process_telegram_queue())

        try:
            while True:
                config = await self.load_config_async()
                interval_min = config.get("telegram_summary_interval_min", 1)

                try:
                    # Within an async loop, we call the async method directly
                    await self.collect_and_send()
                except Exception as e:
                    logger.error(f"Error during aggregation cycle: {e}", exc_info=True)

                await asyncio.sleep(interval_min * 60)
        finally:
            # Гарантируем закрытие при выходе из run
            await self.close_services()

if __name__ == "__main__":
    aggregator = StatusAggregator()
    try:
        asyncio.run(aggregator.run())
    except KeyboardInterrupt:
        pass
