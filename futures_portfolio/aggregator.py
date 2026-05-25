import asyncio
import json
import os
import logging
import time
import glob
import sys
import io
from typing import Dict, Any, List, Set, Tuple
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
        # Абсолютное разрешение путей для запуска из любого каталога (PM2 root safety)
        if not os.path.isabs(config_path):
            self.config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), config_path))
        else:
            self.config_path = config_path
            
        self.connector = None
        self.notifier = None
        self.telegram_queue: asyncio.Queue = asyncio.Queue()

    def load_config(self) -> Dict[str, Any]:
        """Синхронная загрузка конфигурации (SSOT)"""
        return safe_load_json_sync(self.config_path, {})

    async def load_config_async(self) -> Dict[str, Any]:
        """Асинхронная обертка для совместимости"""
        return await asyncio.to_thread(self.load_config)

    async def init_services(self, config: Dict[str, Any]) -> None:
        """Ленивая инициализация коннекторов и нотификаторов с защитой от дублирования"""
        if self.connector is None:
            # Импортируем локально, чтобы избежать циклических зависимостей при вызове из супервайзера
            from connector import BinanceConnector
            api_key = os.environ.get("BINANCE_API_KEY", config.get("api_key", ""))
            secret_key = os.environ.get("BINANCE_SECRET_KEY", config.get("secret_key", ""))
            self.connector = BinanceConnector(api_key=api_key, secret_key=secret_key, testnet=config.get("testnet", True))
            logger.info("Binance Connector initialized in Aggregator.")

        if self.notifier is None:
            from notifier import TelegramNotifier
            self.notifier = TelegramNotifier(config=config)
            logger.info("Telegram Notifier initialized in Aggregator.")

    async def close_services(self) -> None:
        """Безопасное закрытие ресурсов"""
        if self.notifier:
            try:
                await self.notifier.close()
            except Exception as e:
                logger.error(f"Error closing notifier: {e}")
            self.notifier = None
        if self.connector:
            try:
                if hasattr(self.connector, "close"):
                    await self.connector.close()
            except Exception as e:
                logger.error(f"Error closing connector: {e}")
            self.connector = None

    async def process_telegram_queue(self) -> None:
        """Фоновый воркер обработки очереди Telegram сообщений"""
        while True:
            try:
                msg = await self.telegram_queue.get()
                if self.notifier:
                    await self.notifier.send_message(msg, force_direct=True)
                self.telegram_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in telegram queue processor: {e}")
                await asyncio.sleep(1)

    def _generate_swarm_section(self, state_files: List[str], config: Dict[str, Any], filter_set: Set[str], is_combat: bool = True) -> Tuple[List[str], float, float, float]:
        """
        Генерирует блок строк для определенной группы ботов (COMBAT или INCUBATOR).
        Возвращает (список строк, общий TPV, общая прибыль, общий сейф).
        """
        summary_lines = []
        total_tpv = 0.0
        total_profit = 0.0
        total_safe = 0.0

        real_whitelist = config.get("real_whitelist", [])
        active_tickers = set(config.get("tickers", []))

        for sf in state_files:
            filename = os.path.basename(sf)
            ticker = filename.replace("paper_state_", "").replace(".json", "")

            # Фильтр: если мы в режиме COMBAT, берем только тех, кто в filter_set
            if is_combat:
                if ticker not in filter_set: continue
            else:
                if ticker in filter_set: continue

            state = safe_load_json_sync(sf, {})
            if not state:
                continue

            tpv = state.get("tpv", state.get("initial_capital", 0.0))
            profit = state.get("total_pnl", state.get("last_profit", 0.0))
            cycles = state.get("rebalance_cycles", 0)
            siphoned = state.get("siphoning_reserve", 0.0)

            # Extract shares
            share_long = state.get("share_long_pct", 0.0)
            share_short = state.get("share_short_pct", 0.0)
            share_virt = state.get("share_virt_pct", 0.0)

            # РАСЧЕТ ЭФФЕКТИВНОСТИ (Profit per Cycle)
            min_cycles = config.get("min_cycles_for_rank", 20)
            efficiency = profit / max(cycles, min_cycles)

            total_tpv += tpv
            total_profit += profit
            total_safe += siphoned

            is_active = ticker in active_tickers
            status_icon = "🟢" if is_active else "💤"
            vetted_icon = " 🛡️" if ticker in real_whitelist else ""

            line = f"{status_icon} <b>{ticker}</b>{vetted_icon}: <code>{profit:+.2f}</code> USDT | L:{share_long:.1f}% S:{share_short:.1f}% V:{share_virt:.1f}% | {cycles} cyc"
            if siphoned > 0:
                line += f" 🛡️<code>{siphoned:.2f}</code>"

            summary_lines.append((efficiency, line))

        # Сортировка по эффективности
        summary_lines.sort(key=lambda x: x[0], reverse=True)
        lines = [item[1] for item in summary_lines]

        return lines, total_tpv, total_profit, total_safe

    async def collect_and_send(self) -> None:
        """Асинхронное ядро сбора метрик и отправки сводки"""
        config = await self.load_config_async()
        await self.init_services(config)

        # Поиск стейтов в директории скрипта
        base_dir = os.path.dirname(self.config_path)
        state_files = glob.glob(os.path.join(base_dir, "paper_state_*.json"))
        
        if not state_files:
            logger.info("No active paper states found for aggregation.")
            return

        live_swarm_set = set(config.get("live_swarm", []))

        combat_lines, combat_tpv, combat_profit, combat_safe = self._generate_swarm_section(state_files, config, live_swarm_set, is_combat=True)
        incubator_lines, incubator_tpv, incubator_profit, incubator_safe = self._generate_swarm_section(state_files, config, live_swarm_set, is_combat=False)

        msg_lines = [f"STAT: SWARM REBALANCE SUMMARY\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"]
        
        msg_lines.append(f"COMBAT SWARM (Active: {len(combat_lines)})")
        if combat_lines:
            msg_lines.extend(combat_lines)
            msg_lines.append(f"-> Total Combat TPV: {combat_tpv:.2f}$ | PnL: {combat_profit:+.2f}$")
            if combat_safe > 0:
                msg_lines.append(f"-> Total SAFE Reserve: {combat_safe:.2f} USDT")
        else:
            msg_lines.append("No active combat bots.")
            
        msg_lines.append(f"\nINCUBATOR SWARM (Testing: {len(incubator_lines)})")
        if incubator_lines:
            msg_lines.extend(incubator_lines)
            msg_lines.append(f"-> Total Incubator TPV: {incubator_tpv:.2f}$ | PnL: {incubator_profit:+.2f}$")
        else:
            msg_lines.append("No bots in incubator.")

        alloc_text = f"\nTotal Pool Capitalization: {combat_tpv + incubator_tpv:.2f}$"
        msg_lines.append(alloc_text)

        full_message = "\n".join(msg_lines)
        
        if config.get("telegram_enabled", True) and self.notifier:
            await self.notifier.send_message(full_message, force_direct=True)
        else:
            logger.info(f"Aggregation complete (Telegram disabled):\n{full_message}")

    def execute_aggregation_cycle(self) -> None:
        """
        Главная точка входа для внешних менеджеров процессов (PM2 / Supervisor).
        Она обязана быть синхронной, атомарной и полностью очищать ресурсы после выполнения.
        """
        try:
            # Создаем или получаем loop для синхронного запуска в рамках одного шага
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.collect_and_send())
            finally:
                # Гарантируем закрытие коннекторов и сессий aiohttp внутри этого вызова
                loop.run_until_complete(self.close_services())
                loop.close()
            logger.info("Explicit aggregation cycle executed successfully via PM2 hook.")
        except Exception as e:
            logger.error(f"Critical failure inside execute_aggregation_cycle: {e}", exc_info=True)

    async def run(self) -> None:
        """Точка входа для долгоживущего daemon-процесса (если запускается напрямую)"""
        logger.info("Status Aggregator daemon mode started.")
        config = await self.load_config_async()
        await self.init_services(config)
        
        # Запускаем фоновый обработчик очереди Telegram
        queue_task = asyncio.create_task(self.process_telegram_queue())

        try:
            while True:
                current_cfg = await self.load_config_async()
                interval_min = current_cfg.get("telegram_summary_interval_min", 1)

                try:
                    await self.collect_and_send()
                except Exception as e:
                    logger.error(f"Error during daemon aggregation loop: {e}", exc_info=True)

                await asyncio.sleep(interval_min * 60)
        except asyncio.CancelledError:
            logger.info("Daemon loop canceled.")
        finally:
            queue_task.cancel()
            await asyncio.gather(queue_task, return_exceptions=True)
            await self.close_services()

if __name__ == "__main__":
    # Нам нужно проверить: если этот скрипт вызывается напрямую модулем, запускаем daemon.
    # Если диспетчер импортирует класс и вызывает execute_aggregation_cycle, управление туда не дойдет.
    aggregator = StatusAggregator()
    try:
        asyncio.run(aggregator.run())
    except KeyboardInterrupt:
        logger.info("Aggregator shutdown manually.")
