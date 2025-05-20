# c:\Python\Prosperous_Bot\src\signal_bot.py

import asyncio
import json
import logging
import os
from adaptive_agent.datetime import datetime
from adaptive_agent.pathlib import Path

from src.data_loader import DataLoader
from src.signal_generator import process_all_data
from src.update_distribution import update_distribution
from src.graphs import plot_indicators
from src.utils import save_to_csv, ensure_directory, load_symbols  # Добавлен импорт load_symbols
from src.strategy import MLStrategy, RuleBasedStrategy, HybridStrategy  # Исправлен импорт

# ----
# Константы путей / файлов
# ----
CONFIG_PATH = "config_signal.json"
REBALANCE_CONFIG_PATH = "config.json"
STATE_FILE = "bot_state.json"
GRAPH_DIR = Path(r"C:/Python/Prosperous_Bot/graphs")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SignalBot:
    """Сигнальный бот: получает данные, генерирует сигналы, сохраняет графики."""

    def __init__(self):
        self.load_config()  # Начальная загрузка конфига и чистка
        self.load_state()   # Подгружаем прошлое состояние сигнала из bot_state.json
        self._config_mtime = Path(CONFIG_PATH).stat().st_mtime  # Для проверки hot-reload
        self.update_asset_distribution()  # Применяем распределение активов по сохранённым сигналам
        self.data_loader = DataLoader(self.config)
        self.last_state_save = datetime.now()
        self.last_symbol_refresh = datetime.now()  # Добавлен атрибут для отслеживания времени последнего обновления символов

    # ----
    # Работа с конфигом
    # ----
    def load_config(self):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        syms = load_symbols(self.config)
        self._symbols = syms
        self.config['binance_pairs'] = syms
        self.config.setdefault("last_signals", {})
        
        # 1. Добавляем новые пары, 2. Удаляем устаревшие сигналы
        ls = self.config.setdefault("last_signals", {})
        for pair in self._symbols:
            ls.setdefault(pair, {"signal": "NEUTRAL", "time": datetime.now().isoformat()})
        for old in list(ls):
            if old not in self._symbols:
                del ls[old]
        logger.info("Configuration loaded")
        # --- глобальная очистка: удаляем в graphs всё, чего нет в self._symbols ---
        existing = set()
        for f in GRAPH_DIR.iterdir():
            for suffix in ("_indicators_adx.html", "_data.csv", "_indicators_signals.html"):
                if f.name.endswith(suffix):
                    existing.add(f.name[:-len(suffix)])
        removed = existing - set(self._symbols)
        if removed:
            logger.info("Initial cleanup of obsolete files: %s", removed)
            self._cleanup_obsolete(removed)

    def save_config(self):
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4)
        logger.info("Configuration saved")
        self._config_mtime = Path(CONFIG_PATH).stat().st_mtime  # Сброс _config_mtime после сохранения

    # ----
    # Распределение активов для ребалансировщика
    # ----
    def update_asset_distribution(self):
        try:
            with open(REBALANCE_CONFIG_PATH, "r", encoding="utf-8") as f:
                rebalance_cfg = json.load(f)

            for symbol, info in self.config["last_signals"].items():
                sig = info["signal"].upper()
                dist_map = self.config["asset_distributions"].get(sig)
                if not dist_map:
                    continue
                base = symbol.replace("USDT", "")
                for pair in (f"{base}_USDT", f"{base}5L_USDT", f"{base}5S_USDT"):
                    if pair in dist_map and pair in rebalance_cfg["asset_distribution"]:
                        rebalance_cfg["asset_distribution"][pair] = dist_map[pair]

            with open(REBALANCE_CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(rebalance_cfg, f, indent=4)
            logger.info("Asset distribution updated")
        except Exception as e:
            logger.error("Asset distribution update error: %s", e)

    # ----
    # Обработка одного инструмента: сохранить CSV + график
    # ----
    async def process_single_symbol(self, symbol, df, signals):
        ensure_directory(GRAPH_DIR)
        csv_path = os.path.join(GRAPH_DIR, f"{symbol}_data.csv")
        try:
            await asyncio.to_thread(save_to_csv, df, csv_path)
            await asyncio.to_thread(plot_indicators, df, symbol, signals)
        except Exception as e:
            logger.error("Error saving/plotting for %s: %s", symbol, e, exc_info=True)

    # ----
    # Генерация сигналов для всех инструментов
    # ----
    async def process_signals(self, market_data):
        results = process_all_data(market_data, self.config, self.config["last_signals"])
        tasks = []
        state_dirty = False  # Новый флаг для отслеживания изменений
        config_dirty = False  # Новый флаг для отслеживания изменений
        for symbol, (df, new_signal, signals) in results.items():
            tasks.append(asyncio.create_task(self.process_single_symbol(symbol, df, signals)))
            if new_signal and new_signal != self.config["last_signals"][symbol]["signal"]:
                self.config["last_signals"][symbol] = {"signal": new_signal, "time": datetime.now().isoformat()}
                state_dirty = True  # Устанавливаем флаг, если есть изменения
                config_dirty = True  # Устанавливаем флаг, если есть изменения
                logger.info("New %s signal for %s", new_signal, symbol)
        await asyncio.gather(*tasks)
        if state_dirty:
            self.save_state()  # Сохраняем состояние один раз после всех изменений
        if config_dirty:
            self.save_config()  # Сохраняем конфигурацию один раз после всех изменений
            self.update_asset_distribution()  # Обновляем только при конфиг-изменениях

    # ----
    # Получение данных с биржи
    # ----
    async def check_signals(self):
        tasks = [
            self.data_loader.get_realtime_data(sym, self.config["interval"], self.config["lookback_period"])
            for sym in self.config["binance_pairs"]
        ]
        data = await asyncio.gather(*tasks)
        market = {sym: df for sym, df in zip(self.config["binance_pairs"], data) if df is not None}
        if market:
            await self.process_signals(market)

    # ----
    # Сохранение / загрузка состояния
    # ----
    def save_state(self):
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(self.config["last_signals"], f, indent=4)
        logger.info("State saved")

    def load_state(self):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # оставляем только актуальные пары
            self.config["last_signals"] = {k: v for k, v in raw.items() if k in self._symbols}
            # добавляем недостающие
            for s in self._symbols:
                self.config["last_signals"].setdefault(s, {"signal": None, "time": None})
            logger.info("State loaded")
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("No previous state or invalid file – starting fresh")

    # ----
    # Hot-reload: проверяем, изменён ли CONFIG_PATH на диске
    # ----
    async def reload_config_if_changed(self):
        # --- динамический TTL ещё до проверки mtime ---
        if self.config.get('symbol_source') == 'dynamic':
            # обновляем список только по истечении TTL, чтобы не дергать API каждую итерацию
            ttl = self.config.get('symbol_cache_ttl', 1800)
            if (datetime.now() - self.last_symbol_refresh).total_seconds() > ttl:
                new_syms = load_symbols(self.config)
                # удаляем устаревшие файлы
                removed = set(self._symbols) - set(new_syms)
                if removed:
                    logger.info("Dynamic cleanup of obsolete files: %s", removed)
                    self._cleanup_obsolete(removed)
                # если список символов изменился — сохраняем и обновляем состояние
                if set(new_syms) != set(self._symbols):
                    self._symbols = new_syms
                    self.config['binance_pairs'] = new_syms
                    self._sync_last_signals(new_syms)
                    self.save_config()
                    self.save_state()
                    self.update_asset_distribution()
                # обновляем отметку последнего запроса к API
                self.last_symbol_refresh = datetime.now()

        # --- hot-reload файла ---
        mtime = Path(CONFIG_PATH).stat().st_mtime
        if mtime == self._config_mtime:
            return  # выходим, если сам файл не менялся

        old_syms = set(self._symbols)
        self.load_config()  # Перезагружаем – обновит self.config и self._symbols
        self._config_mtime = mtime

        new_syms = set(self._symbols)
        removed = old_syms - new_syms
        added = new_syms - old_syms

        if removed:
            self._cleanup_obsolete(removed)
        if added or removed:
            # Держим ключи last_signals в актуальном состоянии
            self._sync_last_signals(new_syms)
            self.save_config()  # Сохраняем согласованность на диске
            self.save_state()  # Немедленно переписываем bot_state.json
            self.update_asset_distribution()

    # ----
    # Удаляем html/csv-остатки для символов, которые больше не отслеживаем
    # ----
    def _cleanup_obsolete(self, symbols):
        for sym in symbols:
            for suffix in ("_indicators_adx.html", "_data.csv", "_indicators_signals.html"):
                f = GRAPH_DIR / f"{sym}{suffix}"
                if f.exists():
                    try:
                        f.unlink()
                        logger.info("Removed obsolete file %s", f)
                    except Exception as e:
                        logger.warning("Cannot delete %s: %s", f, e)

    # ----
    # Синхронизация last_signals
    # ----
    def _sync_last_signals(self, symbols):
        """Добавляет/удаляет ключи в self.config['last_signals']."""
        added = False
        ls = self.config["last_signals"]
        for s in symbols:
            if s not in ls:
                ls[s] = {"signal": "NEUTRAL", "time": None}
                added = True
        for s in list(ls.keys()):
            if s not in symbols:
                del ls[s]
                added = True
        return added

    # ----
    # Главный цикл
    # ----
    async def run(self):
        self.load_state()
        interval_sec = int(self.config["interval"].rstrip("m")) * 60
        while True:
            try:
                await self.reload_config_if_changed()  # <‑‑ НОВОЕ
                await self.check_signals()
                # Сохраняем состояние раз в 5 мин
                if (datetime.now() - self.last_state_save).total_seconds() > 300:
                    self.save_state()
                    self.last_state_save = datetime.now()
            except Exception as e:
                logger.error("Main loop error: %s", e, exc_info=True)
            await asyncio.sleep(interval_sec)  # оставить только одну паузу

if __name__ == "__main__":
    asyncio.run(SignalBot().run())