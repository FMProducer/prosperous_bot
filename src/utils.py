# c:\Python\Prosperous_Bot\src\utils.py

import requests
import logging
from datetime import datetime, timedelta

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Кэш для топ-символов
top_symbols_cache = {
    'data': None,
    'expiry': None
}

def get_binance_top_symbols(limit=20, cache_ttl=1800, min_quote_volume=10, excluded=None):
    """Возвращает top-N пар с максимальным объёмом за сутки (USDT) и минимальным объёмом торгов, исключая указанные символы."""
    global top_symbols_cache

    # Проверяем, есть ли актуальные данные в кэше
    if top_symbols_cache['data'] and top_symbols_cache['expiry'] > datetime.now():
        logger.info("Using cached top symbols")
        return top_symbols_cache['data']

    url = "https://api.binance.com/api/v3/ticker/24hr"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Проверяем, что запрос успешен
        data = response.json()
        excluded = set(excluded or [])
        usdt_pairs = [
            item for item in data
            if item["symbol"].endswith("USDT")
            and item["symbol"] not in excluded  # фильтруем полные имена
            and float(item["quoteVolume"]) >= min_quote_volume * 1_000_000
        ]
        sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x["quoteVolume"]), reverse=True)
        top_symbols = [pair["symbol"] for pair in sorted_pairs[:limit]]

        # Обновляем кэш
        top_symbols_cache['data'] = top_symbols
        top_symbols_cache['expiry'] = datetime.now() + timedelta(seconds=cache_ttl)
        logger.info("Top symbols fetched and cached")
        return top_symbols
    except requests.RequestException as e:
        logger.warning("Ошибка сети или API-лимит при получении топ-символов с Binance: %s", e)
        return []  # Возвращаем пустой список, если запрос не удался
    except Exception as e:
        logger.error("Ошибка получения топ-символов с Binance: %s", e, exc_info=True)
        return []

def save_to_csv(df, path):
    """
    Сохраняет DataFrame в CSV файл, предварительно
    перенося индекс (timestamp) в столбец 'timestamp'.
    """
    df_to_save = df.copy()
    # Если индекс не назван, выносим его под именем 'timestamp'
    idx_name = df_to_save.index.name or 'timestamp'
    df_to_save.reset_index(inplace=True)
    # Переименуем колонку index в 'timestamp' если нужно
    if idx_name != 'timestamp':
        df_to_save.rename(columns={idx_name: 'timestamp'}, inplace=True)
    df_to_save.to_csv(path, index=False)
    logger.info("Data saved to %s", path)

def ensure_directory(directory):
    """Создаёт директорию, если она не существует."""
    directory.mkdir(parents=True, exist_ok=True)
    logger.info("Directory %s ensured", directory)

def load_symbols(config):
    """Загружает символы из конфигурации или динамически, если указано."""
    if config.get('symbol_source') == 'dynamic':
        return get_binance_top_symbols(
            config.get('symbol_limit', 20),
            config.get('symbol_cache_ttl', 1800),
            config.get('min_quote_volume', 10),
            config.get('excluded_symbols', [])
        )
    else:
        return config.get('binance_pairs', [])