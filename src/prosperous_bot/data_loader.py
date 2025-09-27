import time
import pandas as pd
import aiohttp
import asyncio
import logging
import ta
from ta.trend import ADXIndicator  # <-- NEW: импорт индикатора ADX
from typing import Optional
import json

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class DataLoader:
    """Загружает свечные данные с Binance и рассчитывает технические индикаторы.

    Параметры берутся из переданного при инициализации словаря ``config``.
    Используется кэш (~5 мин.) и экспоненциальный бэкофф при ошибках сети.
    """

    def __init__(self, config: dict):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache: dict[str, tuple[pd.DataFrame, float]] = {}
        self.cache_ttl = 300  # 5 минут — время жизни кэша
        self.max_retries = 3
        self.retry_delay = 5  # секунд

    # ------------------------------------------------------------------
    # Работа с HTTP‑сессией
    # ------------------------------------------------------------------
    async def get_session(self) -> aiohttp.ClientSession:
        """Возвращает открытую aiohttp‑сессию (создаёт при необходимости)."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close(self):
        """Корректно закрывает текущую HTTP‑сессию."""
        if self.session and not self.session.closed:
            await self.session.close()

    # ------------------------------------------------------------------
    # Публичный метод: получить данные с учётом кэша и ретраев
    # ------------------------------------------------------------------
    async def get_realtime_data(self, symbol: str, interval: str, lookback_period: int) -> Optional[pd.DataFrame]:
        """Возвращает DataFrame со свечами + индикаторами для *symbol*.

        * ``interval`` — таймфрейм Binance (например, ``'5m'``)
        * ``lookback_period`` — количество часов истории
        """
        cache_key = f"{symbol}_{interval}_{lookback_period}"
        now = time.time()

        # ---------- проверяем кэш ----------
        if cache_key in self.cache:
            df_cached, ts_cached = self.cache[cache_key]
            if now - ts_cached < self.cache_ttl:
                logging.debug("Using cached data for %s", symbol)
                return df_cached

        # ---------- если нет кэша — делаем запросы с ретраями ----------
        for attempt in range(self.max_retries):
            try:
                df = await self._get_realtime_data_binance(symbol, interval, lookback_period)
                if df is not None and not df.empty:
                    self.cache[cache_key] = (df, now)
                return df
            except aiohttp.ClientError as e:
                logging.error("Connection error (%s/%s): %s", attempt + 1, self.max_retries, e)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))  # эксп. бэкофф
                    await self.close()  # пересоздадим сессию
                else:
                    logging.error("Max retries reached – giving up.")
                    return None
            except Exception as e:
                logging.error("Unexpected error: %s", e, exc_info=True)
                return None

    # ------------------------------------------------------------------
    # Внутренний метод: реальный запрос на Binance + расчёт индикаторов
    # ------------------------------------------------------------------
    async def _get_realtime_data_binance(self, symbol: str, interval: str, lookback_period: int) -> Optional[pd.DataFrame]:
        logging.debug("Fetching data for %s", symbol)

        try:
            current_time = int(time.time() * 1000)
            start_time = current_time - (lookback_period * 60 * 60 * 1000)
            all_data: list[list] = []
            url = "https://api.binance.com/api/v3/klines"
            session = await self.get_session()

            # Binance отдаёт максимум 1000 свечей за один запрос
            while start_time < current_time:
                end_time = min(start_time + 1000 * 60 * 1000, current_time)
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": start_time,
                    "endTime": end_time,
                    "limit": 1000,
                }
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        logging.error("HTTP %s – %s", resp.status, await resp.text())
                        raise aiohttp.ClientError(f"HTTP {resp.status}")
                    all_data.extend(await resp.json())
                start_time = end_time

            if not all_data:
                logging.error("No data fetched for %s", symbol)
                return None

            # ---------- формируем DataFrame ----------
            df = pd.DataFrame(
                all_data,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_asset_volume",
                    "number_of_trades",
                    "taker_buy_base_asset_volume",
                    "taker_buy_quote_asset_volume",
                    "ignore",
                ],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df = df[["open", "high", "low", "close", "volume"]].astype(float)

            # ---------- рассчитываем индикаторы ----------
            # EMA
            df["EMA1"] = ta.trend.ema_indicator(df["close"], window=self.config["ema1_window"])
            df["EMA2"] = ta.trend.ema_indicator(df["close"], window=self.config["ema2_window"])
            df["EMA3"] = ta.trend.ema_indicator(df["close"], window=self.config["ema3_window"])

            # SMA
            df["MA1"] = ta.trend.sma_indicator(df["close"], window=self.config["ma1_window"])
            df["MA2"] = ta.trend.sma_indicator(df["close"], window=self.config["ma2_window"])

            # Bollinger Bands
            bb = ta.volatility.BollingerBands(
                df["close"],
                window=self.config["bb_window"],
                window_dev=self.config["bb_std_dev"],
            )
            df["BB_High"] = bb.bollinger_hband()
            df["BB_Low"] = bb.bollinger_lband()
            df["BB_Mid"] = bb.bollinger_mavg()
            df["bb_width"] = (df["BB_High"] - df["BB_Low"]) / df["BB_Mid"]

            # ADX (НОВЫЙ индикатор)
            adx = ADXIndicator(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                window=self.config["adx_window"],
            )
            df["ADX"] = adx.adx()

            logging.debug("Fetched %s rows for %s. Columns: %s", len(df), symbol, list(df.columns))
            return df

        except aiohttp.ClientError:
            raise  # пробросим наружу — обработается в вызывающем методе
        except Exception as e:
            logging.error("Unexpected error while fetching %s: %s", symbol, e, exc_info=True)
            return None
