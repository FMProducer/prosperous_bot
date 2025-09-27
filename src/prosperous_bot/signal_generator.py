import logging
import pandas as pd

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------------------------------
#  Настройки фильтров ADX
#   – нижний порог берём из params['adx_threshold'] (уже 30)
#   – верхний лимит фиксируем на 55, чтобы не торговать в фазе «истощения»
#   – slope_window = 3: требуем, чтобы ADX рос последние 3 свечи
# --------------------------------------------------
ADX_UPPER_LIMIT = 55
ADX_SLOPE_WINDOW = 3


def generate_signal(data: pd.DataFrame, params: dict, last_signal: str):
    """Генерация buy/sell с двойным фильтром ADX (диапазон + наклон)."""

    required_columns = [
        "close", "MA1", "MA2", "BB_Mid",
        "EMA1", "EMA2", "EMA3", "bb_width", "ADX",
    ]
    if not all(col in data.columns for col in required_columns):
        raise KeyError("Missing required columns in data")

    data["Buy_Signal"], data["Sell_Signal"] = 0, 0

    # ------------------ Фильтр силы тренда ------------------
    adx_in_range = (data["ADX"] > params["adx_threshold"]) & (data["ADX"] < ADX_UPPER_LIMIT)
    adx_slope_pos = data["ADX"].diff().rolling(window=ADX_SLOPE_WINDOW).mean() > 0
    trend_strength = adx_in_range & adx_slope_pos

    # ------------------ Условия ------------------
    buy_conditions = [
        data["EMA1"] > data["EMA2"],
        data["EMA1"] > data["EMA3"],
        data["EMA2"] > data["EMA3"],
        data["close"] > data["BB_Mid"],
        data["close"] > data["MA1"],
        data["EMA1"] > data["BB_Mid"],
        data["EMA2"] > data["BB_Mid"],
        data["EMA3"] > data["BB_Mid"],
        data["MA1"] > data["BB_Mid"],
        data["MA2"] > data["BB_Mid"],
        data["bb_width"] > params["volatility_threshold"],
        data["MA1"] > data["MA2"],
        trend_strength,
    ]

    sell_conditions = [
        data["EMA1"] < data["EMA2"],
        data["EMA1"] < data["EMA3"],
        data["EMA2"] < data["EMA3"],
        data["close"] < data["BB_Mid"],
        data["close"] < data["MA1"],
        data["EMA1"] < data["BB_Mid"],
        data["EMA2"] < data["BB_Mid"],
        data["EMA3"] < data["BB_Mid"],
        data["MA1"] < data["BB_Mid"],
        data["MA2"] < data["BB_Mid"],
        data["bb_width"] > params["volatility_threshold"],
        data["MA1"] < data["MA2"],
        trend_strength,
    ]

    buy_mask = pd.concat(buy_conditions, axis=1).all(axis=1)
    sell_mask = pd.concat(sell_conditions, axis=1).all(axis=1)

    new_signal = None
    signals = []
    for ts in data.index:
        if buy_mask.loc[ts] and last_signal != "buy":
            new_signal = "buy"; signals.append(("buy", ts)); data.loc[ts, "Buy_Signal"] = 1; last_signal = "buy"
        elif sell_mask.loc[ts] and last_signal != "sell":
            new_signal = "sell"; signals.append(("sell", ts)); data.loc[ts, "Sell_Signal"] = 1; last_signal = "sell"

    logging.info("Signal %s, total points %d", new_signal, len(signals))
    return data, new_signal, signals


def process_all_data(all_data: dict, params: dict, last_signals: dict):
    return {sym: generate_signal(df, params, last_signals.get(sym, {}).get("signal"))
            for sym, df in all_data.items()}
