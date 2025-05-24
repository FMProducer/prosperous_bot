
import pandas as pd

def process_all_data(all_data: dict[str, pd.DataFrame], config: dict, last_signals: dict) -> dict[str, tuple[pd.DataFrame, str]]:
    result = {}
    for symbol, df in all_data.items():
        df = df.copy()
        df['signal'] = 0  # dummy signal logic
        signal_type = "NEUTRAL"

        # Пример логики на основе последнего сигнала (можно заменить на RuleBasedStrategy или ML)
        last = last_signals.get(symbol, {}).get("signal", "NEUTRAL")

        if df['close'].iloc[-1] > df['close'].iloc[-2]:  # simple logic
            signal_type = "BUY" if last != "BUY" else "NEUTRAL"
        elif df['close'].iloc[-1] < df['close'].iloc[-2]:
            signal_type = "SELL" if last != "SELL" else "NEUTRAL"

        result[symbol] = (df, signal_type)

    return result
