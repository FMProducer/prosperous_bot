import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_distribution(symbol: str, signal: str):
    """Обновляет распределение активов в config.json в зависимости от полученного сигнала.

    Параметры
    ----------
    symbol : str
        Тикер (например "GALAUSDT"). Проверяется только постфикс «USDT».
    signal : str
        Строка "BUY" / "SELL" / "NEUTRAL" (регистрозависимость не важна).
    """

    config_path = "config_signal.json"          # таблица распределений
    rebalance_config_path = "config.json"        # фактическая конфигурация ребалансировщика

    # -------------------- читаем файлы --------------------
    try:
        with open(config_path, "r", encoding="utf‑8") as f:
            signal_config = json.load(f)
        with open(rebalance_config_path, "r", encoding="utf‑8") as f:
            rebalance_config = json.load(f)
    except FileNotFoundError as e:
        logger.error("Config file missing: %s", e)
        print(f"Config file missing: {e}", file=sys.stderr)
        return
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in config: %s", e)
        print(f"Invalid JSON in config: {e}", file=sys.stderr)
        return

    # -------------------- какие пары нужно трогать? --------------------
    if not symbol.endswith("USDT"):
        print(f"Unsupported symbol format: {symbol}", file=sys.stderr)
        return

    base = symbol.replace("USDT", "")
    pairs_to_update = [f"{base}_USDT", f"{base}5L_USDT", f"{base}5S_USDT"]

    # -------------------- берём новое распределение --------------------
    signal = signal.upper()
    new_distribution = signal_config.get("asset_distributions", {}).get(signal)
    if new_distribution is None:
        print(f"Invalid signal: {signal}", file=sys.stderr)
        return

    # -------------------- обновляем только нужные пары --------------------
    for pair in pairs_to_update:
        if pair in new_distribution and pair in rebalance_config.get("asset_distribution", {}):
            rebalance_config["asset_distribution"][pair] = new_distribution[pair]

    # -------------------- сохраняем результат --------------------
    with open(rebalance_config_path, "w", encoding="utf‑8") as f:
        json.dump(rebalance_config, f, indent=4, ensure_ascii=False)

    logger.info("Asset distribution updated for %s (%s)", symbol, signal)


# --------------------------------------------------------------
# CLI‑обёртка
# --------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python update_distribution.py <symbol> <BUY|SELL|NEUTRAL>", file=sys.stderr)
    else:
        update_distribution(sys.argv[1], sys.argv[2])
