#!/usr/bin/env python3
"""
Main Freqtrade bot script.
"""
import logging
import sys
import importlib
from typing import Any
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

# Ensure freqtrade is importable (e.g. if running from third_party folder)
try:
    import freqtrade
except ImportError:
    # Add 'third_party' to sys.path so 'freqtrade' package can be found
    third_party_dir = Path(__file__).resolve().parent.parent
    if str(third_party_dir) not in sys.path:
        sys.path.insert(0, str(third_party_dir))

# ---------------------------------------------------------------------------------
# 1. MONKEY PATCH: CCXT (Чтобы скачивал 10 колонок)
# ---------------------------------------------------------------------------------
import ccxt
# Пытаемся импортировать async поддержку явно
try:
    import ccxt.async_support
except ImportError:
    pass

def custom_parse_ohlcv(self, ohlcv, market=None):
    res = [
        self.safe_integer(ohlcv, 0),  # timestamp
        self.safe_number(ohlcv, 1),   # open
        self.safe_number(ohlcv, 2),   # high
        self.safe_number(ohlcv, 3),   # low
        self.safe_number(ohlcv, 4),   # close
        self.safe_number(ohlcv, 5),   # volume
    ]
    # Добавляем 4 доп. колонки
    res.append(self.safe_number(ohlcv, 7))   # quote_volume (6)
    res.append(self.safe_integer(ohlcv, 8))  # num_trades (7)
    res.append(self.safe_number(ohlcv, 9))   # taker_base (8)
    res.append(self.safe_number(ohlcv, 10))  # taker_quote (9)
    return res

# Применяем патч к CCXT
ccxt.binance.parse_ohlcv = custom_parse_ohlcv
if hasattr(ccxt, 'async_support'):
    ccxt.async_support.binance.parse_ohlcv = custom_parse_ohlcv
try:
    from ccxt.async_support.binance import binance as async_binance
    async_binance.parse_ohlcv = custom_parse_ohlcv
except ImportError:
    pass

print("PATCH 1/2: CCXT patched to return 10 columns.")

# ---------------------------------------------------------------------------------
# 2. MONKEY PATCH: FREQTRADE CONVERTER (Чтобы создавал DataFrame с 10 колонками)
# ---------------------------------------------------------------------------------
from freqtrade.data import converter  # type: ignore

def custom_ohlcv_to_dataframe(ohlcv: list, timeframe: str, pair: str, *,
                            fill_missing: bool = True, drop_incomplete: bool = True) -> pd.DataFrame:
    cols = ["date", "open", "high", "low", "close", "volume", 
            "quote_volume", "num_trades", "taker_base", "taker_quote"]
    df = pd.DataFrame(ohlcv, columns=cols)
    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
    if drop_incomplete and not df.empty:
        df.drop(df.tail(1).index, inplace=True)
    return df

# Подменяем функцию в модуле FreqTrade
converter.ohlcv_to_dataframe = custom_ohlcv_to_dataframe
print("PATCH 2/2: Freqtrade Converter patched to support extended columns.")

# ---------------------------------------------------------------------------------
# MAIN BOT LOGIC
# ---------------------------------------------------------------------------------

# check min. python version
if sys.version_info < (3, 11):
    sys.exit("Freqtrade requires Python version >= 3.11")

from freqtrade import __version__
from freqtrade.commands import Arguments  # type: ignore
from freqtrade.constants import DOCS_LINK  # type: ignore
from freqtrade.exceptions import ConfigurationError, FreqtradeException, OperationalException  # type: ignore
from freqtrade.loggers import setup_logging_pre  # type: ignore
from freqtrade.system import (  # type: ignore
    asyncio_setup,
    gc_set_threshold,
    print_version_info,
    set_mp_start_method,
)

logger = logging.getLogger("freqtrade")

def main(sysargv: list[str] | None = None) -> None:
    return_code: Any = 1
    try:
        setup_logging_pre()
        asyncio_setup()
        arguments = Arguments(sysargv)
        args = arguments.get_parsed_arg()

        if args.get("version") or args.get("version_main"):
            print_version_info()
            return_code = 0
        elif "func" in args:
            logger.info(f"freqtrade {__version__}")
            gc_set_threshold()
            set_mp_start_method()
            return_code = args["func"](args)
        else:
            raise OperationalException(
                "Usage of Freqtrade requires a subcommand to be specified.\n"
                "Run `freqtrade trade [options...]`."
            )

    except SystemExit as e:
        return_code = e
    except KeyboardInterrupt:
        logger.info("SIGINT received, aborting ...")
        return_code = 0
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
    except FreqtradeException as e:
        logger.error(str(e))
        return_code = 2
    except Exception:
        logger.exception("Fatal exception!")
    finally:
        sys.exit(return_code)

if __name__ == "__main__":
    main()
