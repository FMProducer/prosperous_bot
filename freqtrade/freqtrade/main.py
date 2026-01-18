#!/usr/bin/env python3
"""
Main Freqtrade bot script.
Read the documentation to know what cli arguments you need.
"""

import logging
import sys
from typing import Any

import ccxt
# 1. Override the CCXT parser for Binance
# Original: ccxt.binance.parse_ohlcv
def custom_parse_ohlcv(self, ohlcv, market=None):
    # Standard fields
    res = [
        self.safe_integer(ohlcv, 0),  # timestamp
        self.safe_number(ohlcv, 1),   # open
        self.safe_number(ohlcv, 2),   # high
        self.safe_number(ohlcv, 3),   # low
        self.safe_number(ohlcv, 4),   # close
        self.safe_number(ohlcv, 5),   # volume
        # --- ADDED FIELDS ---
        self.safe_number(ohlcv, 7),   # quote_volume (index 7)
        self.safe_integer(ohlcv, 8),  # num_trades (index 8)
        self.safe_number(ohlcv, 9),   # taker_base_vol (index 9)
        self.safe_number(ohlcv, 10),  # taker_quote_vol (index 10)
    ]
    return res

# Apply the patch to the binance class in the ccxt library
ccxt.binance.parse_ohlcv = custom_parse_ohlcv
# Also patch the async version if you use websockets or async mode
ccxt.pro.binance.parse_ohlcv = custom_parse_ohlcv


# check min. python version
if sys.version_info < (3, 11):  # pragma: no cover  # noqa: UP036
    sys.exit("Freqtrade requires Python version >= 3.11")

from freqtrade import __version__
from freqtrade.commands import Arguments
from freqtrade.constants import DOCS_LINK
from freqtrade.exceptions import ConfigurationError, FreqtradeException, OperationalException
from freqtrade.loggers import setup_logging_pre
from freqtrade.system import (
    asyncio_setup,
    gc_set_threshold,
    print_version_info,
    set_mp_start_method,
)


logger = logging.getLogger("freqtrade")


def main(sysargv: list[str] | None = None) -> None:
    """
    This function will initiate the bot and start the trading loop.
    :return: None
    """

    return_code: Any = 1
    try:
        setup_logging_pre()
        asyncio_setup()
        arguments = Arguments(sysargv)
        args = arguments.get_parsed_arg()

        # Call subcommand.
        if args.get("version") or args.get("version_main"):
            print_version_info()
            return_code = 0
        elif "func" in args:
            logger.info(f"freqtrade {__version__}")
            gc_set_threshold()
            set_mp_start_method()
            return_code = args["func"](args)
        else:
            # No subcommand was issued.
            raise OperationalException(
                "Usage of Freqtrade requires a subcommand to be specified.\n"
                "To have the bot executing trades in live/dry-run modes, "
                "depending on the value of the `dry_run` setting in the config, run Freqtrade "
                "as `freqtrade trade [options...]`.\n"
                "To see the full list of options available, please use "
                "`freqtrade --help` or `freqtrade <command> --help`."
            )

    except SystemExit as e:  # pragma: no cover
        return_code = e
    except KeyboardInterrupt:
        logger.info("SIGINT received, aborting ...")
        return_code = 0
    except ConfigurationError as e:
        logger.error(
            f"Configuration error: {e}\n"
            f"Please make sure to review the documentation at {DOCS_LINK}."
        )
    except FreqtradeException as e:
        logger.error(str(e))
        return_code = 2
    except Exception:
        logger.exception("Fatal exception!")
    finally:
        sys.exit(return_code)


if __name__ == "__main__":  # pragma: no cover
    main()
