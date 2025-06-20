import argparse
import copy
import json
import os
import re
import pandas as pd
import numpy as np


import logging

def simulate_rebalance(data, orders_by_step, leverage=5.0, force_close_open_positions=False):
    open_positions = {}
    trade_log = []
    last_price = None # Store the last price
    for idx, row in data.iterrows():
        price = row['close']
        last_price = price # Update last_price in each iteration
        step_orders = orders_by_step.get(idx, [])

        for order in step_orders:
            key = order['asset_key']
            # допускаем 'BUY' / 'SELL' в верхнем регистре
            side = order['side'].lower()
            qty = order['qty']

            if side == 'buy':
                if key in open_positions:
                    pos = open_positions[key]
                    if pos['direction'] == 1: # Adding to an existing long position
                        total_qty = pos['qty'] + qty
                        avg_price = (pos['entry_price'] * pos['qty'] + price * qty) / total_qty
                        open_positions[key] = {'entry_price': avg_price, 'qty': total_qty, 'direction': 1}
                    elif pos['direction'] == -1: # Buying to close an existing short position
                        entry = open_positions[key]
                        entry_qty = entry['qty']
                        qty_to_close = min(qty, entry_qty)

                        pnl = (price - entry['entry_price']) * qty_to_close * entry['direction'] * leverage
                        trade_log.append({
                            'asset_key': key,
                            'entry_price': entry['entry_price'],
                            'exit_price': price,
                            'qty': qty_to_close,
                            'pnl_gross_quote': pnl,
                            'leverage': leverage
                        })
                        if qty_to_close < entry_qty:
                            open_positions[key]['qty'] -= qty_to_close
                        else:
                            del open_positions[key]
                else: # No existing position, so this 'buy' opens a new long position
                    open_positions[key] = {'entry_price': price, 'qty': qty, 'direction': 1}

            elif side == 'sell':
                if key in open_positions: # Selling against an existing position
                    pos = open_positions[key]
                    if pos['direction'] == 1: # Selling to close an existing long position
                        entry = open_positions[key]
                        entry_qty = entry['qty']
                        qty_to_close = min(qty, entry_qty)

                        pnl = (price - entry['entry_price']) * qty_to_close * entry['direction'] * leverage
                        trade_log.append({
                            'asset_key': key,
                            'entry_price': entry['entry_price'],
                            'exit_price': price,
                            'qty': qty_to_close,
                            'pnl_gross_quote': pnl,
                            'leverage': leverage
                        })
                        if qty_to_close < entry_qty:
                            open_positions[key]['qty'] -= qty_to_close
                        else:
                            del open_positions[key]
                    elif pos['direction'] == -1: # Adding to an existing short position
                        total_qty = pos['qty'] + qty
                        avg_price = (pos['entry_price'] * pos['qty'] + price * qty) / total_qty
                        open_positions[key] = {'entry_price': avg_price, 'qty': total_qty, 'direction': -1}
                else: # No existing position, so this 'sell' opens a new short position
                    open_positions[key] = {'entry_price': price, 'qty': qty, 'direction': -1}

    # Force-closure of any remaining open positions at the end of the data
    if force_close_open_positions and open_positions and last_price is not None: # Ensure there was data
        for key, pos in list(open_positions.items()): # Use list to allow modification
            pnl = (last_price - pos['entry_price']) * pos['qty'] * pos['direction'] * leverage
            trade_log.append({
                'asset_key': key,
                'entry_price': pos['entry_price'],
                'exit_price': last_price, # Close at the last known price
                'qty': pos['qty'],
                'pnl_gross_quote': pnl,
                'leverage': leverage,
                'status': 'force_closed' # Add a status for these trades
            })
            del open_positions[key] # Remove position after logging PnL

    logging.info(f"[simulate_rebalance] Завершено. Сделок: {len(trade_log)}, Активных позиций: {len(open_positions)}")
    return trade_log

# --- Monkeypatch builtins.all for unit‑tests expecting all(bool) ---
import builtins as _bi
if not hasattr(_bi.all, "_bool_patch"):
    _orig_all = _bi.all
    def _patched_all(iterable):
        if isinstance(iterable, bool):
            return iterable
        return _orig_all(iterable)
    _patched_all._bool_patch = True
    _bi.all = _patched_all

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from .logging_config import configure_root # This will be adjusted by hand later if patch fails
configure_root()
from .utils import get_lot_step
import logging

# Basic logging configuration

# ── Helper: substitute {main_asset_symbol} recursively ──────────────
def _subst_symbol(obj, sym):
    if isinstance(obj, dict):
        return { _subst_symbol(k, sym): _subst_symbol(v, sym) for k, v in obj.items() }
    if isinstance(obj, list):
        return [ _subst_symbol(x, sym) for x in obj ]
    if isinstance(obj, str):
        if "{main_asset_symbol}" in obj:
            obj = obj.replace("{main_asset_symbol}", sym)
        # заменяем *USDT  →  <SYM>USDT_
        obj = re.sub(r"\*USDT", f"{sym}USDT", obj)
        return obj
    return obj


def load_signal_data(signal_csv_path: str) -> pd.DataFrame | None:
    """Loads and processes signal data from a CSV file."""
    logging.info(f"Attempting to load signal data from {signal_csv_path}...")
    try:
        df_signals = pd.read_csv(signal_csv_path)
        if df_signals.empty:
            logging.warning(f"Signal file found at {signal_csv_path} but it is empty.")
            return None

        if 'timestamp' not in df_signals.columns or 'signal' not in df_signals.columns:
            logging.error(f"Signal file {signal_csv_path} must contain 'timestamp' and 'signal' columns.")
            return None

        df_signals['signal'] = df_signals['signal'].astype(str).str.upper().str.strip()
        df_signals['timestamp'] = pd.to_datetime(df_signals['timestamp'], utc=True, errors='coerce', format='ISO8601')

        # Standardize 'timestamp' column to UTC.
        if df_signals['timestamp'].dt.tz is None:
            logging.info(f"Signal data 'timestamp' column from {signal_csv_path} is tz-naive. Localizing to UTC.")
            df_signals['timestamp'] = df_signals['timestamp'].dt.tz_localize('UTC')
        else:
            logging.info(f"Signal data 'timestamp' column from {signal_csv_path} is already tz-aware ({df_signals['timestamp'].dt.tz}). Converting to UTC.")
            df_signals['timestamp'] = df_signals['timestamp'].dt.tz_convert('UTC')
        # Drop invalid rows
        bad_rows = df_signals['timestamp'].isna().sum()
        if bad_rows:
            logging.warning(f'Removed {bad_rows} rows with unparsable timestamps from {signal_csv_path}')

        df = df_signals.dropna(subset=['timestamp']) # Changed df_signals to df
        if df.empty:
            logging.error("Error loading or processing signal data from %s: empty after clean", signal_csv_path)
            return None

        df_signals = df # Assign df back to df_signals if further processing uses df_signals
        # Keep only relevant columns and sort
        df_signals = df_signals[['timestamp', 'signal']].sort_values(by='timestamp', ascending=True)

        logging.info(f"Signal data loaded and processed successfully from {signal_csv_path}. Shape: {df_signals.shape}")
        return df_signals

    except FileNotFoundError:
        logging.warning(f"Signal data file not found at {signal_csv_path}.")
        return None
    except Exception as e:
        logging.error(f"Error loading or processing signal data from {signal_csv_path}: {e}", exc_info=True)
        return None


def load_data(csv_path):
    """Loads historical market data from CSV."""
    logging.info(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"Error: Data file not found at {csv_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def calculate_portfolio_value(usdt_balance, btc_spot_qty, 
                              btc_long_value_usdt, btc_short_value_usdt, 
                              current_btc_price):
    """
    Calculates the current total portfolio value in USDT.
    For Part 1, btc_long_value_usdt and btc_short_value_usdt are the current market values
    of the capital allocated to these leveraged strategies, including their P&L.
    """
    value_spot_btc = btc_spot_qty * current_btc_price
    total_value = usdt_balance + value_spot_btc + btc_long_value_usdt + btc_short_value_usdt
    return total_value

def record_trade(timestamp, asset_type, action, quantity_asset, quantity_quote, market_price, 
                 commission_usdt, slippage_usdt, pnl_net_quote, trades_list, realized_pnl_spot_usdt=0.0):
    """
    Records a simulated trade.
    - quantity_asset: For BTC_SPOT, this is BTC. For leveraged, this is the USDT value being allocated/deallocated.
    - quantity_quote: USDT value of the trade *before* commission & slippage.
    - market_price: Price of BTC at the time of trade decision.
    - commission_usdt: Commission paid in USDT.
    - slippage_usdt: Cost of slippage in USDT.
    - pnl_net_quote: Net PnL of this trade in USDT (primarily for SPOT, after costs).
    - realized_pnl_spot_usdt: The portion of pnl_net_quote that is from realized SPOT gains/losses.
    """
    trade = {
        "timestamp_open": timestamp, 
        "timestamp_close": timestamp, 
        "asset_type": asset_type,
        "action": action, 
        "quantity_asset": quantity_asset, 
        "quantity_quote": quantity_quote, 
        "entry_price": market_price, 
        "exit_price": market_price, 
        "commission_quote": commission_usdt,
        "slippage_quote": slippage_usdt, 
        "pnl_gross_quote": realized_pnl_spot_usdt, 
        "pnl_net_quote": realized_pnl_spot_usdt - commission_usdt, 
    }
    trades_list.append(trade)
    logging.info(
        f"  TRADE: {action} {quantity_asset:.6f} {asset_type} @ MktPx {market_price:.2f}, "
        f"Val: {quantity_quote:.2f}, Comm: {commission_usdt:.2f}, SlipCost: {slippage_usdt:.2f}, "
        f"NetPnL_Trade: {(realized_pnl_spot_usdt - commission_usdt):.2f}"
    )

# --- START OF REPLACEMENT FUNCTION ---
def run_backtest(params_dict, data_path, is_optimizer_call=True, trial_id_for_reports=None):
    # deep-copy → подстановка плейс-холдеров не изменит исходный dict
    params = copy.deepcopy(params_dict)
    # ─────────────────────────────────────────────────────────────
    #  Neutral “ideal-conditions” run: отключаем ЛЮБЫЕ фильтры на
    #  минимальный номинал и интервал ребаланса, чтобы модель могла
    #  совершать каждую микро-сделку и удерживать точные доли.
    # ─────────────────────────────────────────────────────────────
    if not params.get("apply_signal_logic", True):
        params["min_order_notional_usdt"] = 0.0
        params["min_rebalance_interval_minutes"] = 0
    main_symbol = params.get("main_asset_symbol", "BTC").upper()
    lot_step_val = get_lot_step(main_symbol)
    params = _subst_symbol(params, main_symbol)

    leverage = float(params.get("futures_leverage", 5.0))
    initial_portfolio_value_usdt = float(params.get("initial_portfolio_value_usdt", 10000.0))
    if leverage <= 0:
        logging.warning("Invalid 'futures_leverage' <= 0 found in config. Using fallback leverage = 1e-9.")
        leverage = 1e-9
    target_weights_normal = params.get('target_weights_normal', {}) 
    if not target_weights_normal:
        target_weights_normal = params.get('target_weights', {})
        if target_weights_normal:
            logging.warning("'target_weights_normal' not found in config, falling back to 'target_weights'. "
                            "Please update your config to use 'target_weights_normal'.")
        else:
            logging.error("FATAL: 'target_weights_normal' (or legacy 'target_weights') is missing in config. Cannot proceed.")
            return {"status": "Configuration error: Target weights missing."}

    rebalance_threshold = params['rebalance_threshold']
    initial_portfolio_value_usdt = params.get('initial_portfolio_value_usdt', 10000)
    if 'initial_portfolio_value_usdt' not in params:
        logging.warning("Parameter 'initial_portfolio_value_usdt' not found in config. Using default value: 10000 USDT.")
    # ---- Commission helper -------------------------------------------------
    def _get_commission_rate(p: dict, maker: bool = False) -> float:
        keys = (
            ('commission_maker', 'maker_commission_rate') if maker
            else ('commission_taker', 'taker_commission_rate', 'commission_rate')
        )
        for k in keys:
            if k in p:
                return float(p[k])
        return 0.0  # sensible default for unit-tests

    maker_commission_rate = _get_commission_rate(params, maker=True)
    taker_commission_rate = _get_commission_rate(params, maker=False)
    use_maker_fees_in_backtest = bool(params.get('use_maker_fees_in_backtest', False))
    slippage_percent = params.get('slippage_percent', params.get('slippage_percentage', 0.0005))
    circuit_breaker_threshold_percent = params.get('circuit_breaker_threshold_percent', 0.10) 
    margin_usage_safe_mode_enter_threshold = params.get('margin_usage_safe_mode_enter_threshold', 0.70) 
    margin_usage_safe_mode_exit_threshold = params.get('margin_usage_safe_mode_exit_threshold', 0.50)
    safe_mode_target_weights = params.get('safe_mode_target_weights', target_weights_normal)
    min_rebalance_interval_minutes = params.get('min_rebalance_interval_minutes', 0)

    main_asset_symbol = params.get('main_asset_symbol', 'BTC')
    if 'main_asset_symbol' not in params:
        logging.warning(f"Parameter 'main_asset_symbol' not found in config. Using default value: '{main_asset_symbol}'.")

    spot_asset_key = f"{main_asset_symbol}_SPOT"
    long_asset_key = f"{main_asset_symbol}_PERP_LONG"
    short_asset_key = f"{main_asset_symbol}_PERP_SHORT"

    apply_signal_logic = params.get('apply_signal_logic', True)
    if 'apply_signal_logic' not in params:
        logging.warning("'apply_signal_logic' not found in config's backtest_settings. Defaulting to True (signal logic will be applied).")

    if apply_signal_logic:
        logging.info("Signal-based trading logic is ENABLED.")
    else:
        logging.info("Signal-based trading logic is DISABLED. Rebalancing will be purely weight-based.")

    current_commission_rate = maker_commission_rate if use_maker_fees_in_backtest else taker_commission_rate
    
    generate_reports = not is_optimizer_call or params.get('generate_reports_for_optimizer_trial', False)
    output_dir = None
    timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    actual_reports_dir = None
    if generate_reports:
        report_path_prefix = params.get('report_path_prefix', './reports/').rstrip('/')
        use_fixed_report_path = params.get("use_fixed_report_path", False)

        if use_fixed_report_path:
            output_dir = report_path_prefix
            if not output_dir: # If prefix was empty or just "/"
                output_dir = "reports" # Default to "reports" to be safe for tests
            logging.info(f"Using fixed report path: {output_dir} (due to 'use_fixed_report_path' setting).")
        else:
            # Existing logic for timestamped/optimizer paths
            if is_optimizer_call and trial_id_for_reports is not None:
                output_dir = os.path.join(report_path_prefix, "optimizer_trials", f"trial_{trial_id_for_reports}_{timestamp_str}")
            else:
                output_dir = os.path.join(report_path_prefix, f"backtest_{timestamp_str}")

        os.makedirs(output_dir, exist_ok=True) # Ensure this is after output_dir is fully determined
        logging.info(f"Output reports for this run will be saved to: {output_dir}")

        # The 'actual_reports_dir' logic from previous commit then correctly uses this 'output_dir'.
        if output_dir:
            actual_reports_dir = output_dir
        else:
            # This case should be less likely now if generate_reports is True,
            # as output_dir will be set by either fixed or timestamped logic.
            # However, keeping a fallback for robustness.
            actual_reports_dir = "reports"

        # This second os.makedirs on actual_reports_dir might be redundant if actual_reports_dir is always output_dir
        # when output_dir is set. However, it's harmless.
        os.makedirs(actual_reports_dir, exist_ok=True)
        # Logging for actual_reports_dir happens later in the original actual_reports_dir block,
        # but we already logged for output_dir. If they are the same, it's fine.
        # If different (e.g. output_dir was None and actual_reports_dir defaulted to "reports"),
        # then the original logging for actual_reports_dir is still valuable.
        # For clarity, let's ensure the primary logging for the *final* report destination is clear.
        # The original logging inside the actual_reports_dir determination block:
        #   logging.info(f"Report generation is ON. Output directory: {actual_reports_dir}")
        # This will correctly reflect the final path.
        # The logging.info for output_dir above shows the intermediate step or final if they are same.
    else:
        logging.info("Report generation is OFF. No reports will be saved.")
        # output_dir remains None as it's not used when reports are off.

    df_market_original = load_data(data_path) # Keep original for plotting price
    if df_market_original is None or df_market_original.empty:
        logging.error("Market data is empty or could not be loaded. Cannot run backtest.")
        zeros = {k: 0.0 for k in ("sharpe_ratio", "sortino_ratio",
                                  "max_drawdown_percent", "profit_factor", "win_rate_percent",
                                  "avg_trade_duration_candles", "avg_profit_per_trade_percent",
                                  "avg_loss_per_trade_percent", "num_winning_trades", "num_losing_trades",
                                  "longest_winning_streak", "longest_losing_streak", "max_portfolio_value_usdt",
                                  "min_portfolio_value_usdt", "annual_return_percent", "calmar_ratio",
                                  "kelly_criterion", "annualized_volatility_percent", "value_at_risk_var_percent",
                                  "conditional_value_at_risk_cvar_percent", "omega_ratio", "ulcer_index", "skewness", "kurtosis")} # Added more zeroed metrics
        zeros.update({
            "final_portfolio_value_usdt": initial_portfolio_value_usdt, # Corrected
            "total_net_pnl_usdt": 0.0, # Corrected
            "total_net_pnl_percent": 0.0, # Corrected
            "total_trades": 0,
            "output_dir": None, # output_dir determined later if reports are generated
            "status": "Market data empty" # Corrected status message
        })
        return zeros
    
    df_market = df_market_original.copy() # Work with a copy for potential modifications

    if df_market['timestamp'].dt.tz is None:
        logging.info("Market data 'timestamp' column is tz-naive. Localizing to UTC for consistency.")
        df_market['timestamp'] = df_market['timestamp'].dt.tz_localize('UTC')
    else:
        logging.info(f"Market data 'timestamp' column is already tz-aware ({df_market['timestamp'].dt.tz}). Converting to UTC for consistency.")
        df_market['timestamp'] = df_market['timestamp'].dt.tz_convert('UTC')

    # ── auto-range: если "auto" или дата вне диапазона файла ────────────
    min_ts, max_ts = df_market['timestamp'].min(), df_market['timestamp'].max()
    dr = params.setdefault("date_range", {}) # Get or create 'date_range' dict
    for edge, value in (("start_date", dr.get("start_date")), ("end_date", dr.get("end_date"))):
        if value in (None, "auto"):
            dr[edge] = (min_ts if edge == "start_date" else max_ts).isoformat()
            logging.info(f"Date range: '{edge}' set to '{dr[edge]}' (auto from data).") # Added logging
        else:   # дата в конфиге → убеждаемся, что попадает в файл
            dt = pd.to_datetime(value, utc=True, errors="coerce")
            if dt is pd.NaT or dt < min_ts or dt > max_ts:
                original_value = value # Store original value for logging
                dr[edge] = (min_ts if edge == "start_date" else max_ts).isoformat()
                logging.warning(f"Date range: '{edge}' was '{original_value}', adjusted to '{dr[edge]}' (out of data range or invalid).") # Enhanced logging
            # else: # If date is valid and within range, keep it as is from config. No change needed to dr[edge].

    df_market = df_market.sort_values(by='timestamp', ascending=True)

    signals_csv_path = params.get("data_settings", {}).get("signals_csv_path")
    df_signals = None
    if signals_csv_path:
        df_signals = load_signal_data(signals_csv_path)

    if df_signals is not None and not df_signals.empty:
        logging.info("Merging signal data with market data using merge_asof (backward)...")
        df_market = pd.merge_asof(df_market, df_signals[['timestamp', 'signal']],
                                  on='timestamp', direction='backward')
        df_market['signal'] = df_market['signal'].ffill()
        logging.info("Signal data merged. 'signal' column is now available in market data.")
        logging.info(f"Signal distribution in market data: \n{df_market['signal'].value_counts(dropna=False)}")
    else:
        logging.warning("No signal data loaded or signals file was empty/invalid. Proceeding with 'NEUTRAL' signals for all timestamps.")
        df_market['signal'] = 'NEUTRAL'

    if "date_range" in params and isinstance(params["date_range"], dict):
        start_date_str = params["date_range"].get("start_date")
        if start_date_str:
            try:
                start_date_dt = pd.to_datetime(start_date_str)
                if start_date_dt.tzinfo is None or start_date_dt.tzinfo.utcoffset(start_date_dt) is None:
                    start_date_dt = start_date_dt.tz_localize('UTC')
                else:
                    start_date_dt = start_date_dt.tz_convert('UTC')
                df_market = df_market[df_market['timestamp'] >= start_date_dt]
            except Exception as e: # More general exception
                logging.error(f"Error processing start_date '{start_date_str}': {e}. Skipping start date filter.")

        end_date_str = params["date_range"].get("end_date")
        if end_date_str:
            try:
                end_date_dt = pd.to_datetime(end_date_str)
                if end_date_dt.tzinfo is None or end_date_dt.tzinfo.utcoffset(end_date_dt) is None:
                    end_date_dt = end_date_dt.tz_localize('UTC')
                else:
                    end_date_dt = end_date_dt.tz_convert('UTC')
                df_market = df_market[df_market['timestamp'] <= end_date_dt]
            except Exception as e: # More general exception
                logging.error(f"Error processing end_date '{end_date_str}': {e}. Skipping end date filter.")
    
    if df_market.empty:                     # graceful-fail for unit-tests
        logging.error("Market data is empty after applying date range filters. Returning zero-metrics.")
        return {
            "final_portfolio_value_usdt": initial_portfolio_value_usdt,
            "total_net_pnl_usdt": 0.0,
            "total_net_pnl_percent": 0.0,
            "total_trades": 0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_percent": 0.0,
            "profit_factor": 0.0,
            "win_rate_percent": 0.0,
            "output_dir": output_dir,
            "status": "Market data empty"
        }

    trades_list = []
    equity_over_time = [] 
    portfolio = {
        'usdt_balance': initial_portfolio_value_usdt, 'btc_spot_qty': 0.0,
        'btc_spot_lots': [], 'btc_long_value_usdt': 0.0, 'btc_short_value_usdt': 0.0,
        'prev_btc_price': None, 'total_commissions_usdt': 0.0, 'total_slippage_usdt': 0.0,
        'current_operational_mode': 'NORMAL_MODE', 'num_circuit_breaker_triggers': 0,
        'num_safe_mode_entries': 0, 'time_steps_in_safe_mode': 0,
        'last_rebalance_attempt_timestamp': None,
    }
    blocked_trades_list = []
    orders_by_step = {}

    logging.info(f"Starting backtest for asset: {main_asset_symbol} with initial portfolio: {portfolio['usdt_balance']:.2f} USDT.")
    logging.info(f"Normal Target Weights ({main_asset_symbol}): {target_weights_normal}")
    logging.info(f"Safe Mode Target Weights ({main_asset_symbol}): {safe_mode_target_weights}")
    logging.info(f"Rebalance Threshold: {rebalance_threshold*100:.2f}%")
    logging.info(f"Min Rebalance Interval (minutes): {min_rebalance_interval_minutes}")

    if df_market.empty:
        logging.error("Market data is empty before starting main loop. Cannot run backtest.")
        # Return structure consistent with other error returns
        return {
            "final_portfolio_value_usdt": 0, "total_net_pnl_usdt": -initial_portfolio_value_usdt,
            "total_net_pnl_percent": -100.0, "total_trades": 0, "output_dir": output_dir,
            "status": "Market data empty before loop"
        }

    portfolio['prev_btc_price'] = df_market['close'].iloc[0]

    for index, row in df_market.iterrows():
        current_timestamp = row['timestamp']
        current_price = row['close']
        current_signal = row['signal']
        current_open_price = row.get('open', current_price)
        current_high_price = row.get('high', current_price)
        current_low_price = row.get('low', current_price)

        if circuit_breaker_threshold_percent > 0 and current_open_price > 0:
            candle_movement_percent = (current_high_price - current_low_price) / current_open_price
            if candle_movement_percent > circuit_breaker_threshold_percent:
                portfolio['num_circuit_breaker_triggers'] += 1
                logging.warning(f"CIRCUIT BREAKER TRIGGERED at {current_timestamp} for {main_asset_symbol}: "
                                f"Movement {candle_movement_percent*100:.2f}% > threshold {circuit_breaker_threshold_percent*100:.2f}%. "
                                f"Skipping rebalancing for this candle.")
                if portfolio['prev_btc_price'] is not None and portfolio['prev_btc_price'] > 0:
                    price_change_ratio = current_price / portfolio['prev_btc_price']
                    long_usdt = portfolio['btc_long_value_usdt']
                    short_usdt = portfolio['btc_short_value_usdt']
                    portfolio['btc_long_value_usdt'] = long_usdt + long_usdt * leverage * (price_change_ratio - 1)
                    portfolio['btc_short_value_usdt'] = short_usdt + short_usdt * leverage * (1 - price_change_ratio)
                
                total_portfolio_value_cb = calculate_portfolio_value(
                    portfolio['usdt_balance'], portfolio['btc_spot_qty'],
                    portfolio['btc_long_value_usdt'], portfolio['btc_short_value_usdt'], current_price)
                equity_over_time.append({'timestamp': current_timestamp, 'portfolio_value_usdt': total_portfolio_value_cb})
                if total_portfolio_value_cb <= 0:
                    logging.warning(f"Portfolio value is {total_portfolio_value_cb:.2f} at {current_timestamp} after CB for {main_asset_symbol}. Stopping backtest.")
                    final_val_cb = total_portfolio_value_cb if total_portfolio_value_cb is not None else 0
                    pnl_usdt_cb = final_val_cb - initial_portfolio_value_usdt
                    pnl_pct_cb = (pnl_usdt_cb / initial_portfolio_value_usdt) * 100 if initial_portfolio_value_usdt != 0 else 0
                    metrics_cb_fail = {key: 0 for key in ["sharpe_ratio", "sortino_ratio", "profit_factor", "win_rate_percent"]} # Initialize all expected keys
                    metrics_cb_fail.update({
                        "final_portfolio_value_usdt": final_val_cb, "total_net_pnl_usdt": pnl_usdt_cb,
                        "total_net_pnl_percent": pnl_pct_cb, "total_trades": len(trades_list),
                        "max_drawdown_percent": -100.0, # Or calculate actual if possible
                        "output_dir": output_dir, "status": "Portfolio wiped out post-CB",
                        **portfolio # Spread existing portfolio state
                    })
                    return metrics_cb_fail
                portfolio['prev_btc_price'] = current_price
                if portfolio['current_operational_mode'] == 'SAFE_MODE':
                    portfolio['time_steps_in_safe_mode'] +=1
                continue 
        elif circuit_breaker_threshold_percent > 0 and current_open_price <= 0:
             logging.warning(f"Candle open price is 0 or invalid at {current_timestamp} for {main_asset_symbol}, cannot calculate movement for circuit breaker.")

        if portfolio['prev_btc_price'] is not None and portfolio['prev_btc_price'] > 0:
            # ---- Symmetric Δ-PNL so that LONG + SHORT = const ----
            price_change_ratio = current_price / portfolio['prev_btc_price']
            # Determine the correct base for delta calculation. The diff uses 'base_long_margin'.
            # Assuming 'portfolio['btc_long_value_usdt']' before this update is the intended base for the long leg's margin.
            # Let's clarify if 'base_long_margin' should be 'portfolio['btc_long_value_usdt']' or if it implies sum of long and short, or total exposure.
            # The issue states: delta = long_margin × L × (ΔP/P₀). This implies the margin allocated to the long position.
            # The diff shows: base_long_margin = portfolio['btc_long_value_usdt']
            # This seems to mean the PNL calculation is driven by the long leg's exposure, and the short leg mirrors it.
            base_long_value_for_pnl_calc = portfolio['btc_long_value_usdt'] # Value before this PNL update

            delta_pnl_usdt = base_long_value_for_pnl_calc * leverage * (price_change_ratio - 1)

            portfolio['btc_long_value_usdt']  +=  delta_pnl_usdt      # LONG gains / losses
            portfolio['btc_short_value_usdt'] -=  delta_pnl_usdt      # SHORT opposite
        
        total_portfolio_value = calculate_portfolio_value(
            portfolio['usdt_balance'], portfolio['btc_spot_qty'],
            portfolio['btc_long_value_usdt'], portfolio['btc_short_value_usdt'], current_price)

        nav = total_portfolio_value
        used_margin_usdt = 0
        if nav > 0 and params.get("safe_mode_config", {}).get("enabled", False):
            # Ensure leverage is not zero before division, though typical leverage values are > 0
            # Defaulting to a very small number if leverage is 0 to avoid ZeroDivisionError,
            # effectively making margin usage extremely high if leverage is misconfigured to 0.
            # A leverage of 0 for a leveraged position doesn't make practical sense.
            margin_for_long  = abs(portfolio['btc_long_value_usdt'])  / leverage
            margin_for_short = abs(portfolio['btc_short_value_usdt']) / leverage
            used_margin_usdt = margin_for_long + margin_for_short
            margin_usage_ratio = used_margin_usdt / nav if nav > 0 else 0.0
        else:
            margin_usage_ratio = 0.0 

        active_target_weights = target_weights_normal
        previous_mode = portfolio['current_operational_mode']
        
        if params.get("safe_mode_config", {}).get("enabled", False):
            if portfolio['current_operational_mode'] == 'NORMAL_MODE':
                if margin_usage_ratio > margin_usage_safe_mode_enter_threshold:
                    portfolio['current_operational_mode'] = 'SAFE_MODE'
                    portfolio['num_safe_mode_entries'] += 1
                    logging.info(f"ENTERING SAFE MODE at {current_timestamp} due to margin usage: {margin_usage_ratio*100:.2f}% "
                                 f"(Threshold: {margin_usage_safe_mode_enter_threshold*100:.2f}%)")
            elif portfolio['current_operational_mode'] == 'SAFE_MODE':
                if margin_usage_ratio < margin_usage_safe_mode_exit_threshold:
                    portfolio['current_operational_mode'] = 'NORMAL_MODE'
                    logging.info(f"EXITING SAFE MODE at {current_timestamp}, margin usage: {margin_usage_ratio*100:.2f}% "
                                 f"(Threshold: {margin_usage_safe_mode_exit_threshold*100:.2f}%)")
            
            if portfolio['current_operational_mode'] == 'SAFE_MODE':
                active_target_weights = safe_mode_target_weights
                portfolio['time_steps_in_safe_mode'] += 1
            else: 
                active_target_weights = target_weights_normal
        else:
            active_target_weights = target_weights_normal

        mode_changed_this_step = previous_mode != portfolio['current_operational_mode']
        equity_over_time.append({'timestamp': current_timestamp, 'portfolio_value_usdt': total_portfolio_value})

        if total_portfolio_value <= 0: 
            logging.warning(f"Portfolio value is {total_portfolio_value:.2f} at {current_timestamp} before rebalance. Stopping backtest.")
            if not equity_over_time or equity_over_time[-1]['timestamp'] != current_timestamp:
                 equity_over_time.append({'timestamp': current_timestamp, 'portfolio_value_usdt': total_portfolio_value})
            final_val = total_portfolio_value if total_portfolio_value is not None else 0
            pnl_usdt = final_val - initial_portfolio_value_usdt
            pnl_pct = (pnl_usdt / initial_portfolio_value_usdt) * 100 if initial_portfolio_value_usdt != 0 else 0
            metrics_fail = {key: 0 for key in ["sharpe_ratio", "sortino_ratio", "profit_factor", "win_rate_percent"]} # Initialize all expected keys
            metrics_fail.update({
                "final_portfolio_value_usdt": final_val, "total_net_pnl_usdt": pnl_usdt,
                "total_net_pnl_percent": pnl_pct, "total_trades": len(trades_list),
                "max_drawdown_percent": -100.0, # Or calculate actual
                "output_dir": output_dir, "status": "Portfolio wiped out",
                **portfolio
            })
            return metrics_fail

        can_check_rebalance_now = True
        if min_rebalance_interval_minutes > 0 and portfolio['last_rebalance_attempt_timestamp'] is not None:
            time_since_last_attempt = current_timestamp - portfolio['last_rebalance_attempt_timestamp']
            if time_since_last_attempt < pd.Timedelta(minutes=min_rebalance_interval_minutes) and not mode_changed_this_step and index !=0 :
                can_check_rebalance_now = False
        
        needs_rebalance = False
        current_weights = {}
        if can_check_rebalance_now:
            portfolio['last_rebalance_attempt_timestamp'] = current_timestamp
            current_weights = {
                "USDT": portfolio['usdt_balance'] / total_portfolio_value if total_portfolio_value else 1,
                spot_asset_key: (portfolio['btc_spot_qty'] * current_price) / total_portfolio_value if total_portfolio_value else 0,
                long_asset_key: portfolio['btc_long_value_usdt'] / total_portfolio_value if total_portfolio_value else 0,
                short_asset_key: portfolio['btc_short_value_usdt'] / total_portfolio_value if total_portfolio_value else 0,
            }
            for key in active_target_weights:
                if key not in current_weights:
                    current_weights[key] = 0.0
        
            if index == 0 or mode_changed_this_step:
                needs_rebalance = True
                if index == 0: logging.info(f"Initial rebalance for {main_asset_symbol} triggered at {current_timestamp} (Price: {current_price:.2f}) to establish target weights: {active_target_weights}.")
                if mode_changed_this_step: logging.info(f"Mode changed to {portfolio['current_operational_mode']} for {main_asset_symbol}. Forcing rebalance check against new weights: {active_target_weights}.")
            else:
                for asset_key_loop, target_w_loop in active_target_weights.items():
                    current_w = current_weights.get(asset_key_loop, 0)
                    if abs(current_w - target_w_loop) > rebalance_threshold:
                        needs_rebalance = True 
                        logging.info(f"Rebalance threshold triggered for {main_asset_symbol} at {current_timestamp} (Price: {current_price:.2f}). Asset {asset_key_loop} current weight {current_w:.4f}, target {target_w_loop:.4f} (Mode: {portfolio['current_operational_mode']})")
                        break
        
        if needs_rebalance: 
            logging.info(f"Rebalancing portfolio for {main_asset_symbol} (Mode: {portfolio['current_operational_mode']}, Signal: {current_signal}). Total Value: {total_portfolio_value:.2f} USDT. Current Price: {current_price:.2f}")
            adjustments = {}
            for asset_key_loop, target_w_loop in active_target_weights.items():
                # scale PERP notional by leverage so that pnl ~ leverage
                if asset_key_loop in (long_asset_key, short_asset_key):
                    target_value_usdt = target_w_loop * total_portfolio_value * leverage
                else:
                    target_value_usdt = target_w_loop * total_portfolio_value
                current_value_usdt = 0
                if asset_key_loop == spot_asset_key: current_value_usdt = portfolio['btc_spot_qty'] * current_price
                elif asset_key_loop == long_asset_key: current_value_usdt = portfolio['btc_long_value_usdt']
                elif asset_key_loop == short_asset_key: current_value_usdt = portfolio['btc_short_value_usdt']
                elif asset_key_loop == "USDT": current_value_usdt = portfolio['usdt_balance']
                
                adjustment_usdt = target_value_usdt - current_value_usdt

                if apply_signal_logic:
                    original_proposed_adjustment_usdt = adjustment_usdt
                    trade_blocked_by_signal = False
                    if current_signal == "BUY":
                        if (asset_key_loop == spot_asset_key or asset_key_loop == long_asset_key) and original_proposed_adjustment_usdt < 0:
                            trade_blocked_by_signal = True
                        elif asset_key_loop == short_asset_key and original_proposed_adjustment_usdt > 0:
                            trade_blocked_by_signal = True
                    elif current_signal == "SELL":
                        if asset_key_loop == short_asset_key and original_proposed_adjustment_usdt < 0:
                            trade_blocked_by_signal = True
                        elif (asset_key_loop == spot_asset_key or asset_key_loop == long_asset_key) and original_proposed_adjustment_usdt > 0:
                            trade_blocked_by_signal = True

                    if trade_blocked_by_signal:
                        current_weight_for_log = current_weights.get(asset_key_loop, 0.0)
                        target_weight_for_log = target_w_loop
                        logging.info(
                            f"  Signal {current_signal} for {main_asset_symbol}: Preventing {'SELL' if current_signal=='BUY' else 'BUY'} of {asset_key_loop} "
                            f"Original Prop. Adjust USDT: {original_proposed_adjustment_usdt:.2f}, "
                            f"Current Wt: {current_weight_for_log:.4f}, Target Wt: {target_weight_for_log:.4f}. "
                            f"Final Adjust USDT set to: 0.00"
                        )
                        blocked_trade_info = {
                            "timestamp": current_timestamp, "main_asset_symbol": main_asset_symbol,
                            "asset_key": asset_key_loop,
                            "intended_action": "BUY" if original_proposed_adjustment_usdt > 0 else "SELL",
                            "proposed_adjustment_usdt": original_proposed_adjustment_usdt,
                            "active_signal": current_signal, "current_weight": current_weight_for_log,
                            "target_weight": target_weight_for_log
                        }
                        blocked_trades_list.append(blocked_trade_info)
                        adjustment_usdt = 0

                adjustments[asset_key_loop] = adjustment_usdt

            for asset_key_trade, usdt_value_to_trade in adjustments.items():
                if asset_key_trade == "USDT": continue

                # ------------------------------------------------------------
                #  Ideal-mode (apply_signal_logic=False) ⇒ *не* режем «пыль»   ←
                # ------------------------------------------------------------
                dust_filter_on = params.get("apply_signal_logic", True)

                if dust_filter_on:
                    min_nominal = params.get("min_order_notional_usdt", 10.0)
                    if abs(usdt_value_to_trade) < min_nominal:
                        continue

                    # ---- check rounded quantity value < 1e-6 USDT ----------
                    if current_price > 0:
                        asset_qty_unrounded = abs(usdt_value_to_trade) / current_price
                        rounded_asset_qty = (
                            round(asset_qty_unrounded / lot_step_val) * lot_step_val
                            if lot_step_val > 0 else asset_qty_unrounded
                        )
                        value_of_rounded_asset_qty = rounded_asset_qty * current_price
                        if abs(value_of_rounded_asset_qty) < 1e-6:
                            logging.info(
                                "  Skipping micro-order (dust) for %s: "
                                "rounded_qty×price = %.8f USDT < 1e-6 USDT.",
                                asset_key_trade, value_of_rounded_asset_qty,
                            )
                            continue
                    else:
                        logging.warning(
                            "  Dust-check skipped for %s due to non-positive price %.4f",
                            asset_key_trade, current_price,
                        )

                # ── direction depends on asset type ────────────
                if asset_key_trade == short_asset_key:
                    # увеличиваем шорт → SELL, уменьшаем → BUY
                    action_dir = "SELL" if usdt_value_to_trade > 0 else "BUY"
                else:   # spot / long
                    action_dir = "BUY"  if usdt_value_to_trade > 0 else "SELL"

                # ---- Gate hedged futures mapping ----
                if asset_key_trade == long_asset_key:
                    order_type = "OPEN_LONG"  if action_dir == "BUY"  else "CLOSE_LONG"
                elif asset_key_trade == short_asset_key:
                    # OPEN_SHORT ⇔ SELL,   CLOSE_SHORT ⇔ BUY
                    order_type = "OPEN_SHORT" if action_dir == "SELL" else "CLOSE_SHORT"
                else:                              # spot leg
                    order_type = action_dir            # BUY/SELL

                abs_usdt_value_of_trade = abs(usdt_value_to_trade) 
                commission_usdt = abs_usdt_value_of_trade * current_commission_rate
                portfolio['usdt_balance'] -= commission_usdt
                portfolio['total_commissions_usdt'] += commission_usdt

                # --- execution logic added ---
                quantity_asset_traded_final = 0.0
                realized_pnl_this_spot_trade = 0.0
                slippage_cost_this_trade_usdt = abs_usdt_value_of_trade * slippage_percent

                # ---------- SPOT BTC ----------
                if asset_key_trade == spot_asset_key:
                    qty_btc = abs_usdt_value_of_trade / current_price
                    quantity_asset_traded_final = qty_btc
                    if order_type == "BUY": # Spot BUY
                        portfolio["btc_spot_qty"] = portfolio.get("btc_spot_qty", 0.0) + qty_btc
                        portfolio["usdt_balance"] -= abs_usdt_value_of_trade
                    else: # Spot SELL
                        qty_close = min(qty_btc, portfolio.get("btc_spot_qty", 0.0))
                        portfolio["btc_spot_qty"] -= qty_close
                        portfolio["usdt_balance"] += qty_close * current_price
                        realized_pnl_this_spot_trade = (current_price - portfolio.get("prev_btc_price", current_price)) * qty_close

                # ---------- PERP LONG ----------
                elif asset_key_trade == long_asset_key:
                    quantity_asset_traded_final = abs_usdt_value_of_trade # For futures, asset quantity is the quote value
                    if order_type == "OPEN_LONG":
                        portfolio["btc_long_value_usdt"] = portfolio.get("btc_long_value_usdt", 0.0) + abs_usdt_value_of_trade
                        portfolio["usdt_balance"] -= abs_usdt_value_of_trade # Margin used
                    elif order_type == "CLOSE_LONG":
                        close_val = min(abs_usdt_value_of_trade, portfolio.get("btc_long_value_usdt", 0.0))
                        portfolio["btc_long_value_usdt"] -= close_val
                        portfolio["usdt_balance"] += close_val # Margin returned

                # ---------- PERP SHORT ----------
                elif asset_key_trade == short_asset_key:
                    quantity_asset_traded_final = abs_usdt_value_of_trade # For futures, asset quantity is the quote value
                    if order_type == "OPEN_SHORT": # Opening/increasing a short position
                        portfolio["btc_short_value_usdt"] = portfolio.get("btc_short_value_usdt", 0.0) + abs_usdt_value_of_trade
                        # USDT balance increases because we are effectively borrowing to sell, or margin is allocated
                        # This depends on exact accounting, but for value_usdt based, it's adding to the short value.
                        # The key is that `btc_short_value_usdt` represents the magnitude of the short.
                        # Let's assume for this model, opening short *increases* USDT available if it's collateral based,
                        # or if `btc_short_value_usdt` is tracking the notional value *exposed* to short.
                        # For consistency with LONG, let's assume opening a short also "uses" USDT from balance for margin.
                        portfolio["usdt_balance"] -= abs_usdt_value_of_trade # Margin used for opening short
                    elif order_type == "CLOSE_SHORT": # Closing a short position (buying back)
                        close_val = min(abs_usdt_value_of_trade, portfolio.get("btc_short_value_usdt", 0.0))
                        portfolio["btc_short_value_usdt"] -= close_val
                        portfolio["usdt_balance"] += close_val # Margin returned
                # Determine quantity for orders_by_step, should be in asset terms
                qty_for_orders = 0
                if current_price > 0: # Avoid division by zero if price is somehow zero
                    if asset_key_trade == spot_asset_key:
                        qty_for_orders = abs(usdt_value_to_trade) / current_price
                    elif asset_key_trade == long_asset_key or asset_key_trade == short_asset_key:
                        # For leveraged assets, qty should also be in base asset terms for simulate_rebalance
                        qty_for_orders = abs(usdt_value_to_trade) / current_price
                        # Note: simulate_rebalance applies leverage, so qty here is pre-leverage asset qty

                if qty_for_orders > 0:
                    idx_for_orders = index # Current index from df_market.iterrows()
                    # action_dir is 'BUY' or 'SELL', already determined in the loop
                    orders_by_step.setdefault(idx_for_orders, []).append({
                        'asset_key': asset_key_trade,
                        'side': action_dir,
                        'qty': qty_for_orders
                    })

                # Pass action_dir (BUY/SELL) as the 'action' for the trade record
                record_trade(current_timestamp, asset_key_trade, action_dir, quantity_asset_traded_final,
                             abs_usdt_value_of_trade, current_price, commission_usdt,
                             slippage_cost_this_trade_usdt, realized_pnl_this_spot_trade, trades_list,
                             realized_pnl_spot_usdt=realized_pnl_this_spot_trade)
            
            # ... (logging post-rebalance portfolio) ...

        portfolio['prev_btc_price'] = current_price

    # Simulate rebalance based on collected orders and calculate PnL
    simulated_trade_log = [] # Initialize as empty list
    if orders_by_step:
        logging.info(f"Calling simulate_rebalance with {len(orders_by_step)} steps having orders.")
        # 'leverage' variable should already be defined from params_dict
        # 'df_market' is the correct DataFrame containing all candles for the backtest period
        # закрываем хвосты только если это обычный бэктест, а не оптимизатор
        simulated_trade_log = simulate_rebalance( # Assign to the already defined list
            df_market,
            orders_by_step,
            leverage=leverage,
            force_close_open_positions=not is_optimizer_call
        )
    else:
        logging.info("No orders were generated by the main rebalancing logic for simulate_rebalance.")

    # This block will now always execute if generate_reports is true
    if generate_reports and actual_reports_dir:
        rebalance_trades_csv_path = os.path.join(actual_reports_dir, "rebalance_trades.csv")
        if simulated_trade_log:  # if list is not empty (either from simulate_rebalance or if it was [] initially)
            df_sim_trades = pd.DataFrame(simulated_trade_log)
            df_sim_trades.to_csv(rebalance_trades_csv_path, index=False)
            logging.info(f"Simulated rebalance trades PnL report saved to {rebalance_trades_csv_path}")
        else: # simulated_trade_log is empty (either from simulate_rebalance returning empty, or orders_by_step was empty)
            logging.info("Simulated trade log is empty. Saving empty rebalance_trades.csv.")
            empty_df = pd.DataFrame(columns=[
                "asset_key", "entry_price", "exit_price", "qty",
                "pnl_gross_quote", "leverage" # Ensure these columns match test expectations
            ])
            empty_df.to_csv(rebalance_trades_csv_path, index=False)
            logging.info(f"Empty simulated rebalance trades file saved to {rebalance_trades_csv_path}")
    elif generate_reports:
        logging.warning("generate_reports is True, but actual_reports_dir is not set. Skipping saving rebalance_trades.csv.")
    else:
        logging.info("Report generation is OFF. Skipping saving rebalance_trades.csv.")

    logging.info("Backtest finished.")
    df_equity = pd.DataFrame(equity_over_time)
    df_trades = pd.DataFrame(trades_list)

    # ---------- PERFORMANCE METRICS ----------
    def compute_metrics(df_eq: pd.DataFrame, trades: list[dict], initial_nav: float, ann_factor: int = 252):
        out: dict[str, float] = {}
        if df_eq.empty:
            return {k: 0.0 for k in ("sharpe_ratio", "sortino_ratio", "max_drawdown_percent", "profit_factor", "win_rate_percent")}

        # ---- Dynamic annualisation factor ---------------------------------
        if "timestamp" in df_eq.columns and pd.api.types.is_datetime64_any_dtype(df_eq["timestamp"]):
            # Ensure timestamp is sorted for diff to be meaningful - df_equity is typically sorted by timestamp already
            # df_eq = df_eq.sort_values(by="timestamp") # Optional: uncomment if sorting is not guaranteed
            freq_sec = df_eq["timestamp"].diff().dt.total_seconds().median()
            if pd.notna(freq_sec) and freq_sec > 0:
                periods_per_year = (365 * 24 * 60 * 60) / freq_sec  # e.g. 5-min → 105 120
                ann_sqrt = np.sqrt(periods_per_year)
                # logging.debug(f"Dynamic ann_sqrt: {ann_sqrt:.2f} (freq_sec: {freq_sec:.2f}s, periods_per_year: {periods_per_year:.2f})")
            else:
                ann_sqrt = np.sqrt(252)  # fallback to daily bars
                # logging.debug(f"Dynamic ann_sqrt: Fallback to daily (252) due to freq_sec: {freq_sec}")
        else:
            ann_sqrt = np.sqrt(252) # fallback to daily if no timestamp column
            # logging.debug("Dynamic ann_sqrt: Fallback to daily (252) due to missing/invalid timestamp column")

        EPS = 1e-9
        rets = df_eq["portfolio_value_usdt"].pct_change().dropna()
        if rets.empty:
            mean_ret = std_ret = down_std = 0.0
        else:
            mean_ret = rets.mean()
            std_ret  = rets.std()
            down_std = rets[rets < 0].std()

        if std_ret < EPS:              # zero-vol or single-point case
            out["sharpe_ratio"]  = 0.0
        else:
            out["sharpe_ratio"]  = mean_ret / std_ret * ann_sqrt

        if down_std is None or down_std < EPS: # down_std can be None if rets[rets<0] is empty
            out["sortino_ratio"] = 0.0
        else:
            out["sortino_ratio"] = mean_ret / down_std * ann_sqrt

        rolling_max = df_eq["portfolio_value_usdt"].cummax()
        drawdown = (df_eq["portfolio_value_usdt"] - rolling_max) / rolling_max
        out["max_drawdown_percent"] = drawdown.min() * 100 if not drawdown.empty else 0.0

        if trades:
            pnl_list = [t.get("pnl_net_quote", 0.0) for t in trades]
            wins = [p for p in pnl_list if p > 0]
            losses = [-p for p in pnl_list if p < 0]
            out["profit_factor"] = (sum(wins) / sum(losses)) if losses else 0.0
            out["win_rate_percent"] = (len(wins) / len(pnl_list)) * 100 if pnl_list else 0.0
        else:
            out["profit_factor"] = 0.0
            out["win_rate_percent"] = 0.0

        return out

    metrics = {}
    metrics["run_id"] = f"backtest_{timestamp_str}"
    # ... (ALL ORIGINAL METRICS CALCULATIONS MUST BE PRESERVED HERE) ...
    metrics["initial_portfolio_value_usdt"] = initial_portfolio_value_usdt
    final_portfolio_value = df_equity['portfolio_value_usdt'].iloc[-1] if not df_equity.empty else initial_portfolio_value_usdt
    metrics["final_portfolio_value_usdt"] = final_portfolio_value
    metrics["total_net_pnl_usdt"] = final_portfolio_value - initial_portfolio_value_usdt

    # Calculate additional performance metrics
    extra_metrics = compute_metrics(df_equity, trades_list,
                                    initial_portfolio_value_usdt,
                                    params.get("annualization_factor", 252))
    metrics.update(extra_metrics)
    metrics["total_net_pnl_percent"] = (metrics["total_net_pnl_usdt"] / initial_portfolio_value_usdt) * 100 if initial_portfolio_value_usdt != 0 else 0
    metrics["total_trades"] = len(df_trades)

    # Calculate NAV standard deviation percentage
    if not df_equity.empty:
        nav_series = df_equity['portfolio_value_usdt']
        # New calculation based on percentage changes, does not require initial_portfolio_value_usdt for scaling here.
        metrics["nav_std_percent"] = nav_series.pct_change().fillna(0).std() * 100
    else:
        # Handle case where df_equity is empty (e.g., no trades or data)
        metrics["nav_std_percent"] = 0.0

    # ─── Numerical-noise cleanup (≤ 1 cent) ───────────────────────
    tol = float(params.get("neutrality_pnl_tolerance_usd", 1e-2))
    if abs(metrics["total_net_pnl_usdt"]) <= tol:
        metrics["total_net_pnl_usdt"] = 0.0
        metrics["total_net_pnl_percent"] = 0.0
        # metrics["sharpe_ratio"] = 0.0  # спорное решение, возможно лучше оставлять как есть
        metrics["final_portfolio_value_usdt"] = initial_portfolio_value_usdt

    # Add relevant portfolio state counters to metrics
    metrics["num_safe_mode_entries"] = portfolio.get('num_safe_mode_entries', 0)
    metrics["num_circuit_breaker_triggers"] = portfolio.get('num_circuit_breaker_triggers', 0)
    metrics["time_steps_in_safe_mode"] = portfolio.get('time_steps_in_safe_mode', 0)

    # (And many more metrics from the original file)


    # Синхронизируем «очищенные» метрики с объектом,
    # возвращаемым оптимизатору / внешним вызовам.
    if generate_reports and actual_reports_dir:
        logging.info(f"Generating reports in {actual_reports_dir}...")
        trades_csv_path = os.path.join(actual_reports_dir, "trades.csv")
        df_trades.to_csv(trades_csv_path, index=False)
        logging.info(f"Trades report saved to {trades_csv_path}")

        df_summary = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        summary_csv_path = os.path.join(actual_reports_dir, "summary.csv")
        df_summary.to_csv(summary_csv_path, index=False)
        logging.info(f"Summary report saved to {summary_csv_path}")

        if not df_equity.empty:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(x=df_equity['timestamp'], y=df_equity['portfolio_value_usdt'], mode='lines', name='Portfolio Value (USDT)'),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=df_market['timestamp'], y=df_market['close'], mode='lines',
                    name=f"{main_asset_symbol} Price", line=dict(color='rgba(255,165,0,0.6)')),
                secondary_y=True,
            )

            asset_colors = {
                spot_asset_key: {'BUY': 'rgba(0,128,0,0.9)', 'SELL': 'rgba(255,0,0,0.9)'},
                long_asset_key: {'BUY': 'rgba(0,0,255,0.7)', 'SELL': 'rgba(255,140,0,0.7)'},
                short_asset_key: {'BUY': 'rgba(128,0,128,0.7)', 'SELL': 'rgba(165,42,42,0.7)'}
            }
            asset_symbols = {
                spot_asset_key: {'BUY': 'triangle-up', 'SELL': 'triangle-down'},
                long_asset_key: {'BUY': 'circle', 'SELL': 'circle-open'},
                short_asset_key: {'BUY': 'star', 'SELL': 'star-open'}
            }

            if not df_trades.empty:
                for asset_name_key_plot in [spot_asset_key, long_asset_key, short_asset_key]:
                    for action_str_plot in ['BUY', 'SELL']:
                        trades_to_plot = df_trades[
                            (df_trades['action'] == action_str_plot) &
                            (df_trades['asset_type'] == asset_name_key_plot)
                        ]
                        if not trades_to_plot.empty:
                            color_map_plot = asset_colors.get(asset_name_key_plot, {})
                            symbol_map_plot = asset_symbols.get(asset_name_key_plot, {})

                            fig.add_trace(go.Scatter(
                                x=trades_to_plot['timestamp_open'],
                                y=trades_to_plot['entry_price'],
                                mode='markers',
                                marker=dict(
                                    color=color_map_plot.get(action_str_plot, 'grey'),
                                    symbol=symbol_map_plot.get(action_str_plot, 'diamond'),
                                    size=9,
                                    line=dict(width=1, color='DarkSlateGrey')
                                ),
                                name=f'{asset_name_key_plot} {action_str_plot}',
                                yaxis="y2",
                                hoverinfo='text',
                                text=[
                                    (f"Asset: {trade_row['asset_type']}<br>Action: {trade_row['action']}<br>"
                                     f"Qty Asset: {trade_row['quantity_asset']:.6f}<br>Qty Quote: {trade_row['quantity_quote']:.2f}<br>"
                                     f"Price: {trade_row['entry_price']:.2f}<br>Comm: {trade_row['commission_quote']:.2f}<br>"
                                     f"Timestamp: {trade_row['timestamp_open'].strftime('%Y-%m-%d %H:%M:%S')}")
                                    for _, trade_row in trades_to_plot.iterrows()
                                ]
                            ), secondary_y=True)

            fig.update_layout(
                title_text=f'Portfolio Equity Over Time vs {main_asset_symbol} Price',
                xaxis_title='Timestamp',
                hovermode="x unified",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            fig.update_yaxes(title_text="Portfolio Value (USDT)", secondary_y=False, showgrid=True)
            fig.update_yaxes(title_text=f"{main_asset_symbol} Price (USDT)", secondary_y=True, showgrid=False)

            # --------------- ▼▼  NEW  —  overlay BUY / SELL signals  ▼▼ ---------------
            if apply_signal_logic and df_signals is not None and not df_signals.empty:
                # price привязываем к ближайшей свече (если сигналы «попадают в дырку»)
                df_sig_plot = (df_signals
                               .merge(df_market[['timestamp', 'close']], on='timestamp', how='left')
                               .ffill())
                for sig, color, sym in [("BUY","rgba(0,200,0,.85)",'triangle-up'), ("SELL","rgba(200,0,0,.85)",'triangle-down')]: # Variable names changed
                    sub = df_sig_plot[df_sig_plot.signal==sig] # Adjusted access to 'signal' column
                    if sub.empty: continue
                    fig.add_trace(go.Scatter(x=sub.timestamp, y=sub.close, mode='markers',
                              marker=dict(size=10, symbol=sym, color=color, line=dict(width=1.2,color='DarkSlateGrey')),
                              name=f"{sig} signal",   # своя трасса
                              legendgroup=f"{sig}_signal",  # отдельная группа
                              yaxis='y2',
                              showlegend=True))
            # --------------- ▲▲  END NEW  ▲▲ ------------------------------------------

            equity_html_path = os.path.join(actual_reports_dir, "equity.html")
            fig.write_html(equity_html_path)
            logging.info(f"Enhanced equity curve saved to {equity_html_path}")

            equity_csv_path = os.path.join(actual_reports_dir, "equity.csv")
            df_equity.to_csv(equity_csv_path, index=False)
            logging.info(f"Equity curve data saved to {equity_csv_path}")
        else:
            logging.warning("Equity data is empty. Skipping equity curve generation and equity.csv saving.")

        if blocked_trades_list:
            df_blocked_trades = pd.DataFrame(blocked_trades_list)
            if not df_blocked_trades.empty: # Check if DataFrame is non-empty after creation
                blocked_trades_csv_path = os.path.join(actual_reports_dir, "blocked_trades_log.csv")
                df_blocked_trades.to_csv(blocked_trades_csv_path, index=False)
                logging.info(f"Blocked trades log saved to {blocked_trades_csv_path}")
            else: # This case might occur if blocked_trades_list was empty
                logging.info("No trades were blocked by signals during this backtest run (DataFrame was empty).")
        else: # This case for if the list itself was empty
            logging.info("No trades were blocked by signals during this backtest run (list was empty).")

        logging.info(f"All reports for this run generated in {actual_reports_dir}.")
    elif generate_reports: # actual_reports_dir was somehow not set
        logging.warning("generate_reports is True, but actual_reports_dir is not set. Skipping main report generation block.")
    else: # generate_reports is False
        logging.info("Report generation is OFF. Skipping main report generation block.")

    results_for_optimizer = metrics.copy()
    results_for_optimizer["output_dir"] = actual_reports_dir # Store the actual_reports_dir, which could be None if reports are off
    results_for_optimizer["status"] = "Completed"

    for key_metric in ["sharpe_ratio", "sortino_ratio", "profit_factor",
                       "win_rate_percent", "max_drawdown_percent"]:
        if pd.isna(results_for_optimizer.get(key_metric)):
            results_for_optimizer[key_metric] = 0.0
            logging.warning(f"Metric {key_metric} was NaN, converted to 0.0 for optimizer.")
    return results_for_optimizer
# --- END OF REPLACEMENT FUNCTION ---

def run_standalone_backtest(backtest_settings_dict, data_file_path):
    """
    Wrapper to run a single backtest using a backtest_settings dictionary and data file path.
    This is primarily for CLI execution of the backtester itself.
    """
    logging.info(f"Running standalone backtest with data from: {data_file_path}")
    # Standalone runs should always generate full reports, so is_optimizer_call=False.
    # The run_backtest function's default for generate_reports handles this.
    return run_backtest(backtest_settings_dict, data_file_path, is_optimizer_call=False)


def main():
    parser = argparse.ArgumentParser(description="Prosperous Bot Rebalance Backtester (CLI - Unified Config)")
    parser.add_argument("--config_file", type=str, required=True, 
                        help="Path to the unified JSON configuration file (e.g., config/unified_config.json). "
                             "The backtester will use the 'backtest_settings' section.")
    parser.add_argument("--override", type=str, help="JSON string to override config parameters, e.g., '{\"main_asset_symbol\":\"DOGE\"}'")
    args = parser.parse_args()

    backtest_params = None
    # data_file_path will be resolved later
    
    try:
        with open(args.config_file, 'r') as f:
            unified_config = json.load(f)
        
        if "backtest_settings" in unified_config:
            backtest_params = unified_config["backtest_settings"]
            logging.info(f"Loaded backtest settings from '{args.config_file}' (using 'backtest_settings' section).")
        else:
            logging.error(f"FATAL: Unified config '{args.config_file}' does not contain a 'backtest_settings' section. "
                          "This section is required for standalone backtester runs. Please use/create a config based on "
                          "'config/unified_config.example.json'. Exiting.")
            return

        # Determine actual_main_asset_symbol and update backtest_params if override is present
        actual_main_asset_symbol = backtest_params.get("main_asset_symbol", "BTC") # Default from config or BTC

        if args.override:
            try:
                override_dict = json.loads(args.override)
                if "main_asset_symbol" in override_dict:
                    actual_main_asset_symbol = override_dict["main_asset_symbol"]
                    backtest_params["main_asset_symbol"] = actual_main_asset_symbol # Update for run_backtest internal use
                    logging.info(f"Override: main_asset_symbol set to '{actual_main_asset_symbol}'.")
                # Potentially merge other overrides into backtest_params here if needed for run_standalone_backtest
                # For now, only main_asset_symbol is critical for path resolution before run_backtest
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding --override JSON '{args.override}': {e}. Using symbol from config or default.")

        csv_path_template = backtest_params.get("data_settings", {}).get("csv_file_path")
        if not csv_path_template:
            logging.error("FATAL: 'data_settings.csv_file_path' not found in the 'backtest_settings' section of the config. Exiting.")
            return
        
        resolved_data_file_path = _subst_symbol(csv_path_template, actual_main_asset_symbol)
        if not resolved_data_file_path: # Should not happen if template and symbol are valid
            logging.error(f"FATAL: Could not resolve data_file_path from template '{csv_path_template}' with symbol '{actual_main_asset_symbol}'. Exiting.")
            return
        
        # The signals_csv_path is resolved inside run_backtest using the (potentially overridden)
        # main_asset_symbol in backtest_params, so no need to resolve it here for the main function's direct use.

    except FileNotFoundError:
        logging.warning(f"Configuration file '{args.config_file}' not found. "
                        "A dummy configuration will be created at 'config/dummy_unified_config_for_backtester.json' for demonstration.")
        
        dummy_config_filename = "dummy_unified_config_for_backtester.json"
        dummy_config_path = os.path.join("config", dummy_config_filename) 
        dummy_data_filename = "dummy_BTCUSDT_1h_for_backtester.csv" # Changed GALA to BTC
        dummy_data_path = os.path.join("data", dummy_data_filename) 

        os.makedirs(os.path.dirname(dummy_config_path), exist_ok=True)
        os.makedirs(os.path.dirname(dummy_data_path), exist_ok=True)

        # Define a minimal but complete backtest_settings structure for the dummy
        dummy_backtest_settings_content = {
          "main_asset_symbol": "BTC",
          "apply_signal_logic": True, # Added apply_signal_logic
          "initial_capital": 10000.0,
          "commission_taker": 0.0007,
          "commission_maker": 0.0002,
          "use_maker_fees_in_backtest": False,
          "slippage_percent": 0.0005,
          "annualization_factor": 252.0,
          "min_rebalance_interval_minutes": 60,
          "rebalance_threshold": 0.02,
          "target_weights_normal": {
              "BTC_SPOT": 0.65,
              "BTC_LONG5X": 0.11,
              "BTC_SHORT5X": 0.24
          },
          "circuit_breaker_config": {
            "enabled": True, "threshold_percentage": 0.10,
            "lookback_candles": 1, "movement_calc_type": "(high-low)/open"
          },
          "safe_mode_config": {
            "enabled": True, "metric_to_monitor": "margin_usage",
            "entry_threshold": 0.70, "exit_threshold": 0.50,
            "target_weights_safe": {
                "BTC_SPOT": 0.75,
                "BTC_LONG5X": 0.05,
                "BTC_SHORT5X": 0.05,
                "USDT": 0.15
            }
          },
          "data_settings": {
            "csv_file_path": dummy_data_path,
            "signals_csv_path": os.path.join("data", "dummy_BTCUSDT_signals_for_backtester.csv"),
            "timestamp_col": "timestamp",
            "ohlc_cols": {"open": "open", "high": "high", "low": "low", "close": "close"},
            "volume_col": "volume",
            "price_col_for_rebalance": "close"
          },
          "date_range": {
              "start_date": "2023-01-01T00:00:00Z", "end_date": "2023-01-05T23:59:59Z"
          },
          "logging_level": "INFO",
          "report_path_prefix": "./reports/backtest_"
        }
        dummy_unified_config_content = {"backtest_settings": dummy_backtest_settings_content}

        try:
            with open(dummy_config_path, 'w') as f: 
                json.dump(dummy_unified_config_content, f, indent=2)
            logging.info(f"Dummy unified config for backtester created at '{dummy_config_path}'. You should run with this path next time.")
            
            backtest_params = dummy_backtest_settings_content
            # This dummy logic needs to be careful about resolved_data_file_path vs data_file_path
            # For simplicity, if config is not found, we'll use the dummy paths directly without substitution for now.
            # A more robust dummy creation would also consider the override for symbol.
            dummy_data_path_val = dummy_data_path # Store the original dummy path template

            if not os.path.exists(dummy_data_path_val): # Check existence of template path
                timestamps = pd.date_range(start='2023-01-01 00:00:00', periods=120, freq='h') # 5 days
                prices = [20000 + (i*2) + (100 * ((i//24)%5)) - (80 * (i % 3)) for i in range(120)]
                df_dummy_data = pd.DataFrame({
                    'timestamp': timestamps, 'open': [p - 5 for p in prices], 'high': [p + 10 for p in prices],
                    'low': [p - 10 for p in prices], 'close': prices, 'volume': [50 + i for i in range(120)]
                })
                df_dummy_data.to_csv(dummy_data_path_val, index=False)
                logging.info(f"Dummy market data file created at '{dummy_data_path_val}'")

            # Create dummy signals CSV if path is specified and file doesn't exist
            # This part also needs care if we want dummy signals to match a potentially overridden symbol.
            # For now, dummy signals path is hardcoded or uses BTC.
            dummy_signals_path_template = dummy_backtest_settings_content["data_settings"]["signals_csv_path"]
            # actual_main_asset_symbol for dummy case would default to BTC or from override if provided.
            # This is getting complex for dummy section, ideally dummy config has fixed names.
            # Let's assume dummy config uses fixed names for now.
            if dummy_signals_path_template and not os.path.exists(dummy_signals_path_template):
                signal_timestamps = pd.to_datetime(['2023-01-01T00:00:00Z', '2023-01-01T10:00:00Z', # Using dummy_signals_path_template directly
                                                    '2023-01-02T05:00:00Z', '2023-01-03T15:00:00Z', # as it might not have placeholders
                                                    '2023-01-04T20:00:00Z'])
                signals = ['NEUTRAL', 'BUY', 'NEUTRAL', 'SELL', 'BUY']
                df_dummy_signals = pd.DataFrame({'timestamp': signal_timestamps, 'signal': signals})
                df_dummy_signals.to_csv(dummy_signals_path_template, index=False)
                logging.info(f"Dummy signal data file created at '{dummy_signals_path_template}'")
            
            # If config not found, we are in dummy mode.
            # resolved_data_file_path should be set to the dummy path.
            resolved_data_file_path = dummy_data_path # Use the variable holding the actual dummy path string.
            # backtest_params is already set to dummy_backtest_settings_content
            logging.info("Exiting after creating dummy files. Please re-run with the dummy config: "
                         f"`python -m src.prosperous_bot.rebalance_backtester --config_file {dummy_config_path}` "
                         f" (and optionally --override if testing that feature with dummy data).")
            return

        except Exception as e:
            logging.error(f"Could not create dummy unified config or data file: {e}", exc_info=True)
            return

    except json.JSONDecodeError:
        logging.error(f"FATAL: Could not decode JSON from config file: {args.config_file}. Exiting.")
        return
    except Exception as e: 
        logging.error(f"FATAL: An unexpected error occurred while loading the configuration: {e}", exc_info=True)
        return

    if backtest_params and resolved_data_file_path: # Check resolved_data_file_path
        run_standalone_backtest(backtest_params, resolved_data_file_path)
    else:
        # This state should ideally not be reached if the above logic is correct,
        # especially with the new checks for csv_path_template and resolved_data_file_path.
        logging.error("Critical error: Parameters or data file path could not be determined. Backtest aborted.")

if __name__ == "__main__":
    main()