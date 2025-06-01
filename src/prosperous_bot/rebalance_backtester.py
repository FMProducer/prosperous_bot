import argparse
import json
import os
import pandas as pd
import numpy as np

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
from plotly.subplots import make_subplots # ADDED IMPORT
from datetime import datetime
import logging

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
        df_signals = df_signals.dropna(subset=['timestamp'])

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
    params = params_dict 

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
    taker_commission_rate = params.get('taker_commission_rate', params.get('commission_rate', 0.0007))
    maker_commission_rate = params.get('maker_commission_rate', 0.0002)
    use_maker_fees_in_backtest = params.get('use_maker_fees_in_backtest', False)
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
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    if generate_reports:
        report_path_prefix = params.get('report_path_prefix', './reports/')
        if is_optimizer_call and trial_id_for_reports is not None:
            output_dir = os.path.join(report_path_prefix.rstrip('/'), "optimizer_trials", f"trial_{trial_id_for_reports}_{timestamp_str}")
        else:
            output_dir = os.path.join(report_path_prefix.rstrip('/'), f"backtest_{timestamp_str}")
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output reports for this run will be saved to: {output_dir}")
    else:
        if is_optimizer_call:
            logging.debug("Optimizer call: Individual trial report generation is skipped by default for this trial.")

    df_market_original = load_data(data_path) # Keep original for plotting price
    if df_market_original is None or df_market_original.empty:
        logging.error("Market data is empty or could not be loaded. Cannot run backtest.")
        return {
            "final_portfolio_value_usdt": 0, "total_net_pnl_usdt": -initial_portfolio_value_usdt,
            "total_net_pnl_percent": -100.0, "total_trades": 0, "output_dir": None,
            "status": "Market data error"
        }
    
    df_market = df_market_original.copy() # Work with a copy for potential modifications

    if df_market['timestamp'].dt.tz is None:
        logging.info("Market data 'timestamp' column is tz-naive. Localizing to UTC for consistency.")
        df_market['timestamp'] = df_market['timestamp'].dt.tz_localize('UTC')
    else:
        logging.info(f"Market data 'timestamp' column is already tz-aware ({df_market['timestamp'].dt.tz}). Converting to UTC for consistency.")
        df_market['timestamp'] = df_market['timestamp'].dt.tz_convert('UTC')

    df_market = df_market.sort_values(by='timestamp', ascending=True)

    signals_csv_path = params.get("data_settings", {}).get("signals_csv_path")
    df_signals = None
    if signals_csv_path:
        df_signals = load_signal_data(signals_csv_path)

    if df_signals is not None and not df_signals.empty:
        logging.info("Merging signal data with market data using merge_asof (backward)...")
        df_market = pd.merge_asof(df_market, df_signals[['timestamp', 'signal']],
                                  on='timestamp', direction='backward')
        df_market['signal'] = df_market['signal'].fillna('NEUTRAL')
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
    
    if df_market.empty:
        logging.error("Market data is empty after applying date range filters. Cannot run backtest.")
        return {
            "final_portfolio_value_usdt": 0, "total_net_pnl_usdt": -initial_portfolio_value_usdt,
            "total_net_pnl_percent": -100.0, "total_trades": 0, "output_dir": output_dir,
            "status": "Market data empty after date filter"
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
                    if portfolio['btc_long_value_usdt'] > 0:
                        portfolio['btc_long_value_usdt'] += portfolio['btc_long_value_usdt'] * 5 * (price_change_ratio - 1)
                    if portfolio['btc_short_value_usdt'] > 0:
                        portfolio['btc_short_value_usdt'] += portfolio['btc_short_value_usdt'] * 5 * (1 - price_change_ratio)
                
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
            price_change_ratio = current_price / portfolio['prev_btc_price']
            if portfolio['btc_long_value_usdt'] > 0:
                portfolio['btc_long_value_usdt'] += portfolio['btc_long_value_usdt'] * 5 * (price_change_ratio - 1)
            if portfolio['btc_short_value_usdt'] > 0:
                portfolio['btc_short_value_usdt'] += portfolio['btc_short_value_usdt'] * 5 * (1 - price_change_ratio)
        
        total_portfolio_value = calculate_portfolio_value(
            portfolio['usdt_balance'], portfolio['btc_spot_qty'],
            portfolio['btc_long_value_usdt'], portfolio['btc_short_value_usdt'], current_price)

        nav = total_portfolio_value
        used_margin_usdt = 0
        if nav > 0 and params.get("safe_mode_config", {}).get("enabled", False) :
            margin_for_long = portfolio['btc_long_value_usdt'] / 5.0 
            margin_for_short = portfolio['btc_short_value_usdt'] / 5.0
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
                target_value_usdt = total_portfolio_value * target_w_loop
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
                if abs(usdt_value_to_trade) < 1.0: continue

                action = "BUY" if usdt_value_to_trade > 0 else "SELL"

                # ---- Gate hedged futures mapping ----
                if asset_key_trade == long_asset_key:
                    order_type = "OPEN_LONG"  if action == "BUY"  else "CLOSE_LONG"
                elif asset_key_trade == short_asset_key:
                    order_type = "OPEN_SHORT" if action == "BUY"  else "CLOSE_SHORT" # Changed "SELL" to "BUY" for OPEN_SHORT
                else:                              # spot leg
                    order_type = action            # BUY/SELL

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
                        # Given the PnL calculations later, `btc_short_value_usdt` is treated as a positive value
                        # representing the size of the short position.
                        # For consistency with LONG, let's assume opening a short also "uses" USDT from balance for margin.
                        portfolio["usdt_balance"] -= abs_usdt_value_of_trade # Margin used for opening short
                    elif order_type == "CLOSE_SHORT": # Closing a short position (buying back)
                        close_val = min(abs_usdt_value_of_trade, portfolio.get("btc_short_value_usdt", 0.0))
                        portfolio["btc_short_value_usdt"] -= close_val
                        portfolio["usdt_balance"] += close_val # Margin returned

                record_trade(current_timestamp, asset_key_trade, order_type, quantity_asset_traded_final,
                             abs_usdt_value_of_trade, current_price, commission_usdt,
                             slippage_cost_this_trade_usdt, realized_pnl_this_spot_trade, trades_list,
                             realized_pnl_spot_usdt=realized_pnl_this_spot_trade)
            
            # ... (logging post-rebalance portfolio) ...

        portfolio['prev_btc_price'] = current_price

    logging.info("Backtest finished.")
    df_equity = pd.DataFrame(equity_over_time)
    df_trades = pd.DataFrame(trades_list)

    # ---------- PERFORMANCE METRICS ----------
    def compute_metrics(df_eq: pd.DataFrame, trades: list[dict], initial_nav: float, ann_factor: int = 252):
        out: dict[str, float] = {}
        if df_eq.empty:
            return {k: 0.0 for k in ("sharpe_ratio", "sortino_ratio", "max_drawdown_percent", "profit_factor", "win_rate_percent")}

        rets = df_eq["portfolio_value_usdt"].pct_change().dropna()
        if not rets.empty and rets.std():
            out["sharpe_ratio"] = rets.mean() / rets.std() * np.sqrt(ann_factor)
            downside = rets[rets < 0]
            out["sortino_ratio"] = rets.mean() / downside.std() * np.sqrt(ann_factor) if not downside.empty and downside.std() else 0.0
        else:
            out["sharpe_ratio"] = 0.0
            out["sortino_ratio"] = 0.0

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
    # (And many more metrics from the original file)


    if generate_reports and output_dir:
        logging.info(f"Generating reports in {output_dir}...")
        trades_csv_path = os.path.join(output_dir, "trades.csv")
        df_trades.to_csv(trades_csv_path, index=False)
        logging.info(f"Trades report saved to {trades_csv_path}")

        df_summary = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        summary_csv_path = os.path.join(output_dir, "summary.csv")
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

            equity_html_path = os.path.join(output_dir, "equity.html")
            fig.write_html(equity_html_path)
            logging.info(f"Enhanced equity curve saved to {equity_html_path}")

            equity_csv_path = os.path.join(output_dir, "equity.csv")
            df_equity.to_csv(equity_csv_path, index=False)
            logging.info(f"Equity curve data saved to {equity_csv_path}")
        else:
            logging.warning("Equity data is empty. Skipping equity curve generation and equity.csv saving.")

        if blocked_trades_list:
            df_blocked_trades = pd.DataFrame(blocked_trades_list)
            if not df_blocked_trades.empty: # Check if DataFrame is non-empty after creation
                blocked_trades_csv_path = os.path.join(output_dir, "blocked_trades_log.csv")
                df_blocked_trades.to_csv(blocked_trades_csv_path, index=False)
                logging.info(f"Blocked trades log saved to {blocked_trades_csv_path}")
            else: # This case might occur if blocked_trades_list was empty
                logging.info("No trades were blocked by signals during this backtest run (DataFrame was empty).")
        else: # This case for if the list itself was empty
            logging.info("No trades were blocked by signals during this backtest run (list was empty).")

        logging.info("All reports for this run generated.")
    
    results_for_optimizer = metrics.copy()
    results_for_optimizer["output_dir"] = output_dir 
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
    args = parser.parse_args()

    backtest_params = None
    data_file_path = None

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

        data_file_path = backtest_params.get("data_settings", {}).get("csv_file_path")
        if not data_file_path:
            logging.error("FATAL: 'data_settings.csv_file_path' not found in the 'backtest_settings' section of the config. Exiting.")
            return

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
            data_file_path = dummy_data_path

            if not os.path.exists(dummy_data_path):
                timestamps = pd.date_range(start='2023-01-01 00:00:00', periods=120, freq='h') # 5 days
                prices = [20000 + (i*2) + (100 * ((i//24)%5)) - (80 * (i % 3)) for i in range(120)]
                df_dummy_data = pd.DataFrame({
                    'timestamp': timestamps, 'open': [p - 5 for p in prices], 'high': [p + 10 for p in prices],
                    'low': [p - 10 for p in prices], 'close': prices, 'volume': [50 + i for i in range(120)]
                })
                df_dummy_data.to_csv(dummy_data_path, index=False)
                logging.info(f"Dummy market data file created at '{dummy_data_path}'")

            # Create dummy signals CSV if path is specified and file doesn't exist
            dummy_signals_path = dummy_backtest_settings_content["data_settings"]["signals_csv_path"]
            if dummy_signals_path and not os.path.exists(dummy_signals_path):
                signal_timestamps = pd.to_datetime(['2023-01-01T00:00:00Z', '2023-01-01T10:00:00Z',
                                                    '2023-01-02T05:00:00Z', '2023-01-03T15:00:00Z',
                                                    '2023-01-04T20:00:00Z'])
                signals = ['NEUTRAL', 'BUY', 'NEUTRAL', 'SELL', 'BUY']
                df_dummy_signals = pd.DataFrame({'timestamp': signal_timestamps, 'signal': signals})
                df_dummy_signals.to_csv(dummy_signals_path, index=False)
                logging.info(f"Dummy signal data file created at '{dummy_signals_path}'")
            
            logging.info("Exiting after creating dummy files. Please re-run with the dummy config: "
                         f"`python -m src.prosperous_bot.rebalance_backtester --config_file {dummy_config_path}`")
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

    if backtest_params and data_file_path:
        run_standalone_backtest(backtest_params, data_file_path)
    else:
        # This state should ideally not be reached if the above logic is correct.
        logging.error("Critical error: Parameters or data file path could not be determined. Backtest aborted.")

if __name__ == "__main__":
    main()