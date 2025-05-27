import argparse
import json
import os
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import logging

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
        "timestamp_open": timestamp, # For now, all trades are "opened" and "closed" instantly in rebalancing
        "timestamp_close": timestamp, # For PnL calc, this is the time of realization
        "asset_type": asset_type,
        "action": action, # BUY or SELL
        "quantity_asset": quantity_asset, 
        "quantity_quote": quantity_quote, # Gross value of asset moved, at market_price
        "entry_price": market_price, # Market price at time of trade
        "exit_price": market_price, # For rebalancing trades, entry and exit are effectively at the same time
        "commission_quote": commission_usdt,
        "slippage_quote": slippage_usdt, # Total slippage cost for this trade event
        "pnl_gross_quote": realized_pnl_spot_usdt, # PnL before commission for this trade (primarily for SPOT)
        "pnl_net_quote": realized_pnl_spot_usdt - commission_usdt, # PnL after commission (slippage already in execution)
        # "duration_hours": 0 # Duration can be complex for rebalancing trades.
    }
    trades_list.append(trade)
    logging.info(
        f"  TRADE: {action} {quantity_asset:.6f} {asset_type} @ MktPx {market_price:.2f}, "
        f"Val: {quantity_quote:.2f}, Comm: {commission_usdt:.2f}, SlipCost: {slippage_usdt:.2f}, "
        f"NetPnL_Trade: {(realized_pnl_spot_usdt - commission_usdt):.2f}"
    )

def run_backtest(params_dict, data_path, is_optimizer_call=True, trial_id_for_reports=None):
    """
    Main backtesting logic for Part 1.
    Accepts a parameters dictionary and the data path.
    If is_optimizer_call is True, report generation can be conditional.
    trial_id_for_reports can be used to create unique report subfolders for optimizer trials if needed.
    """
    params = params_dict # Parameters are now passed as a dictionary

    target_weights = params['target_weights']
    rebalance_threshold = params['rebalance_threshold']
    initial_portfolio_value_usdt = params['initial_portfolio_value_usdt']
    # Old commission_rate might still be in params, prioritize new ones
    taker_commission_rate = params.get('taker_commission_rate', params.get('commission_rate', 0.0007)) # Default to a typical taker fee
    maker_commission_rate = params.get('maker_commission_rate', 0.0002) # Default to a typical maker fee
    use_maker_fees_in_backtest = params.get('use_maker_fees_in_backtest', False)
    
    slippage_percentage = params.get('slippage_percentage', 0.0005) # Keep existing default if not specified
    
    # New risk control parameters
    circuit_breaker_threshold_percent = params.get('circuit_breaker_threshold_percent', 0.10) # Default 10%
    margin_usage_safe_mode_enter_threshold = params.get('margin_usage_safe_mode_enter_threshold', 0.70) # Default 70%
    margin_usage_safe_mode_exit_threshold = params.get('margin_usage_safe_mode_exit_threshold', 0.50) # Default 50%
    safe_mode_target_weights = params.get('safe_mode_target_weights', target_weights) # Default to normal weights if not specified
    min_rebalance_interval_minutes = params.get('min_rebalance_interval_minutes', 0)


    # Determine actual commission rate to use for this backtest run
    current_commission_rate = maker_commission_rate if use_maker_fees_in_backtest else taker_commission_rate
    
    # Determine if reports should be generated for this specific run
    # For optimizer calls, this might be False to save time/space, unless explicitly requested for debugging.
    # For standalone runs (is_optimizer_call=False), reports are usually desired.
    generate_reports = not is_optimizer_call or params.get('generate_reports_for_optimizer_trial', False)
    output_dir = None

    if generate_reports:
        report_path_prefix = params.get('report_path_prefix', './reports/backtest_')
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        if is_optimizer_call and trial_id_for_reports is not None:
            # Create a specific subdirectory for this trial's reports if requested
            output_dir = os.path.join(report_path_prefix + "optimizer_trials", f"trial_{trial_id_for_reports}_{timestamp_str}")
        else:
            output_dir = f"{report_path_prefix}{timestamp_str}"
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output reports for this run will be saved to: {output_dir}")
    else:
        if is_optimizer_call:
            logging.debug("Optimizer call: Individual trial report generation is skipped by default for this trial.")

    df_market = load_data(data_path)
    if df_market is None or df_market.empty:
        logging.error("Market data is empty or could not be loaded. Cannot run backtest.")
        # Return a structure indicating failure, which optimizer can use
        return {
            "final_portfolio_value_usdt": 0,
            "total_net_pnl_usdt": -initial_portfolio_value_usdt,
            "total_net_pnl_percent": -100.0,
            "total_trades": 0,
            "output_dir": None,
            "status": "Market data error"
        }
    
    trades_list = []
    equity_over_time = [] # To store {'timestamp': ts, 'portfolio_value_usdt': val}

    # Portfolio State for Part 1:
    # 'btc_long_value_usdt' and 'btc_short_value_usdt' represent the USDT capital
    # allocated to these strategies, including their P&L. They are not contract counts.
    portfolio = {
        'usdt_balance': initial_portfolio_value_usdt,
        'btc_spot_qty': 0.0,
        'btc_spot_lots': [], # For FIFO PnL calculation: list of {'qty': float, 'price_per_unit': float}
        'btc_long_value_usdt': 0.0,
        'btc_short_value_usdt': 0.0,
        'prev_btc_price': None,
        'total_commissions_usdt': 0.0,
        'total_slippage_usdt': 0.0,
        # New state variables
        'current_operational_mode': 'NORMAL_MODE', # NORMAL_MODE or SAFE_MODE
        'num_circuit_breaker_triggers': 0,
        'num_safe_mode_entries': 0,
        'time_steps_in_safe_mode': 0,
        'last_rebalance_attempt_timestamp': None, # For min_rebalance_interval_minutes
    }

    logging.info(f"Starting backtest with initial portfolio: {portfolio['usdt_balance']:.2f} USDT.")
    logging.info(f"Target Weights: {target_weights}")
    logging.info(f"Rebalance Threshold: {rebalance_threshold*100:.2f}%")

    # Set initial prev_btc_price for the first iteration's P&L calculation.
    # If df_market is not empty, this will be the first closing price.
    portfolio['prev_btc_price'] = df_market['close'].iloc[0]

    for index, row in df_market.iterrows():
        current_timestamp = row['timestamp']
        current_btc_price = row['close']
        current_open_price = row['open']
        current_high_price = row['high']
        current_low_price = row['low']

        # --- 1. Circuit Breaker Check ---
        if current_open_price > 0:
            candle_movement_percent = (current_high_price - current_low_price) / current_open_price
            if candle_movement_percent > circuit_breaker_threshold_percent:
                portfolio['num_circuit_breaker_triggers'] += 1
                logging.warning(f"CIRCUIT BREAKER TRIGGERED at {current_timestamp}: "
                                f"Movement {candle_movement_percent*100:.2f}% > threshold {circuit_breaker_threshold_percent*100:.2f}%. "
                                f"Skipping rebalancing for this candle.")
                # Still need to update P&L and prev_btc_price, but skip rebalancing and mode checks for this candle
                # P&L Update for Leveraged Positions (copied from below, simplified for skip)
                if portfolio['prev_btc_price'] is not None and portfolio['prev_btc_price'] > 0:
                    price_change_ratio = current_btc_price / portfolio['prev_btc_price']
                    if portfolio['btc_long_value_usdt'] > 0:
                        portfolio['btc_long_value_usdt'] += portfolio['btc_long_value_usdt'] * 5 * (price_change_ratio - 1)
                    if portfolio['btc_short_value_usdt'] > 0:
                        portfolio['btc_short_value_usdt'] += portfolio['btc_short_value_usdt'] * 5 * (1 - price_change_ratio)
                
                # Update total portfolio value for equity curve
                total_portfolio_value_cb = calculate_portfolio_value(
                    portfolio['usdt_balance'], portfolio['btc_spot_qty'],
                    portfolio['btc_long_value_usdt'], portfolio['btc_short_value_usdt'],
                    current_btc_price
                )
                equity_over_time.append({'timestamp': current_timestamp, 'portfolio_value_usdt': total_portfolio_value_cb})
                if total_portfolio_value_cb <= 0:
                    logging.warning(f"Portfolio value is {total_portfolio_value_cb:.2f} at {current_timestamp} after CB. Stopping backtest.")
                    # Return structure indicating failure
                    # (Copying the structure from the main portfolio wipeout condition)
                    final_val_cb = total_portfolio_value_cb if total_portfolio_value_cb is not None else 0
                    pnl_usdt_cb = final_val_cb - initial_portfolio_value_usdt
                    pnl_pct_cb = (pnl_usdt_cb / initial_portfolio_value_usdt) * 100 if initial_portfolio_value_usdt != 0 else 0
                    return {
                        "final_portfolio_value_usdt": final_val_cb, "total_net_pnl_usdt": pnl_usdt_cb,
                        "total_net_pnl_percent": pnl_pct_cb, "total_trades": len(trades_list),
                        "output_dir": output_dir, "status": "Portfolio wiped out post-CB",
                        **portfolio # Include all portfolio state for debugging
                    }

                portfolio['prev_btc_price'] = current_btc_price
                if portfolio['current_operational_mode'] == 'SAFE_MODE': # Count time in safe mode even if CB triggered
                    portfolio['time_steps_in_safe_mode'] +=1
                continue # Skip to the next candle
        else: # Handle open_price == 0 if it can occur
            logging.warning(f"Candle open price is 0 at {current_timestamp}, cannot calculate movement for circuit breaker.")


        # --- 2. P&L Update for Leveraged Positions (Part 1 Simplification) ---
        # P&L is calculated based on the change in BTC price from the previous bar,
        # applied to the current value of the leveraged segments.
        if portfolio['prev_btc_price'] is not None and portfolio['prev_btc_price'] > 0:
            price_change_ratio = current_btc_price / portfolio['prev_btc_price'] # e.g., 1.01 for 1% rise
            
            if portfolio['btc_long_value_usdt'] > 0:
                # Change in value = Current Value * Leverage * (Price Ratio - 1)
                pnl_long = portfolio['btc_long_value_usdt'] * 5 * (price_change_ratio - 1)
                portfolio['btc_long_value_usdt'] += pnl_long
            
            if portfolio['btc_short_value_usdt'] > 0:
                # Change in value = Current Value * Leverage * (1 - Price Ratio)
                pnl_short = portfolio['btc_short_value_usdt'] * 5 * (1 - price_change_ratio)
                portfolio['btc_short_value_usdt'] += pnl_short
        
        # --- Portfolio Valuation ---
        total_portfolio_value = calculate_portfolio_value(
            portfolio['usdt_balance'], portfolio['btc_spot_qty'],
            portfolio['btc_long_value_usdt'], portfolio['btc_short_value_usdt'],
            current_btc_price
        )
        # equity_over_time is appended after mode checks and potential rebalancing for the current bar

        # --- 3. Margin Usage & Safe Mode Check ---
        nav = total_portfolio_value # Net Asset Value
        used_margin_usdt = 0
        if nav > 0: # Proceed only if portfolio has value
            # Margin for leveraged positions (assuming 5x leverage for both as per current model)
            # This is a simplified view; real margin depends on exchange rules, entry prices, etc.
            margin_for_long = portfolio['btc_long_value_usdt'] / 5.0 
            margin_for_short = portfolio['btc_short_value_usdt'] / 5.0
            used_margin_usdt = margin_for_long + margin_for_short
            margin_usage_ratio = used_margin_usdt / nav if nav > 0 else 0.0
        else:
            margin_usage_ratio = 0.0 # No value, no margin usage meaningful

        active_target_weights = None # Will be set based on mode
        previous_mode = portfolio['current_operational_mode']

        if portfolio['current_operational_mode'] == 'NORMAL_MODE':
            if margin_usage_ratio > margin_usage_safe_mode_enter_threshold:
                portfolio['current_operational_mode'] = 'SAFE_MODE'
                portfolio['num_safe_mode_entries'] += 1
                logging.info(f"ENTERING SAFE MODE at {current_timestamp} due to margin usage: {margin_usage_ratio*100:.2f}% "
                             f"(Threshold: {margin_usage_safe_mode_enter_threshold*100:.2f}%)")
            # active_target_weights will be set below
        elif portfolio['current_operational_mode'] == 'SAFE_MODE':
            if margin_usage_ratio < margin_usage_safe_mode_exit_threshold:
                portfolio['current_operational_mode'] = 'NORMAL_MODE'
                logging.info(f"EXITING SAFE MODE at {current_timestamp}, margin usage: {margin_usage_ratio*100:.2f}% "
                             f"(Threshold: {margin_usage_safe_mode_exit_threshold*100:.2f}%)")
            # active_target_weights will be set below
        
        if portfolio['current_operational_mode'] == 'SAFE_MODE':
            active_target_weights = safe_mode_target_weights
            portfolio['time_steps_in_safe_mode'] += 1
        else: # NORMAL_MODE
            active_target_weights = target_weights # Original target weights

        # If mode changed, it might trigger an immediate rebalance to new targets
        mode_changed_this_step = previous_mode != portfolio['current_operational_mode']

        # Append equity AFTER potential mode changes but BEFORE rebalancing for this bar
        equity_over_time.append({'timestamp': current_timestamp, 'portfolio_value_usdt': total_portfolio_value})

        if total_portfolio_value <= 0: # Check again after P&L and before rebalancing
            logging.warning(f"Portfolio value is {total_portfolio_value:.2f} at {current_timestamp} before rebalance. Stopping backtest.")
            # Ensure equity_over_time has the last value if it's not the first bar
            if not equity_over_time or equity_over_time[-1]['timestamp'] != current_timestamp:
                 equity_over_time.append({'timestamp': current_timestamp, 'portfolio_value_usdt': total_portfolio_value})
            # Return structure indicating failure/poor result for optimizer
            final_val = total_portfolio_value if total_portfolio_value is not None else 0
            pnl_usdt = final_val - initial_portfolio_value_usdt
            pnl_pct = (pnl_usdt / initial_portfolio_value_usdt) * 100 if initial_portfolio_value_usdt != 0 else 0
            return {
                "final_portfolio_value_usdt": final_val,
                "total_net_pnl_usdt": pnl_usdt,
                "total_net_pnl_percent": pnl_pct,
                "total_trades": len(trades_list),
                "output_dir": output_dir, # output_dir might be None
                "status": "Portfolio wiped out"
            }

        # --- Min Rebalance Interval Check ---
        # This check must happen BEFORE the decision to rebalance (needs_rebalance=True)
        # and BEFORE last_rebalance_attempt_timestamp is updated for the current check.
        can_check_rebalance_now = True
        if min_rebalance_interval_minutes > 0 and portfolio['last_rebalance_attempt_timestamp'] is not None:
            time_since_last_attempt = current_timestamp - portfolio['last_rebalance_attempt_timestamp']
            if time_since_last_attempt < pd.Timedelta(minutes=min_rebalance_interval_minutes):
                logging.debug(f"Rebalance check skipped at {current_timestamp} due to min_rebalance_interval_minutes. "
                             f"Time since last attempt: {time_since_last_attempt}. Required: {min_rebalance_interval_minutes} min.")
                can_check_rebalance_now = False
        
        # Always update the last attempt timestamp for the current candle *before* deciding if we need to rebalance.
        # This means the interval starts from the last time we *could* have rebalanced.
        portfolio['last_rebalance_attempt_timestamp'] = current_timestamp

        needs_rebalance = False # Default to false for this candle
        if can_check_rebalance_now:
            # --- Rebalancing Check ---
            current_weights = {
                "USDT": portfolio['usdt_balance'] / total_portfolio_value if total_portfolio_value else 1,
                "BTC_SPOT": (portfolio['btc_spot_qty'] * current_btc_price) / total_portfolio_value if total_portfolio_value else 0,
            "BTC_LONG5X": portfolio['btc_long_value_usdt'] / total_portfolio_value if total_portfolio_value else 0,
            "BTC_SHORT5X": portfolio['btc_short_value_usdt'] / total_portfolio_value if total_portfolio_value else 0,
        }
        
        needs_rebalance = False
        # First bar always rebalances to establish initial positions from 100% USDT.
        # Also, if mode changed, force a rebalance check against the new active_target_weights.
        if index == 0 or mode_changed_this_step:
            needs_rebalance = True
            if index == 0:
                logging.info(f"Initial rebalance triggered at {current_timestamp} (Price: {current_btc_price:.2f}) to establish target weights: {active_target_weights}.")
            if mode_changed_this_step:
                logging.info(f"Mode changed to {portfolio['current_operational_mode']}. Forcing rebalance check against new weights: {active_target_weights}.")
        else:
            # Normal rebalance check based on threshold against current active_target_weights
            for asset_key, target_w in active_target_weights.items(): 
                current_w = current_weights.get(asset_key, 0)
                if abs(current_w - target_w) > rebalance_threshold:
                    needs_rebalance = True # Set to true if threshold met
                    logging.info(f"Rebalance threshold triggered at {current_timestamp} (Price: {current_btc_price:.2f}). Asset {asset_key} current weight {current_w:.4f}, target {target_w:.4f} (Mode: {portfolio['current_operational_mode']})")
                    break
        
        if needs_rebalance: # This 'needs_rebalance' is now conditional on can_check_rebalance_now
            logging.info(f"Rebalancing portfolio (Mode: {portfolio['current_operational_mode']}). Total Value: {total_portfolio_value:.2f} USDT. Current Price: {current_btc_price:.2f}")
            logging.debug(f"  Current Weights before rebalance: {current_weights}")
            logging.debug(f"  Target Weights for rebalance: {active_target_weights}")

            # Store desired changes before applying them to avoid using post-trade values mid-rebalance
            adjustments = {} # Stores the DELTA in USDT value for each asset class

            for asset_key, target_w in active_target_weights.items(): # Use active_target_weights
                target_value_usdt = total_portfolio_value * target_w
                current_value_usdt = 0
                if asset_key == "BTC_SPOT":
                    current_value_usdt = portfolio['btc_spot_qty'] * current_btc_price
                elif asset_key == "BTC_LONG5X":
                    current_value_usdt = portfolio['btc_long_value_usdt']
                elif asset_key == "BTC_SHORT5X":
                    current_value_usdt = portfolio['btc_short_value_usdt']
                
                adjustments[asset_key] = target_value_usdt - current_value_usdt

            # Execute trades based on adjustments
            for asset_key, usdt_value_to_trade in adjustments.items():
                if abs(usdt_value_to_trade) < 1.0: # Minimum trade value (e.g., 1 USDT)
                    continue

                action = "BUY" if usdt_value_to_trade > 0 else "SELL"
                abs_usdt_value_of_trade = abs(usdt_value_to_trade) 

                # Apply Commission: Subtracted from USDT balance for all trades
                # Use current_commission_rate determined at the start of run_backtest
                commission_usdt = abs_usdt_value_of_trade * current_commission_rate 
                portfolio['usdt_balance'] -= commission_usdt
                portfolio['total_commissions_usdt'] += commission_usdt
                
                quantity_asset_traded_final = 0
                slippage_cost_this_trade_usdt = 0
                realized_pnl_this_spot_trade = 0.0 # For SPOT PnL

                if asset_key == "BTC_SPOT":
                    if action == "BUY":
                        price_after_slippage = current_btc_price * (1 + slippage_percentage)
                        slippage_cost_this_trade_usdt = abs_usdt_value_of_trade * slippage_percentage # Cost of slippage
                        portfolio['total_slippage_usdt'] += slippage_cost_this_trade_usdt
                        
                        # Amount of BTC we can buy with the allocated USDT value (abs_usdt_value_of_trade is the value at market price)
                        quantity_asset_traded_final = abs_usdt_value_of_trade / price_after_slippage
                        
                        portfolio['btc_spot_qty'] += quantity_asset_traded_final
                        portfolio['usdt_balance'] -= abs_usdt_value_of_trade # Cost of BTC (commission already deducted from USDT)
                        
                        # Add to lots for FIFO PnL tracking
                        # Cost per unit for this lot includes slippage effect.
                        # The 'value' of BTC bought (abs_usdt_value_of_trade) at market price, 
                        # but we paid abs_usdt_value_of_trade * (1+slippage_percentage) effectively.
                        # So, effective price per unit is current_btc_price * (1+slippage_percentage)
                        portfolio['btc_spot_lots'].append({'qty': quantity_asset_traded_final, 
                                                           'price_per_unit': price_after_slippage})
                    else: # SELL SPOT
                        price_after_slippage = current_btc_price * (1 - slippage_percentage)
                        slippage_value_impact_usdt = abs_usdt_value_of_trade * slippage_percentage # Value lost to slippage
                        portfolio['total_slippage_usdt'] += slippage_value_impact_usdt

                        # We want to sell 'abs_usdt_value_of_trade' worth of BTC (at current market price)
                        btc_to_sell_nominal = abs_usdt_value_of_trade / current_btc_price
                        actual_btc_sold = min(btc_to_sell_nominal, portfolio['btc_spot_qty'])
                        quantity_asset_traded_final = actual_btc_sold
                        
                        usdt_received_after_slippage_and_commission = actual_btc_sold * price_after_slippage
                        portfolio['btc_spot_qty'] -= actual_btc_sold
                        portfolio['usdt_balance'] += usdt_received_after_slippage_and_commission # Add proceeds
                        
                        # FIFO PnL Calculation for SPOT SELL
                        temp_qty_to_sell = actual_btc_sold
                        realized_pnl_this_spot_trade = 0
                        
                        new_lots = []
                        for lot in portfolio['btc_spot_lots']:
                            if temp_qty_to_sell <= 0:
                                new_lots.append(lot)
                                continue
                            
                            qty_from_lot = min(lot['qty'], temp_qty_to_sell)
                            realized_pnl_this_spot_trade += qty_from_lot * (price_after_slippage - lot['price_per_unit'])
                            
                            lot['qty'] -= qty_from_lot
                            temp_qty_to_sell -= qty_from_lot
                            
                            if lot['qty'] > 1e-9: # Avoid floating point residuals for qty
                                new_lots.append(lot)
                        portfolio['btc_spot_lots'] = new_lots
                        
                        # If we sold less than intended due to insufficient BTC, adjust abs_usdt_value_of_trade for record
                        abs_usdt_value_of_trade = actual_btc_sold * current_btc_price


                elif asset_key == "BTC_LONG5X" or asset_key == "BTC_SHORT5X":
                    slippage_cost_this_trade_usdt = abs_usdt_value_of_trade * slippage_percentage
                    portfolio['usdt_balance'] -= slippage_cost_this_trade_usdt # Slippage as extra cost from USDT
                    portfolio['total_slippage_usdt'] += slippage_cost_this_trade_usdt

                    target_value_var = 'btc_long_value_usdt' if asset_key == "BTC_LONG5X" else 'btc_short_value_usdt'
                    quantity_asset_traded_final = abs_usdt_value_of_trade 

                    if action == "BUY": 
                        portfolio[target_value_var] += abs_usdt_value_of_trade
                        portfolio['usdt_balance'] -= abs_usdt_value_of_trade 
                    else: 
                        actual_decrease_usdt = min(abs_usdt_value_of_trade, portfolio[target_value_var])
                        quantity_asset_traded_final = actual_decrease_usdt
                        
                        portfolio[target_value_var] -= actual_decrease_usdt
                        portfolio['usdt_balance'] += actual_decrease_usdt 
                        abs_usdt_value_of_trade = actual_decrease_usdt 
                
                # PnL for non-spot "trades" in trades.csv is the direct cost (comm+slip) as value change PnL is in equity curve
                pnl_for_this_trade_record = realized_pnl_this_spot_trade if asset_key == "BTC_SPOT" else 0
                
                record_trade(current_timestamp, asset_key, action, quantity_asset_traded_final, 
                             abs_usdt_value_of_trade, current_btc_price, commission_usdt, 
                             slippage_cost_this_trade_usdt, pnl_for_this_trade_record, trades_list,
                             realized_pnl_spot_usdt=realized_pnl_this_spot_trade)
            
            # --- Post-Rebalance Values Update ---
            # Recalculate total portfolio value after rebalancing for accurate equity curve update
            total_portfolio_value_after_rebalance = calculate_portfolio_value(
                portfolio['usdt_balance'], portfolio['btc_spot_qty'],
                portfolio['btc_long_value_usdt'], portfolio['btc_short_value_usdt'],
                current_btc_price
            )
            # Update the last entry in equity_over_time for this timestamp
            if equity_over_time and equity_over_time[-1]['timestamp'] == current_timestamp:
                equity_over_time[-1]['portfolio_value_usdt'] = total_portfolio_value_after_rebalance
            
            logging.info(f"  Post-Rebalance Portfolio: USDT: {portfolio['usdt_balance']:.2f}, SPOT_QTY: {portfolio['btc_spot_qty']:.6f}, LONG_VAL: {portfolio['btc_long_value_usdt']:.2f}, SHORT_VAL: {portfolio['btc_short_value_usdt']:.2f}")
            logging.info(f"  New Total Value after rebalance: {total_portfolio_value_after_rebalance:.2f} USDT")

        portfolio['prev_btc_price'] = current_btc_price # Update for next iteration's P&L

    # --- Reporting ---
    logging.info("Backtest finished.")
    
    df_equity = pd.DataFrame(equity_over_time)
    df_trades = pd.DataFrame(trades_list)

    # Calculate Advanced KPIs
    metrics = {}
    # Ensure timestamp_str is available or generate if not (e.g. if generate_reports was false)
    current_timestamp_str = timestamp_str if generate_reports and 'timestamp_str' in locals() else datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics["run_id"] = f"backtest_{current_timestamp_str}"
    metrics["strategy_name"] = "Rebalancing Strategy" # Placeholder
    metrics["date_range_start"] = df_equity['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M:%S') if not df_equity.empty else "N/A"
    metrics["date_range_end"] = df_equity['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S') if not df_equity.empty else "N/A"
    metrics["initial_portfolio_value_usdt"] = initial_portfolio_value_usdt
    
    final_portfolio_value = df_equity['portfolio_value_usdt'].iloc[-1] if not df_equity.empty else initial_portfolio_value_usdt
    metrics["final_portfolio_value_usdt"] = final_portfolio_value
    
    total_net_pnl_usdt = final_portfolio_value - initial_portfolio_value_usdt
    metrics["total_net_pnl_usdt"] = total_net_pnl_usdt
    metrics["total_net_pnl_percent"] = (total_net_pnl_usdt / initial_portfolio_value_usdt) * 100 if initial_portfolio_value_usdt != 0 else 0

    # Max Drawdown
    if not df_equity.empty:
        roll_max = df_equity['portfolio_value_usdt'].cummax()
        drawdown = df_equity['portfolio_value_usdt'] / roll_max - 1.0
        metrics["max_drawdown_percent"] = drawdown.min() * 100
    else:
        metrics["max_drawdown_percent"] = 0

    # Sharpe & Sortino Ratios (assuming daily returns for annualization, adjust if data is different)
    # For simplicity, let's assume data frequency implies periods for return calculation.
    # If data is hourly, these are hourly returns. Annualization factor needs care.
    # Defaulting risk_free_rate to 0, annualization_factor to sqrt(252) for daily-like returns.
    risk_free_rate_per_period = params.get("risk_free_rate_annual", 0.0) / 252 # Example: daily risk-free
    annualization_factor = params.get("annualization_factor", 252) # sqrt(252) for Sharpe/Sortino std dev
    
    if not df_equity.empty and len(df_equity) > 1:
        df_equity['returns'] = df_equity['portfolio_value_usdt'].pct_change().fillna(0)
        mean_return = df_equity['returns'].mean()
        std_return = df_equity['returns'].std()
        
        if std_return != 0:
            metrics["sharpe_ratio"] = ((mean_return - risk_free_rate_per_period) / std_return) * (annualization_factor**0.5)
        else:
            metrics["sharpe_ratio"] = 0.0
            
        downside_returns = df_equity['returns'][df_equity['returns'] < 0]
        std_downside_return = downside_returns.std(ddof=0) # ddof=0 for population std dev if using all returns as population
        if std_downside_return != 0 and not pd.isna(std_downside_return):
            metrics["sortino_ratio"] = ((mean_return - risk_free_rate_per_period) / std_downside_return) * (annualization_factor**0.5)
        else:
            metrics["sortino_ratio"] = 0.0
    else:
        metrics["sharpe_ratio"] = 0.0
        metrics["sortino_ratio"] = 0.0

    # Trade-based metrics
    metrics["total_trades"] = len(df_trades)
    if not df_trades.empty:
        spot_trades_pnl = df_trades[df_trades['asset_type'] == 'BTC_SPOT']['pnl_net_quote']
        winning_trades_spot = spot_trades_pnl[spot_trades_pnl > 0]
        losing_trades_spot = spot_trades_pnl[spot_trades_pnl < 0]

        metrics["winning_trades"] = len(winning_trades_spot)
        metrics["losing_trades"] = len(losing_trades_spot)
        metrics["win_rate_percent"] = (metrics["winning_trades"] / len(spot_trades_pnl)) * 100 if len(spot_trades_pnl) > 0 else 0
        
        gross_profit = winning_trades_spot.sum()
        gross_loss = abs(losing_trades_spot.sum())
        metrics["profit_factor"] = gross_profit / gross_loss if gross_loss != 0 else float('inf')

        metrics["average_trade_pnl_usdt"] = spot_trades_pnl.mean() if not spot_trades_pnl.empty else 0.0
        metrics["average_winning_trade_usdt"] = winning_trades_spot.mean() if not winning_trades_spot.empty else 0.0
        metrics["average_losing_trade_usdt"] = losing_trades_spot.mean() if not losing_trades_spot.empty else 0.0
        avg_win_val = metrics["average_winning_trade_usdt"]
        avg_loss_val = abs(metrics["average_losing_trade_usdt"])
        metrics["ratio_avg_win_avg_loss"] = avg_win_val / avg_loss_val if avg_loss_val != 0 else float('inf')
    else:
        metrics["winning_trades"] = 0
        metrics["losing_trades"] = 0
        metrics["win_rate_percent"] = 0.0
        metrics["profit_factor"] = 0.0
        metrics["average_trade_pnl_usdt"] = 0.0
        metrics["average_winning_trade_usdt"] = 0.0
        metrics["average_losing_trade_usdt"] = 0.0
        metrics["ratio_avg_win_avg_loss"] = 0.0

    metrics["average_trade_duration_hours"] = "N/A" 
    metrics["longest_trade_duration_hours"] = "N/A" 

    metrics["total_commissions_usdt"] = portfolio['total_commissions_usdt']
    metrics["total_slippage_usdt"] = portfolio['total_slippage_usdt'] 

    # Config parameters for reference in summary (ensure all new ones are included)
    metrics["config_target_weights_normal"] = str(params.get('target_weights', {})) # Original normal weights
    metrics["config_target_weights_safe"] = str(params.get('safe_mode_target_weights', {}))
    metrics["config_rebalance_threshold"] = params.get('rebalance_threshold')
    metrics["config_taker_commission_rate"] = taker_commission_rate
    metrics["config_maker_commission_rate"] = maker_commission_rate
    metrics["config_used_maker_fees"] = use_maker_fees_in_backtest
    metrics["config_slippage_percentage"] = slippage_percentage
    metrics["config_price_source"] = "close" 
    metrics["config_slippage_model"] = "percentage" 
    metrics["config_risk_free_rate_annual"] = params.get("risk_free_rate_annual", 0.0)
    metrics["config_annualization_factor"] = annualization_factor
    metrics["config_circuit_breaker_threshold_percent"] = circuit_breaker_threshold_percent
    metrics["config_margin_usage_safe_mode_enter_threshold"] = margin_usage_safe_mode_enter_threshold
    metrics["config_margin_usage_safe_mode_exit_threshold"] = margin_usage_safe_mode_exit_threshold
    metrics["config_min_rebalance_interval_minutes"] = min_rebalance_interval_minutes
    
    # New operational stats
    metrics["num_circuit_breaker_triggers"] = portfolio['num_circuit_breaker_triggers']
    metrics["num_safe_mode_entries"] = portfolio['num_safe_mode_entries']
    metrics["time_steps_in_safe_mode"] = portfolio['time_steps_in_safe_mode']


    if generate_reports and output_dir:
        logging.info(f"Generating reports in {output_dir}...")
        trades_csv_path = os.path.join(output_dir, "trades.csv")
        df_trades.to_csv(trades_csv_path, index=False)
        logging.info(f"Trades report saved to {trades_csv_path}")

        # Create Summary DataFrame from metrics dictionary
        df_summary = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        summary_csv_path = os.path.join(output_dir, "summary.csv")
        df_summary.to_csv(summary_csv_path, index=False)
        logging.info(f"Summary report saved to {summary_csv_path}")

        if not df_equity.empty:
            fig = go.Figure(data=[go.Scatter(x=df_equity['timestamp'], y=df_equity['portfolio_value_usdt'], mode='lines')])
            fig.update_layout(title='Portfolio Equity Over Time', xaxis_title='Timestamp', yaxis_title='Portfolio Value (USDT)')
            equity_html_path = os.path.join(output_dir, "equity.html")
            fig.write_html(equity_html_path)
            logging.info(f"Equity curve saved to {equity_html_path}")
        else:
            logging.warning("Equity data is empty. Skipping equity curve generation.")
        logging.info("All reports for this run generated.")
    
    # The 'results_for_optimizer' dictionary should directly use the snake_case keys from 'metrics'
    results_for_optimizer = metrics.copy() # Start with all metrics
    results_for_optimizer["output_dir"] = output_dir # Add output_dir if generated
    results_for_optimizer["status"] = "Completed" # Add status
    
    # Ensure the primary metrics used by optimizer are definitely there and correctly named
    results_for_optimizer["final_portfolio_value_usdt"] = metrics["final_portfolio_value_usdt"]
    results_for_optimizer["total_net_pnl_usdt"] = metrics["total_net_pnl_usdt"]
    results_for_optimizer["total_net_pnl_percent"] = metrics["total_net_pnl_percent"]
    results_for_optimizer["total_trades"] = metrics["total_trades"]
    results_for_optimizer["max_drawdown_percent"] = metrics["max_drawdown_percent"]
    results_for_optimizer["sharpe_ratio"] = metrics["sharpe_ratio"]
    results_for_optimizer["sortino_ratio"] = metrics["sortino_ratio"]
    results_for_optimizer["profit_factor"] = metrics["profit_factor"]
    results_for_optimizer["win_rate_percent"] = metrics["win_rate_percent"]
    results_for_optimizer["total_commissions_usdt"] = metrics["total_commissions_usdt"]
    results_for_optimizer["total_slippage_usdt"] = metrics["total_slippage_usdt"]
    results_for_optimizer["num_circuit_breaker_triggers"] = metrics["num_circuit_breaker_triggers"]
    results_for_optimizer["num_safe_mode_entries"] = metrics["num_safe_mode_entries"]
    results_for_optimizer["time_steps_in_safe_mode"] = metrics["time_steps_in_safe_mode"]
    # results_for_optimizer["config_min_rebalance_interval_minutes"] = metrics["config_min_rebalance_interval_minutes"] # Already part of metrics copy
    
    return results_for_optimizer

def run_backtest_from_config_file(params_file_path, data_file_path):
    """
    Wrapper to run a single backtest using file paths for parameters and data.
    This is primarily for CLI execution of the backtester itself.
    """
    logging.info(f"Attempting to load parameters from: {params_file_path}")
    try:
        with open(params_file_path, 'r') as f:
            params_config = json.load(f)
    except FileNotFoundError:
        logging.error(f"FATAL: Parameters file not found at {params_file_path}. Exiting.")
        return None
    except json.JSONDecodeError:
        logging.error(f"FATAL: Could not decode JSON from parameters file: {params_file_path}. Exiting.")
        return None
    
    # Standalone runs should always generate reports.
    return run_backtest(params_config, data_file_path, is_optimizer_call=False)


def main():
    parser = argparse.ArgumentParser(description="Prosperous Bot Rebalance Backtester (CLI)")
    parser.add_argument("params_file", type=str, help="Path to the parameters JSON file (e.g., config/params.json)")
    parser.add_argument("data_file", type=str, help="Path to the input CSV data file (e.g., data/BTCUSDT_1h_data.csv)")
    args = parser.parse_args()

    # --- Dummy File Creation (for standalone testing via CLI) ---
    # Ensures that if someone runs `python rebalance_backtester.py params.json data.csv` and those don't exist,
    # they get created with default values for a quick test run.
    config_dir = os.path.dirname(args.params_file) if os.path.dirname(args.params_file) else "."
    data_dir = os.path.dirname(args.data_file) if os.path.dirname(args.data_file) else "."
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(args.params_file):
        logging.warning(f"Parameter file '{args.params_file}' not found. Creating a dummy one for testing.")
        dummy_params_content = {
          "target_weights": {"BTC_SPOT": 0.60, "BTC_LONG5X": 0.20, "BTC_SHORT5X": 0.20}, # Example normal weights
          "rebalance_threshold": 0.01, 
          "initial_portfolio_value_usdt": 10000.0,
          # "commission_rate": 0.001, # Old way, will be overridden by taker/maker if they exist
          "taker_commission_rate": 0.0007,
          "maker_commission_rate": 0.0002,
          "use_maker_fees_in_backtest": False,
          "slippage_percentage": 0.0005,
          "report_path_prefix": "./reports/backtest_",
          # New advanced params for dummy file
          "circuit_breaker_threshold_percent": 0.10, # 10% candle movement
          "margin_usage_safe_mode_enter_threshold": 0.70, # 70% margin usage
          "margin_usage_safe_mode_exit_threshold": 0.50,  # 50% margin usage
          "safe_mode_target_weights": {"BTC_SPOT": 0.80, "BTC_LONG5X": 0.05, "BTC_SHORT5X": 0.15} # Example safe weights
        }
        try:
            with open(args.params_file, 'w') as f: json.dump(dummy_params_content, f, indent=2)
            logging.info(f"Dummy parameters file created at '{args.params_file}'")
        except Exception as e:
            logging.error(f"Could not create dummy parameters file: {e}")
            return

    if not os.path.exists(args.data_file):
        logging.warning(f"Data file '{args.data_file}' not found. Creating a dummy one for testing.")
        timestamps = pd.date_range(start='2023-01-01 00:00:00', periods=100, freq='h')
        prices = [20000 + (i*10) + (500 * (i % 5)) - (300 * (i % 3)) for i in range(100)]
        df_dummy = pd.DataFrame({
            'timestamp': timestamps, 'open': [p - 10 for p in prices], 'high': [p + 20 for p in prices],
            'low': [p - 20 for p in prices], 'close': prices, 'volume': [100 + i*2 for i in range(100)]
        })
        try:
            df_dummy.to_csv(args.data_file, index=False)
            logging.info(f"Dummy data file created at '{args.data_file}'")
        except Exception as e:
            logging.error(f"Could not create dummy data file: {e}")
            return
    # --- End of Dummy File Creation ---

    run_backtest_from_config_file(args.params_file, args.data_file)

if __name__ == "__main__":
    main()
