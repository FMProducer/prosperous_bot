import argparse
import json
import os
import pandas as pd
import optuna
from datetime import datetime
import logging
import copy # For deep copying params

# Attempt to import the backtester; handle potential ImportError if structure changes
try:
    from src.prosperous_bot.rebalance_backtester import run_backtest
except ImportError:
    # Fallback if the script is run from a different relative path or package structure issues
    # This assumes rebalance_backtester.py is in the same directory for a simple fallback
    try:
        from rebalance_backtester import run_backtest
    except ImportError:
        logging.error("CRITICAL: Could not import 'run_backtest' from rebalance_backtester.py. Ensure it's accessible.")
        run_backtest = None # Ensure it's defined to avoid NameError later, but it will fail

import functools
import operator

# Basic logging configuration for the optimizer
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# Optuna visualization imports - handled gracefully if not available
try:
    from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice, plot_contour
    optuna_visualization_available = True
except ImportError:
    logging.warning("Optuna visualization modules not found. Plot generation will be skipped. "
                    "Install plotly for Optuna visualizations: `pip install plotly`")
    optuna_visualization_available = False


# --- Available metrics from rebalance_backtester.py (snake_case) ---
# 'run_id', 'strategy_name', 'date_range_start', 'date_range_end',
# 'initial_portfolio_value_usdt', 'final_portfolio_value_usdt',
# 'total_net_pnl_usdt', 'total_net_pnl_percent', 'max_drawdown_percent',
# 'sharpe_ratio', 'sortino_ratio', 'profit_factor', 'total_trades',
# 'winning_trades', 'losing_trades', 'win_rate_percent',
# 'average_trade_pnl_usdt', 'average_winning_trade_usdt',
# 'average_losing_trade_usdt', 'ratio_avg_win_avg_loss',
# 'total_commissions_usdt', 'total_slippage_usdt',
# 'config_target_weights', 'config_rebalance_threshold', 'config_commission_rate',
# 'config_slippage_percentage', 'config_price_source', 'config_slippage_model',
# 'config_risk_free_rate_annual', 'config_annualization_factor',
# 'output_dir', 'status'
# --------------------------------------------------------------------

def set_nested_value(d, path, value):
    """
    Sets a value in a nested dictionary based on a dot-separated path.
    Example: set_nested_value(my_dict, "a.b.c", 10)
    """
    keys = path.split('.')
    current_level = d
    for i, key in enumerate(keys):
        if i == len(keys) - 1:
            current_level[key] = value
        else:
            if key not in current_level or not isinstance(current_level[key], dict):
                # This case should ideally not happen if using a valid template from unified_config
                logging.warning(f"Path {path} is creating intermediate dictionary for key '{key}' "
                                f"or overwriting a non-dict value. Ensure template is complete.")
                current_level[key] = {}
            current_level = current_level[key]

def objective(trial, base_backtest_settings, optimization_space, data_file_path, optimizer_settings):
    """
    Optuna objective function.
    Runs a backtest with parameters suggested by Optuna, overriding values in base_backtest_settings.
    """
    if run_backtest is None:
        logging.error("run_backtest function is not available. Cannot run trial.")
        raise optuna.exceptions.TrialPruned("run_backtest function not imported.")

    # Create a deep copy of the base backtest settings for this trial
    current_backtest_params = copy.deepcopy(base_backtest_settings)
    
    # Get main_asset_symbol from the base settings to format paths
    main_asset_symbol = base_backtest_settings.get('main_asset_symbol', 'BTC')
    if 'main_asset_symbol' not in base_backtest_settings:
        logging.warning(f"Optimizer: 'main_asset_symbol' not found in base_backtest_settings, defaulting to '{main_asset_symbol}'. "
                        "Ensure it's defined in unified_config.json's backtest_settings section.")

    # Store suggested params separately for logging/best_params.json
    suggested_params_for_log = {}

    # Suggest parameters based on the optimization_space configuration
    for p_config in optimization_space:
        original_param_path = p_config['path']
        # Substitute {main_asset_symbol} placeholder in the path
        param_path_substituted = original_param_path.format(main_asset_symbol=main_asset_symbol)

        # Optuna uses this name for its internal storage of suggested params.
        # Using the substituted path ensures Optuna correctly tracks params if main_asset_symbol changes,
        # though typically it's fixed for one optimization study.
        param_name_for_trial = param_path_substituted
        
        suggested_value = None
        if p_config['type'] == 'float':
            suggested_value = trial.suggest_float(
                param_name_for_trial, 
                p_config['low'], 
                p_config['high'], 
                log=p_config.get('log', False),
                step=p_config.get('step') 
            )
        elif p_config['type'] == 'int':
            suggested_value = trial.suggest_int(
                param_name_for_trial, 
                p_config['low'], 
                p_config['high'], 
                log=p_config.get('log', False),
                step=p_config.get('step', 1) # Default step to 1 for int
            )
        elif p_config['type'] == 'categorical':
            suggested_value = trial.suggest_categorical(param_name_for_trial, p_config['choices'])
        else:
            logging.warning(f"Unsupported parameter type '{p_config['type']}' for original path '{original_param_path}' (substituted: '{param_path_substituted}'). Skipping suggestion.")
            continue
        
        set_nested_value(current_backtest_params, param_path_substituted, suggested_value)
        suggested_params_for_log[param_path_substituted] = suggested_value # Log with the substituted path

    logging.info(f"Trial {trial.number}: Testing with suggested params for {main_asset_symbol}: {suggested_params_for_log}")

    try:
        # Determine if individual trial reports should be generated
        # This can be set in the fixed part of backtest_settings if needed, e.g. current_backtest_params.get('generate_reports_for_optimizer_trial', False)
        # For this refactor, we assume it's part of the backtest_settings if desired.
        backtest_results = run_backtest(
            params_dict=current_backtest_params, 
            data_path=data_file_path, 
            is_optimizer_call=True, # Ensures it knows it's part of an optimization
            trial_id_for_reports=trial.number # For unique report subfolders if reports are generated
        )
        
        if backtest_results is None or backtest_results.get("status") != "Completed":
            logging.warning(f"Trial {trial.number} failed or did not complete. Status: {backtest_results.get('status', 'Unknown error') if backtest_results else 'None'}.")
            raise optuna.exceptions.TrialPruned(f"Backtest failed or did not complete. Status: {backtest_results.get('status', 'None')}")
        
        metric_to_optimize = optimizer_settings.get('metric_to_optimize', 'total_net_pnl_percent')
        optimization_value = backtest_results.get(metric_to_optimize)

        if optimization_value is None:
            logging.error(f"Trial {trial.number}: Metric '{metric_to_optimize}' not found in backtest results. "
                          f"Available keys: {list(backtest_results.keys())}. Pruning trial.")
            raise optuna.exceptions.TrialPruned(f"Metric '{metric_to_optimize}' not found in backtest results.")
        
        logging.info(f"Trial {trial.number} completed. Metric ('{metric_to_optimize}'): {optimization_value:.4f}")
        return float(optimization_value) 

    except optuna.exceptions.TrialPruned as e:
        logging.info(f"Trial {trial.number} pruned: {e}")
        raise 
    except Exception as e:
        logging.error(f"Trial {trial.number}: Unexpected error during backtest execution: {e}", exc_info=True)
        raise optuna.exceptions.TrialPruned(f"Unexpected error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Prosperous Bot Rebalance Strategy Optimizer (Unified Config)")
    parser.add_argument("--config_file", type=str, required=True,
                        help="Path to the unified JSON configuration file (e.g., config/unified_config.json).")
    parser.add_argument("--n_trials", type=int, default=None, 
                        help="Number of optimization trials. Overrides value in unified_config.json if provided.")
    parser.add_argument("--data_file_path", type=str, default=None,
                        help="Path to the data CSV file. Overrides data_settings.csv_file_path in unified_config.json if provided.")
    args = parser.parse_args()

    if run_backtest is None:
        logging.critical("Optimizer cannot run because the backtester function could not be imported. Exiting.")
        return

    # --- Load Unified Configuration ---
    try:
        with open(args.config_file, 'r') as f:
            unified_config = json.load(f)
        logging.info(f"Unified configuration loaded from {args.config_file}")
    except FileNotFoundError:
        logging.error(f"FATAL: Unified configuration file not found at {args.config_file}. Exiting.")
        return
    except json.JSONDecodeError:
        logging.error(f"FATAL: Could not decode JSON from unified configuration file: {args.config_file}. Exiting.")
        return
    
    optimizer_settings = unified_config.get('optimizer_settings')
    backtest_settings_template = unified_config.get('backtest_settings')

    if not optimizer_settings or not backtest_settings_template:
        logging.error("FATAL: 'optimizer_settings' or 'backtest_settings' missing in the unified configuration file. Exiting.")
        return

    data_file_path = args.data_file_path or backtest_settings_template.get("data_settings", {}).get("csv_file_path")
    if not data_file_path:
        logging.error("FATAL: Data file path ('data_settings.csv_file_path') not found in config and not provided via CLI. Exiting.")
        return
    # If CLI provided a data_file_path, update it in the template for consistency if run_backtest uses it from there
    if args.data_file_path:
        if "data_settings" not in backtest_settings_template:
            backtest_settings_template["data_settings"] = {}
        backtest_settings_template["data_settings"]["csv_file_path"] = args.data_file_path
        logging.info(f"Using data file path from CLI override: {args.data_file_path}")


    # --- Create Output Directory ---
    report_path_prefix = backtest_settings_template.get('report_path_prefix', './reports/') # Use backtest prefix for base
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Specific optimizer run output directory
    optimizer_run_output_dir = os.path.join(report_path_prefix.rstrip('/'), f"optimizer_run_{timestamp_str}")
    os.makedirs(optimizer_run_output_dir, exist_ok=True)
    logging.info(f"Optimizer outputs for this run will be saved to: {optimizer_run_output_dir}")
    
    # Potentially update report_path_prefix in backtest_settings_template for individual trial reports
    # This ensures trial reports go into a subfolder of the main optimizer run output dir
    backtest_settings_template['report_path_prefix'] = os.path.join(optimizer_run_output_dir, "trials/")


    # --- Optuna Study Setup ---
    study_direction = optimizer_settings.get('direction', 'maximize')
    study_name = optimizer_settings.get('study_name', f"rebalance_optimization_{timestamp_str}")
    
    # Use n_trials from CLI if provided, otherwise from config, else default
    n_trials_from_config = optimizer_settings.get('n_trials', 100)
    n_trials_to_run = args.n_trials if args.n_trials is not None else n_trials_from_config
    if args.n_trials is not None:
        logging.info(f"Using n_trials from CLI override: {args.n_trials}. (Config was: {n_trials_from_config})")


    # TODO: Implement Sampler and Pruner selection based on config string
    # For now, using Optuna defaults (TPE Sampler, MedianPruner if not specified or if specified string is not handled)
    sampler = None # Optuna default (TPE)
    pruner = None  # Optuna default (MedianPruner)
    # Example for future:
    # if optimizer_settings.get("sampler_type") == "Random": sampler = optuna.samplers.RandomSampler()
    # if optimizer_settings.get("pruner_type") == "Hyperband": pruner = optuna.pruners.HyperbandPruner()


    study = optuna.create_study(direction=study_direction, study_name=study_name, sampler=sampler, pruner=pruner)
    
    objective_with_args = lambda trial: objective(
        trial, 
        backtest_settings_template, 
        optimizer_settings.get("optimization_space", []), 
        data_file_path,
        optimizer_settings 
    )

    logging.info(f"Starting Optuna optimization with {n_trials_to_run} trials. "
                 f"Objective: {study_direction} '{optimizer_settings.get('metric_to_optimize', 'total_net_pnl_percent')}'.")

    try:
        study.optimize(objective_with_args, n_trials=n_trials_to_run, timeout=optimizer_settings.get('timeout_seconds'))
    except Exception as e:
        logging.error(f"Optimization process encountered an error: {e}", exc_info=True)
    finally:
        logging.info("Optimization finished or stopped.")

        if study.trials:
            trials_df = study.trials_dataframe()
            trials_df = trials_df.sort_values(by="value", ascending=(study_direction == "minimize"))
            optimization_log_path = os.path.join(optimizer_run_output_dir, "optimization_log.csv")
            trials_df.to_csv(optimization_log_path, index=False)
            logging.info(f"Optimization log saved to {optimization_log_path}")

            # Create the best_backtest_config by applying best Optuna params to the template
            best_backtest_config_final = copy.deepcopy(backtest_settings_template)
            # Remove the specific path for trial reports from the final best config
            best_backtest_config_final['report_path_prefix'] = report_path_prefix 
            
            # Optuna's study.best_params contains keys that are the param_name_for_trial (substituted paths).
            for param_path_substituted, value in study.best_params.items():
                # We need to ensure set_nested_value uses the same substituted path
                set_nested_value(best_backtest_config_final, param_path_substituted, value)

            best_params_data = {
                "best_value": study.best_value,
                "best_optuna_params_substituted": study.best_params, # These keys include the substituted main_asset_symbol
                "best_backtest_config": best_backtest_config_final,
                "best_trial_number": study.best_trial.number
            }
            best_params_path = os.path.join(optimizer_run_output_dir, "best_params.json")
            with open(best_params_path, 'w') as f:
                json.dump(best_params_data, f, indent=2)
            logging.info(f"Best parameters and full backtest config saved to {best_params_path}")
            logging.info(f"Best trial ({study.best_trial.number}): Value = {study.best_value:.4f}, Optuna Params = {study.best_params}")

            if optuna_visualization_available:
                logging.info("Generating Optuna visualization plots...")
                try:
                    history_plot = plot_optimization_history(study)
                    history_plot_path = os.path.join(optimizer_run_output_dir, "optimization_history.html")
                    history_plot.write_html(history_plot_path)
                    logging.info(f"Optimization history plot saved to {history_plot_path}")

                    importances_plot = plot_param_importances(study)
                    importances_plot_path = os.path.join(optimizer_run_output_dir, "param_importances.html")
                    importances_plot.write_html(importances_plot_path)
                    logging.info(f"Parameter importances plot saved to {importances_plot_path}")

                    slice_plot = plot_slice(study)
                    slice_plot_path = os.path.join(optimizer_run_output_dir, "slice_plot.html")
                    slice_plot.write_html(slice_plot_path)
                    logging.info(f"Slice plot saved to {slice_plot_path}")
                    
                    # For contour plot, use the param names directly from optimization_space
                    # as Optuna stores them with their full path in study.best_params
                    param_paths_for_contour = [p['path'] for p in optimizer_settings.get("optimization_space", [])]

                    if len(param_paths_for_contour) > 1:
                        try:
                            param_importances_data_tuples = optuna.importance.get_param_importances(study, normalizer=None) # Get raw importances
                            # Sort params by importance (descending)
                            sorted_important_param_paths = [item[0] for item in sorted(param_importances_data_tuples.items(), key=lambda x: x[1], reverse=True)]
                            params_for_contour_plot = sorted_important_param_paths[:2]
                        except Exception as e_imp:
                            logging.warning(f"Could not determine parameter importances for contour plot automatically, "
                                            f"defaulting to first two from optimization_space. Error: {e_imp}")
                            params_for_contour_plot = param_paths_for_contour[:2]
                        
                        if len(params_for_contour_plot) == 2:
                            contour_plot_fig = plot_contour(study, params=params_for_contour_plot)
                            contour_plot_path = os.path.join(optimizer_run_output_dir, "contour_plot.html")
                            contour_plot_fig.write_html(contour_plot_path)
                            logging.info(f"Contour plot for params {params_for_contour_plot} saved to {contour_plot_path}")
                        else:
                             logging.info("Contour plot requires at least two parameters. Skipping contour plot.")
                    else:
                        logging.info("Only one parameter was optimized or available. Skipping contour plot.")

                except Exception as e:
                    logging.error(f"Error generating Optuna plots: {e}", exc_info=True)
            else:
                logging.info("Optuna visualization plots skipped as plotly is not available.")
        else:
            logging.warning("No trials were completed. Reports and plots cannot be generated.")
        
        logging.info(f"All optimizer outputs saved in {optimizer_run_output_dir}")
        logging.info(f"To run the backtester with the best found parameters, create a new JSON config file "
                     f"using the 'best_backtest_config' section from '{best_params_path}' "
                     f"and run: python src/prosperous_bot/rebalance_backtester.py <your_new_config.json> {data_file_path}")


if __name__ == "__main__":
    # --- Example Dummy Unified Config File Creation (for ease of testing) ---
    # This replaces the old dummy optimizer config and backtester config creation.
    # The user should now primarily use a unified_config.json.
    dummy_unified_config_path = "config/unified_config.example.json" # Matches the file created in previous task
    if not os.path.exists(dummy_unified_config_path):
        os.makedirs(os.path.dirname(dummy_unified_config_path), exist_ok=True)
        logging.info(f"Attempting to create {dummy_unified_config_path} as it was not found.")
        dummy_unified_content = {
          "optimizer_settings": {
            "n_trials": 10, # Reduced for quick dummy run
            "metric_to_optimize": "sharpe_ratio",
            "direction": "maximize",
            "sampler_type": "TPE", # Example
            "pruner_type": "MedianPruner", # Example
            "optimization_space": [
              {
                "path": "rebalance_threshold", # Corresponds to backtest_settings.rebalance_threshold
                "type": "float", "low": 0.01, "high": 0.05, "step": 0.01
              },
              {
                "path": "circuit_breaker_config.threshold_percentage", # Nested path
                "type": "float", "low": 0.05, "high": 0.15
              },
              {
                "path": "target_weights_normal.BTC_SPOT", 
                "type": "float", "low": 0.1, "high": 0.8 
              },
              {
                "path": "target_weights_normal.BTC_LONG5X",
                "type": "float", "low": 0.0, "high": 0.4
              },
              {
                "path": "target_weights_normal.BTC_SHORT5X",
                "type": "float", "low": 0.0, "high": 0.4
              }
              // Add other parameters like slippage_percent, commission rates etc. as needed
            ]
          },
          "backtest_settings": { 
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
              "enabled": True,
              "threshold_percentage": 0.10, 
              "lookback_candles": 1,
              "movement_calc_type": "(high-low)/open"
            },
            "safe_mode_config": { 
              "enabled": True,
              "metric_to_monitor": "margin_usage",
              "entry_threshold": 0.70,
              "exit_threshold": 0.50,
              "target_weights_safe": { "BTC_SPOT": 0.75, "BTC_LONG5X": 0.05, "BTC_SHORT5X": 0.05, "USDT": 0.15 }
            },
            "data_settings": {
              "csv_file_path": "data/BTCUSDT_default_1h_dummy.csv", 
              "timestamp_col": "timestamp",
              "ohlc_cols": {"open": "open", "high": "high", "low": "low", "close": "close"},
              "volume_col": "volume",
              "price_col_for_rebalance": "close" # Default, can be optimized
            },
            "date_range": { # Optional, if not present, backtester uses full data range
                "start_date": "2023-01-01T00:00:00Z", "end_date": "2023-03-31T23:59:59Z"
            },
            "logging_level": "INFO",
            "report_path_prefix": "./reports/", # Base path for reports
            "generate_reports_for_optimizer_trial": False # Usually False for speed
          }
        }
        try:
            with open(dummy_unified_config_path, 'w') as f:
                json.dump(dummy_unified_content, f, indent=2)
            logging.info(f"Dummy unified config created at {dummy_unified_config_path}")
        except Exception as e:
            logging.error(f"Could not create dummy unified config: {e}")

    # Dummy data file for the dummy unified config, if it doesn't exist
    # Ensure this matches the csv_file_path in the dummy_unified_content
    dummy_data_for_optimizer_path = "data/BTCUSDT_default_1h_dummy.csv" 
    if not os.path.exists(dummy_data_for_optimizer_path):
        os.makedirs(os.path.dirname(dummy_data_for_optimizer_path), exist_ok=True)
        timestamps = pd.date_range(start='2023-01-01 00:00:00', periods=200, freq='h')
        prices = [20000 + (i*5) + (600 * (i % 7)) - (400 * (i % 4)) for i in range(200)] 
        df_dummy_data_opt = pd.DataFrame({ # Renamed variable to avoid conflict
            'timestamp': timestamps, 'open': [p - 10 for p in prices], 'high': [p + 20 for p in prices],
            'low': [p - 20 for p in prices], 'close': prices, 'volume': [100 + i*2 for i in range(200)]
        })
        try:
            df_dummy_data_opt.to_csv(dummy_data_for_optimizer_path, index=False) # Use correct variable
            logging.info(f"Dummy data file for optimizer created at {dummy_data_for_optimizer_path}")
        except Exception as e:
            logging.error(f"Could not create dummy data file for optimizer: {e}")
    
    logging.info("Running main() for rebalance_optimizer.py with unified_config. "
                 "To run with specific args, use CLI: \n"
                 "python -m src.prosperous_bot.rebalance_optimizer --config_file config/unified_config.example.json")
    main()
