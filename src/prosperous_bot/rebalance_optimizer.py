import argparse
import json
import os
import pandas as pd
import optuna
from datetime import datetime
import logging
import copy # For deep copying params
import math

# Attempt to import the backtester; handle potential ImportError if structure changes
try:
    # Explicit relative import for files within the same package
    from .rebalance_backtester import run_backtest
except ImportError as e1:
    logging.warning(f"Relative import '.rebalance_backtester' failed: {e1}. Trying original fallbacks.")
    try:
        from prosperous_bot.rebalance_backtester import run_backtest
    except ImportError as e2:
        logging.warning(f"Absolute import 'prosperous_bot.rebalance_backtester' failed: {e2}. Trying direct import.")
        # Fallback if the script is run from a different relative path or package structure issues
        # This assumes rebalance_backtester.py is in the same directory for a simple fallback
        try:
            from rebalance_backtester import run_backtest # This would only work if src/prosperous_bot was directly on PYTHONPATH
        except ImportError as e3:
            logging.error(f"CRITICAL: Could not import 'run_backtest' from rebalance_backtester.py. All attempts failed. Last error: {e3}", exc_info=True)
            run_backtest = None # Ensure it's defined to avoid NameError later, but it will fail

# Basic logging configuration for the optimizer
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# Optuna visualization imports - handled gracefully if not available
try:
    from optuna.visualization import (
        plot_optimization_history, plot_param_importances, plot_slice, 
        plot_contour, plot_parallel_coordinate
    )
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

    # --- START MODIFICATIONS ---
    data_path_resolved = data_file_path.format(main_asset_symbol=main_asset_symbol)
    logging.info(f"Trial {trial.number}: Raw data_file_path from config/CLI: '{data_file_path}', "
                 f"Resolved to: '{data_path_resolved}' using main_asset_symbol: '{main_asset_symbol}'.")

    if not os.path.exists(data_path_resolved):
        logging.warning(f"Trial {trial.number}: Data file not found at {data_path_resolved}. Pruning trial.")
        raise optuna.exceptions.TrialPruned(f"Data file not found: {data_path_resolved}")
    # --- END MODIFICATIONS ---

    # Store suggested params separately for logging/best_params.json
    suggested_params_for_log = {}

    # Get main_asset_symbol from base_backtest_settings
    main_asset_symbol = base_backtest_settings.get('main_asset_symbol', 'BTC')

    # Define choices for spot_pct
    spot_pct_choices = [0.80,0.75,0.70,0.65,0.60,0.55,0.50,0.45,0.40,0.35,0.30,0.25,0.20,0.15,0.10,0.05]
    spot_pct = trial.suggest_categorical("spot_pct", spot_pct_choices)

    # Calculate long_pct and short_pct based on spot_pct
    # spot + long + short = 1
    # spot * 1 + long * 5 - short * 5 = 0 (delta neutral)
    # spot + long * 5 - (1 - spot - long) * 5 = 0
    # spot + 5long - 5 + 5spot + 5long = 0
    # 6spot + 10long - 5 = 0
    # 10long = 5 - 6spot => long = (5 - 6spot) / 10
    # short = 1 - spot - long = 1 - spot - (5 - 6spot)/10 = (10 - 10spot - 5 + 6spot)/10 = (5 - 4spot)/10
    long_pct = (5 - 6 * spot_pct) / 10
    short_pct = (5 - 4 * spot_pct) / 10

    # Pruning condition
    if long_pct < 0 or short_pct < 0 or not math.isfinite(long_pct+short_pct):
        logging.info(f"Trial {trial.number}: Pruning trial because calculated long_pct ({long_pct:.4f}), short_pct ({short_pct:.4f}) is negative or non-finite for spot_pct {spot_pct:.2f}.")
        raise optuna.TrialPruned("Calculated long/short percentages are negative or non-finite.")

    # Set target weights in current_backtest_params
    tw = current_backtest_params.setdefault("target_weights_normal", {})
    tw[f"{main_asset_symbol}_SPOT"]        = round(spot_pct, 4)
    tw[f"{main_asset_symbol}_PERP_LONG"]   = round(long_pct, 4)
    tw[f"{main_asset_symbol}_PERP_SHORT"]  = round(short_pct, 4)

    # Log these params
    suggested_params_for_log[f"target_weights_normal.{main_asset_symbol}_SPOT"] = spot_pct
    suggested_params_for_log[f"target_weights_normal.{main_asset_symbol}_PERP_LONG"] = long_pct
    suggested_params_for_log[f"target_weights_normal.{main_asset_symbol}_PERP_SHORT"] = short_pct
    suggested_params_for_log["spot_pct"] = spot_pct # Log the primary suggested param

    # Suggest parameters based on the optimization_space configuration
    for p_config in optimization_space:
        # spot_pct уже обработали
        if p_config['path'] == "spot_pct":
            continue

        # сигнальные / safe_mode параметры игнорируем в neutral-режиме
        # Assuming apply_signal_logic is available in current_backtest_params
        apply_signal_logic = current_backtest_params.get('apply_signal_logic', False) # Default to False for neutral assumption
        if not apply_signal_logic and (
            ".signal" in p_config["path"] or "safe_mode" in p_config["path"]
        ):
            continue

        original_param_path = p_config['path']
        # Substitute {main_asset_symbol} placeholder in the path
        # Ensure main_asset_symbol is available (defined above from base_backtest_settings)
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
            data_path=data_path_resolved, # MODIFIED HERE
            is_optimizer_call=True, # Ensures it knows it's part of an optimization
            trial_id_for_reports=trial.number # For unique report subfolders if reports are generated
        )
        
        if backtest_results is None or backtest_results.get("status") != "Completed":
            logging.warning(f"Trial {trial.number} failed or did not complete. Status: {backtest_results.get('status', 'Unknown error') if backtest_results else 'None'}.")
            raise optuna.exceptions.TrialPruned(f"Backtest failed or did not complete. Status: {backtest_results.get('status', 'None')}")
        
        # 'backtest_results' is the variable holding the results from run_backtest
        # 'current_backtest_params' is the variable holding the configuration for this trial

        pnl = abs(backtest_results["total_net_pnl_percent"]) # Renamed from pnl_abs for consistency with existing code
        std = backtest_results.get("nav_std_percent", 0.0) # Renamed from std_nav for consistency

        # Use initial_portfolio_value_usdt as per patch for commission calculation's denominator
        initial_portfolio_value = current_backtest_params.get("initial_portfolio_value_usdt")
        if initial_portfolio_value is None or initial_portfolio_value == 0:
            logging.warning(f"Trial {trial.number}: 'initial_portfolio_value_usdt' is missing, zero, or invalid in backtest settings (value: {initial_portfolio_value}). Using 1.0 for commission calculation denominator to avoid error, but this may skew results if 'total_commissions_usdt' is large.")
            effective_denominator_for_commission = 1.0
        else:
            effective_denominator_for_commission = initial_portfolio_value

        comm = backtest_results.get("total_commissions_usdt", 0.0) / effective_denominator_for_commission
        value = -(pnl + 0.5*std + 0.1*comm) # pnl, std names kept as per surrounding code
        
        logging.info(f"Trial {trial.number} completed. Objective value: {value:.4f} (Components: PnL_abs_perc: {pnl:.4f}, NAV_std_perc: {std:.4f}, Commission_ratio: {comm:.4f})")
        return value

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
            # Save trials_df_sorted to optimization_log.csv first
            trials_df = study.trials_dataframe()
            # Ensure 'study_direction' string is used for sorting direction
            is_minimization = (study_direction.lower() == "minimize")
            trials_df_sorted = trials_df.sort_values(by="value", ascending=is_minimization)
            optimization_log_path = os.path.join(optimizer_run_output_dir, "optimization_log.csv")
            trials_df_sorted.to_csv(optimization_log_path, index=False)
            logging.info(f"Optimization log saved to {optimization_log_path}")

            try:
                # Attempt to access best_trial, which can raise ValueError if no trials completed successfully
                _ = study.best_trial  # Access to check for ValueError

                logging.info(f"Best trial found: Trial {study.best_trial.number}, Value: {study.best_value:.4f}")

                # Create the best_backtest_config by applying best Optuna params to the template
                best_backtest_config_final = copy.deepcopy(backtest_settings_template)
                # Remove the specific path for trial reports from the final best config
                best_backtest_config_final['report_path_prefix'] = report_path_prefix

                # Optuna's study.best_params contains keys that are the param_name_for_trial (substituted paths).
                for param_path_substituted, value in study.best_params.items():
                    # We need to ensure set_nested_value uses the same substituted path
                    set_nested_value(best_backtest_config_final, param_path_substituted, value)

                # --- Start of patch addition: Clean up placeholder keys ---
                # From patch: # — удаляем placeholder-ключи (если вдруг остались) —
                if "target_weights_normal" in best_backtest_config_final and \
                   isinstance(best_backtest_config_final["target_weights_normal"], dict):
                    best_backtest_config_final["target_weights_normal"] = {
                        k: v for k, v in best_backtest_config_final["target_weights_normal"].items()
                        if "{" not in k
                    }
                # --- End of patch addition ---

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
                    try: # INNER TRY for plots - ALL plotting logic should be inside this try
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

                        parallel_coordinate_plot = plot_parallel_coordinate(study)
                        parallel_coordinate_plot_path = os.path.join(optimizer_run_output_dir, "parallel_coordinate.html")
                        parallel_coordinate_plot.write_html(parallel_coordinate_plot_path)
                        logging.info(f"Parallel coordinate plot saved to {parallel_coordinate_plot_path}")
                        
                        # Get main_asset_symbol for potential substitution in paths, consistent with 'objective'
                        main_asset_symbol_for_plot = backtest_settings_template.get('main_asset_symbol', 'BTC')

                        # Get all *substituted* parameter paths that were part of the optimization space
                        all_substituted_param_paths_in_space = [
                            p_config['path'].format(main_asset_symbol=main_asset_symbol_for_plot)
                            for p_config in optimizer_settings.get("optimization_space", [])
                        ]

                        if len(all_substituted_param_paths_in_space) > 1:
                            params_for_contour_plot = []
                            try:
                                # get_param_importances returns a dict: {param_name_substituted: importance_value}
                                param_importances_dict = optuna.importance.get_param_importances(study)

                                sorted_important_params = sorted(
                                    param_importances_dict.items(),
                                    key=lambda item: item[1],
                                    reverse=True
                                )
                                # Take the names (substituted paths) of the top 2
                                params_for_contour_plot = [param_name for param_name, importance in sorted_important_params[:2]]

                                if len(params_for_contour_plot) < 2 and len(all_substituted_param_paths_in_space) >=2 :
                                    logging.warning(f"Could not get two important params from importances (got {len(params_for_contour_plot)}). "
                                                    f"Falling back to first two from defined optimization space (substituted).")
                                    params_for_contour_plot = all_substituted_param_paths_in_space[:2]

                            except Exception as e_imp:
                                logging.warning(f"Could not determine parameter importances for contour plot automatically, "
                                                f"defaulting to first two from optimization_space (substituted). Error: {e_imp}")
                                params_for_contour_plot = all_substituted_param_paths_in_space[:2]

                            if len(params_for_contour_plot) == 2:
                                contour_plot_fig = plot_contour(study, params=params_for_contour_plot)
                                contour_plot_path = os.path.join(optimizer_run_output_dir, "contour_plot.html")
                                contour_plot_fig.write_html(contour_plot_path)
                                logging.info(f"Contour plot for params {params_for_contour_plot} saved to {contour_plot_path}")
                            else:
                                 logging.info("Contour plot requires at least two parameters with importance or from space. Skipping contour plot.")
                        else:
                            logging.info("Fewer than two parameters were optimized or available in space. Skipping contour plot.")

                    except Exception as e: # INNER EXCEPT for plots
                        logging.error(f"Error generating Optuna plots: {e}", exc_info=True)
                # This 'else' correctly corresponds to 'if optuna_visualization_available:'
                else:
                    logging.info("Optuna visualization plots skipped as plotly is not available.")

            # This 'except ValueError' correctly corresponds to the OUTER TRY
            except ValueError:
                logging.warning("Could not determine best parameters (e.g., all trials pruned or failed). "
                                "Skipping generation of best_params.json and visualization plots. "
                                f"Check '{optimization_log_path}' for trial details.")
        else: # This is the else for 'if study.trials:'
            logging.warning("No trials were recorded in the study. Reports and plots cannot be generated.")
        
        # These messages should be at the end of the 'finally' block
        logging.info(f"All optimizer outputs for this run are in {optimizer_run_output_dir}")
        # The message below might rely on 'best_params_path' which is defined inside the try block.
        # It's safer to make it more general if best_params.json might not exist.
        # Or, ensure best_params_path is defined (e.g. as None) outside the try if used here.
        # For now, let's assume it's okay if this message is not shown when ValueError occurs.
        # A better approach might be to conditionally phrase this last message.
        try:
            # Attempt to reference best_params_path only if it was likely created.
            # This is a bit of a workaround. A cleaner way would be to set best_params_path = None earlier.
            if 'best_params_path' in locals() and os.path.exists(best_params_path):
                 logging.info(f"To run the backtester with the best found parameters, create a new JSON config file "
                              f"using the 'best_backtest_config' section from '{best_params_path}', then run: \n"
                              f"python -m prosperous_bot.rebalance_backtester --config_file <your_new_config.json>")
            elif not study.trials: # If no trials, best_params_path won't exist.
                pass # Message already logged about no trials.
            else: # Trials exist, but ValueError likely occurred.
                 logging.info("If best parameters were found, 'best_params.json' in the output directory can be used to run the backtester.")
        except NameError: # best_params_path might not be defined if all trials failed.
            logging.info("Check 'optimization_log.csv' for trial details. If successful trials exist, 'best_params.json' may be available.")


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
                "path": "target_weights_normal.{main_asset_symbol}_SPOT",
                "type": "float", "low": 0.1, "high": 0.8 
              },
              {
                "path": "target_weights_normal.{main_asset_symbol}_PERP_LONG",
                "type": "float", "low": 0.0, "high": 0.4
              },
              {
                "path": "target_weights_normal.{main_asset_symbol}_PERP_SHORT",
                "type": "float", "low": 0.0, "high": 0.4
              }
            ]
          },
          "backtest_settings": { 
            "main_asset_symbol": "BTC",
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
              "BTC_PERP_LONG": 0.11,
              "BTC_PERP_SHORT": 0.24
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
              "target_weights_safe": {
                "BTC_SPOT": 0.75,
                "BTC_PERP_LONG": 0.05,
                "BTC_PERP_SHORT": 0.05,
                "USDT": 0.15
              }
            },
            "data_settings": {
              "csv_file_path": "graphs/{main_asset_symbol}USDT_data.csv",
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
    # Path changed to 'graphs/' directory and to reflect the {main_asset_symbol}USDT pattern (assuming BTC here for the dummy)
    dummy_data_for_optimizer_path = "graphs/BTCUSDT_default_1h_dummy.csv"
    if not os.path.exists(dummy_data_for_optimizer_path):
        # Ensure the directory exists before trying to create the file
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
