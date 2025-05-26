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

def objective(trial, base_params_config, data_file_path):
    """
    Optuna objective function.
    Runs a backtest with parameters suggested by Optuna.
    """
    if run_backtest is None:
        logging.error("run_backtest function is not available. Cannot run trial.")
        raise optuna.exceptions.TrialPruned("run_backtest function not imported.")

    current_params = copy.deepcopy(base_params_config['fixed_params'])
    
    # Suggest parameters based on the optimization_params configuration
    for param_name, p_config in base_params_config.get('optimization_params', {}).items():
        if p_config['type'] == 'float':
            current_params[param_name] = trial.suggest_float(param_name, p_config['low'], p_config['high'], log=p_config.get('log', False))
        elif p_config['type'] == 'int':
            current_params[param_name] = trial.suggest_int(param_name, p_config['low'], p_config['high'], log=p_config.get('log', False))
        # Add 'categorical' or other types if needed later
        else:
            logging.warning(f"Unsupported parameter type '{p_config['type']}' for {param_name}. Skipping suggestion.")

    logging.info(f"Trial {trial.number}: Testing with params: { {k: current_params[k] for k in base_params_config.get('optimization_params', {}).keys()} }")

    try:
        # Pass trial number for potential individual report naming
        # is_optimizer_call=True by default in run_backtest if called with params_dict
        # generate_reports_for_optimizer_trial can be set in fixed_params if needed
        backtest_results = run_backtest(current_params, data_file_path, trial_id_for_reports=trial.number)
        
        if backtest_results is None or backtest_results.get("status") != "Completed":
            logging.warning(f"Trial {trial.number} failed or did not complete. Status: {backtest_results.get('status', 'Unknown error') if backtest_results else 'None'}.")
            # For maximization, return a very small number; for minimization, a very large one.
            # Or, prune the trial. Pruning is often better if the failure implies invalid params.
            raise optuna.exceptions.TrialPruned(f"Backtest failed or did not complete. Status: {backtest_results.get('status', 'None')}")

        optimizer_settings = base_params_config.get('optimizer_settings', {})
        metric_to_optimize = optimizer_settings.get('metric_to_optimize', 'total_net_pnl_percent')
        
        optimization_value = backtest_results.get(metric_to_optimize)

        if optimization_value is None:
            logging.error(f"Trial {trial.number}: Metric '{metric_to_optimize}' not found in backtest results. "
                          f"Available keys: {list(backtest_results.keys())}. Pruning trial.")
            raise optuna.exceptions.TrialPruned(f"Metric '{metric_to_optimize}' not found in backtest results.")
        
        # Handle cases where drawdown might be positive (e.g. 0 if no loss) but we want to maximize (e.g. -5 is better than -10)
        # If optimizing max_drawdown_percent and direction is 'maximize', a less negative (larger) value is better.
        # No special transformation needed here if users understand that maximizing -5 is better than -10.
        # Optuna handles maximization/minimization directly.

        logging.info(f"Trial {trial.number} completed. Metric ('{metric_to_optimize}'): {optimization_value:.4f}")
        return float(optimization_value) # Ensure it's a float

    except optuna.exceptions.TrialPruned as e:
        logging.info(f"Trial {trial.number} pruned: {e}")
        raise # Re-raise to let Optuna handle it
    except Exception as e:
        logging.error(f"Trial {trial.number}: Unexpected error during backtest execution: {e}", exc_info=True)
        # Prune trial on unexpected error
        raise optuna.exceptions.TrialPruned(f"Unexpected error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Prosperous Bot Rebalance Strategy Optimizer")
    parser.add_argument("--config_file", type=str, required=True, 
                        help="Path to the optimizer JSON configuration file. This file defines fixed_params, "
                             "optimization_params (search space), and optimizer_settings (metric, direction).")
    parser.add_argument("--data_file", type=str, required=True, 
                        help="Path to the input CSV data file (e.g., data/BTCUSDT_1h_data.csv)")
    parser.add_argument("--n_trials", type=int, default=100, 
                        help="Number of optimization trials to run.")
    parser.add_argument("--output_dir_prefix", type=str, default="./reports/optimizer_", 
                        help="Prefix for the optimizer's output directory.")
    args = parser.parse_args()

    if run_backtest is None:
        logging.critical("Optimizer cannot run because the backtester function could not be imported. Exiting.")
        return

    # --- Load Optimizer Configuration ---
    try:
        with open(args.config_file, 'r') as f:
            optimizer_config = json.load(f)
        logging.info(f"Optimizer configuration loaded from {args.config_file}")
    except FileNotFoundError:
        logging.error(f"FATAL: Optimizer configuration file not found at {args.config_file}. Exiting.")
        return
    except json.JSONDecodeError:
        logging.error(f"FATAL: Could not decode JSON from optimizer configuration file: {args.config_file}. Exiting.")
        return
    
    # --- Create Output Directory ---
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    optimizer_output_dir = f"{args.output_dir_prefix.rstrip('_')}_{timestamp_str}"
    os.makedirs(optimizer_output_dir, exist_ok=True)
    logging.info(f"Optimizer outputs will be saved to: {optimizer_output_dir}")

    # --- Optuna Study Setup ---
    opt_settings = optimizer_config.get('optimizer_settings', {})
    study_direction = opt_settings.get('direction', 'maximize')
    study_name = opt_settings.get('study_name', f"rebalance_optimization_{timestamp_str}")

    study = optuna.create_study(direction=study_direction, study_name=study_name)
    
    # Define the objective function with fixed arguments (base_params_config, data_file_path)
    objective_with_args = lambda trial: objective(trial, optimizer_config, args.data_file)

    logging.info(f"Starting Optuna optimization with {args.n_trials} trials. Objective: {study_direction} '{opt_settings.get('metric_to_optimize', 'total_net_pnl_percent')}'.")

    try:
        study.optimize(objective_with_args, n_trials=args.n_trials, timeout=opt_settings.get('timeout_seconds', None))
    except Exception as e:
        logging.error(f"Optimization process encountered an error: {e}", exc_info=True)
    finally:
        logging.info("Optimization finished or stopped.")

        # --- Reporting ---
        if study.trials: # Check if there are any trials to report
            # Save all trial data
            trials_df = study.trials_dataframe()
            # Sort by value for easier viewing of best trials, matching study direction
            trials_df = trials_df.sort_values(by="value", ascending=(study_direction == "minimize"))
            optimization_log_path = os.path.join(optimizer_output_dir, "optimization_log.csv")
            trials_df.to_csv(optimization_log_path, index=False)
            logging.info(f"Optimization log saved to {optimization_log_path}")

            # Save best parameters
            best_params_data = {
                "best_value": study.best_value,
                "best_params_suggested": study.best_params, # These are only the ones Optuna suggested
                "full_config_for_best_trial": { # Combine fixed and best suggested
                    **optimizer_config.get('fixed_params', {}), 
                    **study.best_params
                },
                "best_trial_number": study.best_trial.number
            }
            best_params_path = os.path.join(optimizer_output_dir, "best_params.json")
            with open(best_params_path, 'w') as f:
                json.dump(best_params_data, f, indent=2)
            logging.info(f"Best parameters saved to {best_params_path}")
            logging.info(f"Best trial ({study.best_trial.number}): Value = {study.best_value:.4f}, Params = {study.best_params}")

            # Generate and save Optuna plots if available
            if optuna_visualization_available:
                logging.info("Generating Optuna visualization plots...")
                try:
                    # Optimization History Plot
                    history_plot = plot_optimization_history(study)
                    history_plot_path = os.path.join(optimizer_output_dir, "optimization_history.html")
                    history_plot.write_html(history_plot_path)
                    logging.info(f"Optimization history plot saved to {history_plot_path}")

                    # Parameter Importances Plot
                    importances_plot = plot_param_importances(study)
                    importances_plot_path = os.path.join(optimizer_output_dir, "param_importances.html")
                    importances_plot.write_html(importances_plot_path)
                    logging.info(f"Parameter importances plot saved to {importances_plot_path}")

                    # Slice Plot
                    slice_plot = plot_slice(study)
                    slice_plot_path = os.path.join(optimizer_output_dir, "slice_plot.html")
                    slice_plot.write_html(slice_plot_path)
                    logging.info(f"Slice plot saved to {slice_plot_path}")

                    # Contour Plot (if more than 1 parameter)
                    optimized_params = list(optimizer_config.get('optimization_params', {}).keys())
                    if len(optimized_params) > 1:
                        # Try to get parameter importances to pick best two, otherwise default to first two
                        try:
                            param_importances_data = optuna.importance.get_param_importances(study)
                            # Sort params by importance (descending) and pick top two
                            sorted_important_params = [item[0] for item in sorted(param_importances_data.items(), key=lambda x: x[1], reverse=True)]
                            params_for_contour = sorted_important_params[:2]
                        except Exception as e_imp:
                            logging.warning(f"Could not determine parameter importances for contour plot automatically, defaulting to first two. Error: {e_imp}")
                            params_for_contour = optimized_params[:2]
                        
                        if len(params_for_contour) == 2 : # Ensure we have exactly two
                            contour_plot = plot_contour(study, params=params_for_contour)
                            contour_plot_path = os.path.join(optimizer_output_dir, "contour_plot.html")
                            contour_plot.write_html(contour_plot_path)
                            logging.info(f"Contour plot for params {params_for_contour} saved to {contour_plot_path}")
                        else:
                            logging.info("Contour plot requires at least two parameters to compare. Skipping contour plot.")
                    else:
                        logging.info("Only one parameter was optimized. Skipping contour plot.")

                except Exception as e:
                    logging.error(f"Error generating Optuna plots: {e}", exc_info=True)
            else:
                logging.info("Optuna visualization plots skipped as plotly is not available.")
        else:
            logging.warning("No trials were completed. Reports and plots cannot be generated.")
        
        logging.info(f"All optimizer outputs saved in {optimizer_output_dir}")

        # Suggest running the best params directly if user wants to see full reports
        logging.info(f"To run the backtester with the best found parameters and generate full reports, you can use:")
        logging.info(f"python src/prosperous_bot/rebalance_backtester.py <path_to_a_params_file_based_on_best_params.json> {args.data_file}")
        logging.info(f"(Note: You might need to create a new .json file using 'full_config_for_best_trial' from 'best_params.json')")


if __name__ == "__main__":
    # --- Example Dummy Optimizer Config File Creation (for ease of testing) ---
    dummy_optimizer_config_path = "config/optimizer_params_example.json"
    if not os.path.exists(dummy_optimizer_config_path):
        os.makedirs(os.path.dirname(dummy_optimizer_config_path), exist_ok=True)
        dummy_opt_config = {
          "fixed_params": {
            "target_weights": { "BTC_SPOT": 0.65, "BTC_LONG5X": 0.11, "BTC_SHORT5X": 0.24 },
            "initial_portfolio_value_usdt": 10000.0,
            "report_path_prefix": "./reports/optimizer_trial_reports/trial_", # For individual trial reports (if enabled)
            "generate_reports_for_optimizer_trial": False # Usually false to speed up optimization
          },
          "optimization_params": {
            "rebalance_threshold": {"type": "float", "low": 0.001, "high": 0.05, "log": True},
            "commission_rate": {"type": "float", "low": 0.0005, "high": 0.0015},
            "slippage_percentage": {"type": "float", "low": 0.0001, "high": 0.001}
          },
          "optimizer_settings": {
            "metric_to_optimize": "total_net_pnl_percent", 
            # Common choices: "total_net_pnl_percent", "sharpe_ratio", "sortino_ratio", 
            # "final_portfolio_value_usdt", "profit_factor".
            # For "max_drawdown_percent", if you want less negative to be better, use "maximize".
            "direction": "maximize", # "maximize" or "minimize"
            "study_name": "example_btc_rebalance_opt",
            "timeout_seconds": 3600 # Optional: 1 hour timeout for the whole study
          }
        }
        # For documentation:
        # metric_to_optimize: Must be one of the snake_case keys returned by rebalance_backtester.py.
        # Common ones include:
        # - total_net_pnl_percent (higher is better)
        # - final_portfolio_value_usdt (higher is better)
        # - sharpe_ratio (higher is better)
        # - sortino_ratio (higher is better)
        # - profit_factor (higher is better)
        # - max_drawdown_percent (higher is better, as -5% is > -10%)
        # - total_net_pnl_usdt (higher is better)
        # - total_trades (can be used with 'minimize' or 'maximize' depending on goal)
        # - total_commissions_usdt (usually 'minimize')
        # - total_slippage_usdt (usually 'minimize')
        #
        # direction: "maximize" if higher values of the metric are better, 
        #            "minimize" if lower values are better.
        try:
            with open(dummy_optimizer_config_path, 'w') as f:
                json.dump(dummy_opt_config, f, indent=2)
            logging.info(f"Dummy optimizer config created at {dummy_optimizer_config_path}")
        except Exception as e:
            logging.error(f"Could not create dummy optimizer config: {e}")
    
    # --- Example Dummy Data File (if not created by backtester's main) ---
    # This assumes the optimizer might be run without first running the backtester's main
    dummy_data_file_path = "data/BTCUSDT_1h_data_optimizer_dummy.csv"
    if not os.path.exists(dummy_data_file_path):
        os.makedirs(os.path.dirname(dummy_data_file_path), exist_ok=True)
        timestamps = pd.date_range(start='2023-01-01 00:00:00', periods=200, freq='h') # More data for optimizer
        prices = [20000 + (i*5) + (600 * (i % 7)) - (400 * (i % 4)) for i in range(200)] 
        df_dummy = pd.DataFrame({
            'timestamp': timestamps, 'open': [p - 10 for p in prices], 'high': [p + 20 for p in prices],
            'low': [p - 20 for p in prices], 'close': prices, 'volume': [100 + i*2 for i in range(200)]
        })
        try:
            df_dummy.to_csv(dummy_data_file_path, index=False)
            logging.info(f"Dummy data file for optimizer created at {dummy_data_file_path}")
        except Exception as e:
            logging.error(f"Could not create dummy data file for optimizer: {e}")

    logging.info("Running main() for rebalance_optimizer.py. If you want to run with specific args, use CLI.")
    # To run from CLI, for example:
    # python src/prosperous_bot/rebalance_optimizer.py --config_file config/optimizer_params_example.json --data_file data/BTCUSDT_1h_data_optimizer_dummy.csv --n_trials 10
    main()
