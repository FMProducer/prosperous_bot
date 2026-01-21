# find_best_seed.py
import argparse
import copy
import json
import logging
import os
import sys
import time
import multiprocessing
import random

import pandas as pd

from config import MasterConfig
from train_ohlcv import main as train_main
from utils import load_config, setup_logging


def train_worker(cfg: MasterConfig):
    """
    A wrapper for train_main that can be called in a separate process.
    It sets up logging for the subprocess.
    """
    # Each process needs its own logging setup.
    setup_logging(f"seed_finder_worker_{cfg.random_seed}", cfg)
    try:
        train_main(cfg=cfg)
    except Exception:
        logging.error(f"Training failed in worker for seed {cfg.random_seed}", exc_info=True)
        sys.exit(1)


def find_best_seed(base_cfg: MasterConfig, seeds: list[int], output_csv: str):
    """
    Iterates through a list of random seeds, trains a model for each,
    collects validation metrics, and saves the results.

    Args:
        base_cfg (MasterConfig): The base configuration object.
        seeds (list[int]): A list of random seeds to test.
        output_csv (str): Path to save the final results CSV.
    """
    results = []
    original_config_name = base_cfg.paths.config_name

    for i, seed in enumerate(seeds):
        trial_name = f"{original_config_name}_seed_{seed}"
        logging.info(f"--- Starting Trial {i+1}/{len(seeds)} | Seed: {seed} | Trial Name: {trial_name} ---")

        # 1. Create a deep copy of the config for this trial
        cfg = copy.deepcopy(base_cfg)
        cfg.random_seed = seed
        cfg.paths.config_name = trial_name

        # Ensure output directories are unique for each trial
        timestamp = time.strftime("date_%Y%m%d_time_%H%M%S")
        session_name = f"{cfg.project_name}_{timestamp}"
        trial_output_dir = os.path.join(cfg.paths.base_output_dir, cfg.paths.config_name)
        cfg.paths.model_dir = os.path.join(trial_output_dir, "saved_models")
        cfg.paths.plot_dir = os.path.join(trial_output_dir, "plots")

        # 2. Train the model in a separate process to ensure isolation
        try:
            logging.info(f"Step 1: Spawning training process for seed {seed}...")
            process = multiprocessing.Process(target=train_worker, args=(cfg,))
            process.start()
            process.join()

            if process.exitcode != 0:
                raise RuntimeError(f"Training process for seed {seed} failed with exit code {process.exitcode}.")

            logging.info(f"Training process for seed {seed} complete.")

            if not os.path.exists(cfg.paths.model_dir) or not os.listdir(cfg.paths.model_dir):
                 raise FileNotFoundError(f"No model directory found in {cfg.paths.model_dir}. Training may have failed early.")

            # After training, find the output directory to load validation metrics.
            # The model is saved in a subfolder with a timestamp. We find the latest one.
            latest_run_dir = sorted(os.listdir(cfg.paths.model_dir))[-1]
            model_run_path = os.path.join(cfg.paths.model_dir, latest_run_dir)

            metrics_path = os.path.join(model_run_path, "metrics.json")
            if not os.path.exists(metrics_path):
                raise FileNotFoundError(f"metrics.json not found in {model_run_path}. Training may have failed.")

            with open(metrics_path, 'r') as f:
                train_summary = json.load(f)

            # Extract metrics from the 'best_validation' dictionary.
            metrics = train_summary.get("best_validation", {})
            if not metrics:
                logging.warning(f"No 'best_validation' metrics found for seed {seed}. The validation gate might not have been passed.")
                continue

            # 4. Collect results
            metrics['seed'] = seed
            results.append(metrics)

        except Exception as e:
            logging.error(f"Error during training or metric collection for seed {seed}: {e}", exc_info=True)
            continue # Skip to the next seed

    # 5. Save and display final results
    if not results:
        logging.warning("No successful trials were completed. No results to save.")
        return

    results_df = pd.DataFrame(results)
    # Rename columns for consistency and better readability
    results_df = results_df.rename(columns={
        "Validation_sharpe": "sharpe",
        "Validation_sortino": "sortino",
        "Validation_max_drawdown": "max_drawdown",
        "Validation_win_rate": "win_rate",
        "Validation_profit_factor": "profit_factor",
        "Validation_trades": "trades",
    })
    results_df = results_df.sort_values(by="sharpe", ascending=False)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results_df.to_csv(output_csv, index=False)

    logging.info("\n--- Experiment Complete ---")
    logging.info(f"Results saved to: {output_csv}")
    print("\nSeed Performance Summary (based on validation metrics, sorted by Sharpe):")
    print(results_df[['seed', 'sharpe', 'sortino', 'max_drawdown', 'win_rate', 'profit_factor', 'trades']].to_string(index=False))


if __name__ == "__main__":
    # Required for multiprocessing on Windows and macOS
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Find the best random seed for a given RL trading configuration.")
    parser.add_argument("config_path", type=str, help="Path to the base configuration file (e.g., configs/alpha_aggressive.py).")
    parser.add_argument("--num-seeds", type=int, required=True, help="The number of random 3-digit seeds to generate and test.")
    parser.add_argument("--output-csv", type=str, default="output/seed_search_results.csv", help="Path to save the results CSV file.")
    args = parser.parse_args()

    base_config, _ = load_config(args.config_path, return_module=True)
    setup_logging("seed_finder_session", base_config)

    # Generate a list of unique random 3-digit seeds
    num_seeds_to_generate = args.num_seeds
    if num_seeds_to_generate <= 0:
        raise ValueError("--num-seeds must be a positive integer.")
    
    generated_seeds = sorted(list(set(random.randint(100, 999) for _ in range(num_seeds_to_generate))))
    logging.info(f"Generated {len(generated_seeds)} unique random seeds to test: {generated_seeds}")

    find_best_seed(base_config, generated_seeds, args.output_csv)
