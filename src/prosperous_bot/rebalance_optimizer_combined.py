
# COMBINED VERSION: rebalance_optimizer.py
# Поддерживает CLI, прямой вызов, fallback-импорты, визуализацию, сохранение логов

import argparse
import json
import os
import copy
import logging
from datetime import datetime
from pathlib import Path
import optuna
import pandas as pd

# === Импорт run_backtest ===
try:
    from prosperous_bot.rebalance_backtester import run_backtest
except ImportError:
    try:
        from rebalance_backtester import run_backtest
    except ImportError:
        logging.error("CRITICAL: Could not import 'run_backtest'.")
        run_backtest = None

# === Настройка логирования ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === Optuna визуализация ===
try:
    from optuna.visualization import (
        plot_param_importances, plot_optimization_history,
        plot_parallel_coordinate, plot_contour
    )
    import plotly.io as pio
    optuna_visualization_available = True
except ImportError:
    logging.warning("Plotly не установлен, визуализация Optuna будет пропущена.")
    optuna_visualization_available = False

def set_nested_value(d, path, value):
    keys = path.split('.')
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def objective(trial, base_config, space, data_path, metric):
    if run_backtest is None:
        raise optuna.exceptions.TrialPruned("run_backtest недоступен")

    params = copy.deepcopy(base_config)
    for p in space:
        path = p['path']
        if p['type'] == 'float':
            val = trial.suggest_float(path, p['low'], p['high'], step=p.get('step'))
        elif p['type'] == 'int':
            val = trial.suggest_int(path, p['low'], p['high'], step=p.get('step', 1))
        elif p['type'] == 'categorical':
            val = trial.suggest_categorical(path, p['choices'])
        else:
            continue
        set_nested_value(params, path, val)

    result = run_backtest(params, data_path, is_optimizer_call=True, trial_id_for_reports=trial.number)
    if result is None or result.get("status") != "Completed":
        raise optuna.exceptions.TrialPruned("Backtest failed.")
    return result.get(metric, float("-inf"))

def run_optimizer(config_path, n_trials_override=None, data_path_override=None):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    backtest_config = config["backtest_settings"]
    optimizer_config = config["optimizer_settings"]
    optimization_space = optimizer_config["optimization_space"]

    metric = optimizer_config.get("metric_to_optimize", "sharpe_ratio")
    direction = optimizer_config.get("direction", "maximize")
    n_trials = n_trials_override or optimizer_config.get("n_trials", 100)
    data_path = data_path_override or backtest_config["data_settings"]["csv_file_path"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(backtest_config.get("report_path_prefix", "./reports/")) / f"optimizer_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    backtest_config["report_path_prefix"] = str(report_dir / "trials")

    study = optuna.create_study(direction=direction)
    study.optimize(lambda t: objective(t, backtest_config, optimization_space, data_path, metric), n_trials=n_trials)

    best_params = study.best_params
    best_config = copy.deepcopy(backtest_config)
    for k, v in best_params.items():
        set_nested_value(best_config, k, v)

    with open(report_dir / "best_params.json", "w") as f:
        json.dump(best_config, f, indent=2)

    df = study.trials_dataframe()
    df.to_csv(report_dir / "optimization_log.csv", index=False)

    if optuna_visualization_available:
        try:
            vis_dir = report_dir / "optuna_visuals"
            vis_dir.mkdir(exist_ok=True)

            plots = {
                "optimization_history": plot_optimization_history,
                "param_importance": plot_param_importances,
                "parallel": plot_parallel_coordinate,
                "contour": plot_contour,
            }
            for name, func in plots.items():
                fig = func(study)
                pio.write_html(fig, file=str(vis_dir / f"{name}.html"))
        except Exception as e:
            logging.error(f"Ошибка визуализации: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--n_trials", type=int)
    parser.add_argument("--data_file_path", type=str)
    args = parser.parse_args()

    run_optimizer(args.config_file, args.n_trials, args.data_file_path)

if __name__ == "__main__":
    main()
