# optimize_cfg.py
import argparse
import copy
import datetime as dt
import json
import logging
import os
import time

import optuna
import pandas as pd
import platform

from paper_trader import PaperTrader  # Изменено
from config import MasterConfig
from utils import load_config, setup_logging

def run_papertrade(cfg: MasterConfig, trial_num: int):
    """
    Запускает PaperTrader в режиме базы данных и возвращает метрики производительности.
    """
    # Устанавливаем уникальный путь вывода для этого испытания, чтобы избежать конфликтов
    original_base_dir = cfg.paths.base_output_dir
    original_config_name = cfg.paths.config_name

    trial_output_dir = os.path.join(original_base_dir, f"trial_{trial_num}")
    os.makedirs(trial_output_dir, exist_ok=True)
    cfg.paths.base_output_dir = trial_output_dir

    trader = PaperTrader(cfg=cfg, cfg_mod=None, model_path_override=cfg.paths.model_path)
    trader.run()  # Запускает симуляцию из базы данных

    # --- Сбор и расчет метрик ---
    metrics = {}
    trades_path = os.path.join(trial_output_dir, "paper_trader", "paper_trades.csv")
    equity_path = os.path.join(trial_output_dir, "paper_trader", "paper_equity.csv")

    if os.path.exists(trades_path):
        trades_df = pd.read_csv(trades_path)
        metrics["total_trades"] = len(trades_df)
        if not trades_df.empty:
            winning_trades = trades_df[trades_df["pnl"] > 0].shape[0]
            metrics["accuracy"] = f"{(winning_trades / len(trades_df)) * 100:.2f}%"
        else:
            metrics["accuracy"] = "0.0%"
    else:
        metrics["total_trades"] = 0
        metrics["accuracy"] = "0.0%"

    if os.path.exists(equity_path):
        equity_df = pd.read_csv(equity_path)
        if not equity_df.empty:
            final_balance = equity_df["balance"].iloc[-1]
            initial_balance = cfg.market.initial_balance
            pnl_pct = ((final_balance - initial_balance) / initial_balance) * 100
            metrics["final_balance_change"] = f"{pnl_pct:.2f}%"
        else:
            metrics["final_balance_change"] = "0.0%"
    else:
        metrics["final_balance_change"] = "0.0%"

    # Восстанавливаем исходный путь вывода, чтобы не влиять на другие процессы
    cfg.paths.base_output_dir = original_base_dir
    cfg.paths.config_name = original_config_name
    
    return metrics

def _safe_save_df(df: "pd.DataFrame", opt_dir: str) -> None:
    """
    Save trials as Parquet (if engine available) and ALWAYS as CSV.
    """
    base = os.path.join(opt_dir, "trials")
    csv_path = f"{base}.csv"
    try:
        pq_path = f"{base}.parquet"
        df.to_parquet(pq_path, index=False)
        logging.info(f"[Optuna] saved {pq_path}")
    except Exception as e:
        logging.warning(f"[Optuna] parquet save failed: {e}. CSV will be used.")
    df.to_csv(csv_path, index=False)
    logging.info(f"[Optuna] saved {csv_path}")

def _dump_trials_jsonl(study: "optuna.study.Study", opt_dir: str) -> None:
    """
    Export all trials (number, state, values, params, user_attrs, timings) to JSONL.
    """
    path = os.path.join(opt_dir, "trials.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for t in study.get_trials(deepcopy=False):
            rec = {
                "number": t.number,
                "state": str(t.state) if t.state is not None else None,
                "values": t.values,
                "params": t.params,
                "user_attrs": t.user_attrs,
                "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
                "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else None,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logging.info(f"[Optuna] saved {path}")

def _save_top_tables(df: "pd.DataFrame", opt_dir: str, topn: int = 20) -> None:
    """
    Save top-N trials by PnL (values_0), Accuracy (values_1), and -Trades (values_2).
    """
    cols = df.columns
    targets = [("values_0", "top_by_pnl"), ("values_1", "top_by_accuracy"), ("values_2", "top_by_neg_trades")]
    for val_col, stem in targets:
        if val_col in cols:
            top = df.sort_values(val_col, ascending=False).head(topn)
            top.to_csv(os.path.join(opt_dir, f"{stem}.csv"), index=False)
            top.to_json(os.path.join(opt_dir, f"{stem}.json"), orient="records", indent=2)
            logging.info(f"[Optuna] saved {stem} (top {len(top)})")

def _save_pareto(study: "optuna.study.Study", opt_dir: str) -> None:
    """
    Save Pareto-front trials (numbers, values, params) into pareto_trials.json.
    """
    best_trials = [t for t in study.best_trials if t.values is not None]
    payload = [{"number": t.number, "values": t.values, "params": t.params} for t in best_trials]
    with open(os.path.join(opt_dir, "pareto_trials.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logging.info(f"[Optuna] saved pareto_trials.json ({len(payload)} trials)")

def _save_param_importances(study: "optuna.study.Study", opt_dir: str) -> None:
    """
    Save parameter importances for PnL objective (values[0]) if available.
    """
    try:
        from optuna.importance import get_param_importances
        imp = get_param_importances(study, target=lambda t: t.values[0])
        with open(os.path.join(opt_dir, "param_importances_pnl.json"), "w", encoding="utf-8") as f:
            json.dump(imp, f, ensure_ascii=False, indent=2)
        logging.info("[Optuna] saved param_importances_pnl.json")
    except Exception as e:
        logging.warning(f"[Optuna] param importances unavailable: {e}")

def _save_system_info(opt_dir: str, run_stamp: str) -> None:
    """
    Save environment metadata for reproducibility.
    """
    info = {
        "run_stamp_utc": run_stamp,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "optuna": getattr(optuna, "__version__", None),
        "pandas": getattr(pd, "__version__", None),
    }
    with open(os.path.join(opt_dir, "system_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    logging.info("[Optuna] saved system_info.json")

def objective(trial: optuna.Trial):
    # Reconstruct the config object from the JSON stored in user_attrs
    base_cfg_raw = trial.study.user_attrs["base_cfg"]
    base_cfg_dict = json.loads(base_cfg_raw) if isinstance(base_cfg_raw, str) else base_cfg_raw
    cfg = MasterConfig.model_validate(base_cfg_dict)
    cfg.random_seed = 17 + trial.number

    # --- NEW: Create a unique cache directory for each trial to prevent race conditions ---
    trial_cache_dir = os.path.join(trial.study.user_attrs["opt_dir"], "trial_caches", f"trial_{trial.number}")
    os.makedirs(trial_cache_dir, exist_ok=True)
    # set trial-specific cache dir only if model supports it
    if hasattr(cfg.paths, "extra_cache_dir"):
        cfg.paths.extra_cache_dir = trial_cache_dir
    else:
        trial.set_user_attr("extra_cache_dir", trial_cache_dir)

    # SEARCH SPACE
    cfg.backtest.long_action_threshold = trial.suggest_float("long_thr", 0.001, 0.03, log=True)
    cfg.backtest.short_action_threshold = trial.suggest_float("short_thr", 0.001, 0.03, log=True)
    
    # risk-management knobs
    cfg.backtest.use_risk_management = trial.suggest_categorical("use_rm", [True, False])
    if cfg.backtest.use_risk_management:
        cfg.backtest.stop_loss = trial.suggest_float("stop_loss", 0.005, 0.03)
        cfg.backtest.take_profit = trial.suggest_float("take_profit", 0.01, 0.05)
        cfg.backtest.trailing_stop = trial.suggest_float("trail", 0.001, 0.02)
    else:
        cfg.backtest.stop_loss = 0.0
        cfg.backtest.take_profit = 0.0
        cfg.backtest.trailing_stop = 0.0

    if cfg.backtest.selection_strategy == "ensemble_q_filter":
        cfg.backtest.ensemble_max_sigma = trial.suggest_float("max_sigma", 0.001, 0.015, log=True)

    cfg.paths.config_name = f"{cfg.paths.config_name}_trial{trial.number:05d}"
    # The base output dir for the whole optimization process is already set.
    # The run_papertrade function will handle trial-specific subdirectories.

    # for faster runs: skip plotting and example caching
    cfg.data.plot_examples = 0
    cfg.backtest.plot_backtest_balance_curve = False

    t0 = time.time()
    # Изменено: вызываем run_papertrade вместо run_backtest
    metrics = run_papertrade(cfg=cfg, trial_num=trial.number)
    duration_s = time.time() - t0
    
    # Persist useful attrs for later analysis/audit
    trial.set_user_attr("duration_s", round(duration_s, 3))
    trial.set_user_attr("random_seed", cfg.random_seed)
    trial.set_user_attr("trial_cache_dir", trial_cache_dir)
    for k, v in metrics.items():
        trial.set_user_attr(k, v)

    # TARGET METRICS
    total_pnl = float(metrics.get("final_balance_change", "0.0%").rstrip("%"))
    accuracy = float(metrics.get("accuracy", "0.0%").rstrip("%"))
    num_trades = int(metrics.get("total_trades", 0))
    
    # Optuna пытается максимизировать, поэтому для минимизации количества сделок мы возвращаем отрицательное значение
    return total_pnl, accuracy, -num_trades

def main():
    parser = argparse.ArgumentParser(description="Optimise PaperTrader parameters using historical DB data.")
    parser.add_argument("cfg_path", type=str, help="Path to experiment *.py config")
    parser.add_argument("--trials", type=int, default=100, help="Total Optuna trials")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs. WARNING: High values can lead to race conditions or high memory usage.")
    parser.add_argument("--topn", type=int, default=20, help="Top-N rows to save in summary tables")
    args = parser.parse_args()

    base_cfg = load_config(args.cfg_path)
    
    # --- Важно: Убедитесь, что конфигурация настроена для работы с базой данных ---
    if not hasattr(base_cfg, 'paper') or base_cfg.paper.source != "database":
        logging.error("Config error: `cfg.paper.source` must be set to 'database' for this optimization script.")
        return
        
    run_stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")

    session_name = f"optuna_papertrader_{run_stamp}"
    opt_dir = os.path.join(base_cfg.paths.output_dir, session_name)
    os.makedirs(opt_dir, exist_ok=True)
    
    # --- Важно: Перенаправляем основной путь вывода в директорию оптимизации ---
    base_cfg.paths.base_output_dir = opt_dir

    with open(os.path.join(opt_dir, "orig_master_cfg.json"), "w") as f:
        # Используем model_dump() для получения словаря и json.dumps с default=str для обработки несериализуемых типов
        f.write(json.dumps(base_cfg.model_dump(), default=str, indent=2))

    setup_logging(session_name=session_name, cfg=base_cfg)
    logging.info(f"[Optuna] Output dir: {opt_dir}")

    sampler = optuna.samplers.TPESampler(multivariate=True, warn_independent_sampling=False)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5, interval_steps=2)

    study = optuna.create_study(
        directions=["maximize", "maximize", "maximize"],  # pnl ↑,  accuracy ↑, -trades ↑
        sampler=sampler,
        pruner=pruner,
        study_name=f"papertrade_opt_{run_stamp}",
        storage=f"sqlite:///{os.path.join(opt_dir,'optuna.db')}",
        load_if_exists=False,
    )

    # Используем model_dump() без mode='json' и json.dumps с default=str
    # для надежной сериализации, включая torch.device
    config_dict = base_cfg.model_dump()
    study.set_user_attr("base_cfg", json.dumps(config_dict, default=str))
    study.set_user_attr("opt_dir", opt_dir)

    logging.info(f"[Optuna] starting optimisation -- trials={args.trials} jobs={args.jobs}")
    start_t = time.time()
    study.optimize(objective, n_trials=args.trials, n_jobs=args.jobs, show_progress_bar=True)
    logging.info(f"[Optuna] finished in {(time.time()-start_t)/60:.1f} min")

    df = study.trials_dataframe(attrs=("number", "values", "params", "user_attrs", "state"))
    _safe_save_df(df, opt_dir)
    _dump_trials_jsonl(study, opt_dir)
    _save_top_tables(df, opt_dir, topn=args.topn)
    _save_pareto(study, opt_dir)
    _save_param_importances(study, opt_dir)
    _save_system_info(opt_dir, run_stamp)

    best_trials = [t for t in study.best_trials if t.values is not None]
    if best_trials:
        best = best_trials[0]
        best_cfg_params = dict(best.params)
        with open(os.path.join(opt_dir, "best_papertrade_cfg.json"), "w") as f:
            json.dump(best_cfg_params, f, indent=2)

        logging.info(f"[Optuna] best trial #{best.number}: PnL={best.values[0]:.2f}%, Accuracy={best.values[1]:.2f}%, trades={-best.values[2] if best.values[2] is not None else 'N/A'}")
        logging.info(f"[Optuna] params: {best_cfg_params}")
    else:
        logging.warning("[Optuna] No successful trials found to determine the best parameters.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from optuna.visualization.matplotlib import plot_optimization_history, plot_pareto_front

        ax1 = plot_optimization_history(study, target=lambda t: t.values[0], target_name="Total PnL (%)")
        fig1 = getattr(ax1, "figure", ax1)
        fig1.savefig(os.path.join(opt_dir, "optuna_history.png"), dpi=300)
        plt.close(fig1)

        ax2 = plot_pareto_front(study, target_names=["PnL (%)", "Accuracy (%)", "-Trades"])
        fig2 = getattr(ax2, "figure", ax2)
        fig2.savefig(os.path.join(opt_dir, "pareto.png"), dpi=300)
        plt.close(fig2)

    except Exception as e:
        logging.warning(f"Failed to draw Optuna plots: {e}")

if __name__ == "__main__":
    main()