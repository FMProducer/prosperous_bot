# optimize_cfg.py
import argparse
import copy
import datetime as dt
import json
import logging
import os
import time
import numpy as np

import optuna
import pandas as pd
import platform

from validate_test import run_validation_with_config
from config import MasterConfig, cfg as default_cfg
from utils import load_config, setup_logging



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
    Save top-N trials by:
      - Sharpe (values_0),
      - Sortino (values_1),
      - Profit Factor (values_2).
    """
    cols = df.columns
    targets = [
        ("values_0", "top_by_sharpe"),
        ("values_1", "top_by_sortino"),
        ("values_2", "top_by_profit_factor"),
    ]
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
    Save parameter importances for multi-objective study:
      - Sharpe (values[0])
      - Sortino (values[1])
    """
    try:
        from optuna.importance import get_param_importances
        # Sharpe
        imp_sharpe = get_param_importances(study, target=lambda t: t.values[0])
        with open(os.path.join(opt_dir, "param_importances_sharpe.json"), "w", encoding="utf-8") as f:
            json.dump(imp_sharpe, f, ensure_ascii=False, indent=2)
        logging.info("[Optuna] saved param_importances_sharpe.json")
        # Sortino
        imp_sortino = get_param_importances(study, target=lambda t: t.values[1])
        with open(os.path.join(opt_dir, "param_importances_sortino.json"), "w", encoding="utf-8") as f:
            json.dump(imp_sortino, f, ensure_ascii=False, indent=2)
        logging.info("[Optuna] saved param_importances_sortino.json")
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

def objective(trial: optuna.Trial, base_cfg: MasterConfig, cached_signals: dict = None) -> tuple[float, float, float]:
    # Reconstruct the config object from the JSON stored in user_attrs
    cfg = copy.deepcopy(base_cfg)
    cfg.random_seed = 17 + trial.number

    # --- NEW: Create a unique cache directory for each trial to prevent race conditions ---
    trial_cache_dir = os.path.join(trial.study.user_attrs["opt_dir"], "trial_caches", f"trial_{trial.number}")
    os.makedirs(trial_cache_dir, exist_ok=True)
    # set trial-specific cache dir only if model supports it
    if hasattr(cfg.paths, "extra_cache_dir"):
        cfg.paths.extra_cache_dir = trial_cache_dir
    else:
        trial.set_user_attr("extra_cache_dir", trial_cache_dir)

    # --- TSL Search Space ---
    d_min = trial.suggest_float("d_min", 0.001, 0.005, log=True)
    # d0's lower bound is dynamically set by the value of d_min
    d0 = trial.suggest_float("d0", d_min, 0.02, log=True)
    delta_p_hyst = trial.suggest_float("delta_p_hyst", 0.0005, 0.005, log=True)

    # Inject TSL parameters into the config for backtesting
    cfg.backtest.trailing_stop_min = d_min
    cfg.backtest.trailing_stop = d0
    cfg.backtest.delta_p_hysteresis = delta_p_hyst

    # --- DYNAMIC SEARCH SPACE FROM CONFIG ---
    search_space = getattr(cfg, "optuna_search_space", None)
    if not search_space:
        raw_ss = trial.study.user_attrs.get("optuna_search_space")
        search_space = json.loads(raw_ss) if isinstance(raw_ss, str) else (raw_ss or {})
    suggested_params = {}

    for name, params in search_space.items():
        suggest_type, low, high, log_flag, path = params

        if isinstance(low, str) and low in suggested_params:
            low = suggested_params[low]

        if suggest_type in ("suggest_float", "suggest_int"):
            if low is None or high is None:
                raise ValueError(f"[Optuna] Param '{name}': low/high must be set for {suggest_type}")
            if float(high) < float(low):
                raise ValueError(f"[Optuna] Param '{name}': high({high}) < low({low})")
            if suggest_type == "suggest_float" and log_flag and float(low) <= 0.0:
                raise ValueError(f"[Optuna] Param '{name}': log-scale requires low>0 (got {low})")

        if suggest_type == "suggest_float":
            if log_flag and (float(low) < 0 or float(high) < 0):
                pos_val = trial.suggest_float(name, abs(float(high)), abs(float(low)), log=True)
                value = -pos_val
            else:
                value = trial.suggest_float(name, float(low), float(high), log=bool(log_flag))
        elif suggest_type == "suggest_int":
            value = trial.suggest_int(name, int(low), int(high), log=bool(log_flag))
        elif suggest_type == "suggest_categorical":
            if not isinstance(low, (list, tuple)):
                raise ValueError(f"[Optuna] Param '{name}': categorical choices must be list/tuple")
            value = trial.suggest_categorical(name, list(low))
        else:
            raise ValueError(f"[Optuna] Unknown suggest_type '{suggest_type}' for param '{name}'")

        suggested_params[name] = value

        parts = path.split('.')
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    logging.info(f"[Optuna] trial#{trial.number} params: {json.dumps(suggested_params, ensure_ascii=False)}")

    cfg.backtest.use_risk_management = True
    cfg.backtest.stop_loss = None
    cfg.backtest.take_profit = None

    trial_output_dir = os.path.join(trial.study.user_attrs["opt_dir"], f"trial_{trial.number}")
    os.makedirs(trial_output_dir, exist_ok=True)

    cfg.paths.base_output_dir = trial_output_dir
    cfg.paths.config_name = f"{cfg.paths.config_name}_trial{trial.number:05d}"

    t0 = time.time()
    try:
        if cached_signals:
            metrics = run_validation_with_config(cfg, action_signals=cached_signals)
        else:
            metrics = run_validation_with_config(cfg)

    except Exception as e:
        logging.exception(f"[Optuna] trial#{trial.number} validation failed")
        metrics = {"Validation_sharpe": -1.0, "Validation_sortino": -1.0, "Validation_profit_factor": 0.0}
    
    logging.info("=" * 80)
    logging.info(f"[Optuna Trial #{trial.number}] VALIDATION RESULTS")
    logging.info("=" * 80)
    
    sharpe = metrics.get("Validation_sharpe", 0.0)
    sortino = metrics.get("Validation_sortino", 0.0)
    pf = metrics.get("Validation_profit_factor", 0.0)
    maxdd = metrics.get("Validation_max_drawdown", 0.0)
    winrate = metrics.get("Validation_win_rate", 0.0)
    trades = metrics.get("Validation_total_trades", 0)
    netpnl = metrics.get("Validation_net_pnl", 0.0)
    
    logging.info(f"[Optuna] Sharpe: {sharpe:.4f} | Sortino: {sortino:.4f} | PF: {pf:.4f}")
    logging.info(f"[Optuna] MaxDD: {abs(maxdd):.2%} | WinRate: {winrate:.2%} | Trades: {trades}")
    logging.info(f"[Optuna] Net PnL: {netpnl:.2f} USDT")
    
    logging.info("=" * 80)

    duration_s = time.time() - t0
    
    trial.set_user_attr("duration_s", round(duration_s, 3))
    trial.set_user_attr("random_seed", cfg.random_seed)
    trial.set_user_attr("trial_cache_dir", trial_cache_dir)
    for k, v in metrics.items():
        trial.set_user_attr(k, v)

    try:
        with open(os.path.join(trial_output_dir, "trial_params.json"), "w", encoding="utf-8") as f:
            json.dump(suggested_params, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"[Optuna] cannot save trial_params.json: {e}")

    return sharpe, sortino, pf

def main():
    parser = argparse.ArgumentParser(description="Optimise PaperTrader parameters using historical DB data.")
    parser.add_argument("cfg_path", type=str, help="Path to experiment *.py config")
    parser.add_argument("--trials", type=int, default=100, help="Total Optuna trials")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs. WARNING: High values can lead to race conditions or high memory usage.")
    parser.add_argument("--topn", type=int, default=20, help="Top-N rows to save in summary tables")
    args = parser.parse_args()

    base_cfg = load_config(args.cfg_path)
    
    run_stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")

    session_name = f"optuna_papertrader_{run_stamp}"
    opt_dir = os.path.join(base_cfg.paths.output_dir, session_name)
    os.makedirs(opt_dir, exist_ok=True)
    
    base_cfg.paths.base_output_dir = opt_dir

    with open(os.path.join(opt_dir, "orig_master_cfg.json"), "w") as f:
        f.write(json.dumps(base_cfg.model_dump(), default=str, indent=2))

    setup_logging(session_name=session_name, cfg=base_cfg)
    logging.info(f"[Optuna] Output dir: {opt_dir}")

    sampler = optuna.samplers.TPESampler(multivariate=True, warn_independent_sampling=False)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5, interval_steps=2)

    study = optuna.create_study(
        directions=["maximize", "maximize", "maximize"],
        sampler=sampler,
        pruner=pruner,
        study_name=f"papertrade_opt_{run_stamp}",
        storage=f"sqlite:///{os.path.join(opt_dir,'optuna.db')}",
        load_if_exists=False,
    )

    study.set_user_attr("config_path", args.cfg_path)

    config_dict = base_cfg.model_dump()
    study.set_user_attr("base_cfg", json.dumps(config_dict, default=str))
    ss = getattr(base_cfg, "optuna_search_space", {})
    study.set_user_attr("optuna_search_space", json.dumps(ss, default=str))
    study.set_user_attr("opt_dir", opt_dir)

    logging.info("[Optuna] Pre-calculating model signals (Inference Cache) for validation set...")
    initial_run = run_validation_with_config(base_cfg, save_signals=True)
    cached_signals = initial_run.get("signals")
    if not cached_signals:
        logging.error("[Optuna] Failed to pre-calculate signals. Aborting optimization.")
        return

    logging.info(f"[Optuna] starting optimisation -- trials={args.trials} jobs={args.jobs}")
    start_t = time.time()
    study.optimize(lambda t: objective(t, base_cfg, cached_signals=cached_signals), n_trials=args.trials, n_jobs=args.jobs, show_progress_bar=True)
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

        logging.info(f"[Optuna] best trial #{best.number}: Sharpe={best.values[0]:.2f}, Sortino={best.values[1]:.2f}, ProfitFactor={best.values[2]:.2f}")
        logging.info(f"[Optuna] Best trial params: {best_cfg_params}")
    else:
        logging.warning("[Optuna] No successful trials found to determine the best parameters.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from optuna.visualization.matplotlib import plot_optimization_history, plot_pareto_front

        ax1 = plot_optimization_history(study, target=lambda t: t.values[0], target_name="Sharpe Ratio")
        fig1 = getattr(ax1, "figure", ax1)
        fig1.savefig(os.path.join(opt_dir, "optuna_history.png"), dpi=300)
        plt.close(fig1)

        ax2 = plot_pareto_front(study, target_names=["Sharpe", "Sortino", "Profit Factor"])
        fig2 = getattr(ax2, "figure", ax2)
        fig2.savefig(os.path.join(opt_dir, "pareto.png"), dpi=300)
        plt.close(fig2)

    except Exception as e:
        logging.warning(f"Failed to draw Optuna plots: {e}")

if __name__ == "__main__":
    main()