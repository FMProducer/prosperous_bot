TL;DR: ниже — итоговый патч к `third_party/rl-trading-binance/optimize_cfg.py`, который делает сохранение результатов Optuna максимально информативным и устойчивым: Parquet+CSV (fallback), JSONL всех триалов, топ-таблицы (PnL/Accuracy/−Trades), Pareto-набор, важности параметров, системная мета-инфа, длительность каждого триала и сид, кэш-директория триала. Структура артефактов и вызовы соответствуют README (Quickstart шаги 5–7) и требованиям проекта о сохранении артефактов в `output/<config_name>/`  .

---

## Шаг | Действие | KPI/риск

1 | Добавить Parquet→CSV fallback и дублирующее сохранение CSV | Экспорт никогда не падает без `pyarrow`; всегда есть читаемый CSV
2 | Сохранить `trials.jsonl` + топ-таблицы + Pareto + importances | Быстрый анализ качества/важности и аудит
3 | Логировать `duration_s`, `random_seed`, `trial_cache_dir` в `user_attrs` | Диагностика производительности и воспроизводимость
4 | `system_info.json` (версии, платформа) | Репродуцируемость окружения

---

## Unified diff (1 файл)

```diff
*** a/third_party/rl-trading-binance/optimize_cfg.py
--- b/third_party/rl-trading-binance/optimize_cfg.py
@@
 import argparse
 import copy
 import datetime as dt
 import json
 import logging
 import os
 import time
 
 import optuna
+import pandas as pd
+import platform
 
 from backtest_engine import run_backtest
 from config import MasterConfig
 from utils import load_config, setup_logging
 
+def _safe_save_df(df: "pd.DataFrame", opt_dir: str) -> None:
+    """
+    Save trials as Parquet (if engine available) and ALWAYS as CSV.
+    """
+    base = os.path.join(opt_dir, "trials")
+    csv_path = f"{base}.csv"
+    try:
+        pq_path = f"{base}.parquet"
+        df.to_parquet(pq_path, index=False)
+        logging.info(f"[Optuna] saved {pq_path}")
+    except Exception as e:
+        logging.warning(f"[Optuna] parquet save failed: {e}. CSV will be used.")
+    df.to_csv(csv_path, index=False)
+    logging.info(f"[Optuna] saved {csv_path}")
+
+def _dump_trials_jsonl(study: "optuna.study.Study", opt_dir: str) -> None:
+    """
+    Export all trials (number, state, values, params, user_attrs, timings) to JSONL.
+    """
+    path = os.path.join(opt_dir, "trials.jsonl")
+    with open(path, "w", encoding="utf-8") as f:
+        for t in study.get_trials(deepcopy=False):
+            rec = {
+                "number": t.number,
+                "state": str(t.state) if t.state is not None else None,
+                "values": t.values,
+                "params": t.params,
+                "user_attrs": t.user_attrs,
+                "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
+                "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else None,
+            }
+            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
+    logging.info(f"[Optuna] saved {path}")
+
+def _save_top_tables(df: "pd.DataFrame", opt_dir: str, topn: int = 20) -> None:
+    """
+    Save top-N trials by PnL (values_0), Accuracy (values_1), and -Trades (values_2).
+    """
+    cols = df.columns
+    targets = [("values_0", "top_by_pnl"), ("values_1", "top_by_accuracy"), ("values_2", "top_by_neg_trades")]
+    for val_col, stem in targets:
+        if val_col in cols:
+            top = df.sort_values(val_col, ascending=False).head(topn)
+            top.to_csv(os.path.join(opt_dir, f"{stem}.csv"), index=False)
+            top.to_json(os.path.join(opt_dir, f"{stem}.json"), orient="records", indent=2)
+            logging.info(f"[Optuna] saved {stem} (top {len(top)})")
+
+def _save_pareto(study: "optuna.study.Study", opt_dir: str) -> None:
+    """
+    Save Pareto-front trials (numbers, values, params) into pareto_trials.json.
+    """
+    best_trials = [t for t in study.best_trials if t.values is not None]
+    payload = [{"number": t.number, "values": t.values, "params": t.params} for t in best_trials]
+    with open(os.path.join(opt_dir, "pareto_trials.json"), "w", encoding="utf-8") as f:
+        json.dump(payload, f, ensure_ascii=False, indent=2)
+    logging.info(f"[Optuna] saved pareto_trials.json ({len(payload)} trials)")
+
+def _save_param_importances(study: "optuna.study.Study", opt_dir: str) -> None:
+    """
+    Save parameter importances for PnL objective (values[0]) if available.
+    """
+    try:
+        from optuna.importance import get_param_importances
+        imp = get_param_importances(study, target=lambda t: t.values[0])
+        with open(os.path.join(opt_dir, "param_importances_pnl.json"), "w", encoding="utf-8") as f:
+            json.dump(imp, f, ensure_ascii=False, indent=2)
+        logging.info("[Optuna] saved param_importances_pnl.json")
+    except Exception as e:
+        logging.warning(f"[Optuna] param importances unavailable: {e}")
+
+def _save_system_info(opt_dir: str, run_stamp: str) -> None:
+    """
+    Save environment metadata for reproducibility.
+    """
+    info = {
+        "run_stamp_utc": run_stamp,
+        "python": platform.python_version(),
+        "platform": platform.platform(),
+        "optuna": getattr(optuna, "__version__", None),
+        "pandas": getattr(pd, "__version__", None),
+    }
+    with open(os.path.join(opt_dir, "system_info.json"), "w", encoding="utf-8") as f:
+        json.dump(info, f, ensure_ascii=False, indent=2)
+    logging.info("[Optuna] saved system_info.json")
+
 def objective(trial: optuna.Trial):
     # Reconstruct the config object from the JSON stored in user_attrs
     base_cfg_raw = trial.study.user_attrs["base_cfg"]
     base_cfg_dict = json.loads(base_cfg_raw) if isinstance(base_cfg_raw, str) else base_cfg_raw
     cfg = MasterConfig.model_validate(base_cfg_dict)
     cfg.random_seed = 17 + trial.number
@@
     # for faster runs: skip plotting and example caching
     cfg.data.plot_examples = 0
     cfg.backtest.plot_backtest_balance_curve = False
     # cfg.debug.debug_max_size_data = None
 
-    metrics = run_backtest(cfg=cfg) # model_path_override is not needed for optimization
+    t0 = time.time()
+    metrics = run_backtest(cfg=cfg)  # model_path_override is not needed for optimization
+    duration_s = time.time() - t0
+    # Persist useful attrs for later analysis/audit
+    trial.set_user_attr("duration_s", round(duration_s, 3))
+    trial.set_user_attr("random_seed", cfg.random_seed)
+    trial.set_user_attr("trial_cache_dir", trial_cache_dir)
     for k, v in metrics.items():
         trial.set_user_attr(k, v)
 
     # TARGET METRICS
     total_pnl = float(metrics.get("final_balance_change", "0.0%").rstrip("%"))
@@
 def main():
     parser = argparse.ArgumentParser(description="Optimise BacktestConfig parameters")
     parser.add_argument("cfg_path", type=str, help="Path to experiment *.py config")
     parser.add_argument("--trials", type=int, default=200, help="Total Optuna trials")
     parser.add_argument("--jobs", type=int, default=4, help="Parallel jobs")
+    parser.add_argument("--topn", type=int, default=20, help="Top-N rows to save in summary tables")
     args = parser.parse_args()
 
     base_cfg = load_config(args.cfg_path)
     run_stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
 
     session_name = "optuna_cfg_optimization_results"
     opt_dir = os.path.join(base_cfg.paths.output_dir, session_name)
     os.makedirs(opt_dir, exist_ok=True)
@@
     sampler = optuna.samplers.TPESampler(multivariate=True, warn_independent_sampling=False)
     pruner = optuna.pruners.MedianPruner(n_warmup_steps=5, interval_steps=2)
 
     study = optuna.create_study(
         directions=["maximize", "maximize", "maximize"],  # pnl ↑,  accuracy ↑, -trades ↑
         sampler=sampler,
         pruner=pruner,
         study_name=f"backtest_opt_{run_stamp}",
         storage=f"sqlite:///{os.path.join(opt_dir,'optuna.db')}",
         load_if_exists=False,
     )
@@
     logging.info(f"[Optuna] starting optimisation -- trials={args.trials} jobs={args.jobs}")
     start_t = time.time()
     study.optimize(objective, n_trials=args.trials, n_jobs=args.jobs, show_progress_bar=True)
     logging.info(f"[Optuna] finished in {(time.time()-start_t)/60:.1f} min")
 
-    df = study.trials_dataframe(attrs=("number", "values", "params", "user_attrs", "state"))
-    df.to_parquet(os.path.join(opt_dir, "trials.parquet"), index=False)
+    df = study.trials_dataframe(attrs=("number", "values", "params", "user_attrs", "state"))
+    _safe_save_df(df, opt_dir)
+    _dump_trials_jsonl(study, opt_dir)
+    _save_top_tables(df, opt_dir, topn=args.topn)
+    _save_pareto(study, opt_dir)
+    _save_param_importances(study, opt_dir)
+    _save_system_info(opt_dir, run_stamp)
 
     # best of Pareto front (rank 0) -> take the first one
     best_trials = [t for t in study.best_trials if t.values is not None]
     if best_trials:
         best = best_trials[0]
         best_cfg = dict(best.params)
         with open(os.path.join(opt_dir, "best_backtest_cfg.json"), "w") as f:
             json.dump(best_cfg, f, indent=2)
@@
     except Exception as e:
         logging.warning(f"Failed to draw Optuna plots: {e}")
 
 if __name__ == "__main__":
     main()
```

**Обоснование по проектной доке:**
— Quickstart прямо предусматривает Optuna и отчёт по топ-триалам (шаги 5–7) — теперь артефакты богаче и стабильнее (CSV-fallback) .
— Артефакты сохраняются в `output/<config_name>/...`, как требует SYSTEM_PROMPT/README (репортинг и воспроизводимость)  .
— В `requirements.txt` нет `pyarrow/fastparquet`; fallback предотвращает падения при их отсутствии .

---

### Команды для применения и PR

```bash
git checkout -b feat/optuna-informative-exports
git apply --index changes.patch
git commit -m "feat(optuna): informative exports (CSV/Parquet fallback, JSONL, top tables, Pareto, importances, system info, per-trial duration/seed)"
git push -u origin feat/optuna-informative-exports
gh pr create -t "feat(optuna): informative and robust results saving" -b "Расширен экспорт: Parquet+CSV (fallback), trials.jsonl, top_by_* (CSV/JSON), pareto_trials.json, param_importances_pnl.json, system_info.json; добавлены duration_s/seed/trial_cache_dir. Соответствует README Quickstart и правилам артефактов output/<config_name>/. :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7}" -B prosperous_bot
```

### Быстрый smoke-test (CPU, 5 триалов)

```powershell
python third_party/rl-trading-binance/optimize_cfg.py third_party/rl-trading-binance/configs/alpha.py --trials 5 --jobs 1 --topn 5
```

Должны появиться:
`trials.parquet` + `trials.csv`, `trials.jsonl`, `top_by_*.csv/json`, `pareto_trials.json`, `param_importances_pnl.json`, `system_info.json`, `best_backtest_cfg.json`, а также PNG-графики истории/Парето (как и раньше) .
