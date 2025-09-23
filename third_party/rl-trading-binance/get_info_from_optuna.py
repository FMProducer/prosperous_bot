"""
Show and save top-N Optuna trials for a given config.

Usage:
  python get_info_from_optuna.py configs/alpha.py --n-best-trials 10
  python get_info_from_optuna.py configs/alpha.py --n-best-trials 15 --metric values_0 --direction max
"""

from pathlib import Path
import argparse
import sys
import pandas as pd


def read_parquet_safely(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)  # tries default engine
    except Exception as e:
        # explicit fallback attempts with hints
        try:
            return pd.read_parquet(path, engine="pyarrow")
        except Exception:
            try:
                return pd.read_parquet(path, engine="fastparquet")
            except Exception:
                raise RuntimeError(
                    f"Failed to read {path}. "
                    f"Install a parquet engine: pip install pyarrow OR fastparquet. "
                    f"Original error: {e}"
                )


def main():
    ap = argparse.ArgumentParser(description="Print and save top-N Optuna trials for a given config.")
    ap.add_argument("config_path", help="Path to config, e.g. configs/alpha.py")
    ap.add_argument("--n-best-trials", type=int, default=10, help="How many best trials to show/save")
    ap.add_argument("--metric", default=None, help="Metric column to sort by (default: auto-detect values_0/value)")
    ap.add_argument("--direction", choices=["max", "min"], default="max", help="Optimization direction for metric")
    ap.add_argument("--out-dir", default=None, help="Override output dir (by default inferred from config name)")
    args = ap.parse_args()

    if args.n_best_trials <= 0:
        print("n-best-trials must be positive.", file=sys.stderr)
        sys.exit(2)

    cfg_name = Path(args.config_path).stem
    base_dir = Path(args.out_dir) if args.out_dir else Path("output") / cfg_name / "optuna_cfg_optimization_results"
    trials_path = base_dir / "trials.parquet"

    if not trials_path.exists():
        print(f"Not found: {trials_path}", file=sys.stderr)
        sys.exit(1)

    df = read_parquet_safely(trials_path)

    # Detect metric column
    metric_col = args.metric
    if metric_col is None:
        value_cols = [c for c in df.columns if c.startswith("values_")]
        if len(value_cols) >= 1:
            metric_col = sorted(value_cols)[0]  # usually 'values_0'
        elif "value" in df.columns:
            metric_col = "value"
        else:
            print("Cannot auto-detect metric column. Use --metric (e.g., --metric values_0).", file=sys.stderr)
            sys.exit(3)

    if metric_col not in df.columns:
        print(f"Metric column '{metric_col}' not found in DataFrame columns.", file=sys.stderr)
        print(f"Available columns example: {list(df.columns)[:20]} ...", file=sys.stderr)
        sys.exit(3)

    # Filter only completed trials if 'state' exists
    if "state" in df.columns:
        df = df[df["state"] == "COMPLETE"].copy()

    if df.empty:
        print("No completed trials found.", file=sys.stderr)
        sys.exit(0)

    ascending = (args.direction == "min")
    try:
        top = df.sort_values(by=metric_col, ascending=ascending).head(args.n_best_trials)
    except Exception as e:
        print(f"Failed to sort by {metric_col}: {e}", file=sys.stderr)
        sys.exit(3)

    # Select columns: trial number, metric, params_*
    cols = []
    for candidate in ("number", "trial_id", "trial_number"):
        if candidate in top.columns:
            cols.append(candidate)
            break
    cols.append(metric_col)
    param_cols = [c for c in top.columns if c.startswith("params_")]
    cols.extend(sorted(param_cols))

    # Print to stdout
    print(f"\nConfig: {cfg_name}")
    print(f"Trials file: {trials_path}")
    print(f"Direction: {'maximize' if not ascending else 'minimize'} | Metric: {metric_col}")
    print(top[cols].to_string(index=False))

    # Save to files
    txt_path = base_dir / f"best_{args.n_best_trials}_trials_{metric_col}.txt"
    csv_path = base_dir / f"best_{args.n_best_trials}_trials_{metric_col}.csv"
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(top[cols].to_string(index=False))
            f.write("\n")
        top[cols].to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Warning: failed to save outputs: {e}", file=sys.stderr)

    print(f"\nSaved: {txt_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()

