#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Freeze candidate snapshot for RL Trader.

Creates a reproducible snapshot under:
  output/<candidate_id>/{artifacts,reports,manifest.json}

Actions:
  1) Copy best.pth from: output/<config_name>/saved_models/<run_id>/best.pth
  2) Copy config snapshot from: configs/<config_name>.py
  3) Parse logs to extract seed and [Final Metrics] block
  4) Save metrics as TXT/JSON/CSV
  5) Compute SHA-256 for artifacts
  6) Build manifest.json with Repo-State
  7) Render metrics_summary.png from metrics JSON

Notes:
  - Configuration must come from configs/*.py (project rule).
  - All outputs live under output/ (project rule).
  - Designed to be OS-agnostic; tested on Windows paths.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from datetime import datetime

# matplotlib is in requirements.txt
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../third_party/rl-trading-binance

def build_run_id_from_ts(ts: str) -> str:
    """
    Convert ISO-like timestamp (UTC/local) 'YYYY-MM-DDTHH:MM:SS' into
    saved_models run-id pattern:
      rl_binance_futures_trading_date_YYYYMMDD_time_HHMMSS
    """
    dt = datetime.fromisoformat(ts)
    return f"rl_binance_futures_trading_date_{{dt:%Y%m%d}}_time_{{dt:%H%M%S}}"


def find_best_pth(config_name: str, run_id: str) -> Path:
    p = PROJECT_ROOT / "output" / config_name / "saved_models" / run_id / "best.pth"
    if not p.exists():
        raise FileNotFoundError(f"best.pth not found: {p}")
    return p


def ensure_dirs(dst_root: Path) -> tuple[Path, Path]:
    artifacts = dst_root / "artifacts"
    reports = dst_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    return artifacts, reports


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = src.read_bytes()
    dst.write_bytes(data)


def sha256sum(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


SEED_REGEX = re.compile(r"Random seed set to\s+(\d+)")


def parse_seed_from_log(text: str) -> str | None:
    m = SEED_REGEX.search(text)
    return m.group(1) if m else None


def extract_final_metrics_block(text: str) -> list[str]:
    """
    Extract lines after a line starting with "[Final Metrics]:" until a blank line.
    Supports logs where each subsequent line contains key/value.
    """
    lines = text.splitlines()
    block: list[str] = []
    collecting = False
    for line in lines:
        if line.strip().startswith("[Final Metrics]:"):
            collecting = True
            # Handle case where first metric is on the same line
            if ":" in line or "=" in line:
                first_metric_line = line.split("[Final Metrics]:", 1)[1].strip()
                if first_metric_line:
                    block.append(first_metric_line)
            continue
        if collecting:
            if not line.strip():
                break
            block.append(line)
    if not block:
        raise ValueError("Final Metrics block not found or is empty in log.")
    return block

KV_PATTERNS = [
    re.compile(r"^\s*(?:.+?\[INFO\]\s*:\s*)?([A-Za-z0-9_ ]+?)\s*=\s*(.+?)\s*$"),
    re.compile(r"^\s*(?:.+?\[INFO\]\s*:\s*)?([A-Za-z0-9_ ]+?)\s*:\s*(.+?)\s*$"),
]

def parse_metrics_kv(block_lines: list[str]) -> dict:
    """
    Parse key/value pairs from block lines. Accepts "key = value" and "key: value".
    Keys are normalized with underscores.
    """
    kv: dict[str, str] = {}
    for raw in block_lines:
        line = raw.strip()
        if not line:
            continue
        for pat in KV_PATTERNS:
            m = pat.match(line)
            if m:
                k, v = m.group(1).strip(), m.group(2).strip()
                k = re.sub(r"\s+", "_", k.lower())
                kv[k] = v
                break
    if not kv:
        raise ValueError("No key/value pairs parsed from Final Metrics block.")
    return kv


def write_text(p: Path, lines: list[str]) -> None:
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(p: Path, obj: dict) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(p: Path, kv: dict) -> None:
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        for k in sorted(kv.keys()):
            w.writerow([k, kv[k]])


def render_metrics_png(json_path: Path, out_png: Path, title: str) -> None:
    m = json.loads(json_path.read_text(encoding="utf-8"))
    keys = [
        "final_balance_change", "sharpe", "sortino", "max_drawdown", "accuracy",
        "total_trades", "avg_trade_amount", "trades_per_day", "total_commission", "seed"
    ]
    rows = [(k, m.get(k, "")) for k in keys]
    fig = plt.figure(figsize=(6, 6), dpi=160)
    plt.axis("off")
    tbl = plt.table(cellText=[(k, str(v)) for k, v in rows],
                    colLabels=["Metric", "Value"], loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.2)
    plt.title(title, pad=12)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")


def find_log_with_final_metrics(log_root: Path, ts: datetime | None) -> Path:
    candidates = []
    for p in log_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".log", ".txt"}:
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if "[Final Metrics]:" in txt:
                mtime = datetime.fromtimestamp(p.stat().st_mtime)
                time_diff = abs((mtime - (ts or mtime)).total_seconds())
                candidates.append((p, time_diff))
    if not candidates:
        raise FileNotFoundError(f"No log with '[Final Metrics]:' found under {log_root}")
    candidates.sort(key=lambda t: t[1])
    return candidates[0][0]


def safe_git(cmd: list[str]) -> str | None:
    try:
        import subprocess
        out = subprocess.check_output(cmd, cwd=PROJECT_ROOT, shell=False)
        return out.decode("utf-8", errors="ignore").strip()
    except Exception:
        return None


def collect_repo_state(args) -> dict:
    branch = args.repo_branch or safe_git(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "prosperous_bot"
    sha = args.repo_sha or safe_git(["git", "rev-parse", "HEAD"]) or ""
    title = args.repo_title or (safe_git(["git", "show", "-s", "--format=%s", sha]) if sha else "") or ""
    url = args.repo_url or (f"https://github.com/FMProducer/prosperous_bot/commit/{sha}" if sha else "")
    return {
        "branch": branch,
        "sha": sha,
        "title": title,
        "url": url,
    }

def main():
    ap = argparse.ArgumentParser(description="Freeze RL Trader candidate snapshot")
    ap.add_argument("--config-name", required=False, help="Config name (e.g., alpha). Used if paths are not provided.")
    gid = ap.add_mutually_exclusive_group(required=False)
    gid.add_argument("--run-id", help="Exact saved_models run id. Used if paths are not provided.")
    gid.add_argument("--run-timestamp", help="Timestamp YYYY-MM-DDTHH:MM:SS to derive run-id. Used if paths are not provided.")
    ap.add_argument("--candidate-id", default="candidate_prod_v1", help="Snapshot id under output/")
    ap.add_argument("--repo-branch", default=None)
    ap.add_argument("--repo-sha", default=None)
    ap.add_argument("--repo-url", default=None)
    ap.add_argument("--repo-title", default=None)
    ap.add_argument("--model-path", default=None, help="Absolute path to the model file (e.g., best.pth)")
    ap.add_argument("--config-path", default=None, help="Absolute path to the config file")
    ap.add_argument("--log-path", default=None, help="Absolute path to the log file")

    args = ap.parse_args()

    run_id = "custom_run"
    config_name = "custom_config"

    if args.model_path and args.config_path and args.log_path:
        best_pth = Path(args.model_path)
        cfg_src = Path(args.config_path)
        log_file = Path(args.log_path)
        config_name = cfg_src.stem
        if args.run_id:
            run_id = args.run_id
    else:
        if not args.config_name or not (args.run_id or args.run_timestamp):
             ap.error("Either all --*-path arguments or --config-name and (--run-id or --run-timestamp) must be provided.")
        config_name = args.config_name
        run_id = args.run_id or build_run_id_from_ts(args.run_timestamp)
        cfg_src = PROJECT_ROOT / "configs" / f"{config_name}.py"
        best_pth = find_best_pth(config_name, run_id)
        log_root = PROJECT_ROOT / "output" / config_name / "logs"
        ts = None
        if args.run_timestamp:
            ts = datetime.fromisoformat(args.run_timestamp)
        log_file = find_log_with_final_metrics(log_root, ts)

    if not cfg_src.exists():
        raise FileNotFoundError(f"Config not found: {cfg_src}")
    if not best_pth.exists():
        raise FileNotFoundError(f"Model not found: {best_pth}")
    if not log_file.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")

    dst_root = PROJECT_ROOT / "output" / args.candidate_id
    artifacts_dir, reports_dir = ensure_dirs(dst_root)

    pth_dst = artifacts_dir / "best.pth"
    copy_file(best_pth, pth_dst)
    cfg_dst = artifacts_dir / cfg_src.name
    copy_file(cfg_src, cfg_dst)

    log_text = log_file.read_text(encoding="utf-8", errors="ignore")
    seed = parse_seed_from_log(log_text) or ""
    block = extract_final_metrics_block(log_text)

    base_filename = f"final_metrics_{args.candidate_id}"
    metrics_txt = reports_dir / f"{base_filename}.txt"
    write_text(metrics_txt, ["[Final Metrics]:"] + block)

    # Parse KV and enrich
    kv = parse_metrics_kv(block)
    kv["seed"] = seed
    kv["run_id"] = run_id
    kv["config_name"] = config_name
    if args.run_timestamp:
        kv["timestamp_utc"] = (args.run_timestamp + "Z")

    # Save JSON/CSV
    metrics_json = reports_dir / f"{base_filename}.json"
    metrics_csv = reports_dir / f"{base_filename}.csv"
    write_json(metrics_json, kv)
    write_csv(metrics_csv, kv)

    sha_pth = sha256sum(pth_dst)
    sha_cfg = sha256sum(cfg_dst)

    repo_state = collect_repo_state(args)

    manifest = {
        "candidate_id": args.candidate_id,
        "run_id": run_id,
        "config_name": config_name,
        "seed": seed,
        "repo_state": repo_state,
        "artifacts": {
            "best_pth": str(pth_dst.as_posix()),
            "best_pth_sha256": sha_pth,
            "config_snapshot": str(cfg_dst.as_posix()),
            "config_snapshot_sha256": sha_cfg,
            "metrics_json": str(metrics_json.as_posix()),
            "metrics_csv": str(metrics_csv.as_posix()),
            "metrics_txt": str(metrics_txt.as_posix()),
        },
        "timestamp_utc": kv.get("timestamp_utc", ""),
    }
    (dst_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    summary_png = reports_dir / "metrics_summary.png"
    title = f'{args.candidate_id} — Final Metrics ({kv.get("timestamp_utc","") or run_id})'
    render_metrics_png(metrics_json, summary_png, title)

    print(f"[OK] Snapshot created at: {dst_root}")
    print(f"      Artifacts: {artifacts_dir}")
    print(f"      Reports  : {reports_dir}")
    print(f"      Manifest : {(dst_root / 'manifest.json')}")


if __name__ == "__main__":
    sys.exit(main())
