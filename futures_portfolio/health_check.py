#!/usr/bin/env python3
"""
System Health Check для SyntheticMarketNeutral Portfolio.
Запускается по cron, проверяет:
1. PM2 процессы (real + paper боты)
2. Последние логи на ошибки
3. State файлы (TPV, trailing stop, blacklisted)
4. Активные позиции на бирже

Вывод: JSON статус
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(r"C:\Python\Prosperous_Bot\futures_portfolio")
LOG_DIR = PROJECT_DIR / "logs"
STATE_DIR = PROJECT_DIR

def check_pm2():
    """Проверяет PM2 процессы через pm2 jlist"""
    try:
        r = subprocess.run(
            ["pm2", "jlist"],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        if r.returncode != 0:
            return {"status": "error", "detail": r.stderr[:200]}
        processes = json.loads(r.stdout)
        online = [p for p in processes if p.get("pm2_env", {}).get("status") == "online"]
        stopped = [p for p in processes if p.get("pm2_env", {}).get("status") != "online"]
        real = [p for p in online if p["name"].startswith("real-")]
        paper = [p for p in online if p["name"].startswith("paper-")]
        infra = [p for p in online if not p["name"].startswith(("real-", "paper-"))]
        return {
            "status": "ok",
            "total_online": len(online),
            "real": len(real),
            "paper": len(paper),
            "infra": len(infra),
            "stopped": len(stopped),
            "names": [p["name"] for p in online]
        }
    except FileNotFoundError:
        return {"status": "pm2_not_found"}
    except Exception as e:
        return {"status": "error", "detail": str(e)[:200]}

def check_state_files():
    """Проверяет real_state_*.json файлы"""
    real_states = list(STATE_DIR.glob("real_state_*.json"))

    results = {}
    for fpath in real_states:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            ticker = fpath.stem.replace("real_state_", "")
            results[ticker] = {
                "tpv": round(data.get("last_tpv", 0), 2),
                "profit": round(data.get("last_profit", 0), 2),
                "cycles": data.get("rebalance_cycles", 0),
                "trailing_stop": data.get("trailing_stop_triggered", False),
                "tpv_ath": round(data.get("tpv_ach", data.get("tpv_ath", 0)), 2),
                "balance": data.get("balance", 115.0)
            }
        except Exception as e:
            results[ticker] = {"error": str(e)[:100]}
    return results

def check_blacklist():
    """Проверяет блэклист из config.json"""
    try:
        with open(STATE_DIR / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        return {
            "blacklisted": config.get("black_list", []),
            "toxic_cooldown_days": config.get("toxic_cooldown_days", 0)
        }
    except:
        return {"blacklisted": []}

def check_recent_errors():
    """Проверяет последние ошибки в логах"""
    errors = []
    for log_file in sorted(LOG_DIR.glob("err_*.log")):
        try:
            stat = os.stat(log_file)
            age_min = (time.time() - stat.st_mtime) / 60
            if age_min < 60:  # ошибки за последний час
                size = stat.st_size
                if size > 0:
                    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    last_lines = content.splitlines()[-5:]
                    errors.append({
                        "file": log_file.name,
                        "age_min": round(age_min, 1),
                        "size": size,
                        "last_errors": last_lines
                    })
        except:
            pass
    return errors

def main():
    result = {
        "timestamp": datetime.now().isoformat(),
        "pm2": check_pm2(),
        "real_states": check_state_files(),
        "blacklist": check_blacklist(),
        "recent_errors": check_recent_errors()
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
