#!/usr/bin/env python3
"""Send monitor report to Telegram via SOCKS5 proxy."""
import urllib.request
import urllib.parse
import json
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_DIR = Path(r"C:\Python\Prosperous_Bot\futures_portfolio")
load_dotenv(PROJECT_DIR / ".env")

token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
api_base = os.environ.get("TELEGRAM_API_BASE", "https://api.telegram.org")

# Build message from live state
paper_state_path = PROJECT_DIR / "paper_state_HEIUSDT.json"
real_state_path = PROJECT_DIR / "real_state_HEIUSDT.json"
paper_log_path = PROJECT_DIR / "logs" / "paper_HEIUSDT.log"

import time
now_str = time.strftime("%H:%M")

# Paper state
tpv = pnl = cycles = guard = "N/A"
if paper_state_path.exists():
    with open(paper_state_path) as f:
        ps = json.load(f)
    tpv = ps.get("last_tpv", 0)
    pnl = ps.get("last_profit", 0)
    cycles = ps.get("rebalance_cycles", 0)
    guard = "ON" if ps.get("pnl_guard_active") else "OFF"

# Real state
real_pnl_str = "N/A"
if real_state_path.exists():
    with open(real_state_path) as f:
        rs = json.load(f)
    real_pnl_str = f"{rs.get('last_profit', 0):.2f}"

# Alerts
alerts = []
# Check heartbeat gap
import os as _os
if paper_log_path.exists():
    st = _os.stat(paper_log_path)
    age_min = (time.time() - st.st_mtime) / 60
    if age_min > 5:
        mtime_str = time.strftime("%H:%M", time.localtime(st.st_mtime))
        alerts.append(f"Paper heartbeat STOPPED at {mtime_str} ({age_min:.0f} min gap)")

if not real_state_path.exists():
    alerts.append("No real state — real swarm is empty")

if guard == "ON":
    alerts.append("PnL GUARD is active")

alert_str = "\n".join(f"⚠️ {a}" for a in alerts) if alerts else "✅ No alerts"

msg = (
    f"🤖 Monitor {now_str} | Real PnL: {real_pnl_str} | Paper PnL: {pnl}\n"
    f"🔒 HEIUSDT: TPV={tpv} PnL={pnl} Cycles={cycles} GUARD {guard}\n"
    f"{alert_str}"
)

url = f"{api_base}/bot{token}/sendMessage"
data = urllib.parse.urlencode({
    "chat_id": chat_id,
    "text": msg,
    "parse_mode": "HTML"
}).encode("utf-8")

req = urllib.request.Request(url, data=data, method="POST")
handler = urllib.request.ProxyHandler({
    "https": "socks5://127.0.0.1:10808",
    "http": "socks5://127.0.0.1:10808"
})
opener = urllib.request.build_opener(handler)
resp = opener.open(req, timeout=20)
result = json.loads(resp.read().decode())
print(json.dumps(result, indent=2))
