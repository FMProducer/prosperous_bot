#!/usr/bin/env python3
"""Health check watchdog for Prosperous Bot Dashboard.

Usage:
  python health_watchdog.py              # Check once
  python health_watchdog.py --daemon     # Continuous loop (for cron)

Exit codes:
  0 - Healthy (200 OK)
  1 - Degraded (503) - logs to stdout
  2 - No response - alert
"""

import json
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

# Config - same as dashboard
HOST = "127.0.0.1"
PORT = 8080
TIMEOUT = 5

COOLDOWN_FILE = Path(__file__).parent / ".health_cooldown"
COOLDOWN_SEC = 180  # Min interval between alerts

HEALTH_URL = f"http://{HOST}:{PORT}/health"


def tcp_check() -> bool:
    """TCP connect check (fastest)."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(TIMEOUT)
        result = sock.connect_ex((HOST, PORT))
        sock.close()
        return result == 0
    except OSError:
        return False


def http_check() -> tuple[int, str]:
    """HTTP check returning status code and body."""
    try:
        import urllib.request
        req = urllib.request.Request(HEALTH_URL)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            code = resp.status
            body = resp.read().decode("utf-8")
            return code, body
    except Exception as e:
        return 0, str(e)


def in_cooldown() -> bool:
    """Check if we're in alert cooldown period."""
    if not COOLDOWN_FILE.exists():
        return False
    try:
        last = float(COOLDOWN_FILE.read_text().strip())
        return (time.time() - last) < COOLDOWN_SEC
    except (ValueError, OSError):
        return False


def set_cooldown():
    """Mark alert sent."""
    COOLDOWN_FILE.write_text(str(time.time()))


def main():
    # TCP check first
    if not tcp_check():
        if in_cooldown():
            sys.exit(0)  # Silent during cooldown
        print(f"[{datetime.now().isoformat()}] CRITICAL: Dashboard TCP port {PORT} unreachable")
        set_cooldown()
        sys.exit(2)

    # HTTP check
    code, body = http_check()
    data = {}
    if body:
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            pass

    if code == 200:
        # Healthy
        sys.exit(0)
    elif code == 503:
        if in_cooldown():
            sys.exit(0)
        status = data.get("status", "unknown")
        supervisor = data.get("supervisor", "unknown")
        stale = data.get("stale_bots", 0)
        print(f"[{datetime.now().isoformat()}] WARNING: Dashboard degraded - supervisor={supervisor}, stale_bots={stale}")
        set_cooldown()
        sys.exit(1)
    else:
        if in_cooldown():
            sys.exit(0)
        print(f"[{datetime.now().isoformat()}] ERROR: Dashboard returned {code}")
        set_cooldown()
        sys.exit(2)


if __name__ == "__main__":
    main()