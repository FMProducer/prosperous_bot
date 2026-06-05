#!/usr/bin/env python3
"""
Dashboard — Flask сервер для мониторинга и управления Prosperous Bot.

Запуск:
  python dashboard.py                  # Разработка
  python dashboard.py --production     # Продакшен (waitress)

Требования:
  pip install flask bcrypt psutil pyyaml
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from functools import wraps
from pathlib import Path

import yaml

# ─── Paths ────────────────────────────────────────────────────

DASHBOARD_DIR = Path(__file__).parent
TEMPLATES_DIR = DASHBOARD_DIR / "templates"
STATIC_DIR = DASHBOARD_DIR / "static"

# ─── Load config ──────────────────────────────────────────────

CONFIG_PATH = DASHBOARD_DIR / "config.yaml"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        print(f"[ERROR] config.yaml not found at {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_config()

PROJECT_PATH = Path(config["paths"]["project"])
LOGS_PATH = Path(config["paths"]["logs"])
DATA_PATH = Path(config["paths"]["data"])
PM2_LOGS_PATH = Path(config["paths"]["pm2_logs"])

# ─── Imports ─────────────────────────────────────────────────

import flask
import bcrypt
from flask import (
    Flask, request, redirect, url_for,
    session, flash, render_template, jsonify
)

# ─── App factory ──────────────────────────────────────────────


def create_app():
    app = Flask(
        __name__,
        template_folder=str(TEMPLATES_DIR),
        static_folder=str(STATIC_DIR),
    )
    app.secret_key = config["server"]["secret_key"]
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["PERMANENT_SESSION_LIFETIME"] = 3600

    # Logging
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    ))
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)

    # ─── Decorators ───────────────────────────────────────────

    def login_required(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not session.get("authenticated"):
                flash("Требуется авторизация", "warning")
                return redirect(url_for("login"))
            return f(*args, **kwargs)
        return decorated

    def csrf_protected(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if request.method == "POST":
                token = request.headers.get("X-CSRF-Token", "")
                if token != session.get("csrf_token", ""):
                    return jsonify({"error": "CSRF validation failed"}), 403
            return f(*args, **kwargs)
        return decorated

    # ─── Routes ───────────────────────────────────────────────

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if session.get("authenticated"):
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")

            if (username == config["auth"]["username"] and
                    bcrypt.checkpw(
                        password.encode("utf-8"),
                        config["auth"]["password_hash"].encode("utf-8")
                    )):
                session["authenticated"] = True
                session["username"] = username
                session["csrf_token"] = os.urandom(32).hex()
                session.permanent = True
                app.logger.info(
                    f"User '{username}' logged in from {request.remote_addr}"
                )
                return redirect(url_for("dashboard"))
            else:
                app.logger.warning(
                    f"Failed login for '{username}' from {request.remote_addr}"
                )
                flash("Неверный логин или пароль", "danger")

        return render_template("login.html")

    @app.route("/logout")
    def logout():
        session.clear()
        flash("Вы вышли из системы", "info")
        return redirect(url_for("login"))

    @app.route("/")
    @login_required
    def dashboard():
        return render_template(
            "dashboard.html",
            refresh_sec=config["refresh"]["dashboard_sec"],
            control_enabled=config["control"]["enabled"],
        )

    @app.route("/ticker/<mode>/<ticker>")
    @login_required
    def ticker_detail(mode, ticker):
        return render_template(
            "ticker.html",
            mode=mode, ticker=ticker,
            refresh_sec=config["refresh"]["ticker_sec"],
            control_enabled=config["control"]["enabled"],
        )

    @app.route("/history")
    @login_required
    def history():
        return render_template("history.html")

    @app.route("/logs/<mode>/<ticker>")
    @login_required
    def logs(mode, ticker):
        return render_template(
            "logs.html",
            mode=mode, ticker=ticker,
            refresh_sec=config["refresh"]["logs_sec"],
        )

    @app.route("/settings")
    @login_required
    def settings():
        return render_template("settings.html", config=config)

    # ─── API endpoints ─────────────────────────────────────────

    @app.route("/api/overview")
    @login_required
    def api_overview():
        from data_collector import get_system_overview
        return jsonify(get_system_overview())

    @app.route("/api/ticker/<mode>/<ticker>")
    @login_required
    def api_ticker(mode, ticker):
        from data_collector import get_ticker_details
        return jsonify(get_ticker_details(mode, ticker))

    @app.route("/api/history")
    @login_required
    def api_history():
        from data_collector import get_history_data
        return jsonify(get_history_data())

    @app.route("/api/logs/<mode>/<ticker>")
    @login_required
    def api_logs(mode, ticker):
        from data_collector import get_log_tail
        lines = int(request.args.get("lines", 100))
        offset = int(request.args.get("offset", 0))
        return jsonify(get_log_tail(mode, ticker, lines, offset))

    @app.route("/api/pm2/list")
    @login_required
    def api_pm2_list():
        from data_collector import get_pm2_processes
        return jsonify(get_pm2_processes())

    # ─── Control API ───────────────────────────────────────────

    @app.route("/api/control/pm2", methods=["POST"])
    @login_required
    @csrf_protected
    def api_control_pm2():
        if not config["control"]["enabled"]:
            return jsonify({"error": "Control is disabled"}), 403

        data = request.get_json()
        action = data.get("action")
        target = data.get("target")
        ticker = data.get("ticker")

        allowed = config["control"]["allowed_commands"]
        if action not in allowed:
            return jsonify({"error": f"Action '{action}' not allowed"}), 403

        result = _execute_pm2_action(action, target, ticker)
        app.logger.info(
            f"Control: {action} {target} by {session.get('username')}"
        )
        return jsonify(result)

    @app.route("/api/control/emergency-stop", methods=["POST"])
    @login_required
    @csrf_protected
    def api_emergency_stop():
        data = request.get_json()
        ticker = data.get("ticker")
        mode = data.get("mode", "real")

        if not ticker:
            return jsonify({"error": "Ticker required"}), 403

        result = _execute_emergency_stop(ticker, mode)
        app.logger.warning(
            f"EMERGENCY STOP: {mode} {ticker} by {session.get('username')}"
        )
        return jsonify(result)

    # ─── Control helpers ───────────────────────────────────────

    def _execute_pm2_action(action, target, ticker=None):
        try:
            if action == "pm2_start" and ticker:
                cmd = [
                    "pm2", "start", "main.py",
                    "--name", f"paper-{ticker.lower()}",
                    "--cwd", str(PROJECT_PATH),
                    "--interpreter", "python",
                    "--", "--config", "config.json",
                    "--ticker", ticker,
                ]
            elif action == "pm2_stop":
                cmd = ["pm2", "stop", target]
            elif action == "pm2_restart":
                cmd = ["pm2", "restart", target, "--update-env"]
            elif action == "pm2_delete":
                cmd = ["pm2", "delete", target]
            else:
                return {"error": f"Unknown action: {action}"}

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=15
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {"error": "Command timed out"}
        except Exception as e:
            return {"error": str(e)}

    def _execute_emergency_stop(ticker, mode):
        results = []
        try:
            process_name = f"{mode}-{ticker.lower()}"
            r = subprocess.run(
                ["pm2", "stop", process_name],
                capture_output=True, text=True, timeout=10
            )
            results.append({
                "step": "pm2_stop", "success": r.returncode == 0
            })

            r = subprocess.run(
                ["pm2", "delete", process_name],
                capture_output=True, text=True, timeout=10
            )
            results.append({
                "step": "pm2_delete", "success": r.returncode == 0
            })

            return {"success": True, "steps": results}
        except Exception as e:
            return {"error": str(e), "steps": results}

    # ─── Error handlers ────────────────────────────────────────

    @app.errorhandler(404)
    def not_found(e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Not found"}), 404
        return render_template("dashboard.html",
                               error="Страница не найдена"), 404

    @app.errorhandler(500)
    def server_error(e):
        app.logger.error(f"Server error: {e}")
        if request.path.startswith("/api/"):
            return jsonify({"error": "Internal error"}), 500
        return render_template("dashboard.html",
                               error="Ошибка сервера"), 500

    return app


# ─── Entry point ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--host", default=config["server"]["host"])
    parser.add_argument("--port", type=int, default=config["server"]["port"])
    args = parser.parse_args()

    app = create_app()

    print("=" * 60)
    print("  Prosperous Bot Dashboard")
    print(f"  URL: http://{args.host}:{args.port}")
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    if args.production:
        try:
            from waitress import serve
            print("  [production] Running with waitress...")
            serve(app, host=args.host, port=args.port, threads=4)
        except ImportError:
            print("  [production] waitress not installed.")
            print("  Install: pip install waitress")
            print("  Falling back to Flask dev server...")
            app.run(host=args.host, port=args.port, debug=False)
    else:
        app.run(
            host=args.host, port=args.port,
            debug=config["server"]["debug"],
        )
