const path = require('path');

// Абсолютный путь к python из venv — PM2 всегда использует его
const VENV_PYTHON = path.join(__dirname, '..', '.venv', 'Scripts', 'python.exe');

module.exports = {
  apps: [
    {
      name: "supervisor-service",
      script: "supervisor_service.py",
      interpreter: VENV_PYTHON,
      restart_delay: 30000,
      error_file: "./logs/err_supervisor_service.log",
      out_file: "./logs/out_supervisor_service.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      env: {
        PYTHONUNBUFFERED: "1"
      }
    },
    {
      name: "swarm-aggregator",
      script: "aggregator.py",
      interpreter: VENV_PYTHON,
      restart_delay: 5000,
      error_file: "./logs/err_aggregator.log",
      out_file: "./logs/out_aggregator.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      env: {
        PYTHONUNBUFFERED: "1"
      }
    },
    {
      name: "telegram-sender",
      script: "telegram_sender.py",
      interpreter: VENV_PYTHON,
      restart_delay: 5000,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      env: {
        PYTHONUNBUFFERED: "1"
      }
    }
  ]
};
