const path = require('path');

// Абсолютный путь к python из venv — PM2 всегда использует его
const VENV_PYTHON = path.join(__dirname, '..', '.venv', 'Scripts', 'python.exe');
// Корневая директория пакета — для PYTHONPATH чтобы все импорты работали
const PORTFOLIO_ROOT = __dirname;

module.exports = {
  apps: [
    {
      name: "supervisor-service",
      script: "supervisor/supervisor_service.py",
      interpreter: VENV_PYTHON,
      restart_delay: 30000,
      error_file: "./logs/err_supervisor_service.log",
      out_file: "./logs/out_supervisor_service.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      env: {
        PYTHONUNBUFFERED: "1",
        PYTHONPATH: PORTFOLIO_ROOT
      }
    },
    {
      name: "swarm-aggregator",
      script: "supervisor/aggregator.py",
      interpreter: VENV_PYTHON,
      restart_delay: 5000,
      error_file: "./logs/err_aggregator.log",
      out_file: "./logs/out_aggregator.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      env: {
        PYTHONUNBUFFERED: "1",
        PYTHONPATH: PORTFOLIO_ROOT
      }
    },
    {
      name: "telegram-sender",
      script: "monitoring/telegram_sender.py",
      interpreter: VENV_PYTHON,
      restart_delay: 5000,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      env: {
        PYTHONUNBUFFERED: "1",
        PYTHONPATH: PORTFOLIO_ROOT
      }
    }
  ]
};
