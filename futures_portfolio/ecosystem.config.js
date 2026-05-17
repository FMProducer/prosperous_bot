const path = require('path');

module.exports = {
  apps: [
    {
      name: "supervisor-service",
      script: "supervisor_service.py",
      interpreter: "python",
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
      interpreter: "python",
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
      interpreter: "python",
      restart_delay: 5000,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      env: {
        PYTHONUNBUFFERED: "1"
      }
    }
  ]
};
