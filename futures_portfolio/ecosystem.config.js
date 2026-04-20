module.exports = {
  apps: [
    {
      name: "rebalancer-zec",
      script: "main.py",
      args: "--config config_zec.json",
      interpreter: "python",
      restart_delay: 10000,
      max_restarts: 10,
      error_file: "./logs/pm2_zec_error.log",
      out_file: "./logs/pm2_zec_out.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      env: { NODE_ENV: "production" }
    }
  ]
};
