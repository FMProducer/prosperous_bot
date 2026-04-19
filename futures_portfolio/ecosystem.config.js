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
    },
    {
      name: "rebalancer-hype",
      script: "main.py",
      args: "--config config_hype.json",
      interpreter: "python",
      restart_delay: 10000,
      max_restarts: 10,
      error_file: "./logs/pm2_hype_error.log",
      out_file: "./logs/pm2_hype_out.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      env: { NODE_ENV: "production" }
    },
    {
      name: "rebalancer-pepe",
      script: "main.py",
      args: "--config config_pepe.json",
      interpreter: "python",
      restart_delay: 10000,
      max_restarts: 10,
      error_file: "./logs/pm2_pepe_error.log",
      out_file: "./logs/pm2_pepe_out.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      env: { NODE_ENV: "production" }
    },
    {
      name: "rebalancer-ordi",
      script: "main.py",
      args: "--config config_ordi.json",
      interpreter: "python",
      restart_delay: 10000,
      max_restarts: 10,
      error_file: "./logs/pm2_ordi_error.log",
      out_file: "./logs/pm2_ordi_out.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      env: { NODE_ENV: "production" }
    },
    {
      name: "rebalancer-rave",
      script: "main.py",
      args: "--config config_rave.json",
      interpreter: "python",
      restart_delay: 10000,
      max_restarts: 10,
      error_file: "./logs/pm2_rave_error.log",
      out_file: "./logs/pm2_rave_out.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      env: { NODE_ENV: "production" }
    }
  ]
};
