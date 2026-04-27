const fs = require('fs');
const path = require('path');

// Читаем основной конфиг
const configPath = path.join(__dirname, 'config.json');
let config = {};

try {
    const data = fs.readFileSync(configPath, 'utf8');
    config = JSON.parse(data);
} catch (e) {
    console.error("Could not read config.json, using defaults");
    config = { tickers: ["BTCUSDT"] };
}

const tickers = config.tickers || [];

const apps = tickers.map(ticker => {
    const shortName = ticker.replace('USDT', '').toLowerCase();
    return {
      name: `bot-${shortName}`,
      script: "main.py",
      args: `--config config.json --ticker ${ticker}`,
      interpreter: "python",
      restart_delay: 15000,
      max_restarts: 20,
      error_file: `./logs/err_${ticker}.log`,
      out_file: `./logs/out_${ticker}.log`,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      env: { 
        PYTHONUNBUFFERED: "1" 
      }
    };
});

// Добавляем Супервайзер-сервис в общий список
apps.push({
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
});

module.exports = {
  apps: apps
};
