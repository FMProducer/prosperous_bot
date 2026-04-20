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
    config = { tickers: ["ZECUSDT"] };
}

const tickers = config.tickers || [config.base_ticker || "BTCUSDT"];

module.exports = {
  apps: tickers.map(ticker => {
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
        NODE_ENV: "production",
        PYTHONUNBUFFERED: "1" 
      }
    };
  })
};
