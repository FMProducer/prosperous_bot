# futures_portfolio/tests/test_swarm_integration.py
import pytest
import asyncio
import os
import json
from unittest.mock import patch, AsyncMock
from supervisor import manage_swarm

class SwarmIntegrator:
    def __init__(self, test_config_path="futures_portfolio/config.json"):
        self.config_path = test_config_path
        self.test_ticker = "BTCUSDT"
        self.backup_path = test_config_path + ".bak"

    def setup_mock_config(self):
        if os.path.exists(self.config_path):
            os.rename(self.config_path, self.backup_path)

        config = {
            "testnet": True,
            "paper_mode": True,
            "max_bots": 1,
            "portfolios": [
                {
                    "name": "TestPortfolio",
                    "targets": {
                        "BASE_LONG": {"share": 0.3, "leverage": 1},
                        "BASE_SHORT": {"share": 0.3, "leverage": 1},
                        "VIRTUAL": {"share": 0.4, "leverage": 1}
                    },
                    "rebalance_threshold": 0.01,
                    "check_interval_sec": 1,
                    "max_capital_usdt": 1000.0
                }
            ],
            "tickers": [self.test_ticker],
            "live_swarm": [],
            "base_ticker": self.test_ticker
        }
        with open(self.config_path, "w") as f:
            json.dump(config, f)

    async def check_pm2_status(self, bot_name):
        proc = await asyncio.create_subprocess_shell(
            f"pm2 jlist",
            stdout=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        data = json.loads(stdout.decode())
        return any(b['name'] == bot_name and b['pm2_env']['status'] == 'online' for b in data)

    async def run_full_cycle(self):
        print("\n[1] Starting Swarm Management...")
        # Set fake API key for main.py to start
        os.environ["BINANCE_API_KEY"] = "fake_key"
        os.environ["BINANCE_SECRET_KEY"] = "fake_secret"
        os.environ["MOCK_MODE"] = "1"

        # Мы патчим rank_tickers чтобы не ждать настоящего сканирования
        with patch("supervisor.run_scanner", return_value=[{"symbol": self.test_ticker, "score": 200}]):
            with patch("supervisor.run_backtest", return_value={"profit_pct": 1.0, "max_dd_pct": 1.0}):
                with patch("notifier.TelegramNotifier.send_message", AsyncMock()):
                    await manage_swarm()

        bot_name = f"bot-{self.test_ticker.replace('USDT', '').lower()}"

        print(f"[2] Checking if {bot_name} is active in PM2...")
        is_active = await self.check_pm2_status(bot_name)
        assert is_active, f"Bot {bot_name} failed to start in PM2"

        print("[3] Waiting for first state file generation...")
        state_file = f"futures_portfolio/paper_state_{self.test_ticker}.json"
        for _ in range(30): # 30 секунд таймаут
            if os.path.exists(state_file):
                with open(state_file, "r") as f:
                    state = json.load(f)
                    if state.get("positions", {}).get(f"{self.test_ticker}_LONG", 0) != 0:
                        print("✅ Integration Success: Position detected in state file.")
                        return True
            await asyncio.sleep(1)

        return False # Fallback instead of raising TimeoutError immediately for better cleanup

    def cleanup(self):
        os.system(f"pm2 delete bot-btc")
        if os.path.exists(self.backup_path):
            if os.path.exists(self.config_path):
                os.remove(self.config_path)
            os.rename(self.backup_path, self.config_path)

@pytest.mark.asyncio
async def test_integration_flow():
    tester = SwarmIntegrator()
    tester.setup_mock_config()
    try:
        # We need to make sure manage_swarm uses our config
        # and that PM2 can find main.py.
        # Since we are running from root, futures_portfolio/main.py is the path.
        result = await tester.run_full_cycle()
        assert result is True
    finally:
        tester.cleanup()
