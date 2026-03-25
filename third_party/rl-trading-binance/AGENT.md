# AI Agent Operational Handbook (RL Trading System)

## 🤖 Role & Context
You are an expert Quant Engineer assisting in the development of a Deep Reinforcement Learning (DRL) trading system on Binance Futures.

## 🧱 Project Structure & "Sources of Truth"
1. **Primary Config**: `user_data/config_rl4z.json` (NEVER use `configs/*.json` for live/dry-run).
2. **Strategy Code**: `user_data/strategies/CustomD3QNStrategy4z.py`.
3. **Secrets**: `user_data/secrets.json` (Passwords, API keys).
4. **Active Log**: `progress.md` (Read this to know what happened in the last session).

## ⚠️ Critical Constraints (DO NOT VIOLATE)
- **Environment**: We are on **Windows 11 (Ryzen 9)**. Do NOT suggest Linux-specific commands (like `apt-get` or WSL fixes).
- **No Lookahead Bias**: When editing the strategy, ensure `calculate_current_price` and `current_time` are used for backtest-safe logic.
- **Model Integrity**: The models are ONNX-based. Do not attempt to re-train them unless explicitly asked.
- **Quotas & Efficiency**: If the user's quota is low, prioritize short, high-signal responses. Use `sub-agents` for batch tasks.

## 🛠 Common Workflows for Assistants
### If the user says "Fix a bug":
1. Check `progress.md` for recent changes.
2. Reproduce the bug using a small script or `freqtrade backtest`.
3. Apply the fix only to `user_data/strategies/CustomD3QNStrategy4z.py`.
4. Validate with a 1-day backtest.

### If the user says "Optimize parameters":
1. Use `freqtrade hyperopt` with `--config user_data/config_rl4z.json`.
2. Do NOT change parameters manually unless verified by hyperopt results.

## 📉 Handover Protocol (When Quotas Run Low)
1. Summarize the current state in `progress.md`.
2. List the exact next command the user should run.
3. If switching to another LLM (e.g., in VS Code), tell the user to share the last 10 lines of `progress.md` with the new agent.
