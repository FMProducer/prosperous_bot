# RL Trading System: Prosperous Bot Edition

## 🛠 Core Components
- **Strategy File:** `user_data/strategies/CustomD3QNStrategy4z.py`
- **Primary Config:** `user_data/config_rl4z.json`
- **Progress Log:** `progress.md`

## 📜 Agent Rules
1. **Context Priority:** Always refer to `progress.md` at the start of a session to understand the current state and next steps.
2. **Continuous Documentation:** After every significant change, test, or optimization, update the `progress.md` file. 
   - Add completed tasks to the `## Completed Tasks` section.
   - Refine or add new items to the `## Next Steps` section.
3. **No Lookahead Bias:** Rigorously verify that no future data is used in backtests (refer to `analysis.md` if needed).
4. **Safety First:** Prioritize stable, low-risk parameters over aggressive profit-seeking during optimization unless explicitly instructed.

## 🎯 Current Focus
- Verification of "Safety First" parameters.
- Stabilization of trailing stop-loss logic.
- Preparation for extended backtesting.
