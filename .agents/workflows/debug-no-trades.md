---
description: Отладка отсутствия сделок
---

I am running `CustomD3QNStrategy4z.py` but the bot is not opening trades. Analyze `populate_entry_trend` and `_compute_ensemble_decision` to identify potential blockers. Check:
1. Is the `min_quote_volume_usd` filter too aggressive?
2. Could the `st_regime_15m` filter be blocking signals due to a mismatch with the model's prediction?
3. Is `epsilon_threshold_eff` potentially too high due to the logic in `_update_dynamic_epsilon`?
4. Are the `q_min` / `q_max` normalization parameters loaded correctly?
Provide a checklist to debug this in live/dry-run mode.