---
description: Добавление нового фильтра (Guard)
---

I want to add a new technical indicator as a guard filter to `CustomD3QNStrategy4z.py`.
1. Show me how to add the indicator calculation in `populate_indicators` (ensure it handles NaNs correctly).
2. Show where to insert the check in `populate_entry_trend` so that it filters signals *after* the RL inference but *before* the final decision is recorded.
3. Ensure the implementation is optimized for performance (vectorized if possible).