---
description: Настройка Hyperopt
---

I need to run Hyperopt for `CustomD3QNStrategy4z.py`.
1. Identify all parameters currently marked with `optimize=True` (e.g., `dd_aggression_k`, `supertrend_period`, `rl_long_threshold_opt`).
2. Verify that `populate_entry_trend` uses the `.value` property of these parameters correctly.
3. Suggest a suitable `HyperOptLoss` class configuration that prioritizes Sortino ratio and stability, considering this is an RL strategy.