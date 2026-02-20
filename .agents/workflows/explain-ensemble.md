---
description: Объяснение логики ансамбля
---

Analyze the `CustomD3QNStrategy4z.py` file. Explain in detail the decision-making process of the 2+2 RL Ensemble. Specifically cover:
1. How the `_parallel_inference` mechanism gathers Q-values from the 4 models.
2. How `_update_dynamic_epsilon` adjusts the voting threshold based on drawdown.
3. How the `st_regime_15m` (Supertrend) filter interacts with the RL signals.
4. The logic inside `_compute_ensemble_decision` regarding voting, vetoes, and Q-value normalization.