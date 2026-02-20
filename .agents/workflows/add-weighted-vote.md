---
description: Взвешенное голосование
---

Реализуй взвешенное голосование в _compute_ensemble_decision:
- Вместо votes → total_weight_long/short (sum normalized advantages)
- Порог weight_threshold_long/short = 0.8
- Сохрани veto логику
Дай diff‑патч только для этой функции