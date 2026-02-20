---
description: Исправление фильтра Supertrend
---

Проверь логику режима (Supertrend 15m) в populate_entry_trend:
1. Где вычисляется bullish/bearish
2. Где применяется фильтр к сигналам
3. Сравни с логами: "Regime BEARISH → Filtered by ST (Bullish)"
4. Дай diff‑патч для синхронизации режима и фильтра