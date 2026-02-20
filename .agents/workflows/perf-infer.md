---
description: Оптимизация inference скорости
---

Оптимизируй parallel inference в populate_entry_trend:
1. Проверь ThreadPoolExecutor настройки
2. Предложи batching или кэширование features
3. Дай diff‑патч с замерами времени
Цель: < 20ms на свечу для всех 4 моделей