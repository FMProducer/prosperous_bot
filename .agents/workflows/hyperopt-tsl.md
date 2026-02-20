---
description: Настройка hyperopt для TSL
---

Настрой hyperopt только для TSL параметров в CustomD3QNStrategy4z.py:
1. Измени d0, d_min, hysteresis → optimize=True, space='stoploss'
2. Создай минимальный Hyperopt класс с Edge/Sortino
3. Покажи команды для запуска hyperopt