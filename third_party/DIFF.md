---

## Конкретные замечания и что поправить

| Шаг | Действие                                                                                                                      | KPI/риск                                                              |
| --- | ----------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 1   | **Убрать дублирующий import** в `agent.py`.                                                                                   | Чистота кода; безрисково.                                             |
| 2   | **Обернуть расчёт таргетов в `no_grad()`** (`next_actions`, `next_q_values`, `target_q_values`). Можно вместе с AMP-autocast. | −GPU память, +стабильность; иногда +1–5% итераций/с.                  |
| 3   | **Читать `cudnn.benchmark` из конфига** в `train.py`, а не хардкодом.                                                         | Консистентность конфигов; управляемость.                              |
| 4   | (Опц.) **Компилировать `target_net`** тем же режимом.                                                                         | Мелкий прирост; следить за граф-брейками. ([Документация PyTorch][4]) |
| 5   | (Опц.) Если будете сохранять чекпоинты — **сохранять/грузить `scaler.state_dict()`** вместе с оптимизатором.                  | Корректное возобновление AMP-тренировки. ([Medium][5])                |

---

## Почему это важно (ссылки на практики PyTorch)

* **AMP**: clip-grad требует `scaler.unscale_(optimizer)` до `clip_grad_norm_` — вы выполняете правильно. ([Документация PyTorch][3])
* **`torch.compile`**: режим `"reduce-overhead"` безопасный; `dynamic=True` годится при меняющихся размерах, но может быть медленнее; начиная с PT≥2.2 детект динамики улучшен — экспериментируйте. ([Документация PyTorch][4])

---

## Рекомендуемые минимальные правки (фрагменты)

> Ниже — точечные вставки без полного diff (пути взяты из ваших загруженных файлов; при подготовке PR под репозиторий укажите реальные пути из `third_party/rl-trading-binance/...`).

**`agent.py`** — убрать дублирующий import и обернуть расчёт таргетов:

```python
# удалить дубликат
# from config import PerformanceConfig

# ... внутри learn() перед optimizer.zero_grad():
with torch.no_grad():
    # можно совместить с AMP, если хотите:
    # with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
    next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
    next_q_values = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
    target_q_values = rewards_t + (1 - dones_t) * (self.gamma * next_q_values)
```

(файл содержит текущую AMP-ветку и `unscale_ → clip → step`, их оставляем как есть).

**(Опционально)** компиляция `target_net` рядом с `policy_net`:

```python
if perf_cfg.compile_mode:
    self.target_net = torch.compile(self.target_net,
                                    mode=perf_cfg.compile_mode,
                                    dynamic=perf_cfg.compile_dynamic)
```

**`train.py`** — брать `cudnn.benchmark` из конфига:

```python
if cfg.device.device.type == "cuda":
    torch.backends.cudnn.benchmark = cfg.perf.cudnn_benchmark
```

Сейчас всегда `True` — это расходится с идеей “всё управляется конфигом”.

---