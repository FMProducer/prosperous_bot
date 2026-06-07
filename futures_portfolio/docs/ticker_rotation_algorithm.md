# Алгоритм выбора и ротации тикеров — Supervisor

## Общая архитектура (manage_swarm)

```mermaid
flowchart TD
    A([🚀 Начало цикла manage_swarm]) --> B[Загрузка config.json]
    B --> C[BinanceConnector.verify_connection]
    C --> D[enforce_swarm_consistency\nПроверка биржевых позиций vs PM2]

    D --> E{Есть позиции на бирже\nбез PM2 процесса?}
    E -- Да, state валиден --> F[HEAL: перезапуск\nreal-бота из стейта]
    E -- Да, state невалиден --> G[Закрытие позиций\nна бирже]
    E -- Нет --> H[Чтение signal-флагов\nиз /signals/]
    F --> H
    G --> H

    H --> I{Есть stop_*.flag?}
    I -- Да --> J[Запись тикера в\ntoxic_blacklist\n+ cooldown_sec]
    I -- Нет --> K[Удаление всех\nstop_*.flag и exit_*.flag]
    J --> K

    K --> L[Prune expired\nиз toxic_blacklist]
    L --> M[get_running_bots_info\npm2 jlist — один вызов на цикл]
    M --> N[run_scanner\nrank_tickers.py\nmin_volume=20M]

    N --> O{Scanner\nвернул результаты?}
    O -- Нет --> P[❌ ABORT\nCycle aborted]
    O -- Да --> Q[Фильтрация toxic\nиз scanner_results]

    Q --> R[selective_merge_incubator\nРотация инкубатора paper-ботов]
    R --> S[Выбор Чемпионов\nдля REAL\ncalculate_bot_score]

    S --> T[enforce_invariant_gate\nЗащита: не убирать тикер\nс открытой позицией]
    T --> U[Authoritative Cleanup\nЗакрытие stray-позиций]
    U --> V[PM2: остановка\nлишних ботов]
    V --> W[PM2: запуск\nновых paper + real]
    W --> X[Сохранение live_swarm\n= target_real_bots]
    X --> Y[enforce_invariant_gate\nПовторная проверка]
    Y --> Z[pm2 save]
    Z --> AA[_ensure_real_bots_alive\nФинальная проверка PM2]
    AA --> AB([✅ Cycle Complete])
```

---

## enforce_swarm_consistency — Проверка согласованности роя

```mermaid
flowchart TD
    A[enforce_swarm_consistency] --> B[get_positions\nВсе позиции на бирже]
    B --> C[get_running_bots_info\npm2 jlist]
    C --> D[allowed = live_swarm ∪ real_whitelist]

    D --> E{Для каждой позиции\npos_key на бирже}
    E --> F{ticker ∈ allowed\nИ\nticker ∈ running_real?}
    F -- Да → Всё OK --> E
    F -- Нет --> G{state_path\nсуществует?}
    G -- Да --> H{rebalance_cycles > 0\nИЛИ\nvirt_qty > 0?}
    H -- Да --> I[to_heal_tickers.add\nHEAL — перезапуск]
    H -- Нет --> J{to_close_tickers.add\nЗакрытие позиций]
    G -- Нет --> K{allowed\nпуст?}
    K -- Да --> J
    K -- Нет --> L{ticker ∉ allowed?}
    L -- Да --> J
    L -- Нет --> E

    I --> M[Для каждого healed:\nticker → live_swarm\nstart_bot real]
    J --> N[Для каждого to_close:\nmain.py --stop --real]
    M --> O([return to_heal_tickers])
    N --> O
```

---

## Чтение signal-флагов

```mermaid
flowchart TD
    A[signals_dir.exists?] --> B{glob stop_*.flag}
    B --> C[Для каждого файла:\nticker = stem[5:]\nexpiry = now + cooldown_sec]
    C --> D[toxic_blacklist[ticker] = expiry]
    D --> E[Лог: STOP signal received]
    E --> F[Удалить все stop_*.flag]
    F --> G[Удалить все exit_*.flag]
    G --> H[Prune expired:\nexpiry < now → удалить]
    H --> I([toxic_blacklist обновлён])

    style D fill:#f96
    style G fill:#f96
```

> ⚠️ **BUG**: Обрабатываются только `stop_*.flag`. `exit_*.flag` удаляется **без записи в toxic_blacklist**. Если main.py отправляет "exit" (прибыльный trailing stop), тикер НЕ блокируется.

---

## selective_merge_incubator — Ротация инкубатора (paper)

```mermaid
flowchart TD
    A[selective_merge_incubator\nold_incubator, scanner_results] --> B{old_incubator\nпуст?}
    B -- Да --> C[Первый старт:\nвзять max_bots из сканера]
    C --> Z([return final])

    B -- Нет --> D[scanner_top = первые max_bots из сканера]
    D --> E[Для каждого t ∈ old_incubator:\n_calc_rotation_score → scored_old]

    E --> F[Разделение:\nprofitable = profit > 0\nunprofitable = остальные]

    F --> G[final = profitable\nВСЕГДА остаются]

    G --> H[scanner_new = scanner_top \ old_incubator]
    H --> H2[new_scored = score 0.0 для новых]

    H2 --> I[unprofiled_scored\nsort by score desc]
    I --> J[competitors = unprofiled + new_scored\nsort by score desc]

    J --> K[Заполнить слоты до max_bots:\nбрать лучших из competitors]

    K --> L[to_remove = old \ final]
    L --> M[to_add = final \ old]
    M --> N[Лог: keep / add / remove]
    N --> Z

    style G fill:#6f6
    style H2 fill:#ff9
```

### _calc_rotation_score

```mermaid
flowchart TD
    A[_calc_rotation_score\nticker, perf, min_cycles] --> B{cycles < min_cycles\nИ\nnet_pnl ≤ 0?}
    B -- Да --> C[return -INF\nОтклонён]
    B -- Нет --> D{net_pnl > 0?}
    D -- Да --> E[eff_cycles = max\ncycles, min_cycles]
    E --> F[base_score =\nnet_pnl / eff_cycles\n× log1p cycles]
    F --> G[return base_score]
    D -- Нет --> H[return net_pnl × 0.01\nМаленький отрицательный]
```

---

## Выбор Чемпионов для REAL

```mermaid
flowchart TD
    A[Выбор Чемпионов] --> B[all_evaluated =\nfinal_incubator ∪ current_real_tickers]
    B --> C[Для каждого t ∈ all_evaluated:\nget_bot_efficiency → perf_map]

    C --> D{Для тикера t:}
    D --> E{is_running_real?}
    E -- Да --> F{real_state\n.last_profit < 0?}
    F -- Да --> G[is_in_drawdown = True]
    F -- Нет --> H[is_in_drawdown = False]
    E -- Нет --> I{trailing_stop_paper_timeout_end\n> now?}
    I -- Да --> J[⏳ Skip\nPaper probation]
    I -- Нет --> H

    G --> K[calculate_bot_score]
    H --> K
    J --> D

    K --> L[sort_eff → ready_pool\nsort by score desc]
    L --> M[max_real_slots =\nmax_bots - paper_mode_bots]

    M --> N[target_real_bots =\ncurrent_real_tickers]

    N --> O{len target < max_real_slots?}
    O -- Да --> P[Заполнить из\nincubator_candidates\nлучшие по score]
    P --> Q{Есть кандидаты\nдля замены?}
    O -- Нет --> Q

    Q --> R[Для каждого REAL бота\nworst score first:]
    R --> S{score == INF?\nDrawdown protection}
    S -- Да --> T[🛡️ LOCKED IN COMBAT\nПропустить]
    S -- Нет --> U{is_probation?\nage < probation_hours\nИ cycles < min_cycles}
    U -- Да --> V[🛡️ Hysteresis Guard\nПропустить]
    U -- Нет --> W{profit > 0?}
    W -- Да --> X[🛡️ Profit Guard\nПропустить]
    W -- Нет --> Y{cand_score >\nscore + 0.25?}
    Y -- Да --> Z[♻️ Substitution\nЗамена бота]
    Y -- Нет --> AA[⏭️ Skip\nНедостаточный cushion]

    Z --> AB{len target >\nmax_real_slots?}
    AA --> AB
    T --> AB
    V --> AB
    X --> AB

    AB -- Да --> AC[✂️ Trim worst\nдо max_real_slots]
    AB -- Нет --> AD[target_real_bots\n= итоговый список]
    AC --> AD

    style G fill:#f66
    style T fill:#6f6
    style V fill:#6f6
    style X fill:#6f6
    style Z fill:#ff6
```

---

## calculate_bot_score

```mermaid
flowchart TD
    A[calculate_bot_score\nticker, p, is_running_real\nis_in_drawdown, min_cycles] --> B{is_in_drawdown?}
    B -- Да --> C[return INF\nLOCKED IN COMBAT]
    B -- Нет --> D{cycles < min_cycles\nИ\nне running_real?}
    D -- Да --> E[return -INF\nRejected]
    D -- Нет --> F{net_pnl > 0?}
    F -- Да --> G[eff_cycles = max\ncycles, min_cycles]
    G --> H[base_score =\nnet_pnl / eff_cycles\n× log1p cycles]
    H --> I{is_running_real?}
    I -- Да --> J[sort_eff = base_score\n× 1.2\nБонус 20%]
    I -- Нет --> K[sort_eff = base_score]
    J --> L[return sort_eff]
    K --> L
    F -- Нет --> M[return -INF\nUnprofitable]

    style C fill:#6f6
    style E fill:#f66
    style J fill:#6f6
    style M fill:#f66
```

---

## enforce_invariant_gate — Защита инварианта

```mermaid
flowchart TD
    A[enforce_invariant_gate] --> B[get_positions\nВсе позиции на бирже]
    B --> C[Для каждой позиции:]
    C --> D{qty > 0?}
    D -- Нет --> C
    D -- Да --> E[notional = qty × price]
    E --> F{notional ≥\ndust_threshold?}
    F -- Нет --> C
    F -- Да --> G{ticker ∉ live_swarm?}
    G -- Да --> H[live_swarm.add\nForced retention]
    G -- Нет --> C
    H --> I[config.live_swarm\n= sorted live_swarm]
    I --> J([live_swarm обновлён])

    style H fill:#ff6
```

---

## Полный цикл: от сканера до PM2

```mermaid
flowchart TD
    subgraph SCANNER["🔍 SCANNER (rank_tickers.py)"]
        S1[48h данные] --> S2[Фильтр: >1.5% циклы]
        S2 --> S3[SPIKE TRAP: >10%/1h → toxic]
        S3 --> S4[NET TRAP: >15%/48h → toxic]
        S4 --> S5[Trend Efficiency < 3%]
        S5 --> S6[SCORE = cycles × efficiency]
        S6 --> S7[scanner_results\nsorted by SCORE]
    end

    subgraph INCUBATOR["📋 INCUBATOR (Paper)"]
        I1[selective_merge] --> I2[profitable → always keep]
        I2 --> I3[unprofitable + new → compete]
        I3 --> I4[final_incubator\n≤ max_bots]
    end

    subgraph CHAMPIONS["🏆 CHAMPIONS (Real)"]
        C1[all_evaluated tickers] --> C2[calculate_bot_score]
        C2 --> C3[ready_pool\nscore > -INF]
        C3 --> C4[target_real_bots\nfill empty slots]
        C4 --> C5[Substitution logic\nwith guards]
        C5 --> C6[Trim to max_real_slots]
    end

    subgraph PM2["⚡ PM2 EXECUTION"]
        P1[Stop bots not in target] --> P2[Start new paper bots]
        P2 --> P3[Start new real bots]
        P3 --> P4[Save config.json]
        P4 --> P5[pm2 save]
        P5 --> P6[_ensure_real_bots_alive]
    end

    S7 --> I1
    S7 --> C1
    I4 --> C1
    I4 --> P2
    C6 --> P1
    C6 --> P3
    P6 --> END([✅ END CYCLE])

    style SCANNER fill:#e6f3ff
    style INCUBATOR fill:#fff3e6
    style CHAMPIONS fill:#e6ffe6
    style PM2 fill:#f3e6ff
```

---

## Защитные механизмы (Guards)

```mermaid
flowchart LR
    subgraph GUARDS["🛡️ Protection Guards"]
        direction TB
        G1[Drawdown Protection\nscore=INF → locked]
        G2[Profit Guard\nprofit>0 → no replace]
        G3[Hysteresis Guard\nprobation period]
        G4[Score Cushion\ncand > score+0.25]
        G5[Invariant Gate\nactive position → keep]
        G6[Whitelist Guard\nwhitelist → no stop]
    end

    G1 --> REAL[REAL Bot]
    G2 --> REAL
    G3 --> REAL
    G4 --> REAL
    G5 --> REAL
    G6 --> REAL

    style G1 fill:#6f6
    style G2 fill:#6f6
    style G3 fill:#6f6
    style G4 fill:#ff6
    style G5 fill:#ff6
    style G6 fill:#6f6
```

---

## Известные баги и проблемы

```mermaid
flowchart TD
    BUG1[⚠️ exit-флаг не создаёт toxic\nmain.py → emit_signal exit\nsupervisor игнорирует]
    BUG2[⚠️ main.py пишет toxic в config\nsupervisor перезаписывает\n→ запись теряется]
    BUG3[⚠️ HEAL перезапускает бот\nпосле trailing stop\n→ позиции открываются заново]
    BUG4[⚠️ Profit Guard + Drawdown\n= INF → бот вечно в рое\nдаже после стопа]
    BUG5[⚠️ Invariant Gate добавляет\nобратно в live_swarm\nесли позиция на бирже]

    style BUG1 fill:#f66
    style BUG2 fill:#f66
    style BUG3 fill:#f66
    style BUG4 fill:#f66
    style BUG5 fill:#f66
```

---

## Таблица параметров конфига

| Параметр | Значение | Назначение |
|----------|----------|------------|
| `max_bots` | 20 | Макс. ботов в инкубаторе |
| `paper_mode_bots` | 19 | Из них paper → real_slots = 1 |
| `min_cycles_for_rank` | 60 | Мин. циклов для скоринга |
| `max_replace_per_cycle` | 10 | Макс. замен за цикл |
| `probation_period_days` | 0.01 (~14 min) | Испытательный срок |
| `toxic_cooldown_days` | 0.02 (~29 min) | Кулдаун toxic |
| `equity_trailing_stop_pct` | 5.7% | Трейлинг стоп |
| `equity_trailing_stop_activation_pct` | 6.5% | Активация трейлинга |
| `equity_trailing_stop_timeout_sec` | 60 | Таймаут перед стопом |
| `supervisor_interval_days` | 0.02 (~29 min) | Интервал цикла |
