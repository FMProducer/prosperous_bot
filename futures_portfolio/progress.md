Progress Log — Market-Neutral Futures Portfolio Rebalancer
Формат: Дата → Что сделали → Почему → Результат Читаемые первые 50 строк = последние 50 строк (свежее сверху).

2026-07-11 — B6/B3 SSOT Refactor: Race Condition + Exit/Stop Isolation
Что сделано
Устранены две архитектурные уязвимости:
1. B6 (Race Condition): main.py напрямую писал в config.json в _handle_liquidation_recovery и _handle_liquidation_guard. Параллельные процессы (main.py + supervisor.py) вызывали гонку данных — потеря toxic-записей и live_swarm изменений. Фикс: удалены блоки записи из main.py, эмиссия emit_signal("stop") вместо прямой мутации. Worker → Read-Only + Signal.
2. B3 (Exit/Stop Propagation): supervisor.py не различал exit и stop — прибыльные тикеры попадали в toxic_blacklist вместо probation. Фикс: маршрутизация сигналов — stop→toxic+blacklist (токсичный), exit→probation (мягкий cooldown).
3. P1 (O(1) Lookup): black_list и live_swarm конвертируются в set при загрузке (O(1) проверка in), обратно в sorted list при сохранении.
4. P2 (Async I/O): flag_file.unlink() → asyncio.to_thread — event loop не блокируется при каскадных ликвидациях.
5. P3 (Probation SSOT): trailing_stop_paper_timeout_end из state-файла заменён на probation_* ключи в config.json. Supervisor — единственный контроллер кулдаунов.
6. Silent Failure: stop_bot bare except:pass → конкретные исключения (FileNotFoundError, OSError).

Изменения
main.py: -50 строк (удалены блоки записи config.json)
supervisor.py: +85 строк (B3 маршрутизация, P1 set, P2 async, P3 probation, error handling)
config.json: +2 ключа (probation_paper, probation_real)
tests/test_supervisor.py: +8 тестов, 1 обновлён
tests/test_main.py: +2 теста
CLAUDE.md: Change Log обновлён

Результат
30/30 тестов пройдены за 4.82s. SSOT-паттерн внедрён. Race condition устранён. Exit/stop изоляция работает. Async I/O не блокирует event loop.

2026-07-21 — B1 Fix: TS Flag Persistence (zombie process kill)
Что сделано
Исправлен критический баг B1 (Score 8.3): после срабатывания trailing stop и рестарта supervisor, зомби-процесс never cleaned up. Причина: HEAL REJECTED в enforce_swarm_consistency() сбрасывал TS флаги ДО того, как Reaper Guard успевал убить зомби. Reaper Guard проверял state файл — флаги уже False — не видел зомби.

Фикс: 1 строка в supervisor.py:197-199 — await stop_bot(ticker, is_paper=False) в HEAL REJECTED блоке ДО сброса флагов. stop_bot вызывает pm2 delete, убивая зомби-процесс.

Изменения
supervisor.py:197-199 — +await stop_bot(ticker, is_paper=False)
tests/test_supervisor.py — +2 unit-теста (call order tracking + flag state verification)

Результат
19/19 тестов пройдены (17 существующих + 2 новых B1). Фикс атомарный, минимально инвазивный. B1 помечен как ✅ ИСПРАВЛЕН в ROADMAP.md.

2026-07-21 — Documentation sync (ROADMAP, STATUS, PROJECT_INDEX)
Что сделано
ROADMAP.md описывал "1 REAL bot (INJUSDT)" — актуально: 4 бота (GRASS, UNI, VVV, YFI). STATUS.md устарел по тикерам, PROJECT_INDEX.md по конфиг-параметрам.

Обновлены все три документа до актуального состояния из config.json (SSOT).

Изменения
ROADMAP.md — Текущее состояние: 4 REAL bots, 20 USDT/bot, ~7 min interval, TS OFF, KPI таблица
STATUS.md — Live Swarm: GRASS/UNI/VVV/YFI, TS=0.001% (OFF), guards=6 active, blacklists
PROJECT_INDEX.md — Config params, tickers, changelog entries, updated date

2026-07-21 — Hermes Agent upgrade 0.17.0 → 0.18.2
Что сделано
Portable USB venv (D:\Hermes-USB-Portable-main\data\hermes-agent\venv) обновлён с 0.17.0 до 0.18.2. Config migrated: v24 → v33.

Изменения
hermes-agent: 0.17.0 → 0.18.2
cryptography: 48.0.0 → 46.0.7 (required by hermes 0.18.2)
Config: v24 → v33 (model_catalog.ttl_hours, agent.verify_on_stop)

Ключевые улучшения v0.18.2: stream-stale circuit breaker, PTY session management, approval gate fixes, MCP stability.

2026-07-16 — Dashboard: убийство зомби-процессов + чистка кода
Что сделано
Глобальный min_notional_usdt: 6.1 блокировал GRASSUSDT (Binance 5.03) и VVVUSDT (5.05) при 3-4% отклонениях. Config задавал единый порог для всех тикеров, хотя Binance minimums у каждого свои.

Добавлен per-ticker lookup из exchange_info → MIN_NOTIONAL filter с 2% буфером на динамику Binance minimums.

Исправлен KeyError 'minNotional' — Binance Futures API использует поле 'notional', не 'minNotional'. Исправлено через f.get("minNotional") or f.get("notional").

Изменения
main.py:489 — min_notionals extraction
main.py:941-945 — active_min_notional = max(config_min, exchange_min * 1.02)
config.json — min_notional_usdt: 6.1 → 5.1
tests/test_main.py — 6 unit-тестов

Effective minimums
GRASSUSDT: max(5.1, 5.03*1.02) = 5.13
VVVUSDT: max(5.1, 5.05*1.02) = 5.15
YFIUSDT: max(5.1, 7.00*1.02) = 7.14
UNIUSDT: max(5.1, 7.05*1.02) = 7.19

Fallback (нет exchange info) = config = 5.1 (без буфера)
Результат
4 real-бота online, 0 restarts. YFI/UNI ребалансируют при ~5.1% вместо 5.0% (0.1% разница — допустимая цена буфера).

2026-07-16 — Dashboard: убийство зомби-процессов + чистка кода
Что сделано
Диагностика проблемы логина в dashboard — пароль Rebalancer0ID не проходил, хотя bcrypt hash валидный
Обнаружена причина — 17 зомби-процессов Python от многочисленных background=true запусков занимали порт 8080. Новый dashboard не мог занять порт, curl попадал на старый процесс с устаревшим кодом
Убиты зомби через wmic process where "ProcessId=..." delete
Dashboard запущен через PM2 — работает стабильно
Убран debug-мусор из dashboard.py и login.html (файловые логи, debug_msg, отладочные flash)
Урок
Не использовать terminal(background=true) для серверных процессов без process kill после тестов
PM2 предпочтительнее для long-running процессов на Windows
При отладке Flask на Windows: taskkill /F /PID не работает из MSYS (режет /F), использовать wmic
Изменённые файлы
Файл	Изменение
dashboard/dashboard.py	Убран debug-код из login route, возврана чистая версия
dashboard/templates/login.html	Убран debug-блок
2026-06-03 — Масштабная сессия: отладка и подготовка к реальному запуску
Что сделано
VIRTUAL OFF — отключена виртуальная нога (share 0.35 → 0.0). Чистый хедж 50/50 LONG/SHORT.

Ротация переписана — score-driven selection. Прибыльные боты (profit > 0) всегда остаются в рое. Замене подлежат только убыточные.

Net Move Guard — новый защитный механизм. Если цена >1.5% за 30с — блокировка всех действий.

PnL Guard — блокировка продажи излишков при отрицательном hedge PnL. Логирование только при смене состояния.

config.json защищён — supervisor не перезаписывает tickers. Fallback из сканера при пустом списке.

Исправлен бэктестер — замена available_funds на val_cash. Бэктестер снова работает.

Оптимизация порогов — surplus 1% / deficit 4% (было 1.5% / 3%).

Изолированная маржа → Кросс-маржа — в main.py изменено на CROSS. При хедже 50/50 это безопаснее.

Margin ratio пороги — warning 3.0x, critical 1.5x (было 5.0x / 2.0x).

Результаты бэктестов
Тикер	48ч PnL	MaxDD	Cycles
HEIUSDT	+0.23%	0.23%	10
WLDUSDT	+2.07%	—	—
STORJUSDT	+1.95%	—	—
FORMUSDT	+1.68%	—	—
Результат
Запущен 1 бот на реальный счёт:

HEIUSDT, 180 USDT, x5, кросс-маржа
Margin ratio: 20.74x
PnL: +0.23 USDT (11 циклов)
Известные риски
Liquidation guard не работает при кросс-марже (Binance не даёт liquidationPrice)
При экстремальных движениях >50% возможна потеря всего баланса (крайне маловероятно при хедже)
2026-05-30 — Катастрофа: ликвидация PORTALUSDT SHORT
Событие
PORTALUSDT SHORT x5 ликвидирована биржей при росте цены +17% за 1ч (0.01034 → 0.01213). Бот не смог восстановить позицию. Supervisor restart убил real-portal PM2 процесс без перезапуска. Средства выведены с аккаунта, реальная торговля отключена.

Хронология
15:00  PORTALUSDT = 0.01034, SHORT маржа ≈ 43.50 USDT
15:00-15:06  Цена растёт до 0.01193, unrealized PnL → -21.93 USDT
15:06  Heartbeat: S:0.0% — ликвидирована биржей
15:11  Rebalance #8 пытается восстановить: BUY 15468.5 LONG + SELL 19202.3 SHORT @ 0.01175
15:11:32  Последняя запись в real_PORTALUSDT.log
15:37  Supervisor service restart → убивает все PM2 процессы (SIGINT)
15:37  real-portal (PID 15616) потерян, НЕ перезапущен
15:37  Cycle Complete. REAL Swarm: [] — оба real-бота мертвы
Анализ: корневые причины
Нет predictive liquidation guard — только post-factum (margin <= 0). Бот узнаёт о ликвидации постфактум.
Supervisor теряет real-ботов — при своём рестарте убивает все PM2 процессы через SIGINT, но не перезапускает real-ботов из live_swarm.
Рассинхронизация paper/real (КРИТИЧЕСКИЙ БАГ) — paper mode НЕ симулировал liquidation price. Супервизор отправлял в реальную торговлю тикеры, которые "на бумаге" выглядели безопасно, но в реальных условиях приводили к ликвидации.
Решение: 3 уровня защиты
Уровень 1 — connector.py: get_position_risk()

Для каждой позиции считает: liq_price, distance_pct, unrealized_pnl, margin
Источник: futures_position_information + futures_mark_price от биржи
Уровень 2 — main.py: Guard #4 Liquidation Distance Monitor

Работает для BOTH paper и real
Пороги: liquidation_distance_warn_pct: 15%, liquidation_distance_crit_pct: 8%
Real: emergency market close позиции через PortfolioExecutor.execute_market_order()
Paper: _handle_liquidation_guard() — единая функция, paper симулирует PnL + сброс shadow_state
При открытии позиции в paper mode рассчитываются long_liquidation_price / short_liquidation_price по формуле Binance:
LONG liq  = entry × (1 - 1/leverage + mmr)   где mmr = 0.004 (0.4%)
SHORT liq = entry × (1 + 1/leverage - mmr)
При закрытии позиции liq price сбрасывается в 0.0
Уровень 3 — supervisor.py: _ensure_real_bots_alive()

После каждого swarm цикла проверяет: все ли tickers из live_swarm имеют PM2 процессы?
Если процесс отсутствует → автоперезапуск через start_bot()
Изменённые файлы
Файл	Изменение
connector.py	get_position_risk() — данные о риске ликвидации от биржи
main.py	Simulated liq price в paper mode, Guard #4 paper+real, _handle_liquidation_guard()
supervisor.py	_ensure_real_bots_alive() + liq price в начальный shadow_state
config.json	liquidation_distance_warn_pct: 15.0, liquidation_distance_crit_pct: 8.0
backtest_rebalance.py	Predictive liquidation guard (формула Binance, порог 8%)
CLAUDE.md	Safety Mechanisms: Guard #3 + Guard #5
Решения по дизайну
ISOLATED margin оставлен — cross margin рискованнее для аккаунта (одна позиция может убить весь аккаунт). Isolated = ликвидация только маржи позиции.
Position-level trailing stop НЕ добавляем (FMProducer: не даём шансов манипуляторам/охотникам за стоплоссами)
LIVE TRADING отключена до полного тестирования новых защит
2026-05-31 — Оптимизация стоплоссов + запуск Combat
Что
Optuna-оптимизация параметров стоплосса, 7-дневный бэктест, запуск реальной торговли (Combat).

Почему
Параметры стоплосса были выбраны эмпирически. Нужна систематическая оптимизация.

Результат
Optuna: 101 trial, 92 completed, 44с. Оптимальные параметры:
max_drawdown_limit: 33% → 22%
equity_trailing_stop_pct: 10% → 24%
equity_trailing_stop_activation_pct: 2% → 7.5%
equity_trailing_stop_timeout_sec: 60s (не оптимизировался)
7-дневный бэктест (20 тикеров): 18/20 прибыльных, avg +9.78%, avg DD 7.99%, 0 TS triggers
Combat запущен: live_swarm = [PORTALUSDT, VTHOUSDT], позиции открыты на Binance
Incubator (paper): +14.57 USDT за 8ч (12.7% ROI), 13/30 тикеров прибыльных
Проблема: NFPUSDT -14.56 USDT (28% дамп, TS сработал). Решение: min_cycles_for_rank=10 не пускает в Combat.
2026-05-31 — Настройка мониторинга
Что
Создан health_check.py, cron job для автоматического мониторинга.

Почему
Ручная проверка PM2/логов/state занимает время. Автоматизация = быстрая реакция на проблемы.

Результат
health_check.py — проверяет PM2, real_state, blacklist, ошибки
Cron job каждые 15 мин → Telegram отчёт
2026-05-31 — Рабочие соглашения (OWL ↔ FMProducer)
Что
Описан workflow взаимодействия, приоритеты, ограничения.

Результат
progress.md переписан (дата → что → почему → результат)
MEMORY обновлён (project-specific контекст вместо общих заметок)
Skill prosperous-bot создан
CLAUDE.md актуализирован на основе PDF
experiments.md для структурирования гипотез
2026-05-30 — Рефакторинг калькулятора
Что
calculate_deviations() теперь возвращает dict вместо list. calculate_rebalance() обновлён.

Почему
Bugs: 'str' object has no attribute 'get' при итерации по dict keys вместо list.

Результат
Все backtest движки синхронизированы с calculator.py. run_all_backtests.py — 345 тикеров, 0 ошибок.

2026-05-30 — Voice STT (Hermes Gateway)
Что
Добавлено распознавание голосовых сообщений в Telegram-боте (Vosk + ffmpeg).

Результат
