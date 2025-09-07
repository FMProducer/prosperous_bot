# How-to work (короткая памятка)
**SoT:** весь код/конфиги/отчёты — только в GitHub (`FMProducer/prosperous_bot`). В «Файлах проекта» держим только метадокументы.

## Базовый цикл работы
1) Постановка задачи: цель + ограничения (KPI: Sharpe ≥ 1.5, PF ≥ 1.3, Max DD < 20%).
2) Патчи: unified diff (код + юнит-тесты + влияние на KPI/риски).
3) PR: ветка `feature/*`, PR с диффом, описанием и чек-листом.
4) CI: `pytest --cov` + смоук-бэктест; артефакты — в `reports/`.
5) Ревью: анализ логов CI и метрик; при необходимости — новые дифы.
6) Merge: только при «зелёном» CI и обновлённой документации (roadmap/Manual).

## Сообщения для эффективности
- Начинайте с TL;DR и перечня затрагиваемых файлов.
- Просите диф сразу с тестами и прогнозом KPI.
- Для оптимизаций указывайте период, частоту, плечо, комиссию, целевую метрику.

## Правила
- SoT: не дублируем исходники в «Файлы проекта».
- Секреты: только переменные окружения/Secrets CI.
- Отчеты: `reports/` в репозитории (CSV/HTML/PNG).
- Единый конфиг: `unified_config*.json` (bez `asset_distribution`, bez 5L/5S).
- Без асинхронных обещаний: rezultat — v tekushchem otvete.

## Быстрые команды
- pytest -q --maxfail=1 --disable-warnings --cov=. --cov-report=term-missing
- git checkout -b feature/xyz
- git apply --index changes.patch && git commit -m "feat: xyz"
- git push -u origin feature/xyz
