## Initial setup (CI-first, cross-platform)

**Цель:** среда разворачивается одинаково на Windows/macOS/Linux и не требует `source .venv/bin/activate`. Скрипты для людей:

- POSIX: `scripts/init_env.sh`
- Windows: `scripts/init_env.ps1`

Оба скрипта используют `python -m …` с интерпретатором из `.venv`, а не активацию шелла. Это совпадает с тем, как запускается CI.

### Политика тестов
- **Источник истины — GitHub Actions.** Jules **не** запускает локальные тесты; он только применяет diff и открывает Draft PR (probe для `feature/ci-*` / strict для остальных).
- **Probe (быстрый дым):** ветки `feature/ci-*` → `-m "not integration"`, без порога покрытия.
- **Strict (боевой):** все прочие ветки → весь набор тестов + `--cov-fail-under=90`.

### Быстрые команды для разработчика (локально)
```bash
# POSIX
./scripts/init_env.sh
.venv/bin/python -m pytest -q -m "not integration" --maxfail=1
```
```powershell
# Windows
pwsh -File .\scripts\init_env.ps1
.\.venv\Scripts\python -m pytest -q -m "not integration" --maxfail=1
```

### Триггер CI без коммита в код
- **Actions → Python CI → Run workflow** (если включён `workflow_dispatch`), либо
- «пинг-коммит» в `tests/**` (с path-фильтрами это гарантированно запускает пайплайн).

### Замечания
- Никаких «зашитых» путей вида `C:\…` в тестах; для путей используем утилиты `pathlib`.
- Все параметры — только через `unified_config*.json`; тесты не меняют логику и не вшивают значения.

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
- Единый конфиг: `unified_config*.json` (без `asset_distribution`, без 5L/5S).
- Без асинхронных обещаний: результат — в текущем ответе.

## Jules и атомарность задач
Если часть работы выполняется через Jules, код, скрипты и diff-патчи должны быть подготовлены в удобном для него формате:
 - Делите изменения на небольшие атомарные задачи.
 - Готовьте единый unified diff для каждого логического изменения.
 - Формулируйте описания задач коротко и ясно, без лишнего контекста.
 - Указывайте необходимые шаги (команды) и ожидаемые артефакты.

### Шаблон задания для Jules (вставлять в сообщение модели)
```
[JULES_TASK]
title: "feat: <module>: <short change>"
branch: "feature/<slug>"
scope:
  files: ["path/to/a.py", "path/to/b.py"]
  max_changed_files: 20
  max_diff_lines: 300
rules:
  unified_diff_only: true
  no_binaries_over_mb: 5
  sequential_prs: true
ci:
  run_pytest_cov: "pytest -q --maxfail=1 --disable-warnings --cov=. --cov-report=term-missing"
  run_smoke_backtest: true
artifacts:
  reports_dir: "reports/"
  large_files_as_ci_artifacts_only: true
kpi_impact:
  metrics: ["sharpe_ratio","profit_factor","max_drawdown_percent","win_rate_percent"]
  expectation: "не ухудшить, целью повысить Sharpe; Max DD в допусках"
acceptance:
  - "Открыт PR с описанием и списком файлов"
  - "CI зелёный, отчёты в reports/"
rollback: "revert PR / feature-flag / Safe-Mode при деградации KPI"
deliverables:
  - "unified diff в чате + команды применения"
  - "PR description: цель, KPI, риски, откат, артефакты"
[/JULES_TASK]
```

### Быстрый recovery при зависании UI
- Перезагрузка вкладки/инкогнито; ориентир на GitHub PR/Actions.
- Всегда дублировать дифф текстом и приложить команды `git apply`/`gh pr create`.

## Быстрые команды
- pytest -q --maxfail=1 --disable-warnings --cov=. --cov-report=term-missing
- git checkout -b feature/xyz
- git apply --index changes.patch && git commit -m "feat: xyz"
- git push -u origin feature/xyz
