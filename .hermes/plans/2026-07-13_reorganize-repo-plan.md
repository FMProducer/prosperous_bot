# План: Реорганизация C:\Python и репозитория prosperous_bot

> Дата: 2026-07-13
> Статус: ДИСКУССИЯ — ничего не менять
> Контекст: Jules анализирует репозиторий, параллельно планируем реорганизацию

---

## Текущее состояние — диагноз

### Что имеем на диске (C:\Python)

| Директория                  | Назначение                        | Статус      |
|-----------------------------|-----------------------------------|-------------|
| `Prosperous_Bot/`           | Git-репозиторий (всё в куче)      | Активный    |
| ├─ `futures_portfolio/`     | Market Neutral 2.0                | ★ PROD      |
| ├─ `src/prosperous_bot/`    | Legacy "Trading AI Agent"         | Устаревший  |
| ├─ `freqtrade/`             | Clone фреймворка (454 .py)        | Не используется |
| ├─ `third_party/`           | rl-trading-binance fork           | Эксперимент |
| ├─ `tests/`                 | Старые тесты                      | Legacy      |
| ├─ `output/`, `graphs/`     | Артефакты запусков                | Мусор       |
| ├─ `venv/`                  | Virtualenv                        | Не в git?   |
| └─ root .py                 | Смесь конфигов, утилит, мусора    | Хаос        |
| `CodeLLM/`                  | Архив исследований (114 .py)      | Архив       |
| ``OLD/`                     | Legacy архив (50 .py + zip)       | Архив       |
| `.venv/`                    | Root venv                         | Системный   |
| `Git/`                      | Git for Windows                   | Утилита     |
| `prosperous_bot.bak/`       | Git bare repo backup              | Бэкап       |

### Что имеем в Git

| Метрика                | Значение  | Проблема                        |
|------------------------|-----------|----------------------------------|
| Tracked файлов         | 1 720     | Огромно для одного проекта       |
| Remote веток           | 120       | Катастрофа (auto-generated PRs)  |
| freqtrade в git        | 693 файла | Фреймворк не должен быть в git  |
| venv в git             | 504 файла | НИКОГДА не трекать venv          |
| third_party в git      | 180 файлов| Fork чужого проекта              |
| output/ в git          | 56 файлов | Артефакты, должны быть в .gitignore |
| graphs/ в git          | 32 файла  | Артефакты, должны быть в .gitignore |
| .gitignore             | ~200 строк| Фрагментарный, патч-за-патчем    |

### Корневые проблемы

1. **Один репозиторий — 5+ систем**: active trading, legacy code, research experiments, forked framework, utility scripts
2. **venv трекается в git** — 504 файла мусора в истории
3. **freqtrade клон трекается в git** — 693 файла чужого кода вместо dependency
4. **120 remote веток** — auto-generated PR branches от Jules/Claude never cleaned up
5. **Нет разделения dev/staging/prod** — всё в master
6. **.gitignore фрагментарный** — добавляется по файлам, а не по паттернам

---

## Варианты реорганизации

### Вариант A: Monorepo с чисткой (МИНИМАЛЬНЫЙ РИСК)

**Идея:** Остаёмся в одном репозитории, но наводим порядок.

**Структура:**
```
prosperous_bot/
├── futures_portfolio/     — ★ Активная продакшн-система
│   ├── src/               — Ядро: main, connector, calculator, executor
│   ├── supervisor/        — Supervision: supervisor, aggregator
│   ├── optimization/      — Optuna-скрипты
│   ├── backtest/          — Backtesting engine
│   ├── dashboard/         — Web UI
│   ├── tools/             — Утилиты (telegram, health, etc.)
│   ├── tests/             — Unit-тесты
│   └── config/            — Конфиги (config.json, etc.)
├── legacy/                — Архив: src/prosperous_bot, CodeLLM, `OLD
├── docs/                  — Общая документация
├── .github/workflows/     — CI/CD
├── pyproject.toml         — Dependencies (freqtrade через pip)
├── requirements.txt       — Lock file
└── README.md
```

**Действия:**
1. Удалить venv из git history (git filter-branch или BFG)
2. Удалить freqtrade/ из git, добавить в requirements.txt
3. Удалить output/, graphs/ из git, добавить в .gitignore
4. Переместить legacy код в legacy/ архив
5. Переименовать и разделить futures_portfolio по подмодулям
6. Почистить 120 remote веток (git remote prune)
7. Переписать .gitignore с нуля

**Плюсы:** Минимальные риски, история сохраняется, PM2-пути меняются минимально
**Минусы:** Всё ещё monorepo, legacy в том же репо

---

### Вариант B: Multi-Repo Split (ЧИСТАЯ АРХИТЕКТУРА)

**Идея:** Каждая система — отдельный репозиторий.

**Репозитории:**
```
github.com/FMProducer/
├── prosperous-bot          — ★ Активная продакшн-система (только futures_portfolio)
├── prosperous-bot-legacy   — Архив: src/prosperous_bot, старые тесты
├── prosperous-bot-research — CodeLLM, adaptive filter, ML эксперименты
├── rl-trading-binance      — Fork/ветка (upstream: YuriyKolesnikov)
└── prosperous-bot-shared   — Общие библиотеки (если потребуется)
```

**Действия:**
1. Создать новый чистый repo `prosperous-bot` с futures_portfolio/
2. Мигрировать state files, config, PM2-конфиги
3. Старый repo переименовать в `prosperous-bot-archive`
4. CodeLLM и `OLD` вынести в `prosperous-bot-research`
5. rl-trading-binance оставить как fork

**Плюсы:** Чистое разделение, каждый repo = один проект, простой CI/CD
**Минусы:** Потеря git-истории (или сложная миграция), PM2-пути меняются, больше репозиториев для поддержки

---

### Вариант C: Monorepo + Git Submodules (ГИБРИДНЫЙ)

**Идея:** Один репозиторий, но внешние зависимости через submodules.

**Структура:**
```
prosperous_bot/
├── futures_portfolio/     — ★ Продакшн (основной код)
├── .gitmodules            — freqtrade как submodule
├── docs/                  — Документация
├── scripts/               — Утилиты запуска
├── pyproject.toml
└── README.md
```

**Действия:**
1. freqtrade → git submodule (или pip dependency)
2. third_party → git submodule (rl-trading-binance)
3. Остальное — чистка и реструктуризация

**Плюсы:** Гибкость, внешние проекты изолированы
**Минусы:** Submodules — pain point (забыл update, detached HEAD, etc.)

---

### Вариант D: Hybrid — Чистка + Archive Branch (ПРАГМАТИЧНЫЙ)

**Идея:** Чистим master, legacy уходим в отдельную ветку, не трогаем структуру глубоко.

**Действия:**
1. **Создать ветку `archive/legacy-2025`** со всем legacy-кодом (src/, CodeLLM, `OLD)
2. **Удалить legacy из master** (git rm --cached + .gitignore)
3. **Удалить venv из git** (BFG Repo-Cleaner)
4. **Удалить freqtrade из git**, добавить в requirements.txt
5. **Удалить output/, graphs/ из git**
6. **Переписать .gitignore** — глобальные паттерны вместо конкретных файлов
7. **git remote prune** — удалить 120 мёртвых веток
8. **Переименовать корень** futures_portfolio/ → project root (поднять файлы вверх)

**Плюсы:** Быстро, безопасно, история в archive-ветке, минимальные изменения путей
**Минусы:** Не идеальная архитектура, но практично

---

## Рекомендация: Вариант D (Прагматичный) + постепенная миграция к B

Причины:
1. **Prod работает** — любая радикальная миграция = риск downtime
2. **PM2-пути захардкожены** — смена структуры = перенастройка всех процессов
3. **120 веток** — сначала почистить, потом думать о split
4. **Jules уже анализирует** — дождаться инвентаря, потом действовать
5. **Legacy нужен для справки** — archive-ветка сохраняет историю

---

## Пошаговый план (Вариант D)

### Этап 1: Подготовка (без изменений в prod)

- [ ] Дождаться результатов Jules (REPOSITORY_INVENTORY.md)
- [ ] Сверить инвентарь Jules с нашим анализом C:\Python
- [ ] Определить точный список файлов для удаления из git
- [ ] Согласовать с пользователем — какие файлы точно не нужны

### Этап 2: Чистка Git History (BFG Repo-Cleaner)

- [ ] Удалить venv/ из истории (504 файла)
- [ ] Удалить freqtrade/ из истории (693 файла)
- [ ] Удалить output/, graphs/ из истории (88 файлов)
- [ ] Удалить miniconda3/ из истории
- [ ] Удалить vosk-model-*/ из истории
- [ ] Удалить ffmpeg-*/ из истории
- [ ] **ВНИМАНИЕ:** BFG перезаписывает историю → все clone'ы сломаются
- [ ] **План B:** Если BFG слишком рискованно — просто git rm + .gitignore (история останется)

### Этап 3: Чистка веток

- [ ] `git remote prune origin` — удалить невалидные remote tracking branches
- [ ] Вручную: удалить все auto-generated PR branches (120 штук)
- [ ] Оставить: master, архивные ветки, feature-ветки в работе

### Этап 4: Архивация legacy

- [ ] Создать ветку `archive/legacy-2025` из текущего master
- [ ] Удалить из master: `src/prosperous_bot/`, `CodeLLM/`, ``OLD/``
- [ ] Удалить из master: `tests/` (старые тесты, futures_portfolio/tests остаются)
- [ ] Удалить из master: `third_party/` (перенести в archive-ветку)

### Этап 5: Переписать .gitignore

- [ ] Заменить ~200 строк конкретных паттернов на ~30 глобальных правил
- [ ] Ключевые правила:
  ```
  venv*/
  .venv*/
  miniconda3/
  __pycache__/
  *.pyc
  output/
  graphs/
  temp_plots/
  htmlcov/
  .coverage
  *.log
  *.pkl
  *.npz
  *.db
  *.sqlite*
  state_*.json
  paper_state_*.json
  real_state_*.json
  shadow_state_*.json
  paper_shadow_*.json
  .env
  secrets.txt
  ```

### Этап 6: Реорганизация futures_portfolio (опционально)

- [ ] Поднять ключевые файлы в корень проекта:
  ```
  futures_portfolio/main.py → ./main.py
  futures_portfolio/config.json → ./config.json
  futures_portfolio/supervisor.py → ./supervisor.py
  ```
- [ ] Остальное — по поддиректориям
- [ ] **Только после согласования с пользователем**

### Этап 7: Обновление документации

- [ ] README.md — новая структура
- [ ] CLAUDE.md — обновить пути
- [ ] Obsidian — обновить C-Python-Full-Inventory.md
- [ ] .hermes/ — обновить планы и скиллы

---

## Риски и митигация

| Риск                                    | Вероятность | Митигация                              |
|-----------------------------------------|-------------|----------------------------------------|
| PM2-процессы сломаются после реорганизации | Высокая     | Не трогаем prod-пути до согласования  |
| Git history сломается (BFG)             | Средняя     | Использовать git rm + .gitignore вместо BFG |
| Legacy код потеряется                   | Низкая      | archive-ветка + бэкап на диске         |
| Jules найдёт что-то неожиданное         | Средняя     | Дождаться результатов перед действиями |
| Забытые зависимости между системами      | Средняя     | Инвентарь Jules покажет перекрёстные ссылки |

---

## Открытые вопросы

1. **BFG или git rm?** — BFG чище (удаляет из истории), git rm проще (история остаётся)
2. **freqtrade как pip dependency или submodule?** — pip проще, submodule гибче
3. **Нужен ли split на несколько репозиториев?** — Зависит от планов на развитие
4. **Трогать ли prod-пути?** — futures_portfolio/ может остаться как есть
5. **rl-trading-binance — submodule или fork?** — Сейчас fork, submodule изолированнее

---

## Следующие шаги

1. **Ждём Jules** — пусть закончит инвентарь
2. **Сверяем результаты** — наш анализ vs Jules
3. **Согласовываем план** — выбираем вариант и этапы
4. **Действуем** — поэтапно, с откатом на каждом шаге
