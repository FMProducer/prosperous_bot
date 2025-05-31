# BTC‑Neutral Rebalance Module — **v2 Road‑map**

*(Trading AI Agent | «режим деньги нейросеть» | обновлено под 90 % unit‑coverage + Policy Layer)*

---

## 0 ⸺ Цели обновления

- **Unit‑tests ≥ 90 % coverage.**  
- **Policy Layer**: контроль секретов, ресурсов, побочных эффектов.  
- Интегрировать оба блока в CI/CD и runtime Trading AI Agent.

---

## 1 ⸺ Структура проекта (добавления)

adaptive\_agent/

├── policy\_layer.py          \# runtime‑политики

├── tests/                   \# unit‑тесты (pytest)

│   ├── conftest.py

│   ├── test\_portfolio.py

│   ├── test\_exchange.py

│   ├── test\_policy.py

│   └── … (другое)

├── exchange\_gate.py

├── portfolio\_manager.py

├── rebalance\_engine.py

└── …

---

## 2 ⸺ Unit‑тесты 90 %

| Шаг | Действие | KPI |
| :---- | :---- | :---- |
| **2.1** | Создать `tests/` и подключить **pytest + pytest‑cov** | baseline |
| **2.2** | Минимум один тест‑модуль **на каждый публичный класс/функцию** | покрытие по файлу ≥90 % |
| **2.3** | Использовать **mocks/fixtures** для внешних API (Gate, WebSocket) → без сетевых вызовов | side‑effects = 0 |
| **2.4** | В `pyproject.toml` добавить:   \`\[tool.pytest.ini\_options\] addopts \= "--cov=adaptive\_agent \--cov-fail-under=90"\` | build fail \< 90 % |
| **2.5** | GitHub Actions job: `pytest -q && coverage xml` → badge | CI‑enforced |

\[tool.pytest.ini\_options\]

addopts \= "--cov=adaptive\_agent \--cov-report=term-missing \--cov-fail-under=90"

*Документация*: `--cov-fail-under` ломает build, если покрытие \< порога【turn0search1】【turn0search0】.

---

## 3 ⸺ Policy Layer (`policy_layer.py`)

**Задача** — предотвратить утечки секретов, ограничить ресурсы и побочные эффекты.

| Под‑модуль | Функция | Реализация |
| :---- | :---- | :---- |
| **SecretsGuard** | Проверка config/ENV на шаблоны ключей (AWS, Gate), интеграция **GitHub Secret Scanning** API | regex‑скан \+ fail‑fast【turn0search2】【turn0search6】 |
| **ResourceGuard** | Ограничения CPU/RAM (cgroups), `ulimit`, Docker `mem_limit`, `pids_limit` | docker‑flags \+ psutil runtime мониторинг |
| **PolicyEngine** | Политики OPA/Rego:  — дисковый путь \`read‑only\`, — network egress → \`gate.io\` only | embed OPA (wasm)【turn0search4】 |
| **ContainerGuard** | Запуск контейнера с `--cap-drop ALL`, `no-new-privileges`, rootFS ro | Docker Security Cheat Sheet правила【turn0search5】 |
| **CIHook** | Pre‑commit скрипт: `detect-secrets scan --baseline` .secrets.baseline\`\` | prevents commit secrets |

### Runtime API

from adaptive\_agent.policy\_layer import guard

guard.check\_secrets(cfg)

guard.enforce\_resources()

guard.load\_policies("policy.rego").validate(event)

---

## 4 ⸺ CI/CD Pipeline

1. **Stage 1** — Lint (`ruff`) \+ `pytest --cov`.  
2. **Stage 2** — **detect‑secrets** \+ OPA policy test (`opa test`).  
3. **Stage 3** — Docker build with hardened flags (`--cap-drop ALL`, `--read-only`).  
4. **Stage 4** — Deploy if coverage ≥ 90 % и policy tests pass.

GitHub‑Actions secret‑scanning работает автоматически → alerts repo admins【turn0search2】.

---

## 5 ⸺ Док‑тесты & Mock‑Best‑Practices

- Использовать **pytest‑mocker** \+ fixtures для `aiohttp.ClientSession` → без сетевых побочек【turn0search7】.  
- Для ENV: `monkeypatch.setenv("GATE_KEY","test")` → тесты без настоящих секретов【turn0search9】.

---

## 6 ⸺ Risk Matrix (Policy Layer)

| Риск | Политика | Инструмент |
| :---- | :---- | :---- |
| Утечка API‑ключей | RegEx‑скан \+ GitHub secret‑scanner | SecretsGuard |
| Неограниченный RAM | `docker run --memory 512m --memory-swap 512m` | ResourceGuard |
| Запрет exec/privileged | `--cap-drop ALL --no-new-privileges` | ContainerGuard |
| Подмена цены WS | OPA правило: `input.price_source in ["binance"]` | PolicyEngine |
| Разрастание логов | log‑rotate policy 50 MB | ResourceGuard |

---

## 7 ⸺ Обновление Road‑map Milestones

| ✔ | Шаг |
| :---- | :---- |
| ☐ Unit‑coverage ≥ 90 % (CI gate) |  |
| ☐ Policy Layer modules coded |  |
| ☐ OPA policies \+ detect‑secrets baseline |  |
| ☐ Docker hardening flags в `docker-compose.yml` |  |
| ☐ Policy tests pass (`opa test`) |  |
| ☐ Coverage badge в README |  |
| ☐ SecretsGuard & ResourceGuard активны в runtime |  |

---

*Документ создан:* 2025-05-18 18:34 UTC  
