# Repo-State Header

* Ветка (default): `prosperous_bot`
* HEAD: `6fcaf1e0da9b314abdaca1754ff7fe155926b832` — «docs: new repo» (27 Sep 2025) ([GitHub][1])
* Ссылка на коммит: [https://github.com/FMProducer/prosperous_bot/commit/6fcaf1e0da9b314abdaca1754ff7fe155926b832](https://github.com/FMProducer/prosperous_bot/commit/6fcaf1e0da9b314abdaca1754ff7fe155926b832) ([GitHub][1])

**Что видно по репозиторию сейчас:** каталог `third_party/rl-trading-binance` присутствует в HEAD (GitHub скрывает часть дерева из-за объёма, но корневые файлы и сама папка зафиксированы в коммите) ([GitHub][1]). История ветки подтверждает дату и актуальность HEAD на 27 Sep 2025 ([GitHub][2]).

---

## TL;DR

Ваш переход на «raw combined» через `websocket-client` логичен для лёгкого и независимого от SDK подключения. Чтобы эта заготовка была **надёжной в проде** и не «ела» ресурсы на слабом сервере, проверьте и добавьте: корректный split потоков (≤1024 на соединение), `ping/pong` и авто-переподключение (24-часовой лимит), приведение стримов к нижнему регистру, и idempotent upsert в БД (PK/UNIQUE по ключевым полям). Ниже — точечный чек-лист ревью, схемы таблиц PostgreSQL под kline+aggTrade и минимальные правки в клиенте `websocket-client` для стабильности.

---

## Быстрая проверка изменений (review checklist)

**WebSocket (Binance Futures USDT-M):**

* База для маркет-стримов: `wss://fstream.binance.com` (single `/ws/<stream>` или combined `/stream?streams=<s1>/<s2>/…`) ([Центр разработчиков Binance][3]).
* Ограничения: **до 1024** стримов на одно соединение; лимит входящих control-сообщений (PING/PONG/subs) в сек. (10 для деривативов) — превышение → disconnect/бан IP ([Центр разработчиков Binance][3]).
* Сеанс действителен **не более 24 часов**; сервер шлёт `ping` каждые ~3 мин, если за 10 мин не получить `pong` — разрыв. Настройте keep-alive и ротацию соединений по расписанию. ([Центр разработчиков Binance][4])
* Имена стримов **только в нижнем регистре** (например, `btcusdt@kline_1m`, `btcusdt@aggTrade`) ([Центр разработчиков Binance][3]).
* Форматы полезной нагрузки:
  • **Kline** — поле `k` с `t` (open time), `o/h/l/c`, `v`, `n`, `x` (close flag) и т. д. ([Центр разработчиков Binance][5])
  • **aggTrade** — поля `a` (aggregate trade id), `p`, `q`, `T`, `m`, … — пригодно для PK/идемпотентности ([Центр разработчиков Binance][6]).

**Клиент `websocket-client`:**

* Используйте `WebSocketApp(...).run_forever(ping_interval=, ping_timeout=)` — это даёт автослои событий и регулярные ping; **сам по себе** `run_forever` **не** реализует «умный» авто-reconnect, его надо обернуть циклом с backoff/джиттером и таймаутами. ([websocket-client.readthedocs.io][7])

**Распределение стримов:**

* Разделяйте соединения: например, kline-1m отдельно от aggTrade, и шардируйте tickers пачками по N (например, 400–800), чтобы не приближаться к потолку 1024/conn и упростить деградацию при сбоях. Ограничение 1024/соединение — из оф. документации. ([Центр разработчиков Binance][3])

---

## База данных: минимальные, быстрые и безопасные схемы (PostgreSQL)

### 1) Свечи (1m klines, любые интервалы)

```sql
CREATE TABLE IF NOT EXISTS klines_1m (
  symbol        text        NOT NULL,
  open_time_ms  bigint      NOT NULL, -- k.t (ms)
  open_price    numeric(38,18) NOT NULL,
  high_price    numeric(38,18) NOT NULL,
  low_price     numeric(38,18) NOT NULL,
  close_price   numeric(38,18) NOT NULL,
  base_volume   numeric(38,18) NOT NULL, -- k.v
  quote_volume  numeric(38,18) NOT NULL, -- k.q
  trade_count   integer     NOT NULL,    -- k.n
  taker_base    numeric(38,18) NOT NULL, -- k.V
  taker_quote   numeric(38,18) NOT NULL, -- k.Q
  is_closed     boolean     NOT NULL,    -- k.x
  ingest_ts     timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (symbol, open_time_ms)
);
-- Индексы под выборки по времени:
CREATE INDEX IF NOT EXISTS klines_1m_time_idx ON klines_1m (open_time_ms);
```

**Пояснения:** структура точно соответствует payload kline (`k`-объект) и поддерживает идемпотентный `UPSERT` по (symbol, open_time_ms) при доездах обновлений той же свечи (`x=false/true`) ([Центр разработчиков Binance][5]).

### 2) Аггрегированные трейды (aggTrade)

```sql
CREATE TABLE IF NOT EXISTS agg_trades (
  symbol     text        NOT NULL,
  agg_id     bigint      NOT NULL,         -- a
  price      numeric(38,18) NOT NULL,      -- p
  quantity   numeric(38,18) NOT NULL,      -- q
  first_id   bigint      NOT NULL,         -- f
  last_id    bigint      NOT NULL,         -- l
  trade_time_ms bigint   NOT NULL,         -- T
  is_maker   boolean     NOT NULL,         -- m
  ingest_ts  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (symbol, agg_id)
);
CREATE INDEX IF NOT EXISTS agg_trades_time_idx ON agg_trades (trade_time_ms);
```

**Пояснения:** `agg_id` уникален в рамках символа, что даёт идеальный PK и идемпотентную вставку без дублей. Поля соответствуют оф. примеру Binance Futures. ([Центр разработчиков Binance][6])

> Если позже захотите TimescaleDB — просто «переедем» на hypertable без переписывания схемы (та же логика ключей).

---

## Надёжность и ресурсоэффективность: что обязательно добавить в «сырой» WebSocket-клиенте

1. **Keep-alive и авто-переподключение.**
   Запускайте `run_forever(ping_interval=60, ping_timeout=10)` и оборачивайте в цикл с экспоненциальным backoff (например, 1→2→4→8… до 60 с) и джиттером. Binance шлёт `ping` ~каждые 3 мин, 24-часовой TTL соединения — норма, это не баг. ([Центр разработчиков Binance][4])

2. **Партиционирование стримов.**
   Не превышайте **1024** потоков на одно соединение. Делите universe на несколько соединений; kline и aggTrade держите раздельно. Это снижает CPU «накопления» в очереди и уменьшает blast-radius при реконнекте. ([Центр разработчиков Binance][3])

3. **Нижний регистр имён и валидация.**
   Приводите имена стримов к lower-case при формировании `streams=...` (требование API). ([Центр разработчиков Binance][3])

4. **Безопасный парсинг JSON.**
   Используйте `orjson.loads` (если доступен) или `ujson` как ускоритель. Вставки в БД делайте батчами (например, 500–2 000 записей) через `psycopg2.extras.execute_batch` (или `asyncpg` с `executemany`).

5. **Идемпотентные upserts.**

   * `klines_1m`: `INSERT ... ON CONFLICT (symbol, open_time_ms) DO UPDATE SET ...` (перезапись, когда доезжает `x=true`).
   * `agg_trades`: `ON CONFLICT DO NOTHING` по (symbol, agg_id). Поля для ключей — из оф. payload. ([Центр разработчиков Binance][6])

6. **Ротация соединений < 24 ч.**
   Плановый reconnect по крону/таймеру каждые ~12–18 ч, чтобы не ловить «острый» разрыв у всех воркеров одновременно. Это best-practice из общих правил Binance WS. ([Центр разработчиков Binance][4])

7. **Ограничение control-кадров.**
   Не спамьте `ping` чаще лимитов; Binance ограничивает частоту по ping/pong и control-сообщениям. ([Центр разработчиков Binance][4])

---

## Мини-правка клиента `websocket-client` (устойчивый цикл)

Ниже — **паттерн**, который должен быть в вашем модуле при работе с `WebSocketApp` (не привязываюсь к путям в репо):

```python
import time, random, ssl, websocket  # websocket-client
from threading import Event

STOP = Event()

def run_ws(url: str, on_msg, on_open=None, on_close=None, headers=None):
    backoff = 1
    while not STOP.is_set():
        ws = websocket.WebSocketApp(
            url,
            header=headers or [],
            on_message=on_msg,
            on_open=on_open,
            on_close=on_close,
        )
        try:
            # ping_interval поддерживает keep-alive; run_forever сам не «умный» — цикл выше
            ws.run_forever(
                sslopt={"cert_reqs": ssl.CERT_REQUIRED},
                ping_interval=60,
                ping_timeout=10,
                http_proxy_host=None,
                http_proxy_port=None,
            )
        except Exception:
            pass  # залогируйте
        # экспоненциальный backoff с джиттером и потолком
        time.sleep(backoff + random.random())
        backoff = min(backoff * 2, 60)
```

**Почему так:** `run_forever` — не автоматический reconnect; keep-alive + внешний цикл — стандартная рекомендация для long-lived WS в `websocket-client`. ([PyPI][8])

---

## Что именно я смог подтвердить по вашему репозиторию сейчас

* Ветка `prosperous_bot`, HEAD и дата — подтверждены в истории коммитов на GitHub (27 Sep 2025). ([GitHub][2])
* Наличие папки `third_party/rl-trading-binance` в дереве коммита — подтверждено (GitHub помечает коммит как «большой», часть дерева скрыта UI). ([GitHub][1])

> Примечание: страница списка файлов GitHub для вложенных директорий периодически отдаёт «There was an error while loading / Some content is hidden», поэтому программно раскрыть **все** новые файлы (например, конкретный `backends/*` подмодуль на вашей ветке) прямо сейчас не удалось. Это ограничение веб-интерфейса GitHub для больших диффов, а не проблема вашего репо. (См. баннер «Some content is hidden» на странице коммита.) ([GitHub][1])

---

## Предлагаемая структура конфигурации RL-бота (1 файл)

Чтобы держать **всё управление из одного файла** и не задевать `unified_config*.json` вашего ребалансировщика:

```yaml
# third_party/rl-trading-binance/config_ws.yaml
binance:
  base_url: "wss://fstream.binance.com"
  channels:
    klines_1m: true
    agg_trade: true
  shards:
    kline_streams_per_conn: 600   # < 1024
    agg_trade_streams_per_conn: 600
  ping_interval_sec: 60
  ping_timeout_sec: 10
  reconnect_backoff_max_sec: 60

universe:
  symbols: ["BTCUSDT","ETHUSDT", "..."]  # всегда в нижнем регистре при формировании URL

storage:
  dsn: "postgresql://user:pass@host:5432/db"
  batch_size: 1000
  write_timeout_ms: 5000

housekeeping:
  rotate_connections_hours: 12
  log_level: "INFO"
```

---

## План действий (коротко)

| Шаг | Действие                                                                                                  | KPI/риск                                                                                    |
| --- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| 1   | Проверить, что при формировании combined-URL все `streams` в **lowercase**                                | Нет disconnect из-за неверных имён; ↓ошибок парсинга                                        |
| 2   | Включить `ping_interval=60, ping_timeout=10` и внешний цикл-reconnect с backoff ≤60 с                     | Устойчивость 24/7, без ручных рестартов ([Центр разработчиков Binance][4])                  |
| 3   | Шардировать universe так, чтобы ≤1024 потоков на соединение; разделить kline и aggTrade по разным сокетам | Стабильность при пиковой нагрузке, соблюдение лимитов ([Центр разработчиков Binance][3])    |
| 4   | Применить схемы `klines_1m` и `agg_trades` + idempotent `UPSERT`                                          | 0 дублей, гарантированная консистентность ([Центр разработчиков Binance][5])                |
| 5   | Пакетные вставки (500–2000) через `execute_batch` / `asyncpg.executemany`                                 | ↓CPU/IO БД на слабом железе                                                                 |
| 6   | Плановая ротация WS каждые 12–18 ч                                                                        | Предсказуемость, избегаем массового разрыва в ровно 24 ч ([Центр разработчиков Binance][4]) |
| 7   | Наблюдение: логировать disconnect причины, счётчик реконнектов, lag по времени события (`E/T`)            | Быстрая диагностика регрессий                                                               |

---

## Что ещё стоит добавить (по спецификациям Binance)

* Ограничить частоту `ping/pong` и control-сообщений — Binance считает их частью rate limit; не превышайте лимиты. ([Центр разработчиков Binance][4])
* При большом числе символов следите за суммарной частотой сообщений, особенно у `aggTrade` (100 мс апдейты): если один сокет перегружается, дробите ещё. ([Центр разработчиков Binance][6])

---

## Если хотите, я подготовлю точечный PR-патч

Готов оперативно оформить патч в `third_party/rl-trading-binance` с:

* обёрткой `WebSocketApp` (keep-alive + reconnect),
* валидатором стримов (lowercase),
* шардингом по лимиту 1024/conn,
* батч-писателем PostgreSQL с `UPSERT`.

**Но:** чтобы соблюсти ваш «Ultra-strict» режим и не «галлюцинировать» пути, мне нужна одна ссылка на **конкретный файл** вашего WS-модуля (например, текущий `collector.py`) — тогда внесу изменения строго по фактическому пути и покажу `unified diff` с Repo-State Header.

Пока же весь обзор основан на HEAD вашего репозитория и официальной документации Binance (лимиты, форматы, поведение WS), которые я процитировал рядом с соответствующими пунктами. ([GitHub][1])

Если нужно — продолжу и оформлю PR сразу.

[1]: https://github.com/FMProducer/prosperous_bot/commit/6fcaf1e0da9b314abdaca1754ff7fe155926b832 "docs: new repo · FMProducer/prosperous_bot@6fcaf1e · GitHub"
[2]: https://github.com/FMProducer/prosperous_bot/commits/prosperous_bot "Commits · FMProducer/prosperous_bot · GitHub"
[3]: https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams?utm_source=chatgpt.com "Websocket Market Streams - Binance Developer center"
[4]: https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-api-general-info?utm_source=chatgpt.com "Websocket API General Info | Binance Open Platform"
[5]: https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Kline-Candlestick-Streams "Kline Candlestick Streams | Binance Open Platform"
[6]: https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Aggregate-Trade-Streams "Aggregate Trade Streams | Binance Open Platform"
[7]: https://websocket-client.readthedocs.io/en/latest/examples.html?utm_source=chatgpt.com "Examples — websocket-client 1.8.0 documentation"
[8]: https://pypi.org/project/websocket-client/?utm_source=chatgpt.com "websocket-client"
