# Prosperous Bot Dashboard — Пошаговая инструкция

## Что это

Веб-дашборд для мониторинга и управления ботами. Работает отдельно от основного кода — только читает файлы состояния и логи. Ничего не изменяет в торговой системе.

**Доступ:** браузер → `http://IP:8080` → логин/пароль

---

## Шаг 1: Установить зависимости

Открыть **cmd.exe** (не PowerShell) и выполнить:

```cmd
cd C:\Python\Prosperous_Bot\futures_portfolio\dashboard
pip install flask bcrypt psutil pyyaml waitress
```

**Ожидаемый результат:** `Successfully installed ...`

**Если ошибка "pip не найден":**
```cmd
python -m pip install flask bcrypt psutil pyyaml waitress
```

---

## Шаг 2: Установить пароль

### 2.1 Сгенерировать хеш пароля

В cmd.exe:

```cmd
python -c "import bcrypt; print(bcrypt.hashpw(b'ВАШ_ПАРОЛЬ', bcrypt.gensalt()).decode())"
```

**Важно:** замените `ВАШ_ПАРОЛЬ` на свой пароль. Запишите хеш (начинается с `$2b$12$...`).

### 2.2 Обновить config.yaml

Открыть файл `C:\Python\Prosperous_Bot\futures_portfolio\dashboard\config.yaml` в блокноте.

Найти строку:
```yaml
password_hash: "$2b$12$LJ3m4ys3Lk0TSHhPOhQE6OQDMwBqXx0Zz0Zz0Zz0Zz0Zz0Zz0Zz0Z"
```

Заменить на ваш хеш:
```yaml
password_hash: "$2b$12$ВАШ_ГЕНЕРИРОВАННЫЙ_ХЕШ"
```

Также заменить `secret_key`:
```yaml
secret_key: "CHANGE_ME_RANDOM_32_CHARS_MIN"
```
На любую случайную строку (минимум 32 символа).

**Сохранить файл.**

---

## Шаг 3: Проверить пути

В том же `config.yaml` проверить секцию `paths`:

```yaml
paths:
  project: "C:\\Python\\Prosperous_Bot\\futures_portfolio"
  logs: "C:\\Python\\Prosperous_Bot\\futures_portfolio\\logs"
  data: "C:\\Python\\Prosperous_Bot\\futures_portfolio\\data"
  pm2_logs: "C:\\Users\\svsma\\.pm2\\logs"
```

Если ваши пути отличаются — исправить. Двойные обратные слеши (`\\`) обязательны.

---

## Шаг 4: Первый запуск (тест)

В cmd.exe:

```cmd
cd C:\Python\Prosperous_Bot\futures_portfolio\dashboard
python dashboard.py
```

**Ожидаемый результат:**
```
============================================================
  Prosperous Bot Dashboard
  URL: http://0.0.0.0:8080
  Press Ctrl+C to stop
============================================================
```

**Если ошибка "No module named flask":** вернуться к Шагу 1.

**Если ошибка "Address already in use":** порт 8080 занят. Изменить порт в `config.yaml`:
```yaml
server:
  port: 8081
```

---

## Шаг 5: Открыть в браузере

На этом же ПК открыть браузер:

```
http://127.0.0.1:8080
```

**Должна появиться страница входа** с логотипом 🤖 Prosperous Bot.

Ввести:
- Логин: `admin`
- Пароль: тот, что вы установили в Шаге 2

**Если страница не открывается:**
1. Проверить что dashboard.py запущен (консоль не закрыта)
2. Проверить брандмауэр Windows — разрешить Python в сетях
3. Попробовать `http://localhost:8080`

---

## Шаг 6: Доступ с телефона (ZeroTier)

### 6.1 Убедиться что ZeroTier работает

На ПК выполнить в cmd:
```cmd
zerotier-cli status
```
Должно вернуть: `200 info ... online`

### 6.2 Узнать ZeroTier IP

```cmd
zerotier-cli listnetworks
```
Найти IP в формате `10.147.x.x` или `172.x.x.x`.

### 6.3 Открыть на телефоне

В браузере телефона:
```
http://<ZT_IP>:8080
```

**Если не открывается с телефона:**
1. Проверить что телефон в той же ZeroTier сети
2. Проверить брандмауэр Windows — входящие подключения на порт 8080
3. Временно отключить брандмауэр для теста:
   ```cmd
   netsh advfirewall set allprofiles state off
   ```
   **После проверки включить обратно:**
   ```cmd
   netsh advfirewall set allprofiles state on
   ```

---

## Шаг 7: Автозапуск через PM2

Чтобы dashboard запускался автоматически при перезагрузке:

```cmd
cd C:\Python\Prosperous_Bot\futures_portfolio\dashboard
pm2 start dashboard.py --name "dashboard" --interpreter python
pm2 save
```

**Проверить:**
```cmd
pm2 list
```
Должна быть строка `dashboard` со статусом `online`.

---

## Управление ботами через дашборд

### Кнопки управления (на странице тикера)

| Кнопка | Действие | Опасность |
|--------|----------|-----------|
| 🔄 Рестарт | `pm2 restart <bot> --update-env` | Низкая — бот перезапустится |
| ⏹ Стоп | `pm2 stop <bot>` | Средняя — бот остановится, позиции останутся |
| 🚨 Экстренная | `pm2 stop` + `pm2 delete` | **Высокая** — бот удалён из PM2 |

**Важно:** кнопки управления **не закрывают позиции на бирже**. Они только останавливают PM2 процесс. Для закрытия позиций использовать Binance.

### Страницы дашборда

| Страница | URL | Что показывает |
|----------|-----|----------------|
| Обзор | `/` | Все боты, сводка, PM2, алерты |
| Тикер | `/ticker/<mode>/<ticker>` | Детали, позиции, кнопки управления |
| Логи | `/logs/<mode>/<ticker>` | Лог в реальном времени |
| История | `/history` | PnL по дням |
| Настройки | `/settings` | Конфигурация (только чтение) |

---

## Частые проблемы

### "PM2 не отвечает" в таблице процессов

```cmd
pm2 resurrect
pm2 list
```

### "Log file not found"

Проверить что путь в `config.yaml` → `paths.logs` совпадает с реальным расположением логов.

### "No module named bcrypt"

```cmd
pip install bcrypt
```

Если ошибка компиляции — установить Build Tools для Visual Studio или использовать предкомпилированный wheel.

### Страница не обновляется

Проверить консоль браузера (F12 → Console) на ошибки. Обычно проблема в CORS или неправильном пути API.

### Dashboard падает при запуске

Проверить логи в консоли. Частые причины:
- Неправильный YAML-синтаксис в config.yaml
- Отсутствует папка `templates/` или `static/`
- Порт 8080 занят

---

## Безопасность

1. **Смените пароль по умолчанию** (Шаг 2)
2. **Не открывайте порт 8080 в интернет** — только ZeroTier LAN
3. **Кнопки управления требуют подтверждения** — нельзя случайно нажать
4. **Сессия истекает через 1 час** — автоматический выход

---

## Обратная связь

Вопросы и проблемы → FMProducer (Telegram).

---

*Создано: 2026-07-15 | Версия: 1.0*
