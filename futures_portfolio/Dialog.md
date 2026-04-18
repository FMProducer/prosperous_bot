 Ключ VD417ujKKUCUGQmKWU7siM1k2bTcsey8llkwsjAK46dTO4cJaAlmqNVzi0OEZ7My был в открытом репозитории — любой мог его скопировать.
                                                                                                                                                                                                                                    
  Как отозвать:                                                                                                                                                                                                                     
                                                                                                                                                                                                                                    
  1. Войдите в Binance → Управление API (https://www.binance.com/en/my/settings/api-management)                                                                                                                                     
  2. Найдите ключ с именем (смотрите в списке)                                                                                                                                                                                      
  3. Нажмите "Delete" или "Disable"
  4. Создайте новый ключ

  Настройка окружения:

  Windows PowerShell:
  $env:BINANCE_API_KEY="your_new_api_key"
  $env:BINANCE_SECRET_KEY="your_new_secret_key"
  python main.py

  Windows CMD:
  set BINANCE_API_KEY=your_new_api_key
  set BINANCE_SECRET_KEY=your_new_secret_key
  python main.py

  Или создайте файл .env (скопируйте из .env.example):
  copy .env.example .env
  # Отредактируйте .env, вставьте новые ключи

  1. connector.py
                                                                                                                                                                                                                                    
  Новый метод get_margin_ratio() — возвращает:              
  - margin_ratio — коэффициент запаса (больше = безопаснее)
  - total_maint_margin — поддерживающая маржа
  - available_balance — доступный баланс

  2. main.py

  Добавлена проверка маржи каждый цикл:
  - Warning (≤5x): Логирует предупреждение о низкой марже
  - Critical (≤2x): Экстренное закрытие всех позиций и остановка бота

  3. config.json

  Новые параметры безопасности:
  "margin_ratio_warning": 5.0,
  "margin_ratio_critical": 2.0


  1. Безопасность API ключей     
                                                                                                                                                                                                                                    
  - config.json — удалены ключи из файла                                                                                                                                                                                          
  - main.py — читает ключи из переменных окружения BINANCE_API_KEY и BINANCE_SECRET_KEY                                                                                                                                             
  - Созданы .env.example и .gitignore                                                                                                                                                                                               
                                                                                                                                                                                                                                    
  2. Защита от ликвидации (Margin Ratio)                                                                                                                                                                                            
                                                                                                                                                                                                                                    
  - connector.py:89-103 — новый метод get_margin_ratio()
  - config.json — добавлены пороги: margin_ratio_warning: 5.0, margin_ratio_critical: 2.0
  - main.py:165-227 — проверка каждый цикл:
    - ≤5x — предупреждение
    - ≤2x — экстренное закрытие позиций и остановка

  3. Исправление гистерезиса

  - Проблема: initial_tpv менялся при срабатывании сейфа, ломая логику просадки
  - Решение: Добавлена переменная reference_tpv, которая никогда не меняется после инициализации
  - backtest_rebalance.py:125,205 — используется reference_tpv
  - main.py:54,151,270 — reference_tpv сохраняется в state.json