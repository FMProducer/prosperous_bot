# Описание скриптов в `rl-trading-binance-source-code`

Здесь представлено краткое описание каждого Python-скрипта в директории `third_party/rl-trading-binance-source-code`.

- `__init__.py`: Стандартный файл Python, который помечает директорию как пакет.
- `agent.py`: Реализует D3QN (Dueling Double Deep Q-Networks) агента с Prioritized Experience Replay (PER), который отвечает за принятие решений и обучение.
- `backtest_engine.py`: Движок для реалистичного бэктестинга торговой стратегии на исторических данных с учетом комиссий и проскальзывания.
- `baseline_cnn_classifier.py`: Обучает и оценивает базовую модель (CNN классификатор) для сравнения с производительностью RL-агента.
- `config.py`: Центральный файл конфигурации проекта, использующий `pydantic` для определения всех гиперпараметров и настроек.
- `find_best_matching_cnn_configs.py`: Утилита для подбора конфигураций сверточной нейронной сети (CNN) с заданным количеством параметров.
- `get_info_from_optuna.py`: Инструмент командной строки для извлечения и отображения лучших результатов из исследований по оптимизации гиперпараметров, проведенных с помощью Optuna.
- `model.py`: Определяет архитектуру нейронной сети `DuelingQNetwork`, используемую RL-агентом.
- `optimize_cfg.py`: Скрипт для автоматической оптимизации гиперпараметров бэктестинга с использованием библиотеки Optuna.
- `replay_buffer.py`: Реализует буфер `Prioritized Experience Replay` (PER) для более эффективного обучения RL-агента.
- `test_agent.py`: Скрипт для оценки производительности обученного агента на тестовом наборе данных.
- `trading_environment.py`: Кастомная среда, совместимая с Gym, которая симулирует процесс торговли на Binance Futures.
- `train.py`: Главный скрипт для запуска процесса обучения RL-агента.
- `utils.py`: Набор вспомогательных функций, используемых в проекте, включая загрузку данных, нормализацию, логирование и расчет метрик.

## Пошаговая инструкция по применению

1.  **Обучение RL-агента:**
    ```bash
    python train.py configs/alpha.py
    ```

2.  **Оценка агента на тестовом наборе:**
    ```bash
    python test_agent.py configs/alpha.py
    ```

3.  **Запуск реалистичного бэктеста:**
    ```bash
    python backtest_engine.py configs/alpha.py
    ```

4.  **Обучение базовой модели (CNN-классификатора):**
    ```bash
    python baseline_cnn_classifier.py configs/alpha_baseline_cnn.py
    ```

5.  **Запуск оптимизации конфигурации с помощью Optuna:**
    ```bash
    python optimize_cfg.py configs/alpha.py --trials 100 --jobs 1
    ```

6.  **Отображение и сохранение 10 лучших попыток для данной конфигурации:**
    ```bash
    python get_info_from_optuna.py configs/alpha.py --n-best-trials 10
    ```

7.  **Если ваша цель минимизируется (например, минимизация потерь):**
    ```bash
    python get_info_from_optuna.py configs/alpha.py --n-best-trials 10 --direction min
    ```