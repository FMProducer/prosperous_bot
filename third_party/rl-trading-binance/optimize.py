
import optuna
import copy
import logging
import sys
from config import cfg as default_cfg
from train import main as train_main
from backtest_engine import run_backtest

# --- Настройки Оптимизации ---
N_TRIALS = 8  # Количество испытаний для Optuna
TRAIN_EPISODES_PER_TRIAL = 5000 # Уменьшенное количество эпох для ускорения оптимизации

# Настройка логирования для Optuna
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

def objective(trial: optuna.Trial) -> float:
    """
    Целевая функция для одного испытания Optuna.
    Обучает модель и запускает бэктест с предложенным набором гиперпараметров.
    """
    # Создаем глубокую копию конфига, чтобы испытания не влияли друг на друга
    cfg = copy.deepcopy(default_cfg)

    # --- 1. Определение пространства поиска гиперпараметров ---
    cfg.rl.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    cfg.rl.gamma = trial.suggest_float("gamma", 0.9, 0.999)
    cfg.model.dropout_p = trial.suggest_float("dropout_p", 0.05, 0.3)
    cfg.backtest.long_action_threshold = trial.suggest_float("long_action_threshold", 0.1, 0.7)
    cfg.backtest.short_action_threshold = trial.suggest_float("short_action_threshold", 0.1, 0.7)
    cfg.seq.agent_history_len = trial.suggest_int("agent_history_len", 20, 80)

    # --- 2. Настройка параметров для конкретного испытания ---
    # Уникальное имя для сессии, чтобы логи и модели не перемешивались
    cfg.paths.config_name = f"optuna_trial_{trial.number}"
    # Уменьшаем количество эпох для ускорения оптимизации
    cfg.trainlog.episodes = TRAIN_EPISODES_PER_TRIAL
    # Отключаем валидацию во время обучения для ускорения
    cfg.trainlog.validate_model = False

    sharpe_ratio = -1.0  # Дефолтное значение в случае ошибки

    try:
        logging.info(f"--- Начало испытания #{trial.number} ---")
        logging.info(f"Параметры: {trial.params}")

        # --- 3. Обучение модели ---
        logging.info(f"Шаг 1: Обучение модели для испытания #{trial.number}...")
        train_main(cfg=cfg)
        logging.info(f"Обучение для испытания #{trial.number} завершено.")

        # --- 4. Бэктестинг модели ---
        logging.info(f"Шаг 2: Бэктестинг модели для испытания #{trial.number}...")
        metrics = run_backtest(cfg=cfg)
        logging.info(f"Бэктестинг для испытания #{trial.number} завершен.")

        # --- 5. Извлечение результата ---
        if metrics and "sharpe" in metrics:
            sharpe_ratio = float(metrics["sharpe"])
            logging.info(f"Испытание #{trial.number} | Коэффициент Шарпа: {sharpe_ratio}")
        else:
            logging.warning(f"Не удалось получить 'sharpe' из метрик для испытания #{trial.number}.")

    except Exception as e:
        logging.error(f"Ошибка в испытании #{trial.number}: {e}", exc_info=True)
        # В случае ошибки Optuna присвоит этому испытанию худший результат
        return -1.0

    return sharpe_ratio

if __name__ == "__main__":
    # --- Настройка исследования Optuna ---
    # Используем БД для хранения результатов, что позволяет прерывать и возобновлять оптимизацию
    study_name = "rl_trading_optimization"
    storage_name = f"sqlite:///{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,  # Позволяет возобновить прерванное исследование
        direction="maximize",  # Мы хотим максимизировать коэффициент Шарпа
    )

    logging.info(f"Запуск исследования '{study_name}' с {N_TRIALS} испытаниями.")
    logging.info(f"Результаты будут сохранены в: {storage_name}")

    # --- Запуск оптимизации ---
    study.optimize(objective, n_trials=N_TRIALS)

    # --- Вывод результатов ---
    logging.info("\n--- Оптимизация завершена ---")
    logging.info(f"Количество завершенных испытаний: {len(study.trials)}")
    logging.info("Лучшее испытание:")
    best_trial = study.best_trial
    logging.info(f"  Значение (Шарп): {best_trial.value}")
    logging.info("  Лучшие параметры: ")
    for key, value in best_trial.params.items():
        logging.info(f"    {key}: {value}")
