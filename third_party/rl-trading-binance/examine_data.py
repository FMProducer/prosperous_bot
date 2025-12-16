import numpy as np
import os

def examine_npz_data(filepath):
    """
    Загружает и анализирует .npz файл с данными для обучения.

    Args:
        filepath (str): Путь к .npz файлу.
    """
    print(f"--- Анализ файла: {filepath} ---")

    if not os.path.exists(filepath):
        print("\n[ОШИБКА] Файл данных не найден.")
        print("Пожалуйста, убедитесь, что вы скачали датасеты и поместили их в директорию 'third_party/rl-trading-binance/data/'.")
        print("--------------------------------------------------")
        return

    try:
        # Загружаем .npz файл. allow_pickle=True необходим для загрузки _keys_map_
        data_archive = np.load(filepath, allow_pickle=True)

        # Получаем список всех эпизодов (ключей), исключая служебные поля
        episode_keys = [k for k in data_archive.files if not k.startswith('_')]

        if not episode_keys:
            print("\n[ОШИБКА] В файле не найдено эпизодов для анализа.")
            print("--------------------------------------------------")
            data_archive.close()
            return

        print(f"\n1. Общая информация:")
        print(f"   - Всего найдено эпизодов (ключей): {len(episode_keys)}")

        # Выбираем первый эпизод для детального анализа
        sample_key = episode_keys[0]
        sample_episode_data = data_archive[sample_key]

        print(f"\n2. Анализ одного эпизода (ключа: '{sample_key}'):")
        print(f"   - Форма тензора данных (shape): {sample_episode_data.shape}")
        print(f"     - {sample_episode_data.shape[0]} временных шагов (минутных баров)")
        print(f"     - {sample_episode_data.shape[1]} признаков (фичей)")

        # Выбираем один временной срез (бар) для демонстрации.
        # Возьмем бар из середины эпизода.
        bar_index_to_show = sample_episode_data.shape[0] // 2
        one_bar_features = sample_episode_data[bar_index_to_show]

        print(f"\n3. Пример признаков для одного бара (индекс {bar_index_to_show}):")
        feature_names = [
            ("open", "Цена открытия (в USDT)"),
            ("high", "Макс. цена (в USDT)"),
            ("low", "Мин. цена (в USDT)"),
            ("close", "Цена закрытия (в USDT)"),
            ("volume", "Объём в базовой валюте (напр., BTC)"),
            ("quote_volume", "Объём в котируемой валюте (USDT)"),
            ("num_trades", "Количество сделок"),
            ("taker_base", "Объём покупок тейкерами (в базовой валюте)"),
            ("taker_quote", "Объём покупок тейкерами (в USDT)"),
            ("vwap", "Средневзвешенная цена (VWAP)")
        ]

        for i, (name, description) in enumerate(feature_names):
            value = one_bar_features[i]
            print(f"   - Индекс {i} ({name: <12}): {value: <15.6f} # {description}")

        data_archive.close()

    except Exception as e:
        print(f"\n[ОШИБКА] Произошла ошибка при чтении файла: {e}")

    print("\n--- Анализ завершён ---")


if __name__ == "__main__":
    # Путь к файлу данных относительно корня репозитория
    data_file_path = "third_party/rl-trading-binance/data/train_data_fair_8m.npz"
    examine_npz_data(data_file_path)
