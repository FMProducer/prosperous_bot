import os
import re

# Путь к вашим данным (из вашего запроса)
data_dir = r"C:\Python\Prosperous_Bot\third_party\rl-trading-binance\user_data\data\binance\futures"

pairs = set()

if os.path.exists(data_dir):
    print(f"Сканирование папки: {data_dir} ...")
    for f in os.listdir(data_dir):
        # Ищем файлы данных json, feather или json.gz
        if f.endswith(".json") or f.endswith(".feather") or f.endswith(".json.gz"):
            # Пример имени файла: BTC_USDT_USDT-1m.json
            # Regex берет часть до таймфрейма
            match = re.match(r"([a-zA-Z0-9_]+)-[0-9]+[a-z]+", f)
            if match:
                symbol_part = match.group(1) # например BTC_USDT_USDT
                parts = symbol_part.split('_')
                
                # Формируем пару в формате Freqtrade для фьючерсов
                if len(parts) == 3: # Futures: BASE/QUOTE:STAKE
                    pair = f"{parts[0]}/{parts[1]}:{parts[2]}"
                    pairs.add(pair)
                elif len(parts) == 2: # На случай если попадутся спотовые файлы
                    pair = f"{parts[0]}/{parts[1]}"
                    pairs.add(pair)

    # Вывод результата
    sorted_pairs = sorted(list(pairs))
    print("\n--- Скопируйте список ниже и замените им pair_whitelist в конфиге ---\n")
    print('        "pair_whitelist": [')
    for i, pair in enumerate(sorted_pairs):
        comma = "," if i < len(sorted_pairs) - 1 else ""
        print(f'            "{pair}"{comma}')
    print("        ],")
    print(f"\n--- Конец списка (Найдено пар: {len(sorted_pairs)}) ---")

else:
    print(f"ОШИБКА: Папка не найдена: {data_dir}")
