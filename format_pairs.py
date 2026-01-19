import json
import re
import os

# Путь к файлу с парами
FILE_PATH = r'C:\Python\Prosperous_Bot\log.txt'

def normalize_pair(pair_str):
    # Очистка от пробелов и перевод в верхний регистр
    pair_str = pair_str.strip().upper()
    
    # Пропускаем пустые строки
    if not pair_str:
        return None

    # Если пара уже в формате Futures (BASE/QUOTE:SETTLE), например BTC/USDT:USDT
    if re.match(r'^[A-Z0-9]+/[A-Z0-9]+:[A-Z0-9]+$', pair_str):
        return pair_str
        
    # Если пара в формате Spot (BASE/QUOTE), например BTC/USDT -> добавляем :USDT
    if re.match(r'^[A-Z0-9]+/[A-Z0-9]+$', pair_str):
        return f"{pair_str}:USDT"
        
    # Если просто тикер (BASE), например BTC -> превращаем в BTC/USDT:USDT
    # (Предполагаем торговлю к USDT, как в вашем конфиге)
    if re.match(r'^[A-Z0-9]+$', pair_str):
        # Игнорируем слова, похожие на служебные, если они попали в файл
        if pair_str in ['PAIR', 'WHITELIST', 'USDT']:
            return None
        return f"{pair_str}/USDT:USDT"
        
    return None

def main():
    if not os.path.exists(FILE_PATH):
        print(f"Файл не найден: {FILE_PATH}")
        return

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    # Ищем все, что похоже на тикеры или пары (разделители: пробелы, запятые, кавычки, новые строки)
    tokens = re.split(r'[\s,"\'\[\]]+', content)
    
    # Формируем множество уникальных пар
    valid_pairs = {normalize_pair(t) for t in tokens if normalize_pair(t)}
    
    # Сортируем и формируем итоговую структуру
    output_data = {"pair_whitelist": sorted(list(valid_pairs))}
    
    # Записываем результат обратно в файл
    with open(FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Готово! Файл {FILE_PATH} обновлен. Отформатировано {len(output_data['pair_whitelist'])} пар.")

if __name__ == "__main__":
    main()