import torch
import os

# Укажите правильные имена файлов, если они отличаются
f1 = r"C:\Python\Prosperous_Bot\third_party\rl-trading-binance\output\alpha_seed_404_ohlcv_z_LONG_ONLY\saved_models\rl_binance_futures_trading_date_20260127_time_203752\checkpoint_ep00900.pth"
f2 = r"C:\Python\Prosperous_Bot\third_party\rl-trading-binance\output\alpha_seed_404_ohlcv_z_LONG_ONLY\saved_models\rl_binance_futures_trading_date_20260127_time_203752\checkpoint_ep01050.pth"

if os.path.exists(f1) and os.path.exists(f2):
    # Загружаем чекпоинты (на CPU, чтобы не занимать память GPU)
    c1 = torch.load(f1, map_location='cpu')
    c2 = torch.load(f2, map_location='cpu')

    print(f"--- Сравнение {f1} vs {f2} ---")
    
    # Сравниваем ключи верхнего уровня (обычно там epoch, model_state_dict, optimizer_state_dict, metrics)
    keys = set(c1.keys()) | set(c2.keys())
    for k in keys:
        val1 = c1.get(k, "N/A")
        val2 = c2.get(k, "N/A")
        
        # Если это словарь (например, веса), пишем только тип, иначе выводим значение
        if isinstance(val1, dict):
            print(f"{k}: [dict len={len(val1)}] vs [dict len={len(val2)}]")
        elif isinstance(val1, (int, float, str)):
            print(f"{k}: {val1} -> {val2}")
        else:
            print(f"{k}: {type(val1)} -> {type(val2)}")
else:
    print("Файлы не найдены. Проверьте имена.")
