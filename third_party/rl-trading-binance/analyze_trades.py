
import pandas as pd
import os

try:
    # Путь к файлу относительно корня проекта
    csv_path = 'third_party/rl-trading-binance/output/alpha/trades.csv'
    trades_df = pd.read_csv(csv_path)
    
    # Группируем по тикеру и суммируем PnL
    pnl_by_ticker = trades_df.groupby('symbol')['net_pnl_usdt'].sum().sort_values(ascending=False)
    
    print("Наибольший вклад в PnL (топ-10 тикеров):")
    print(pnl_by_ticker.head(10))
    
    print("\nНаибольший убыток (топ-10 тикеров):")
    print(pnl_by_ticker.tail(10))

except FileNotFoundError:
    print(f"Ошибка: trades.csv не найден по пути {os.path.abspath(csv_path)}")
    print("Убедитесь, что вы запускаете скрипт из корневой директории проекта: C:\\Python\\Prosperous_Bot")
except Exception as e:
    print(f"Произошла ошибка: {e}")
