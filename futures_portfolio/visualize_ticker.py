import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import argparse

def plot_ticker_equity(ticker):
    # Пытаемся найти лог или файл состояния. 
    # В этой системе данные об equity часто пишутся в logs/<ticker>.log или подобное.
    # Для примера предполагаем, что у вас есть лог-файл в папке logs
    log_file = "logs/backtest_latest.log"
    if not os.path.exists(log_file):
        print(f"Error: Log file {log_file} not found. Ensure the ticker is running.")
        return

    data = []
    with open(log_file, "r") as f:
        for line in f:
            if "Heartbeat:" in line and "TPV=" in line:
                try:
                    parts = line.split("Heartbeat:")[1]
                    tpv = float(parts.split("TPV=")[1].split("|")[0].strip())
                    timestamp = line.split("INFO:")[0].strip()
                    data.append({"date": timestamp, "tpv": tpv})
                except:
                    continue

    if not data:
        print("No heartbeat data found in logs.")
        return

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S', errors='ignore')

    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')
    plt.plot(df['date'], df['tpv'], color='#00ff00', label=f'{ticker} TPV')
    plt.title(f'Performance Curve: {ticker}', color='white')
    plt.ylabel('TPV (USDT)')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    
    output_file = f"{ticker}_equity.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()
    plot_ticker_equity(args.ticker)
