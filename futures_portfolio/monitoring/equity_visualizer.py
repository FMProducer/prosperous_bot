import os
import json
import time
import csv
import glob
import pandas as pd
import matplotlib.pyplot as plt
import asyncio
from datetime import datetime
from core.notifier import TelegramNotifier
from core.storage import safe_load_json

# Настройки
LOG_DIR = "logs"
HISTORY_FILE = os.path.join(LOG_DIR, "equity_history.csv")
PLOT_FILE = os.path.join(LOG_DIR, "equity_plot.png")
CONFIG_PATH = "config.json"

async def get_total_equity():
    """Максимально честный расчет: учитывает PnL всех активных и закрытых ботов."""
    INITIAL_BOT_CAPITAL = 10000.0
    # Стартовый капитал всего роя (для 10 слотов)
    total_system_value = 10 * INITIAL_BOT_CAPITAL 
    
    # Сюда будем суммировать только отклонения от 10к (чистый профит или убыток)
    total_pnl = 0.0
    global_safe = 0.0
    
    # Загружаем конфиг, чтобы отличить активных от архивных
    config = await safe_load_json(CONFIG_PATH, {})
    active_tickers = config.get("tickers", [])

    # Находим все файлы состояния (это база всех ботов, когда-либо запущенных)
    all_states = glob.glob("state_*.json")
    
    for sf in all_states:
        ticker = os.path.basename(sf).replace("state_", "").replace(".json", "")
        is_active = ticker in active_tickers
        
        try:
            # 1. Считаем SAFE (зафиксированная прибыль)
            s_data = await safe_load_json(sf, {})
            bot_safe = s_data.get("siphoning_reserve", 0.0)
            global_safe += bot_safe
            
            # 2. Считаем Equity (баланс + плавающий PnL)
            pf = f"paper_state_{ticker}.json"
            bot_tpv = INITIAL_BOT_CAPITAL # По умолчанию, если файла нет
            
            if os.path.exists(pf):
                p_data = await safe_load_json(pf, {})
                
                balance = p_data.get("balance", INITIAL_BOT_CAPITAL)
                last_price = p_data.get("last_price", 0.0)
                positions = p_data.get("positions", {})
                
                unrealized_pnl = 0.0
                if last_price > 0:
                    for pos_name, qty in positions.items():
                        if "_LONG" in pos_name:
                            entry = p_data.get("long_entry_price", 0.0)
                            if entry > 0: unrealized_pnl += qty * (last_price - entry)
                        elif "_SHORT" in pos_name:
                            entry = p_data.get("short_entry_price", 0.0)
                            if entry > 0: unrealized_pnl += qty * (entry - last_price)
                
                # TPV этого конкретного бота (активные деньги + его сейф)
                bot_tpv = balance + unrealized_pnl + bot_safe
            
            # 3. КЛЮЧЕВАЯ ЛОГИКА:
            if is_active:
                # Для активного бота: его текущее отклонение от 10к
                # (Если бот новый и у него еще нет paper_state, отклонение будет 0)
                total_pnl += (bot_tpv - INITIAL_BOT_CAPITAL)
            else:
                # Для УДАЛЕННОГО бота: его финальный результат (профит или убыток)
                # Это "реализованный" результат, который остается в истории
                total_pnl += (bot_tpv - INITIAL_BOT_CAPITAL)
                
            status = "ACTIVE" if is_active else "STOPPED"
            print(f"DEBUG: {status} {ticker} Result: {bot_tpv - INITIAL_BOT_CAPITAL:+.2f}")
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    # Итоговое состояние системы = 100к + сумма всех прибылей и убытков
    return (total_system_value + total_pnl), global_safe

def save_history(total, safe):
    """Сохраняет точку данных в CSV."""
    os.makedirs(LOG_DIR, exist_ok=True)
    file_exists = os.path.isfile(HISTORY_FILE)
    
    with open(HISTORY_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "date", "total", "safe"])
        
        writer.writerow([
            int(time.time()),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            round(total, 2),
            round(safe, 2)
        ])

def create_plot():
    """Генерирует график эквити."""
    if not os.path.exists(HISTORY_FILE):
        return False
        
    try:
        df = pd.read_csv(HISTORY_FILE)
        if len(df) < 2: return False
        
        plt.figure(figsize=(10, 6))
        plt.style.use('dark_background')
        
        df['date'] = pd.to_datetime(df['date'])
        
        # Основная линия (Total Capital)
        plt.plot(df['date'], df['total'], color='#00ff00', linewidth=2.5, label='Total Capital (Active + All SAFE)')
        plt.fill_between(df['date'], df['total'], alpha=0.15, color='#00ff00')
        
        # Линия резерва (SAFE)
        plt.plot(df['date'], df['safe'], color='#ff9900', linestyle='--', linewidth=1.5, label='Total SAFE Reserve')
        
        plt.title('Binance Swarm: Global Equity Curve', fontsize=14, color='white', pad=20)
        plt.xlabel('Time', fontsize=10)
        plt.ylabel('USDT', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.2)
        plt.legend(loc='upper left')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(PLOT_FILE)
        plt.close()
        return True
    except Exception as e:
        print(f"Plotting error: {e}")
        return False

async def main():
    notifier = TelegramNotifier()
    
    # 1. Получаем данные по новой формуле
    total, safe = await get_total_equity()
    if total == 0:
        print("No data found to log.")
        return
        
    # 2. Сохраняем
    save_history(total, safe)
    print(f"Logged: Total={total:.2f} (includes Global SAFE={safe:.2f})")
    
    # 3. Рисуем
    if create_plot():
        # 4. Отправляем
        caption = (
            f"📈 <b>Global Swarm Performance</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 Total Capital: <b>{total:.2f} USDT</b>\n"
            f"🛡️ All-Time SAFE: <code>{safe:.2f} USDT</code>\n"
            f"📊 Active Bots: 10\n"
            f"⏱ <i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        )
        await notifier.send_photo(PLOT_FILE, caption=caption)
        print("Report sent to Telegram.")
    else:
        print("Not enough data to plot yet.")

if __name__ == "__main__":
    asyncio.run(main())
