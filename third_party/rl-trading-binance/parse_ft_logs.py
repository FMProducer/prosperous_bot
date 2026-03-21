import re
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Tuple

@dataclass
class TradeExecution:
    pair: str
    direction: str
    signal_price: float = 0.0
    fill_price: float = 0.0
    
    @property
    def slippage_pct(self) -> float:
        if self.signal_price == 0 or self.fill_price == 0:
            return 0.0
        # Отрицательное значение = негативное проскальзывание (убыток)
        # Для Long: если купили дороже сигнала -> плохо (Fill > Signal) -> (Signal - Fill) < 0
        if self.direction == 'long' or self.direction == 'buy':
            return ((self.signal_price - self.fill_price) / self.signal_price) * 100
        # Для Short: если продали дешевле сигнала -> плохо (Fill < Signal) -> (Fill - Signal) < 0
        else: # short / sell
            return ((self.fill_price - self.signal_price) / self.signal_price) * 100

@dataclass
class LogStats:
    raw_signals: int = 0
    filtered_signals: int = 0
    executions: List[TradeExecution] = field(default_factory=list)

def analyze_freqtrade_logs(log_file_path: str | Path) -> None:
    path = Path(log_file_path)
    if not path.exists():
        print(f"Файл {path} не найден.")
        return

    print(f"Анализ лога: {path}")
    stats = LogStats()
    active_orders: Dict[str, TradeExecution] = {}

    # Regex паттерны
    re_raw_signal = re.compile(r'\[RAW RL SIGNAL\]')
    re_filtered_signal = re.compile(r'\[FILTERED SIGNAL\]')
    
    # Парсинг создания ордера 
    # Стандартный формат Freqtrade: "Creating buy order for ETH/USDT" или "Creating long order..."
    # Пытаемся поймать и buy/sell и long/short
    re_order_create = re.compile(r'Creating (long|short|buy|sell) .*? order for (.*?)\s+.*?Rate:\s+([\d.]+)', re.IGNORECASE)
    
    # Парсинг исполнения ордера
    # "Order filled for ETH/USDT. ... executed at 2500.0"
    re_order_fill = re.compile(r'Order filled for (.*?).*?executed at\s+([\d.]+)', re.IGNORECASE)

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if re_raw_signal.search(line):
                stats.raw_signals += 1
                continue
                
            if re_filtered_signal.search(line):
                stats.filtered_signals += 1
                continue

            # Детект попытки входа
            create_match = re_order_create.search(line)
            if create_match:
                direction, pair, rate = create_match.groups()
                # Нормализация направления
                direction = direction.lower()
                if direction == 'buy': direction = 'long'
                if direction == 'sell': direction = 'short'
                
                active_orders[pair] = TradeExecution(
                    pair=pair, 
                    direction=direction, 
                    signal_price=float(rate)
                )
                continue

            # Детект исполнения
            fill_match = re_order_fill.search(line)
            if fill_match:
                pair, fill_rate = fill_match.groups()
                # Иногда в паре может быть мусор, чистим
                pair = pair.strip()
                
                # Ищем совпадающий ордер
                # (Упрощение: предполагаем, что не может быть двух одновременных ордеров по одной паре в ожидании)
                if pair in active_orders:
                    order = active_orders[pair]
                    order.fill_price = float(fill_rate)
                    stats.executions.append(order)
                    del active_orders[pair] # Очищаем после исполнения

    _print_report(stats)

def _print_report(stats: LogStats) -> None:
    print("="*50)
    print(" АНАЛИЗ ПРОСКАЛЬЗЫВАНИЯ И ФИЛЬТРАЦИИ (1m TF)")
    print("="*50)
    
    rejection_rate = 0.0
    if stats.raw_signals > 0:
        rejection_rate = ((stats.raw_signals - stats.filtered_signals) / stats.raw_signals) * 100
        
    print(f"Всего сигналов от D3QN (Raw): {stats.raw_signals}")
    print(f"Пропущено фильтрами объема:   {stats.filtered_signals}")
    print(f"Процент отклонения (Reject):  {rejection_rate:.2f}%\n")

    if not stats.executions:
        print("Исполненные сделки в логе не найдены.")
        return

    total_slippage = 0.0
    print(f"{'Пара':<16} | {'Тип':<6} | {'Signal Price':<14} | {'Fill Price':<14} | {'Slippage %'}")
    print("-" * 75)
    for ex in stats.executions:
        slip = ex.slippage_pct
        total_slippage += slip
        print(f"{ex.pair:<16} | {ex.direction:<6} | {ex.signal_price:<14.5f} | {ex.fill_price:<14.5f} | {slip:+.4f}%")
    
    avg_slippage = total_slippage / len(stats.executions)
    print("-" * 75)
    print(f"Среднее проскальзывание на сделку: {avg_slippage:+.4f}%")
    
    if avg_slippage < -0.05:
        print("\n[ВНИМАНИЕ] Проскальзывание критическое! Спред съедает профит.")

if __name__ == "__main__":
    # Автоматический поиск лога
    log_locations = [
        Path("user_data/logs/freqtrade.log"),
        Path("freqtrade.log")
    ]
    
    found = False
    for p in log_locations:
        if p.exists():
            analyze_freqtrade_logs(p)
            found = True
            break
    
    if not found:
        print("freqtrade.log не найден в стандартных путях.")
