import re
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Tuple
import sys

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
        if self.direction == 'long' or self.direction == 'buy':
            return ((self.signal_price - self.fill_price) / self.signal_price) * 100
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

    re_raw_signal = re.compile(r'\[RAW RL SIGNAL\]')
    re_filtered_signal = re.compile(r'\[FILTERED SIGNAL\]')
    
    # 1. Сигнал на вход (Market или Limit)
    re_signal_found = re.compile(r'(Long|Short) signal found: about create a new trade for (.*?) with .*? price: ([\d.]+)', re.IGNORECASE)
    
    # 2. Исполнение ордера (Включая LIMIT_BUY, LIMIT_SELL)
    # Freqtrade log: "LIMIT_BUY has been fulfilled for Trade(..."
    re_order_fill = re.compile(r'(MARKET_BUY|MARKET_SELL|LIMIT_BUY|LIMIT_SELL) has been fulfilled for Trade\(.*?pair=(.*?), .*?open_rate=([\d.]+).*?\)', re.IGNORECASE)

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if re_raw_signal.search(line):
                stats.raw_signals += 1
                continue
                
            if re_filtered_signal.search(line):
                stats.filtered_signals += 1
                continue

            # Детект сигнала
            sig_match = re_signal_found.search(line)
            if sig_match:
                direction, pair, rate = sig_match.groups()
                direction = direction.lower()
                pair = pair.strip()
                
                active_orders[pair] = TradeExecution(
                    pair=pair, 
                    direction=direction, 
                    signal_price=float(rate)
                )
                continue

            # Детект исполнения
            fill_match = re_order_fill.search(line)
            if fill_match:
                type_str, pair, fill_rate = fill_match.groups()
                pair = pair.strip()
                
                if pair in active_orders:
                    order = active_orders[pair]
                    order.fill_price = float(fill_rate)
                    stats.executions.append(order)
                    del active_orders[pair]
                continue

    _print_report(stats, active_orders)

def _print_report(stats: LogStats, active_orders: Dict[str, TradeExecution]) -> None:
    print("="*60)
    print(" АНАЛИЗ ПРОСКАЛЬЗЫВАНИЯ (Поддержка LIMIT ордеров)")
    print("="*60)
    
    print(f"Всего сигналов от D3QN (Raw): {stats.raw_signals}")
    
    if stats.executions:
        print(f"\n{'Пара':<16} | {'Тип':<6} | {'Signal':<10} | {'Fill':<10} | {'Slip %'}")
        print("-" * 65)
        total_slip = 0.0
        count = 0
        for ex in stats.executions:
            # Исключаем аномалии типа NEIRO
            if abs(ex.slippage_pct) > 50:
                print(f"{ex.pair:<16} | {ex.direction:<6} | {ex.signal_price:<10.5f} | {ex.fill_price:<10.5f} | {ex.slippage_pct:+.2f}% (IGN)")
                continue
                
            slip = ex.slippage_pct
            total_slip += slip
            count += 1
            print(f"{ex.pair:<16} | {ex.direction:<6} | {ex.signal_price:<10.5f} | {ex.fill_price:<10.5f} | {slip:+.4f}%")
        
        if count > 0:
            avg = total_slip / count
            print("-" * 65)
            print(f"Среднее проскальзывание: {avg:+.4f}%")
        else:
            print("\nНет валидных сделок для расчета (только аномалии).")
    else:
        print("\nИсполненные сделки не найдены.")

    # Показываем висящие лимитки
    if active_orders:
        print("\n" + "="*60)
        print(f"⏳ ОЖИДАЮТ ИСПОЛНЕНИЯ (LIMIT ORDERS): {len(active_orders)}")
        print("="*60)
        for pair, order in active_orders.items():
            print(f" - {pair:<16} ({order.direction}) @ {order.signal_price}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_freqtrade_logs(sys.argv[1])
    else:
        log_locations = [Path("user_data/logs/freqtrade.log"), Path("freqtrade.log")]
        for p in log_locations:
            if p.exists():
                analyze_freqtrade_logs(p)
                break
