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
        if self.direction == 'long' or self.direction == 'buy':
            return ((self.signal_price - self.fill_price) / self.signal_price) * 100
        else: # short / sell
            return ((self.fill_price - self.signal_price) / self.signal_price) * 100

@dataclass
class LogStats:
    raw_signals: int = 0
    model_signals: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    filtered_signals: int = 0
    low_volume_skips: int = 0
    executions: List[TradeExecution] = field(default_factory=list)

def analyze_freqtrade_logs(log_file_path: str | Path) -> None:
    path = Path(log_file_path)
    if not path.exists():
        print(f"Файл {path} не найден.")
        return

    print(f"Анализ лога: {path}")
    stats = LogStats()
    active_orders: Dict[str, TradeExecution] = {}

    # Регулярки для новой стратегии 2+2
    re_model_adv = re.compile(r'(L1|L2|S1|S2): adv=')
    re_low_volume = re.compile(r'\[LOW_VOLUME\]')
    re_entry_signal = re.compile(r'act=1 \((ENTRY_LONG|ENTRY_SHORT)\)')
    
    # 1. Сигнал на вход (Freqtrade)
    re_signal_found = re.compile(r'(Long|Short) signal found: about create a new trade for (.*?) with .*? price: ([\d.]+)', re.IGNORECASE)
    
    # 2. Исполнение ордера
    re_order_fill = re.compile(r'(MARKET_BUY|MARKET_SELL|LIMIT_BUY|LIMIT_SELL) has been fulfilled for Trade\(.*?pair=(.*?), .*?open_rate=([\d.]+).*?\)', re.IGNORECASE)

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            # Считаем сигналы от конкретных моделей
            adv_match = re_model_adv.search(line)
            if adv_match:
                model_name = adv_match.group(1)
                stats.model_signals[model_name] += 1
                stats.raw_signals += 1
            
            if re_low_volume.search(line):
                stats.low_volume_skips += 1
                continue

            if re_entry_signal.search(line):
                stats.filtered_signals += 1

            # Детект сигнала Freqtrade
            sig_match = re_signal_found.search(line)
            if sig_match:
                direction, pair, rate = sig_match.groups()
                active_orders[pair.strip()] = TradeExecution(
                    pair=pair.strip(), 
                    direction=direction.lower(), 
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
    print(" АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ 2+2 ENSEMBLE")
    print("="*60)
    
    print(f"Всего сырых Q-прогнозов: {stats.raw_signals}")
    for model, count in sorted(stats.model_signals.items()):
        print(f" - Модель {model}: {count} прогнозов")
    
    print(f"\nОтфильтровано по объему (LOW_VOLUME): {stats.low_volume_skips}")
    print(f"Сигналов прошли фильтры и консенсус: {stats.filtered_signals}")
    
    if stats.executions:
        print(f"\n{'Пара':<16} | {'Тип':<6} | {'Signal':<10} | {'Fill':<10} | {'Slip %'}")
        print("-" * 65)
        total_slip = 0.0
        for ex in stats.executions:
            slip = ex.slippage_pct
            total_slip += slip
            print(f"{ex.pair:<16} | {ex.direction:<6} | {ex.signal_price:<10.5f} | {ex.fill_price:<10.5f} | {slip:+.4f}%")
        
        avg = total_slip / len(stats.executions)
        print("-" * 65)
        print(f"Среднее проскальзывание: {avg:+.4f}%")
    else:
        print("\nИсполненные сделки не найдены.")

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
