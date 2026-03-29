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
        # Prevent division by extremely small numbers or scientific notation errors
        if abs(self.signal_price) < 1e-12:
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
    denied_by_limit: int = 0
    denied_pairs: List[str] = field(default_factory=list)
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
    
    # 1. Сигнал на вход (Freqtrade) - улучшенный парсинг научной нотации (e-05)
    re_signal_found = re.compile(r'(Long|Short) signal found: about create a new trade for (.*?) with .*? price: ([\d.e-]+)', re.IGNORECASE)
    
    # 2. Исполнение ордера
    re_order_fill = re.compile(r'(MARKET_BUY|MARKET_SELL|LIMIT_BUY|LIMIT_SELL) has been fulfilled for Trade\(.*?pair=(.*?), .*?open_rate=([\d.e-]+).*?\)', re.IGNORECASE)

    # 3. Отклонение по лимиту (confirm_trade_entry)
    re_user_denied = re.compile(r'User denied entry for (.*?)\.', re.IGNORECASE)

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
                pair = pair.strip()
                active_orders[pair] = TradeExecution(
                    pair=pair, 
                    direction=direction.lower(), 
                    signal_price=float(rate)
                )
                continue

            # Детект отклонения (Удаляем из активных ожиданий)
            denied_match = re_user_denied.search(line)
            if denied_match:
                pair = denied_match.group(1).strip()
                if pair in active_orders:
                    del active_orders[pair]
                    stats.denied_by_limit += 1
                    if pair not in stats.denied_pairs:
                        stats.denied_pairs.append(pair)
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
    print(f"Отклонено лимитом портфеля (20/20): {stats.denied_by_limit}")
    
    if stats.executions:
        print(f"\n{'Пара':<18} | {'Тип':<6} | {'Signal':<10} | {'Fill':<10} | {'Slip %'}")
        print("-" * 70)
        total_slip = 0.0
        # Filter out extreme outliers (like parsing errors on very small prices)
        valid_executions = [ex for ex in stats.executions if abs(ex.slippage_pct) < 50.0]
        
        for ex in stats.executions:
            slip = ex.slippage_pct
            # Highlight extreme slippage which is usually a parsing error
            warn = " (!)" if abs(slip) > 50.0 else ""
            print(f"{ex.pair:<18} | {ex.direction:<6} | {ex.signal_price:<10.6f} | {ex.fill_price:<10.6f} | {slip:+.4f}%{warn}")
            if abs(slip) < 50.0:
                total_slip += slip
        
        if valid_executions:
            avg = total_slip / len(valid_executions)
            print("-" * 70)
            print(f"Среднее проскальзывание (без ошибок): {avg:+.4f}%")
    else:
        print("\nИсполненные сделки не найдены.")

    if active_orders:
        print("\n" + "="*60)
        print(f"⏳ ОЖИДАЮТ ИСПОЛНЕНИЯ (REAL LIMITS): {len(active_orders)}")
        print("="*60)
        for pair, order in active_orders.items():
            print(f" - {pair:<18} ({order.direction}) @ {order.signal_price}")
    else:
        print("\n[OK] Активных ожидающих ордеров нет.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_freqtrade_logs(sys.argv[1])
    else:
        log_locations = [Path("user_data/logs/freqtrade.log"), Path("freqtrade.log")]
        for p in log_locations:
            if p.exists():
                analyze_freqtrade_logs(p)
                break
