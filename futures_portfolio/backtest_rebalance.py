import pandas as pd
import numpy as np
import asyncio
import aiohttp
import time
import json
import os
import logging
import traceback
import argparse
from typing import Dict, List, Optional, Tuple
from calculator import PortfolioCalculator
from executor import PortfolioExecutor

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Backtest")

class LimitOrderSimulator:
    """
    Симулятор исполнения лимитных ордеров на основе OHLCV данных.
    Использует high/low цены свечи для эмуляции стакана.
    """

    def __init__(self, commission_pct: float = 0.02, timeout_sec: int = 30, offset_pct: float = 0.2):
        self.commission_pct = commission_pct  # Maker комиссия 0.02%
        self.timeout_sec = timeout_sec
        self.offset_pct = offset_pct

        # Статистика для отчётности
        self.stats = {
            "attempted": 0,
            "filled": 0,
            "partial": 0,
            "fallback": 0,
            "total_improvement_usdt": 0.0,
            "total_improvement_pct": 0.0,
            "missed_opportunities": 0
        }

    def simulate_limit_execution(self, side: str, qty: float, mid_price: float,
                                  candle_high: float, candle_low: float,
                                  timeout_factor: float = 1.0) -> Tuple[bool, float, float, str]:
        """
        Симулирует исполнение лимитного ордера на основе свечи.

        Логика:
        - Лимитка выставляется на mid_price ± offset_pct
        - Если цена свечи достигает лимитной цены — ордер исполняется
        - timeout_factor учитывает, что в реальности рынок может не дойти до лимитки за timeout

        Возвращает: (исполнен, qty, fill_price, тип_исполнения)
        """
        self.stats["attempted"] += 1

        # 1. Рассчитываем цену лимитки
        if side == "SELL":
            # Продаём дороже mid
            limit_price = mid_price * (1 + self.offset_pct / 100)
            # Проверяем, достигла ли цена свечи нашего лимита (high >= limit_price)
            if candle_high >= limit_price:
                # Исполнилось! Цена исполнения — лучшая из возможных
                fill_price = max(limit_price, mid_price)  # Не хуже mid
                improvement = fill_price - mid_price
                self.stats["filled"] += 1
                self.stats["total_improvement_usdt"] += improvement * qty
                self.stats["total_improvement_pct"] += (improvement / mid_price) * 100
                return True, qty, fill_price, "LIMIT_FILLED"
            else:
                # Не исполнилось — fallback на market по close
                self.stats["fallback"] += 1
                self.stats["missed_opportunities"] += 1
                return True, qty, candle_high, "FALLBACK_MARKET"
        else:  # BUY
            # Покупаем дешевле mid
            limit_price = mid_price * (1 - self.offset_pct / 100)
            # Проверяем, достигла ли цена свечи нашего лимита (low <= limit_price)
            if candle_low <= limit_price:
                # Исполнилось!
                fill_price = min(limit_price, mid_price)  # Не хуже mid
                improvement = mid_price - fill_price
                self.stats["filled"] += 1
                self.stats["total_improvement_usdt"] += improvement * qty
                self.stats["total_improvement_pct"] += (improvement / mid_price) * 100
                return True, qty, fill_price, "LIMIT_FILLED"
            else:
                # Не исполнилось — fallback на market
                self.stats["fallback"] += 1
                self.stats["missed_opportunities"] += 1
                return True, qty, candle_low, "FALLBACK_MARKET"

    def get_summary(self) -> Dict:
        """Возвращает сводную статистику."""
        fill_rate = self.stats["filled"] / self.stats["attempted"] if self.stats["attempted"] > 0 else 0
        avg_improvement_pct = self.stats["total_improvement_pct"] / self.stats["filled"] if self.stats["filled"] > 0 else 0

        return {
            **self.stats,
            "fill_rate": fill_rate,
            "avg_improvement_pct": avg_improvement_pct
        }


async def download_live_data(symbol: str, data_dir: str, days: int = 2):
    """Загружает данные напрямую с Binance Futures за указанное количество дней"""
    endpoint = f"https://fapi.binance.com/fapi/v1/klines"

    logger.info(f"Step 0: Downloading LIVE data for {symbol} (Last {days} days)...")

    all_data = []
    now = int(time.time() * 1000)
    total_minutes = days * 1440

    # Binance позволяет скачивать по 1500 свечей за раз
    chunk_size = 1440
    num_chunks = int(np.ceil(total_minutes / chunk_size))

    async with aiohttp.ClientSession() as session:
        for i in range(num_chunks):
            # Качаем куски от новых к старым
            end_time = now - i * chunk_size * 60 * 1000
            params = {
                "symbol": symbol,
                "interval": "1m",
                "limit": chunk_size,
                "endTime": end_time
            }
            async with session.get(endpoint, params=params) as resp:
                if resp.status != 200:
                    raise Exception(f"Binance API Error: {resp.status}")
                chunk = await resp.json()
                all_data.extend(chunk)

    # Формируем DataFrame
    df = pd.DataFrame(all_data, columns=['time', 'open', 'high', 'low', 'close', 'vol', 'close_time', 'q_vol', 'trades', 't_base', 't_quote', 'ignore'])
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)

    # Удаляем дубликаты и сортируем
    df = df.drop_duplicates(subset=['time']).sort_values('time')

    # Ограничиваем точным количеством минут
    df = df.tail(total_minutes)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_path = os.path.join(data_dir, f"{symbol}_live_{days}d.feather")
    df.to_feather(file_path)
    logger.info(f"Success. Saved {days}d live data ({len(df)} candles) to {file_path}")
    return file_path

async def run_backtest(config_path: str, data_dir: str, live_mode: bool = False, ticker_override: str = None,
                        days: int = 2, commission: float = 0.0004, use_limit_orders: bool = False,
                        limit_offset_pct: float = 0.2, limit_timeout_sec: int = 30):
    try:
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Проверка настроек лимитных ордеров в конфиге
        if config.get("limit_order_enabled", False) and not use_limit_orders:
            use_limit_orders = True
            limit_offset_pct = config.get("limit_offset_pct", 0.2)
            limit_timeout_sec = config.get("limit_timeout_sec", 30)

        portfolio_cfg = config["portfolios"][0]
        targets = portfolio_cfg["targets"]
        threshold = portfolio_cfg["rebalance_threshold"]
        base_ticker = ticker_override if ticker_override else config.get("base_ticker", "BTCUSDT")

        # 1. Загрузка данных
        if live_mode:
            file_path = await download_live_data(base_ticker, data_dir, days)
        else:
            # Ищем файл в папке данных
            file_path = None
            possible_names = [
                f"{base_ticker.replace('USDT', '_USDT')}_USDT-1m-futures.feather",
                f"{base_ticker}_live_{days}d.feather",
                f"{base_ticker}_live_48h.feather"
            ]

            for name in possible_names:
                p = os.path.join(data_dir, name)
                if os.path.exists(p):
                    file_path = p
                    break

            if not file_path:
                # Попытка найти любой файл, содержащий имя тикера
                files = [f for f in os.listdir(data_dir) if base_ticker in f and f.endswith('.feather')]
                if files:
                    file_path = os.path.join(data_dir, files[0])
                else:
                    logger.error(f"Data file not found for {base_ticker} in {data_dir}")
                    return

        logger.info(f"Starting backtest for {base_ticker} using {file_path}...")
        logger.info(f"Limit Orders Strategy: {'ENABLED' if use_limit_orders else 'DISABLED'} (offset={limit_offset_pct}%, timeout={limit_timeout_sec}s)")
        df = pd.read_feather(file_path)
        df = df.copy().reset_index(drop=True)

        # 2. Инициализация
        initial_capital = portfolio_cfg.get("max_capital_usdt", 10000.0)
        real_balance = initial_capital
        virt_basis_price = df.iloc[0]['close']
        virt_allocated_usdt = real_balance * targets["VIRTUAL"]["share"]
        current_equity = real_balance

        siphoning_threshold_pct = portfolio_cfg.get("siphoning_threshold_pct", 0.0)
        reinvestment_ratio = portfolio_cfg.get("reinvestment_ratio", 0.0)
        equity_trailing_stop_pct = portfolio_cfg.get("equity_trailing_stop_pct", 0.0)
        siphoning_reserve = 0.0
        initial_tpv = initial_capital
        reference_tpv = initial_capital
        tpv_ath = initial_capital

        positions = {f"{base_ticker}_LONG": 0.0, f"{base_ticker}_SHORT": 0.0}

        stats = {
            "rebalance_cycles": 0,
            "total_volume_usdt": 0.0,
            "max_tpv": initial_capital,
            "max_drawdown_pct": 0.0,
            "max_drawdown_duration": 0,
            "current_drawdown_duration": 0,
            "daily_returns": [],
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "trailing_stop_triggered": False,
            "trailing_stop_step": 0,
            # Статистика лимитных ордеров
            "limit_orders_stats": {
                "attempted": 0,
                "filled": 0,
                "fallback": 0,
                "total_improvement_usdt": 0.0,
                "total_improvement_pct": 0.0
            }
        }

        history = []
        prev_tpv = initial_capital
        fee_rate = commission

        # Инициализация симулятора лимитных ордеров
        limit_simulator = None
        if use_limit_orders:
            # Maker комиссия 0.02% для лимитных ордеров
            limit_simulator = LimitOrderSimulator(
                commission_pct=0.02,
                timeout_sec=limit_timeout_sec,
                offset_pct=limit_offset_pct
            )

        # 3. Цикл бэктеста
        for i in range(0, len(df)):
            curr_price = df.iloc[i]['close']
            curr_high = df.iloc[i]['high']
            curr_low = df.iloc[i]['low']

            calc = PortfolioCalculator(positions, curr_price, current_equity, virt_basis_price, virt_allocated_usdt,
                                     base_ticker=base_ticker, siphoning_reserve=siphoning_reserve)

            # Equity Trailing Stop Tracking
            if calc.total_tpv > tpv_ath:
                tpv_ath = calc.total_tpv

            # --- АНАЛИТИКА: PnL Tracking ---
            tpv_change = calc.total_tpv - prev_tpv
            if tpv_change > 0: stats["gross_profit"] += tpv_change
            else: stats["gross_loss"] += abs(tpv_change)
            prev_tpv = calc.total_tpv

            # --- АНАЛИТИКА: Drawdown & Peak Tracking ---
            if calc.total_tpv > stats["max_tpv"]:
                stats["max_tpv"] = calc.total_tpv
                stats["max_drawdown_duration"] = max(stats["max_drawdown_duration"], stats["current_drawdown_duration"])
                stats["current_drawdown_duration"] = 0
            else:
                stats["current_drawdown_duration"] += 1

            drawdown = (stats["max_tpv"] - calc.total_tpv) / stats["max_tpv"] if stats["max_tpv"] > 0 else 0
            if drawdown > stats["max_drawdown_pct"]:
                stats["max_drawdown_pct"] = drawdown

            # Daily Returns for Sharpe
            if i % 1440 == 0 and i > 0:
                day_start_tpv = history[i-1440]["tpv"] if len(history) >= 1440 else initial_capital
                day_return = (calc.total_tpv / day_start_tpv) - 1
                stats["daily_returns"].append(day_return)

            if equity_trailing_stop_pct > 0 and tpv_ath > 0:
                drawdown_from_ath = (1 - calc.total_tpv / tpv_ath) * 100
                if drawdown_from_ath >= equity_trailing_stop_pct:
                    stats["trailing_stop_triggered"] = True
                    stats["trailing_stop_step"] = i
                    logger.warning(f"!!! [STOP] Step {i}: Equity Trailing Stop triggered at {calc.total_tpv:.2f} (ATH: {tpv_ath:.2f}, Drop: {drawdown_from_ath:.2f}%)")
                    break

            # Profit Siphoning (Синхронизирован с Equity)
            if siphoning_threshold_pct > 0:
                if calc.tpv > initial_tpv * (1 + siphoning_threshold_pct / 100):
                    profit = calc.tpv - initial_tpv
                    to_reinvest = profit * reinvestment_ratio
                    to_reserve = profit - to_reinvest
                    
                    # Физически забираем прибыль из рабочего капитала
                    current_equity -= to_reserve
                    siphoning_reserve += to_reserve
                    initial_tpv += to_reinvest
                    
                    # Пересчитываем калькулятор с учетом нового Equity и Резерва
                    calc = PortfolioCalculator(positions, curr_price, current_equity, virt_basis_price, virt_allocated_usdt,
                                             base_ticker=base_ticker, siphoning_reserve=siphoning_reserve, targets=targets)

            # Rebalancing
            current_threshold = -1.0 if i == 0 else threshold
            if i > 0 and calc.tpv < reference_tpv:
                current_threshold *= 2.0

            deviations = calc.calculate_deviations(targets, current_threshold)
            if deviations:
                # Фильтруем отклонения по минимальной стоимости (Notional Guard)
                min_notional = config.get("min_notional_usdt", 6.0)
                filtered_deviations = [d for d in deviations if abs(d["diff_usdt"]) >= min_notional]
                
                if filtered_deviations:
                    stats["rebalance_cycles"] += 1
                    for dev in filtered_deviations:
                        key = dev["symbol"]
                        base_order_qty = dev["diff_usdt"] / curr_price
                        stats["total_volume_usdt"] += abs(dev["diff_usdt"])

                        # Логика исполнения
                        if use_limit_orders and limit_simulator:
                            pos_side = key.split('_')[1]
                            side = ("BUY" if dev["diff_usdt"] > 0 else "SELL") if pos_side == "LONG" else ("SELL" if dev["diff_usdt"] > 0 else "BUY")
                            filled, fill_qty, fill_price, exec_type = limit_simulator.simulate_limit_execution(
                                side=side, qty=abs(base_order_qty), mid_price=curr_price,
                                candle_high=curr_high, candle_low=curr_low
                            )
                            positions[key] += (fill_qty if side == "BUY" else -fill_qty)
                        else:
                            positions[key] += base_order_qty

                    # СИНХРОНИЗАЦИЯ: Фиксируем виртуальный профит в Equity и сбрасываем базис
                    virt_profit = calc.virt_current_value - virt_allocated_usdt
                    current_equity += virt_profit
                    
                    virt_basis_price = curr_price
                    # КРИТИЧЕСКАЯ ПРАВКА: База виртуальной части берется от ВСЕГО TPV, а не от Equity
                    virt_allocated_usdt = calc.tpv * targets["VIRTUAL"]["share"]
                    
                    # Финальный пересчет после всех правок
                    calc = PortfolioCalculator(positions, curr_price, current_equity, virt_basis_price, virt_allocated_usdt,
                                             base_ticker=base_ticker, siphoning_reserve=siphoning_reserve, targets=targets)

            if i % 1000 == 0:
                res_str = f" SAFE:{siphoning_reserve:7.2f}" if siphoning_reserve > 0 else ""
                logger.info(f"Step {i:6d}: TPV={calc.total_tpv:8.2f}{res_str} | L:{calc.share_long_pct}% S:{calc.share_short_pct}% V:{calc.share_virt_pct}% | {base_ticker}={curr_price:8.4f}")

            if i < len(df) - 1:
                next_price = df.iloc[i+1]['close']
                price_diff = next_price - curr_price
                current_equity += (positions[f"{base_ticker}_LONG"] * price_diff) + (positions[f"{base_ticker}_SHORT"] * (-price_diff))

            history.append({"tpv": calc.total_tpv, "equity": current_equity, "reserve": siphoning_reserve})

        # 4. Final Report
        final_total_tpv = history[-1]["tpv"]
        final_reserve = history[-1]["reserve"]
        final_active_tpv = final_total_tpv - final_reserve

        asset_start = df.iloc[0]['close']
        asset_end = df.iloc[-1]['close']
        asset_change_pct = ((asset_end / asset_start) - 1) * 100

        total_profit_usdt = final_total_tpv - initial_capital
        total_profit_pct = (total_profit_usdt / initial_capital) * 100

        # Расчёт комиссий: разные ставки для лимитных и рыночных ордеров
        if use_limit_orders and limit_simulator:
            limit_stats = limit_simulator.get_summary()
            # Объём исполненный через лимитки (maker 0.02%)
            limit_volume = limit_stats["filled"] * (stats["total_volume_usdt"] / stats["rebalance_cycles"]) if stats["rebalance_cycles"] > 0 else 0
            # Fallback объём (taker 0.04%)
            fallback_volume = stats["total_volume_usdt"] - limit_volume
            est_commissions = (limit_volume * 0.0002) + (fallback_volume * 0.0004)
            # Добавляем профит от улучшения цены
            limit_improvement_profit = limit_stats["total_improvement_usdt"]
        else:
            est_commissions = stats["total_volume_usdt"] * fee_rate
            limit_improvement_profit = 0.0
            limit_stats = None

        net_profit_after_fees = total_profit_usdt - est_commissions + limit_improvement_profit

        returns_arr = np.array(stats["daily_returns"])
        sharpe = (np.mean(returns_arr) / np.std(returns_arr)) * np.sqrt(365) if len(returns_arr) > 1 and np.std(returns_arr) > 0 else 0
        recovery_factor = total_profit_usdt / (initial_capital * stats["max_drawdown_pct"]) if stats["max_drawdown_pct"] > 0 else 0
        profit_factor = stats["gross_profit"] / stats["gross_loss"] if stats["gross_loss"] > 0 else 0

        logger.info("\n" + "="*70)
        logger.info("                 ADVANCED MARKET NEUTRAL ANALYTICS")
        logger.info("="*70)
        logger.info(f"Period:             {len(df)} min ({len(df)/1440:.1f} days)")
        logger.info(f"Asset Performance:  {base_ticker} {asset_change_pct:+.2f}%")
        logger.info(f"Strategy Alpha:     {total_profit_pct - asset_change_pct:+.2f}% vs HODL")
        if use_limit_orders:
            logger.info(f"Strategy Mode:      LIMIT ORDERS (offset={limit_offset_pct}%, maker=0.02%)")
        else:
            logger.info(f"Strategy Mode:      MARKET ORDERS (taker={fee_rate*100:.3f}%)")

        logger.info("-" * 70)
        logger.info(f"Initial Capital:    {initial_capital:.2f} USDT")
        logger.info(f"Final Total TPV:    {final_total_tpv:.2f} USDT")
        logger.info(f"  ├── Active TPV:   {final_active_tpv:.2f} USDT")
        logger.info(f"  └── Profit SAFE:  {final_reserve:.2f} USDT")

        logger.info("-" * 70)
        logger.info(f"Gross Profit:       {total_profit_usdt:+.2f} USDT")
        if use_limit_orders and limit_stats:
            logger.info(f"Limit Improvement:  +{limit_improvement_profit:.2f} USDT (from price improvement)")
        logger.info(f"Est. Commissions:   {est_commissions:.2f} USDT")
        logger.info(f"NET PROFIT (FEES):  {net_profit_after_fees:+.2f} USDT ({ (net_profit_after_fees/initial_capital)*100:+.2f}%)")

        logger.info("-" * 70)
        logger.info(f"Max Drawdown:       {stats['max_drawdown_pct']*100:.4f}% (Peak-to-Trough)")
        logger.info(f"Max DD Duration:    {stats['max_drawdown_duration']/1440:.2f} days")
        logger.info(f"Max Run-up:         {(stats['max_tpv']/initial_capital - 1)*100:.4f}%")
        logger.info(f"Recovery Factor:    {recovery_factor:.2f}")

        if stats["trailing_stop_triggered"]:
            logger.warning(f"Trailing Stop:      TRIGGERED at step {stats['trailing_stop_step']}")
        else:
            logger.info(f"Trailing Stop:      Not triggered (Threshold: {equity_trailing_stop_pct}%)")

        logger.info("-" * 70)
        logger.info(f"Sharpe Ratio:       {sharpe:.2f}")
        logger.info(f"Profit Factor:      {profit_factor:.2f}")
        logger.info(f"Rebalance Cycles:   {stats['rebalance_cycles']}")
        logger.info(f"Total Turnover:     {stats['total_volume_usdt']:.2f} USDT")

        # Статистика лимитных ордеров
        if use_limit_orders and limit_stats:
            logger.info("-" * 70)
            logger.info("                 LIMIT ORDERS STATISTICS")
            logger.info("-" * 70)
            logger.info(f"Attempted:          {limit_stats['attempted']}")
            logger.info(f"Filled (LIMIT):     {limit_stats['filled']} ({limit_stats['fill_rate']*100:.1f}%)")
            logger.info(f"Fallback (MARKET):  {limit_stats['fallback']} ({(1-limit_stats['fill_rate'])*100:.1f}%)")
            logger.info(f"Avg Price Improve:  +{limit_stats['avg_improvement_pct']:.3f}%")
            logger.info(f"Total Improvement:  +{limit_stats['total_improvement_usdt']:.2f} USDT")
            logger.info(f"Missed Opps:        {limit_stats['missed_opportunities']} (price didn't reach limit)")

        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--data_dir", default=r"C:\Python\Prosperous_Bot\third_party\rl-trading-binance\user_data\data\binance\futures")
    parser.add_argument("--live", action="store_true", help="Download fresh data and test it")
    parser.add_argument("--ticker", default=None, help="Override ticker for backtest")
    parser.add_argument("--days", type=int, default=2, help="Number of days for backtest (default: 2)")
    parser.add_argument("--commission", type=float, default=0.0004, help="Commission rate (default: 0.0004)")
    parser.add_argument("--limit", action="store_true", help="Use Limit Orders strategy (instead of market)")
    parser.add_argument("--limit-offset", type=float, default=0.2, help="Limit order offset %% (default: 0.2)")
    parser.add_argument("--limit-timeout", type=int, default=30, help="Limit order timeout in seconds (default: 30)")
    args = parser.parse_args()

    asyncio.run(run_backtest(
        args.config,
        args.data_dir,
        args.live,
        args.ticker,
        args.days,
        args.commission,
        use_limit_orders=args.limit,
        limit_offset_pct=args.limit_offset,
        limit_timeout_sec=args.limit_timeout
    ))
