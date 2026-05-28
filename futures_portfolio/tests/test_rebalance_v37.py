import pytest
from decimal import Decimal
from calculator import PortfolioCalculator

def test_portfolio_convergence_and_churn():
    # Initial setup
    initial_tpv = Decimal('100.0')
    targets = {
        "BASE_LONG": {"share": 0.29, "leverage": 5},
        "BASE_SHORT": {"share": 0.36, "leverage": 5},
        "VIRTUAL": {"share": 0.35, "leverage": 1}
    }

    state = {
        "balance": 65.0,
        "l_qty": 5.8,   # 29 USDT margin * 5 / 25 price = 5.8
        "s_qty": 7.2,   # 36 USDT margin * 5 / 25 price = 7.2
        "v_qty": 1.4,   # 35 USDT / 25 price = 1.4
        "price": 25.0,
        "entry": 25.0
    }

    rebalance_counts = {"BASE_LONG": 0, "BASE_SHORT": 0, "VIRTUAL": 0}

    # Simulate price swings
    prices = [25.0, 26.5, 24.0, 28.0, 22.0]

    print("\n--- Ребалансировочный Тест v3.7.0 ---")

    for p in prices:
        price = Decimal(str(p))
        # Эмуляция Real Equity (упрощенно: баланс + PnL фьючерсов)
        pnl_l = Decimal(str(state["l_qty"])) * (price - Decimal(str(state["entry"])))
        pnl_s = Decimal(str(state["s_qty"])) * (Decimal(str(state["entry"])) - price)
        real_equity = Decimal(str(state["balance"])) + pnl_l + pnl_s

        calc = PortfolioCalculator(
            positions={
                "BTCUSDT_LONG": float(state["l_qty"]),
                "BTCUSDT_SHORT": float(state["s_qty"])
            },
            spot_price=float(price),
            real_equity=float(real_equity),
            virt_qty=float(state["v_qty"]),
            long_entry_price=float(state["entry"]),
            short_entry_price=float(state["entry"]),
            virt_entry_price=float(state["entry"]),
            base_ticker="BTCUSDT",
            targets=targets,
            initial_capital=100.0
        )

        # Проверка суммы долей
        total_share = (calc.val_long + calc.val_short + calc.val_virt + calc.val_cash) / calc.tpv
        assert abs(total_share - 1) < 0.0001, f"Sum of shares {total_share} != 1.0 at price {p}"

        actions = calc.calculate_deviations(targets, threshold_surplus=0.025, threshold_deficit=0.025)

        for a in actions:
            leg = "VIRTUAL" if a['type'] == 'VIRTUAL_ORDER' else a['symbol'].split('_')[1]
            key = f"BASE_{leg}" if leg != "VIRTUAL" else "VIRTUAL"
            rebalance_counts[key] += 1
            print(f"[Price: {p}] Rebalancing {key} | Action: {a['diff_usdt']} USDT")

    print(f"\nИтоговая статистика ребалансировок: {rebalance_counts}")
    assert sum(rebalance_counts.values()) > 0, "Бот должен был совершить хотя бы одну сделку"

if __name__ == "__main__":
    test_portfolio_convergence_and_churn()
