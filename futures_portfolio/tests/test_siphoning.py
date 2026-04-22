import unittest
import sys
import os

# Add the project root to sys.path to import calculator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from calculator import PortfolioCalculator

class TestSiphoningLogic(unittest.TestCase):
    def test_recovery_mode_activation(self):
        # Кейс: Депо просело до 9000, в SAFE лежит 500. Начальный капитал 10000.
        # Бот должен использовать все 9500 (9000 активных + 500 сейфа) для ребаланса, а не 9000.

        # total_tpv = real_equity (9000) + (virt_current_value - virt_allocated_usdt) + siphoning_reserve (500)
        # В данном случае, пусть virt_pnl = 0 для простоты.

        calc = PortfolioCalculator(
            positions={"ZECUSDT_LONG": 0.0, "ZECUSDT_SHORT": 0.0},
            spot_price=40.0,
            real_equity=9000.0,
            virt_basis_price=40.0,
            virt_allocated_usdt=3500.0,
            siphoning_reserve=500.0,
            base_ticker="ZECUSDT",
            initial_capital=10000.0,
            targets={
                "BASE_LONG": {"share": 0.3, "leverage": 5},
                "BASE_SHORT": {"share": 0.3, "leverage": 5},
                "VIRTUAL": {"share": 0.4, "leverage": 1}
            }
        )

        # Ожидаем, что total_tpv = 9000 + 0 + 500 = 9500
        self.assertEqual(calc.total_tpv, 9500.0)

        # Так как 9500 < 10000, мы в режиме восстановления
        # Ожидаем, что tpv (активный капитал) подтянет резерв
        self.assertEqual(calc.tpv, 9500.0)
        print("✅ Recovery mode test passed: Siphoning suspended during drawdown (tpv includes SAFE).")

    def test_normal_siphoning_mode(self):
        # Кейс: Депо выросло до 11000, в SAFE лежит 500. Начальный капитал 10000.
        # Бот должен исключить 500 из активного капитала для ребаланса.

        calc = PortfolioCalculator(
            positions={"ZECUSDT_LONG": 0.0, "ZECUSDT_SHORT": 0.0},
            spot_price=40.0,
            real_equity=11000.0,
            virt_basis_price=40.0,
            virt_allocated_usdt=3500.0,
            siphoning_reserve=500.0,
            base_ticker="ZECUSDT",
            initial_capital=10000.0,
            targets={
                "BASE_LONG": {"share": 0.3, "leverage": 5},
                "BASE_SHORT": {"share": 0.3, "leverage": 5},
                "VIRTUAL": {"share": 0.4, "leverage": 1}
            }
        )

        # total_tpv = 11000 + 0 + 500 = 11500
        self.assertEqual(calc.total_tpv, 11500.0)

        # Так как 11500 >= 10000, siphoning активен
        # tpv = total_tpv - siphoning_reserve = 11500 - 500 = 11000
        self.assertEqual(calc.tpv, 11000.0)
        print("✅ Normal mode test passed: Siphoning active when above initial capital.")

if __name__ == '__main__':
    unittest.main()
