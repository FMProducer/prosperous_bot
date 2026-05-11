import unittest
import sys
import os

# Add the project root to sys.path to import calculator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from calculator import PortfolioCalculator

class TestSiphoningLogic(unittest.TestCase):
    def test_recovery_mode_activation(self):
        # Кейс: Депо просело до 9000, в SAFE лежит 500. Начальный капитал 10000.
        # В текущей логике calculator.py, siphoning_reserve просто добавляется к tpv для получения total_tpv.
        # Ребаланс идет по tpv = real_equity + val_virt.

        calc = PortfolioCalculator(
            positions={"ZECUSDT_LONG": 0.0, "ZECUSDT_SHORT": 0.0},
            spot_price=40.0,
            real_equity=9000.0,
            virt_qty=87.5, # 3500 / 40
            siphoning_reserve=500.0,
            base_ticker="ZECUSDT",
            initial_capital=10000.0,
            targets={
                "BASE_LONG": {"share": 0.3, "leverage": 5},
                "BASE_SHORT": {"share": 0.3, "leverage": 5},
                "VIRTUAL": {"share": 0.4, "leverage": 1}
            }
        )

        # tpv = 9000 + (87.5 * 40) = 9000 + 3500 = 12500
        # total_tpv = 12500 + 500 = 13000
        self.assertEqual(float(calc.total_tpv), 13000.0)

    def test_normal_siphoning_mode(self):
        # Кейс: Депо выросло до 11000, в SAFE лежит 500. Начальный капитал 10000.

        calc = PortfolioCalculator(
            positions={"ZECUSDT_LONG": 0.0, "ZECUSDT_SHORT": 0.0},
            spot_price=40.0,
            real_equity=11000.0,
            virt_qty=87.5, # 3500 / 40
            siphoning_reserve=500.0,
            base_ticker="ZECUSDT",
            initial_capital=10000.0,
            targets={
                "BASE_LONG": {"share": 0.3, "leverage": 5},
                "BASE_SHORT": {"share": 0.3, "leverage": 5},
                "VIRTUAL": {"share": 0.4, "leverage": 1}
            }
        )

        # total_tpv = 11000 (real_equity) + 3500 (val_virt) + 500 (siphoning_reserve) = 15000
        self.assertEqual(float(calc.total_tpv), 15000.0)

if __name__ == '__main__':
    unittest.main()
