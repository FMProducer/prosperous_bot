import unittest
from calculator import PortfolioCalculator

class TestRiskEngine(unittest.TestCase):
    def setUp(self):
        self.initial = 10000.0

    def test_siphoning_suspension_in_drawdown(self):
        """Проверка: если эквити < старта, SAFE должен вернуться в TPV"""
        calc = PortfolioCalculator(
            positions={"ZECUSDT_LONG": 100, "ZECUSDT_SHORT": -100},
            spot_price=40.0,
            real_equity=8000.0, # Явная просадка
            virt_basis_price=40.0,
            virt_allocated_usdt=3500.0,
            siphoning_reserve=1000.0,
            initial_capital=self.initial
        )
        # Ожидаем, что tpv будет равен 8000 + 1000 + 0 (virt pnl) = 9000
        self.assertEqual(calc.tpv, 9000.0)

    def test_tpv_calculation_with_profit(self):
        """Проверка: если мы в профите, SAFE должен быть исключен из TPV"""
        calc = PortfolioCalculator(
            positions={"ZECUSDT_LONG": 100, "ZECUSDT_SHORT": -100},
            spot_price=40.0,
            real_equity=11000.0, # Профит
            virt_basis_price=40.0,
            virt_allocated_usdt=3500.0,
            siphoning_reserve=2000.0,
            initial_capital=self.initial
        )
        # Ожидаем, что tpv = 11000 (без учета 2000 резерва)
        self.assertEqual(calc.tpv, 11000.0)

if __name__ == '__main__':
    unittest.main()
