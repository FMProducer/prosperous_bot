# -*- coding: utf-8 -*-
"""
Проверка паритета расчётов paper_trader c бэктест-логикой:
 - размер позиции: cfg.backtest.position_fraction
 - комиссии: cfg.market.transaction_fee
 - проскальзывание: cfg.market.slippage
Тесты лёгкие, без БД/модели.
"""
import importlib.util
import pathlib
import math
import sys

def _load_master_cfg():
    """Загружаем config.py через importlib и РЕГИСТРИРУЕМ его в sys.modules до exec_module."""
    tests_dir = pathlib.Path(__file__).parent
    cfg_path = tests_dir.parent / "config.py"
    name = "rtb_config_for_tests"
    spec = importlib.util.spec_from_file_location(name, cfg_path.as_posix())
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = mod  # <-- важно для dataclasses и строковых аннотаций
    spec.loader.exec_module(mod)  # type: ignore
    # Пытаемся получить cfg; если его нет — пробуем MasterConfig()
    if hasattr(mod, "cfg"):
        return mod.cfg
    if hasattr(mod, "MasterConfig"):
        return mod.MasterConfig()
    raise RuntimeError("Не найден ни `cfg`, ни `MasterConfig` в config.py")

def _load_paper_trader():
    """Динамическая загрузка paper_trader.py с РЕГИСТРАЦИЕЙ в sys.modules до exec_module."""
    tests_dir = pathlib.Path(__file__).parent
    p = tests_dir.parent / "paper_trader.py"
    name = "rtb_paper_trader_for_tests"
    spec = importlib.util.spec_from_file_location(name, p.as_posix())
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = mod  # <-- критично: иначе dataclasses не найдёт модуль по cls.__module__
    spec.loader.exec_module(mod)  # type: ignore
    # пробрасываем MasterConfig
    mod._MASTER_CFG = _load_master_cfg()
    return mod

def test_position_size_parity():
    pt = _load_paper_trader()
    capital = 10_000.0
    entry = 100.0
    master_cfg = _load_master_cfg()
    pf = master_cfg.backtest.position_fraction
    expected_qty = (capital * pf) / entry
    qty = pt._position_size(capital, risk_pct=1.0, entry=entry)
    assert math.isclose(qty, expected_qty, rel_tol=1e-7)

def test_fees_parity():
    pt = _load_paper_trader()
    notional = 1234.56
    master_cfg = _load_master_cfg()
    expected_fee = notional * master_cfg.market.transaction_fee
    fee = pt._fees_cost(notional, fee_bps=40.0)  # игнорируется при наличии MasterConfig
    assert math.isclose(fee, expected_fee, rel_tol=1e-12)

def test_slippage_parity_buy_sell():
    pt = _load_paper_trader()
    price = 200.0
    master_cfg = _load_master_cfg()
    slip = master_cfg.market.slippage
    assert math.isclose(pt._apply_slippage(price, 5.0, "BUY"),  price * (1 + slip), rel_tol=1e-12)
    assert math.isclose(pt._apply_slippage(price, 5.0, "SELL"), price * (1 - slip), rel_tol=1e-12)