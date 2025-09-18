from prosperous_bot.rebalance_backtester import _subst_symbol

def test_subst_symbol_list():
    obj = ["A_{main_asset_symbol}", "B_{main_asset_symbol}"]
    result = _subst_symbol(obj, "BTC")
    assert result == ["A_BTC", "B_BTC"]

def test_subst_symbol_usdt():
    obj = "*USDT"
    result = _subst_symbol(obj, "BTC")
    assert result == "BTCUSDT"
