# exchange_api.py
"""
Shim-модуль для обратной совместимости.
Импортирует ExchangeAPI из нового модуля exchange_gate.py.
"""
from exchange_gate import ExchangeAPI
__all__ = ["ExchangeAPI"]