# src/exchange_api.py
"""Shim-модуль: экспортирует ExchangeAPI из src.exchange_gate.

Работает и когда модуль импортируется как
`import exchange_api` (через корневой shim), и когда как
`import src.exchange_api` внутри пакета."""
from .exchange_gate import ExchangeAPI   # <-- правильная ссылка
__all__ = ["ExchangeAPI"]
