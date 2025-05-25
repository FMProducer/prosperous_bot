"""
Shim-модуль для экспорта ExchangeAPI.

Поддерживает оба варианта импорта:
1)  import exchange_api          (когда 'src' присутствует в PYTHONPATH)
2)  import src.exchange_api      (когда импорт идёт через пакет 'src')
"""
from __future__ import annotations
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent

def _add_src_to_syspath() -> None:
    """Гарантируем, что соседний exchange_gate.py виден при top-level импорте."""
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))

try:
    # ◇ Случай ❷: модуль загружен как часть пакета 'src'
    from .exchange_gate import ExchangeAPI            # type: ignore
except (ImportError, SystemError):
    # ◇ Случай ❶: модуль загружен как top-level 'exchange_api'
    _add_src_to_syspath()
    from exchange_gate import ExchangeAPI             # noqa: E402

__all__ = ["ExchangeAPI"]
