import logging

class DuplicateFilter(logging.Filter):
    """Позволяет скрывать повторяющиеся сообщения INFO."""
    def __init__(self, max_repeats=3):
        super().__init__()
        self._cache = {}
        self.max_repeats = max_repeats

    def filter(self, record: logging.LogRecord) -> bool:
        key = (record.module, record.msg)
        self._cache[key] = self._cache.get(key, 0) + 1
        return self._cache[key] <= self.max_repeats

def configure_root(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )
    logging.getLogger().addFilter(DuplicateFilter())
