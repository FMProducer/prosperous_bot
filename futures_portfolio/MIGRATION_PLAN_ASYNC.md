# План миграции BinanceConnector на AsyncClient

Для повышения производительности и избавления от накладных расходов `asyncio.to_thread`, рекомендуется перейти с синхронного `binance.client.Client` на асинхронный `binance.client.AsyncClient`.

## 1. Изменение инициализации (Паттерн Фабрика)

Поскольку создание `AsyncClient` является асинхронной операцией, его нельзя вызвать напрямую в `__init__`. Рекомендуется использовать метод класса `create()`:

```python
from binance import AsyncClient

class BinanceConnector:
    def __init__(self, api_key: str, secret_key: str, testnet: bool = True, base_ticker: str = "BTCUSDT"):
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        self.base_ticker = base_ticker
        self.client: AsyncClient = None

    @classmethod
    async def create(cls, api_key: str, secret_key: str, testnet: bool = True, base_ticker: str = "BTCUSDT"):
        instance = cls(api_key, secret_key, testnet, base_ticker)
        # Инициализация асинхронного клиента
        instance.client = await AsyncClient.create(
            api_key,
            secret_key,
            testnet=testnet,
            # Настройка URL для обхода блокировок если нужно
        )
        return instance

    async def close(self):
        """Важно: асинхронный клиент требует явного закрытия сессии."""
        if self.client:
            await self.client.close_connection()
```

## 2. Удаление `asyncio.to_thread`

Все методы I/O должны быть переписаны на прямой вызов асинхронных методов клиента.

**Пример `get_positions`:**
```python
<<<<<<< OLD
    @retry_on_network_error(retries=5, delay=3.0)
    async def get_positions(self) -> Dict[str, Dict]:
        positions = await asyncio.to_thread(self.futures_client.futures_position_information)
=======
    @retry_on_network_error(retries=5, delay=3.0)
    async def get_positions(self) -> Dict[str, Dict]:
        positions = await self.client.futures_position_information()
>>>>>>> NEW
```

**Пример `get_mark_prices`:**
```python
<<<<<<< OLD
    @retry_on_network_error(retries=5, delay=3.0)
    async def get_mark_prices(self, tickers: List[str] = None) -> Dict[str, float]:
        prices = await asyncio.to_thread(self.futures_client.futures_mark_price)
=======
    @retry_on_network_error(retries=5, delay=3.0)
    async def get_mark_prices(self, tickers: List[str] = None) -> Dict[str, float]:
        prices = await self.client.futures_mark_price()
>>>>>>> NEW
```

## 3. Обновление декоратора ретраев

Декоратор `retry_on_network_error` уже поддерживает асинхронные функции, поэтому его логика фильтрации статус-кодов (добавленная в текущем патче) продолжит работать корректно с `AsyncClient`.

## 4. Изменения в `main.py`

При запуске бота необходимо изменить способ создания коннектора:

```python
# Было:
# connector = BinanceConnector(api_key=api_key, secret_key=secret_key, testnet=cfg.get("testnet", True))

# Стало:
connector = await BinanceConnector.create(api_key=api_key, secret_key=secret_key, testnet=cfg.get("testnet", True))
```

## Преимущества миграции:
1. **CPU Efficiency**: Исчезает необходимость в переключении контекста между потоками ОС.
2. **Memory**: Меньшее потребление памяти (нет пула потоков).
3. **Latency**: Более быстрый отклик API за счет отсутствия блокировок на уровне GIL при входе/выходе из `to_thread`.
