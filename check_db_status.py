import psycopg2
import os
import datetime

print("Attempting to connect to the database to check data status...")

try:
    # Используем те же параметры подключения, что и в скрипте загрузки
    conn = psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE", "marketdata"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", "")
    )
    cur = conn.cursor()

    # Запрос для получения символов и диапазона дат
    # open_time_ms - это timestamp в миллисекундах
    query = """
        SELECT
            symbol,
            MIN(open_time_ms),
            MAX(open_time_ms),
            COUNT(*)
        FROM
            public.klines_1m
        GROUP BY
            symbol
        ORDER BY
            symbol;
    """
    cur.execute(query)
    results = cur.fetchall()

    if results:
        print("\n" + "="*80)
        print(f"{ 'Symbol':<15} | {'First Record (UTC)':<25} | {'Last Record (UTC)':<25} | {'Row Count':>10}")
        print("-"*80)
        for row in results:
            symbol, min_ts, max_ts, count = row
            # Конвертируем миллисекунды в читаемую дату
            min_time = datetime.datetime.fromtimestamp(min_ts / 1000, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            max_time = datetime.datetime.fromtimestamp(max_ts / 1000, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            print(f"{symbol:<15} | {min_time:<25} | {max_time:<25} | {count:>10,}")
        print("="*80)
    else:
        print("\nNo data found in the 'public.klines_1m' table.")
        print("Please ensure the data loading script has been run successfully.")

    cur.close()
    conn.close()

except psycopg2.OperationalError as e:
    print(f"\n[ERROR] Could not connect to PostgreSQL.")
    print(f"Details: {e}")
    print("\nPlease check the following:")
    print("1. Is the PostgreSQL server running?")
    print("2. Are the environment variables for the connection set correctly?")
    print("   - PGHOST (default: localhost)")
    print("   - PGPORT (default: 5432)")
    print("   - PGDATABASE (default: marketdata)")
    print("   - PGUSER (default: postgres)")
    print("   - PGPASSWORD")
    print("The error message suggests a password might be required or incorrect.")

except psycopg2.errors.UndefinedTable:
    print("\n[ERROR] The table 'public.klines_1m' does not exist.")
    print("Please run the `download_historical_data.py` script first to create the table and load data.")

except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
