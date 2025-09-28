Как собрать это в PowerShell (Windows) — готовые команды

Требуется установленный psql/pg_dump (часть PostgreSQL client). Все команды — «копи-паст».
⚠️ Не хардкодьте пароль в строке — используйте переменную окружения PGPASSWORD.

0) Подключение (URI) и базовые проверки
# 0.1 Укажите свой URI (логин/хост/порт/БД/sslmode при необходимости)
$env:PGPASSWORD = "<ВАШ_ПАРОЛЬ>"
$PGURI = "postgresql://<user>@<host>:5432/<db>?sslmode=prefer"

# 0.2 Проверка версии/таймзоны/локали
psql $PGURI -X -c "SELECT version(); SHOW server_version; SHOW TIME ZONE; SHOW SERVER_ENCODING; SHOW LC_COLLATE;"  # psql справка: meta/CLI :contentReference[oaicite:8]{index=8}

1) Список пользовательских схем, таблиц и представлений
# 1.1 Все схемы (кроме системных)
psql $PGURI -X -A -F "," -P footer=off -c @"
SELECT schema_name
FROM information_schema.schemata
WHERE schema_name NOT IN ('pg_catalog','information_schema')
ORDER BY 1;
"@ | Set-Content schemas.csv

# 1.2 Все таблицы и вьюхи по схемам
psql $PGURI -X -A -F "," -P footer=off -c @"
SELECT table_schema, table_name, table_type
FROM information_schema.tables
WHERE table_schema NOT IN ('pg_catalog','information_schema')
ORDER BY 1,2;
"@ | Set-Content tables_and_views.csv

# 1.3 Материализованные представления
psql $PGURI -X -A -F "," -P footer=off -c @"
SELECT schemaname, matviewname, definition
FROM pg_matviews
ORDER BY 1,2;
"@ | Set-Content matviews.csv  -- pg_matviews описана в официальной документации :contentReference[oaicite:9]{index=9}

2) Колонки с типами/nullable/default (для всех user-таблиц)
psql $PGURI -X -A -F "," -P footer=off -c @"
SELECT table_schema, table_name, ordinal_position, column_name,
       data_type, udt_name, is_nullable, column_default
FROM information_schema.columns
WHERE table_schema NOT IN ('pg_catalog','information_schema')
ORDER BY 1,2,3;
"@ | Set-Content columns.csv   # information_schema.columns — стандартный слой метаданных :contentReference[oaicite:10]{index=10}

3) Ограничения (PK/UK/FK/CHECK) и индексы
# 3.1 Ограничения
psql $PGURI -X -A -F "," -P footer=off -c @"
SELECT n.nspname AS schema, c.relname AS table,
       con.conname AS constraint, con.contype AS type,
       pg_get_constraintdef(con.oid) AS definition
FROM pg_constraint con
JOIN pg_class c ON c.oid = con.conrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname NOT IN ('pg_catalog','information_schema')
ORDER BY 1,2,3;
"@ | Set-Content constraints.csv

# 3.2 Индексы
psql $PGURI -X -A -F "," -P footer=off -c @"
SELECT schemaname, tablename, indexname, indexdef
FROM pg_indexes
WHERE schemaname NOT IN ('pg_catalog','information_schema')
ORDER BY 1,2;
"@ | Set-Content indexes.csv

4) Партиционирование (если используете range/list/hash)
# 4.1 Список partitioned-таблиц и стратегия (range/list/hash)
psql $PGURI -X -A -F "," -P footer=off -c @"
SELECT n.nspname AS schema, c.relname AS table, p.partstrat AS strategy
FROM pg_partitioned_table p
JOIN pg_class c ON c.oid = p.partrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
ORDER BY 1,2;
"@ | Set-Content partitions_master.csv  # pg_partitioned_table описана в каталоге системных таблиц :contentReference[oaicite:11]{index=11}

# 4.2 Связи родитель ↔ дочерние партиции
psql $PGURI -X -A -F "," -P footer=off -c @"
SELECT parent_ns.nspname AS parent_schema, parent.relname AS parent_table,
       child_ns.nspname AS child_schema, child.relname AS child_table
FROM pg_inherits i
JOIN pg_class child ON child.oid = i.inhrelid
JOIN pg_class parent ON parent.oid = i.inhparent
JOIN pg_namespace child_ns ON child_ns.oid = child.relnamespace
JOIN pg_namespace parent_ns ON parent_ns.oid = parent.relnamespace
ORDER BY 1,2,3,4;
"@ | Set-Content partitions_children.csv  # pg_inherits — системный каталог наследования/партиций :contentReference[oaicite:12]{index=12}

5) Размеры, ориентировочные строки, статистика
# 5.1 Размеры и оценка строк
psql $PGURI -X -A -F "," -P footer=off -c @"
SELECT n.nspname AS schema, c.relname AS table,
       pg_size_pretty(pg_total_relation_size(c.oid)) AS total_size,
       s.reltuples::bigint AS est_rows
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
WHERE n.nspname NOT IN ('pg_catalog','information_schema')
  AND c.relkind = 'r'  -- только обычные таблицы
ORDER BY pg_total_relation_size(c.oid) DESC;
"@ | Set-Content table_sizes.csv

6) Расширения и права
# 6.1 Расширения
psql $PGURI -X -A -F "," -P footer=off -c "SELECT extname, extversion FROM pg_extension ORDER BY 1;" | Set-Content extensions.csv

# 6.2 Права на таблицы (кратко для текущего пользователя)
psql $PGURI -X -A -F "," -P footer=off -c @"
SELECT table_schema, table_name, string_agg(DISTINCT privilege_type, ',' ORDER BY privilege_type) AS privileges
FROM information_schema.table_privileges
WHERE grantee = CURRENT_USER
GROUP BY 1,2
ORDER BY 1,2;
"@ | Set-Content table_privileges.csv