import re

def calculate_adv_stats(log_text):
    # Ищем все значения L1 и S2 Adv
    l1_values = [float(m.group(1)) for m in re.finditer(r"L1 Adv: ([\d\.]+)", log_text)]
    s2_values = [float(m.group(1)) for m in re.finditer(r"S2 Adv: ([\d\.]+)", log_text)]

    # Расчет средних
    avg_l1 = sum(l1_values) / len(l1_values) if l1_values else 0.0
    avg_s2 = sum(s2_values) / len(s2_values) if s2_values else 0.0

    print(f"Количество записей: {len(l1_values)}")
    print(f"Среднее L1 Adv: {avg_l1:.6f}")
    print(f"Среднее S2 Adv: {avg_s2:.6f}")

# Вставьте ваши логи в переменную ниже для проверки
log_data = """
2026-01-29 01:30:55,266 - CustomD3QNStrategy4z - INFO - 🔍 BTC/USDT:USDT L1 Adv: 0.00010 ...
... (остальные логи)
"""

if __name__ == "__main__":
    calculate_adv_stats(log_data)
