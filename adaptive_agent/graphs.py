"""Графическая визуализация цены и технических индикаторов, включая ADX.
"""

import os
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import base64
from adaptive_agent.io import BytesIO


def plot_indicators(data, symbol: str, combined_signals):
    """Рисует график с соотношением высот 4:1 (цена : ADX)."""
    logging.info("Plotting indicators for %s", symbol)
    plt.switch_backend("Agg")

    # Используем GridSpec с ratio 4:1
    fig = Figure(figsize=(16, 10), dpi=100)
    gs = GridSpec(5, 1, figure=fig, height_ratios=[4, 0, 0, 0, 1])
    ax_price = fig.add_subplot(gs[:4, 0])  # верхние 4/5
    ax_adx = fig.add_subplot(gs[4, 0], sharex=ax_price)

    # --- PRICE ---
    ax_price.plot(data.index, data["close"], label="Close", color="black", linewidth=1.5)
    ax_price.plot(data.index, data["EMA1"], label="EMA1", color="blue", linewidth=1.2, alpha=0.7)
    ax_price.plot(data.index, data["EMA2"], label="EMA2", color="orange", linewidth=1.2, alpha=0.7)
    ax_price.plot(data.index, data["EMA3"], label="EMA3", color="green", linewidth=1.2, alpha=0.7)
    ax_price.plot(data.index, data["MA1"], label="MA1", color="red", linewidth=1.2, alpha=0.7)
    ax_price.plot(data.index, data["MA2"], label="MA2", color="purple", linewidth=1.2, alpha=0.7)
    ax_price.plot(data.index, data["BB_High"], label="BB High", color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax_price.plot(data.index, data["BB_Low"], label="BB Low", color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax_price.plot(data.index, data["BB_Mid"], label="BB Mid", color="gray", linestyle=":", linewidth=1, alpha=0.5)

    for sig, ts in combined_signals:
        if sig == "buy":
            ax_price.scatter(ts, data.loc[ts, "low"] * 0.99, color="green", marker="^", s=120)
        else:
            ax_price.scatter(ts, data.loc[ts, "high"] * 1.01, color="red", marker="v", s=120)

    ax_price.set_title(f"{symbol} – Price & Indicators")
    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left", fontsize=8)

    # --- ADX ---
    ax_adx.plot(data.index, data["ADX"], color="gray", linewidth=1.2, label="ADX")
    ax_adx.axhline(25, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="ADX threshold")
    ax_adx.set_ylabel("ADX")
    ax_adx.set_xlabel("Time")
    ax_adx.legend(loc="upper left", fontsize=8)

    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout(pad=2)

    html_content = _html(fig, symbol)
    os.makedirs("graphs", exist_ok=True)
    with open(f"graphs/{symbol}_indicators_adx.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    fig.clf(); plt.close(fig)


def _html(fig: Figure, symbol: str) -> str:
    buf = BytesIO(); FigureCanvasAgg(fig).print_png(buf)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"""<html><head><title>{symbol}</title></head><body><img src='data:image/png;base64,{b64}'></body></html>"""
