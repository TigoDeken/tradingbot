"""Plot a pyramid example trade on a candlestick chart."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import pandas as pd
from data_pipeline import get_data
from trade_engine import run_trades

df = get_data()
trades = run_trades(df, close_strength=0.6, max_open_lots=0.4, min_pyramid_bars=3, pyramid_min_profit_r=0.5)

# Pair index 13/14 — both long, both TP
t1 = trades[13]
t2 = trades[14]

print(f"T1: {t1.direction} entry={t1.entry_date} exit={t1.exit_date} stop={t1.stop_price} tp={t1.tp1_price} r={t1.net_r}")
print(f"T2: {t2.direction} entry={t2.entry_date} exit={t2.exit_date} stop={t2.stop_price} tp={t2.tp1_price} r={t2.net_r}")

# Slice price data: 10 bars before T1 entry to 5 bars after T2 exit
start = t1.entry_date - pd.Timedelta(hours=40)
end   = t2.exit_date  + pd.Timedelta(hours=20)
mask  = (df.index >= start) & (df.index <= end)
ohlc  = df.loc[mask].copy()

fig, ax = plt.subplots(figsize=(16, 7), facecolor="#0d1117")
ax.set_facecolor("#0d1117")

# --- Candlesticks ---
for i, (ts, row) in enumerate(ohlc.iterrows()):
    x = mdates.date2num(ts.to_pydatetime())
    o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
    color = "#26a69a" if c >= o else "#ef5350"
    w = 0.055  # bar width in date units
    # Body
    ax.bar(x, abs(c - o), width=w * 0.7, bottom=min(o, c), color=color, zorder=2)
    # Wick
    ax.plot([x, x], [l, h], color=color, linewidth=0.8, zorder=1)

def xpos(dt):
    return mdates.date2num(pd.Timestamp(dt).to_pydatetime())

# --- T1 levels ---
x1_entry = xpos(t1.entry_date)
x1_exit  = xpos(t1.exit_date)
ax.hlines(t1.entry_price, x1_entry, x1_exit, colors="#4fc3f7", linewidths=1.5, linestyles="--", label="Entry T1", zorder=3)
ax.hlines(t1.stop_price,  x1_entry, x1_exit, colors="#ef5350", linewidths=1.2, linestyles=":",  label="Stop T1",  zorder=3)
ax.hlines(t1.tp1_price,   x1_entry, x1_exit, colors="#26a69a", linewidths=1.2, linestyles=":",  label="TP T1",    zorder=3)

# T1 entry marker
ax.scatter(x1_entry, t1.entry_price, marker="^", color="#4fc3f7", s=120, zorder=5)
# T1 exit marker (TP hit)
ax.scatter(x1_exit, t1.exit_price, marker="D", color="#26a69a", s=80, zorder=5)

# --- T2 levels ---
x2_entry = xpos(t2.entry_date)
x2_exit  = xpos(t2.exit_date)
ax.hlines(t2.entry_price, x2_entry, x2_exit, colors="#ffb300", linewidths=1.5, linestyles="--", label="Entry T2 (add)", zorder=3)
ax.hlines(t2.stop_price,  x2_entry, x2_exit, colors="#ff7043", linewidths=1.2, linestyles=":",  label="Stop T2",       zorder=3)
ax.hlines(t2.tp1_price,   x2_entry, x2_exit, colors="#66bb6a", linewidths=1.2, linestyles=":",  label="TP T2",         zorder=3)

# T2 entry marker
ax.scatter(x2_entry, t2.entry_price, marker="^", color="#ffb300", s=120, zorder=5)
# T2 exit marker
ax.scatter(x2_exit, t2.exit_price, marker="D", color="#66bb6a", s=80, zorder=5)

# Shaded zones
x_start_num = mdates.date2num(start.to_pydatetime())
x_end_num   = mdates.date2num(end.to_pydatetime())
ax.axvspan(x1_entry, x2_entry, alpha=0.06, color="#4fc3f7", zorder=0)  # original trade zone
ax.axvspan(x2_entry, x2_exit,  alpha=0.06, color="#ffb300", zorder=0)  # pyramid zone

# Labels
def label(x, y, text, color):
    ax.text(x, y, text, color=color, fontsize=8, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="#0d1117", alpha=0.7, ec="none"), zorder=6)

label(x1_entry + 0.01, t1.entry_price, f" Entry T1  {t1.entry_price:.5f}", "#4fc3f7")
label(x1_entry + 0.01, t1.stop_price,  f" Stop T1   {t1.stop_price:.5f}",  "#ef5350")
label(x1_entry + 0.01, t1.tp1_price,   f" TP T1     {t1.tp1_price:.5f}",   "#26a69a")
label(x2_entry + 0.01, t2.entry_price, f" Entry T2  {t2.entry_price:.5f}", "#ffb300")
label(x2_entry + 0.01, t2.stop_price,  f" Stop T2   {t2.stop_price:.5f}",  "#ff7043")
label(x2_entry + 0.01, t2.tp1_price,   f" TP T2     {t2.tp1_price:.5f}",   "#66bb6a")

# Arrow annotation for "structure stop raised"
ax.annotate(
    "Stop T1 raised\nto structure",
    xy=(x2_entry, t1.stop_price),
    xytext=(x2_entry - 0.15, t1.stop_price - 0.0030),
    color="#ff7043", fontsize=8,
    arrowprops=dict(arrowstyle="->", color="#ff7043", lw=1.0),
    bbox=dict(boxstyle="round,pad=0.3", fc="#0d1117", alpha=0.8, ec="none"),
    zorder=7,
)

# Axes formatting
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=16))
ax.tick_params(colors="#aaaaaa", labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor("#333333")
ax.grid(axis="y", color="#1e2a38", linewidth=0.5)
ax.set_ylabel("Price", color="#aaaaaa", fontsize=9)
ax.set_title(
    f"EURUSD 4H — Pyramid Example (LONG)\n"
    f"T1: +{t1.net_r:.2f}R  |  T2 add: +{t2.net_r:.2f}R  |  Combined: +{t1.net_r + t2.net_r:.2f}R",
    color="#e0e0e0", fontsize=11, pad=10
)

legend_items = [
    mpatches.Patch(color="#4fc3f7", label=f"T1 entry  {t1.entry_price:.5f}"),
    mpatches.Patch(color="#ef5350", label=f"T1 stop   {t1.stop_price:.5f}"),
    mpatches.Patch(color="#26a69a", label=f"T1 TP     {t1.tp1_price:.5f}  → +{t1.net_r:.2f}R"),
    mpatches.Patch(color="#ffb300", label=f"T2 entry  {t2.entry_price:.5f}"),
    mpatches.Patch(color="#ff7043", label=f"T2 stop   {t2.stop_price:.5f}"),
    mpatches.Patch(color="#66bb6a", label=f"T2 TP     {t2.tp1_price:.5f}  → +{t2.net_r:.2f}R"),
]
ax.legend(handles=legend_items, loc="upper left", fontsize=8,
          facecolor="#151b23", edgecolor="#333333", labelcolor="#cccccc")

plt.tight_layout()
import os
path = os.path.abspath("results/pyramid_example.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
print(f"Saved: {path}")
os.startfile(path)
