"""
chart.py — Plot backtest trades on an OHLC candlestick chart.
Reads trade_log.csv and fetches live price data from MT5.

Usage:
    python chart.py                  # all trades, auto date range
    python chart.py --last 30        # last 30 trades only
"""
import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from data_pipeline import get_data

RESULTS_DIR  = Path("results")
PADDING_BARS = 10   # extra bars before first entry and after last exit


def latest_trade_csv() -> Path:
    if RESULTS_DIR.exists():
        runs = sorted(RESULTS_DIR.iterdir(), reverse=True)
        for run in runs:
            p = run / "trade_log.csv"
            if p.exists():
                return p
    fallback = Path("trade_log.csv")
    return fallback


def load_trades(path: Path, last_n: int = None) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"No trade_log.csv found. Run backtest_engine.py first.")
    df = pd.read_csv(path, parse_dates=["entry_date", "exit_date"])
    df["entry_date"] = pd.to_datetime(df["entry_date"], utc=True)
    df["exit_date"]  = pd.to_datetime(df["exit_date"],  utc=True)
    if last_n:
        df = df.tail(last_n).reset_index(drop=True)
    return df


def draw_candles(ax, ohlc: pd.DataFrame) -> None:
    for i, (ts, row) in enumerate(ohlc.iterrows()):
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
        color = "#26a69a" if c >= o else "#ef5350"
        body_bot = min(o, c)
        body_h   = abs(c - o) or 0.00001
        ax.add_patch(mpatches.Rectangle(
            (i - 0.35, body_bot), 0.7, body_h,
            facecolor=color, edgecolor=color, linewidth=0, zorder=3,
        ))
        ax.plot([i, i], [l, h], color=color, linewidth=0.8, zorder=2)


def plot_trades(ohlc: pd.DataFrame, trades: pd.DataFrame, title: str = "") -> None:
    ts_to_i = {ts: i for i, ts in enumerate(ohlc.index)}

    fig, ax = plt.subplots(figsize=(20, 9))
    draw_candles(ax, ohlc)

    wins  = 0
    losses = 0

    for _, t in trades.iterrows():
        ei = ts_to_i.get(t["entry_date"])
        xi = ts_to_i.get(t["exit_date"])
        if ei is None and xi is None:
            continue

        is_long = t["direction"] == "long"
        won     = (t.get("net_r") or 0) > 0
        entry_color = "#1565c0" if is_long else "#b71c1c"
        exit_color  = "#00c853" if won else "#d50000"

        # Entry arrow
        if ei is not None:
            ep = t["entry_price"]
            offset = (ohlc["High"].max() - ohlc["Low"].min()) * 0.015
            if is_long:
                ax.annotate("", xy=(ei, ep), xytext=(ei, ep - offset * 2),
                            arrowprops=dict(arrowstyle="-|>", color=entry_color,
                                           lw=1.8, mutation_scale=14), zorder=6)
            else:
                ax.annotate("", xy=(ei, ep), xytext=(ei, ep + offset * 2),
                            arrowprops=dict(arrowstyle="-|>", color=entry_color,
                                           lw=1.8, mutation_scale=14), zorder=6)

            # SL and TP lines from entry to exit (or to chart edge)
            x_end = xi if xi is not None else len(ohlc) - 1
            ax.plot([ei, x_end], [t["stop_price"], t["stop_price"]],
                    color="#ff6f00", lw=0.6, ls="--", alpha=0.6, zorder=2)
            ax.plot([ei, x_end], [t["tp1_price"], t["tp1_price"]],
                    color="#00897b", lw=0.6, ls="--", alpha=0.6, zorder=2)

        # Exit marker
        if xi is not None:
            ax.plot(xi, t["exit_price"], marker="x", color=exit_color,
                    markersize=10, markeredgewidth=2, zorder=7)

        if won:
            wins += 1
        else:
            losses += 1

    # X-axis: date labels every N bars
    n = len(ohlc)
    step = max(1, n // 15)
    ticks = list(range(0, n, step))
    labels = [ohlc.index[i].strftime("%Y-%m-%d") for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, fontsize=7)
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))

    ax.set_xlim(-1, n)
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.15)
    ax.set_title(title or f"EURUSD H4  |  {wins}W / {losses}L  |  {wins+losses} trades shown")

    # Legend
    legend_items = [
        mpatches.Patch(color="#1565c0", label="Long entry"),
        mpatches.Patch(color="#b71c1c", label="Short entry"),
        mpatches.Patch(color="#00c853", label="Win exit"),
        mpatches.Patch(color="#d50000", label="Loss exit"),
        mpatches.Patch(color="#ff6f00", label="Stop loss"),
        mpatches.Patch(color="#00897b", label="Take profit"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=8)

    plt.tight_layout()
    out = Path("chart_trades.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=int, default=None,
                        help="Show only the last N trades")
    args = parser.parse_args()

    csv_path = latest_trade_csv()
    print(f"Using: {csv_path}")
    trades = load_trades(csv_path, last_n=args.last)
    if trades.empty:
        sys.exit("No trades in trade_log.csv.")

    print(f"Loaded {len(trades)} trades  ({trades['entry_date'].min().date()} to {trades['exit_date'].max().date()})")

    print("Fetching OHLC from MT5...")
    ohlc = get_data()

    # Trim OHLC to trade date range + padding
    first_bar = trades["entry_date"].min()
    last_bar  = trades["exit_date"].max()

    idx = ohlc.index
    start_pos = max(0, idx.get_loc(idx[idx <= first_bar][-1]) - PADDING_BARS)
    end_pos   = min(len(ohlc) - 1, idx.get_loc(idx[idx >= last_bar][0]) + PADDING_BARS)
    ohlc = ohlc.iloc[start_pos:end_pos + 1]

    print(f"Plotting {len(ohlc)} bars...")
    plot_trades(ohlc, trades)


if __name__ == "__main__":
    main()
