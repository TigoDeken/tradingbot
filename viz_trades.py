"""
Visualise 3 selected trades showing how W/M levels are traded.
Draws candlesticks + level grid + entry/stop/TP lines for each trade.
"""

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.transforms import blended_transform_factory
from matplotlib.lines import Line2D

from data_pipeline import get_data
from trade_engine import _wm_levels

RESULTS_DIR = "results/2026-05-21_161036"
TRADE_IDS   = [3, 5, 2]   # long-win, short-win, short-loss


def load_trades():
    df = pd.read_csv(f"{RESULTS_DIR}/trade_log.csv")
    for col in ["entry_date", "exit_date"]:
        df[col] = pd.to_datetime(df[col], utc=True)
    return df[df["trade_id"].isin(TRADE_IDS)].set_index("trade_id").loc[TRADE_IDS]


def draw_candles(ax, df_window):
    if len(df_window) < 2:
        return
    bar_w = (mdates.date2num(df_window.index[1].to_pydatetime()) -
             mdates.date2num(df_window.index[0].to_pydatetime())) * 0.75
    hw = bar_w / 2
    for ts, row in df_window.iterrows():
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
        color    = "#26a69a" if c >= o else "#ef5350"
        body_bot = min(o, c)
        body_h   = max(abs(c - o), 1e-5)
        x = mdates.date2num(ts.to_pydatetime())
        ax.add_patch(mpatches.Rectangle(
            (x - hw, body_bot), bar_w, body_h,
            facecolor=color, edgecolor=color, linewidth=0, zorder=3
        ))
        ax.plot([x, x], [l, body_bot],   color=color, linewidth=0.8, zorder=2)
        ax.plot([x, x], [max(o, c), h],  color=color, linewidth=0.8, zorder=2)


def get_levels_at(weekly_df, monthly_df, bar_time):
    result = {}
    pw = weekly_df[weekly_df.index < bar_time]
    pm = monthly_df[monthly_df.index < bar_time]
    if len(pw) >= 1:
        result["prev_week_H"] = pw.iloc[-1]["High"]
        result["prev_week_L"] = pw.iloc[-1]["Low"]
        result["prev_week_C"] = pw.iloc[-1]["Close"]
    if len(pm) >= 1:
        result["prev_month_H"] = pm.iloc[-1]["High"]
        result["prev_month_L"] = pm.iloc[-1]["Low"]
        result["prev_month_C"] = pm.iloc[-1]["Close"]
    return result


def plot_trade(ax, df, trade, weekly_df, monthly_df):
    entry_ts = trade["entry_date"]
    exit_ts  = trade["exit_date"]

    win_start = entry_ts - pd.Timedelta(weeks=5)
    win_end   = exit_ts  + pd.Timedelta(weeks=1)
    df_win    = df[(df.index >= win_start) & (df.index <= win_end)]
    if df_win.empty:
        return

    draw_candles(ax, df_win)

    lvls = get_levels_at(weekly_df, monthly_df, entry_ts)

    # Transform: x in axes fraction (0-1), y in data coords
    btrans_right = blended_transform_factory(ax.transAxes, ax.transData)
    btrans_left  = blended_transform_factory(ax.transAxes, ax.transData)

    level_styles = {
        "prev_week_H":  ("#4fc3f7", "--", "Prev Wk High"),
        "prev_week_L":  ("#4fc3f7", ":",  "Prev Wk Low"),
        "prev_week_C":  ("#4fc3f7", "-.", "Prev Wk Close"),
        "prev_month_H": ("#ce93d8", "--", "Prev Mo High"),
        "prev_month_L": ("#ce93d8", ":",  "Prev Mo Low"),
        "prev_month_C": ("#ce93d8", "-.", "Prev Mo Close"),
    }
    for key, (col, ls, lbl) in level_styles.items():
        if key in lvls:
            ax.axhline(lvls[key], color=col, linestyle=ls, linewidth=1.0, alpha=0.7, zorder=1)
            ax.text(1.005, lvls[key], lbl, color=col, fontsize=6, va="center",
                    transform=btrans_right, clip_on=False)

    entry_p = trade["entry_price"]
    stop_p  = trade["stop_price"]
    tp_p    = trade["tp1_price"]
    exit_p  = trade["exit_price"]
    reason  = trade["exit_reason"]
    gross_r = trade["gross_r"]
    is_long = trade["direction"] == "long"

    ax.axhline(entry_p, color="#FF8F00", linestyle="-",  linewidth=1.3, zorder=4, alpha=0.9)
    ax.axhline(stop_p,  color="#ef5350", linestyle="--", linewidth=1.0, zorder=4, alpha=0.9)
    ax.axhline(tp_p,    color="#66bb6a", linestyle="--", linewidth=1.0, zorder=4, alpha=0.9)

    ax.text(-0.005, entry_p, "Entry", color="#FF8F00", fontsize=6.5, va="center",
            ha="right", transform=btrans_left, clip_on=False)
    ax.text(-0.005, stop_p,  "Stop",  color="#ef5350", fontsize=6.5, va="center",
            ha="right", transform=btrans_left, clip_on=False)
    ax.text(-0.005, tp_p,    "TP",    color="#66bb6a", fontsize=6.5, va="center",
            ha="right", transform=btrans_left, clip_on=False)

    # Entry arrow
    entry_x  = mdates.date2num(entry_ts.to_pydatetime())
    risk     = abs(entry_p - stop_p)
    arrow_col = "#26a69a" if is_long else "#ef5350"
    dy_offset = risk * 0.8 * (1 if is_long else -1)
    ax.annotate(
        "", xy=(entry_x, entry_p),
        xytext=(entry_x, entry_p - dy_offset),
        arrowprops=dict(arrowstyle="-|>", color=arrow_col, lw=2.2),
        zorder=7
    )

    # Exit marker
    if exit_ts is not None and exit_p is not None:
        exit_x = mdates.date2num(exit_ts.to_pydatetime())
        win    = gross_r > 0
        ax.plot(exit_x, exit_p, marker="X",
                color="#66bb6a" if win else "#ef5350",
                markersize=11, markeredgewidth=2.0, zorder=8)

    # Shaded trade zone
    ex_x      = mdates.date2num(exit_ts.to_pydatetime()) if exit_ts is not None else win_end
    shade_col = "#26a69a18" if is_long else "#ef535018"
    ax.axvspan(entry_x, ex_x, color=shade_col, zorder=0)

    # Axis limits
    x0 = mdates.date2num(df_win.index[0].to_pydatetime())
    x1 = mdates.date2num(df_win.index[-1].to_pydatetime())
    ax.set_xlim(x0, x1)
    p_range = df_win["High"].max() - df_win["Low"].min()
    ax.set_ylim(df_win["Low"].min()  - p_range * 0.10,
                df_win["High"].max() + p_range * 0.10)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=7, color="#aaa")
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.5f"))
    ax.tick_params(axis="y", labelsize=7, labelcolor="#aaa")
    ax.tick_params(axis="x", colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.grid(True, alpha=0.12, color="#888", zorder=0)
    ax.set_facecolor("#131722")

    dir_str = "LONG" if is_long else "SHORT"
    win_str = f"+{gross_r}R  WIN" if gross_r > 0 else f"{gross_r}R  LOSS"
    res_col = "#66bb6a" if gross_r > 0 else "#ef5350"
    ax.set_title(f"Trade {trade.name}  |  {dir_str}  |  Exit: {reason}  |  ",
                 fontsize=9, color="#ddd", loc="left", pad=6)
    ax.set_title(f"  {win_str}", fontsize=9, color=res_col, loc="right", pad=6)


def main():
    print("Fetching M5 EURUSD data from MT5...")
    df = get_data()
    print(f"Got {len(df)} bars")

    _agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    weekly_df  = df.resample("W").agg(_agg).dropna()
    monthly_df = df.resample("ME").agg(_agg).dropna()

    trades = load_trades()
    print(f"Loaded {len(trades)} trades: IDs {list(trades.index)}")

    fig, axes = plt.subplots(3, 1, figsize=(16, 16))
    fig.patch.set_facecolor("#0d1117")

    for ax, (tid, trade) in zip(axes, trades.iterrows()):
        plot_trade(ax, df, trade, weekly_df, monthly_df)

    legend_items = [
        Line2D([0],[0], color="#FF8F00", lw=1.5,              label="Entry"),
        Line2D([0],[0], color="#ef5350", lw=1.0, ls="--",    label="Stop"),
        Line2D([0],[0], color="#66bb6a", lw=1.0, ls="--",    label="TP"),
        Line2D([0],[0], color="#4fc3f7", lw=1.0, ls="--",    label="Prev Wk H/L"),
        Line2D([0],[0], color="#ce93d8", lw=1.0, ls="--",    label="Prev Mo H/L"),
        Line2D([0],[0], marker="X", color="#66bb6a", ms=8, ls="None", label="Exit win"),
        Line2D([0],[0], marker="X", color="#ef5350", ms=8,  ls="None", label="Exit loss"),
    ]
    fig.legend(handles=legend_items, loc="upper center", ncol=7,
               facecolor="#1e222d", edgecolor="#444", labelcolor="white",
               fontsize=8, bbox_to_anchor=(0.5, 1.01))

    fig.suptitle("EURUSD M5  |  W/M Level Strategy  |  3 Trade Examples",
                 fontsize=12, color="white", y=1.04)
    plt.tight_layout(pad=2.0)

    out = "viz_3_trades.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
    # plt.show()


if __name__ == "__main__":
    main()
