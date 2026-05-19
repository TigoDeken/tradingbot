"""
backtest_engine.py — Step 5
Metrics, equity curve, walk-forward validation, parameter optimisation.
"""
import itertools
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data_pipeline import get_data
from constants import PIP, RISK_PCT
from trade_engine  import (
    Trade, run_trades,
    PULLBACK_LOOKBACK, STOP_BUFFER, TP_MODE,
    SLIPPAGE_PIPS, PIP_VALUE, ACCOUNT_BALANCE,
)

INITIAL_BALANCE = 10_000.0
SPLIT_RATIO     = 0.70

PARAM_GRID = {
    "PULLBACK_LOOKBACK":  [3, 4, 5],
    "STOP_BUFFER":        [3, 5, 8, 10],
    "TP_MODE":            ["full", "partial", "trail"],
    "CLOSE_STRENGTH":     [0.0, 0.5, 0.6, 0.7],
}

SEP = "-" * 47


# ── Equity ────────────────────────────────────────────────────────────────────

def build_equity(trades: list, initial: float = INITIAL_BALANCE,
                 risk_pct: float = RISK_PCT) -> pd.DataFrame:
    """Compound risk_pct per trade; return balance + drawdown series."""
    rows, bal, peak = [], initial, initial
    for t in trades:
        risk = bal * risk_pct
        pnl  = (t.net_r or 0.0) * risk
        bal  += pnl
        peak  = max(peak, bal)
        rows.append({
            "trade_id":     t.trade_id,
            "entry_date":   t.entry_date,
            "exit_date":    t.exit_date,
            "exit_reason":  t.exit_reason,
            "direction":    t.direction,
            "net_r":        t.net_r,
            "pnl_usd":      round(pnl, 2),
            "balance":      round(bal, 2),
            "drawdown_pct": round((bal - peak) / peak * 100, 3),
        })
    return pd.DataFrame(rows)


def _dd_duration_days(eq: pd.DataFrame) -> int:
    if eq.empty:
        return 0
    bals  = eq["balance"].values
    dates = pd.to_datetime(eq["exit_date"].values)
    peak  = bals[0]
    dd_start = dates[0]
    in_dd, max_days = False, 0
    for i in range(len(bals)):
        if bals[i] >= peak:
            if in_dd:
                max_days = max(max_days, (dates[i] - dd_start).days)
                in_dd = False
            peak     = bals[i]
            dd_start = dates[i]
        elif not in_dd:
            in_dd = True
    if in_dd:
        max_days = max(max_days, (dates[-1] - dd_start).days)
    return int(max_days)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(trades: list, initial: float = INITIAL_BALANCE) -> dict:
    if not trades:
        return {"total_trades": 0, "expectancy_net_r": 0.0, "_empty": True}

    wins   = [t for t in trades if (t.net_pips or 0) > 0]
    losses = [t for t in trades if (t.net_pips or 0) <= 0]
    n      = len(trades)

    net_r_total   = sum(t.net_r   or 0 for t in trades)
    gross_r_total = sum(t.gross_r or 0 for t in trades)

    # Consecutive streaks
    max_w = max_l = cw = cl = 0
    for t in trades:
        if (t.net_pips or 0) > 0:
            cw += 1; cl = 0; max_w = max(max_w, cw)
        else:
            cl += 1; cw = 0; max_l = max(max_l, cl)

    eq    = build_equity(trades, initial)
    max_dd_pct  = round(eq["drawdown_pct"].min(), 2) if not eq.empty else 0.0
    max_dd_days = _dd_duration_days(eq)

    span_days = (trades[-1].exit_date - trades[0].entry_date).days if n >= 2 else 0
    tpm = round(n / (span_days / 30), 1) if span_days > 0 else 0.0

    # EV decomposition: EV = WR × avg_win_R − LR × |avg_loss_R|
    wr_f = len(wins) / n
    lr_f = len(losses) / n
    _avg_net_win    = float(np.mean([t.net_r   or 0 for t in wins]))   if wins   else 0.0
    _avg_net_loss   = float(np.mean([t.net_r   or 0 for t in losses])) if losses else 0.0
    _avg_gross_win  = float(np.mean([t.gross_r or 0 for t in wins]))   if wins   else 0.0
    _avg_gross_loss = float(np.mean([t.gross_r or 0 for t in losses])) if losses else 0.0
    gross_ev = round(wr_f * _avg_gross_win - lr_f * abs(_avg_gross_loss), 4)
    net_ev   = round(wr_f * _avg_net_win   - lr_f * abs(_avg_net_loss),   4)

    return {
        # Summary
        "total_trades":       n,
        "winning_trades":     len(wins),
        "losing_trades":      len(losses),
        "win_rate_pct":       round(len(wins) / n * 100, 1),
        # Returns
        "gross_pips":         round(sum(t.gross_pips or 0 for t in trades), 1),
        "net_pips":           round(sum(t.net_pips   or 0 for t in trades), 1),
        "gross_r":            round(gross_r_total, 2),
        "net_r":              round(net_r_total,   2),
        "avg_win_pips":       round(np.mean([t.net_pips or 0 for t in wins]),   1) if wins   else 0,
        "avg_win_r":          round(np.mean([t.net_r    or 0 for t in wins]),   2) if wins   else 0,
        "avg_loss_pips":      round(np.mean([t.net_pips or 0 for t in losses]), 1) if losses else 0,
        "avg_loss_r":         round(np.mean([t.net_r    or 0 for t in losses]), 2) if losses else 0,
        "best_pips":          round(max((t.net_pips or 0 for t in wins),   default=0), 1),
        "best_r":             round(max((t.net_r    or 0 for t in wins),   default=0), 2),
        "worst_pips":         round(min((t.net_pips or 0 for t in losses), default=0), 1),
        "worst_r":            round(min((t.net_r    or 0 for t in losses), default=0), 2),
        "expectancy_net_r":   round(net_r_total / n, 4),
        "gross_ev":           gross_ev,
        "net_ev":             net_ev,
        # Risk
        "max_consec_wins":    max_w,
        "max_consec_losses":  max_l,
        "max_drawdown_pct":   max_dd_pct,
        "max_drawdown_days":  max_dd_days,
        "avg_duration_hours": round(np.mean([t.duration_hours or 0 for t in trades]), 1),
        "trades_per_month":   tpm,
        # Slippage
        "slippage_total_pips": n * SLIPPAGE_PIPS,
        "gross_expectancy_r":  round(gross_r_total / n, 4),
    }


def _quick_metrics(trades: list) -> dict:
    """Lightweight subset used inside the optimisation loop."""
    if not trades:
        return {"total_trades": 0, "win_rate_pct": 0, "expectancy_net_r": 0.0,
                "max_drawdown_pct": 0.0, "gross_ev": 0.0, "net_ev": 0.0}
    n     = len(trades)
    wins  = [t for t in trades if (t.net_pips or 0) > 0]
    losses= [t for t in trades if (t.net_pips or 0) <= 0]
    net_r = sum(t.net_r or 0 for t in trades)
    wr_f  = len(wins)   / n
    lr_f  = len(losses) / n
    avg_gw = float(np.mean([t.gross_r or 0 for t in wins]))   if wins   else 0.0
    avg_gl = float(np.mean([t.gross_r or 0 for t in losses])) if losses else 0.0
    avg_nw = float(np.mean([t.net_r   or 0 for t in wins]))   if wins   else 0.0
    avg_nl = float(np.mean([t.net_r   or 0 for t in losses])) if losses else 0.0
    eq = build_equity(trades, risk_pct=RISK_PCT)
    return {
        "total_trades":     n,
        "win_rate_pct":     round(wr_f * 100, 1),
        "expectancy_net_r": round(net_r / n, 4),
        "max_drawdown_pct": round(eq["drawdown_pct"].min(), 2) if not eq.empty else 0.0,
        "gross_ev":         round(wr_f * avg_gw - lr_f * abs(avg_gl), 4),
        "net_ev":           round(wr_f * avg_nw - lr_f * abs(avg_nl), 4),
    }


# ── Display ───────────────────────────────────────────────────────────────────

def print_metrics(m: dict, label: str = "") -> None:
    if label:
        print(f"\n{'=' * 47}")
        print(f"  {label}")
        print(f"{'=' * 47}")
    if m.get("_empty"):
        print("  (no trades)")
        return

    print(f"\nPERFORMANCE SUMMARY")
    print(SEP)
    print(f"  Total trades             {m['total_trades']}")
    print(f"  Winning trades           {m['winning_trades']}")
    print(f"  Losing trades            {m['losing_trades']}")
    print(f"  Win rate                 {m['win_rate_pct']}%")

    print(f"\nRETURNS")
    print(SEP)
    print(f"  Total gross pips         {m['gross_pips']}")
    print(f"  Total net pips           {m['net_pips']}")
    print(f"  Total gross R            {m['gross_r']}")
    print(f"  Total net R              {m['net_r']}")
    print(f"  Avg win                  {m['avg_win_pips']} pips  /  {m['avg_win_r']} R")
    print(f"  Avg loss                 {m['avg_loss_pips']} pips  /  {m['avg_loss_r']} R")
    print(f"  Largest winner           {m['best_pips']} pips  /  {m['best_r']} R")
    print(f"  Largest loser            {m['worst_pips']} pips  /  {m['worst_r']} R")
    print(f"  Expectancy (net R)       {m['expectancy_net_r']}")
    print(f"  Gross EV per trade       {m.get('gross_ev', 'n/a')} R")
    print(f"  Net EV per trade         {m.get('net_ev',   'n/a')} R")

    print(f"\nRISK")
    print(SEP)
    print(f"  Max consecutive losses   {m['max_consec_losses']}")
    print(f"  Max consecutive wins     {m['max_consec_wins']}")
    print(f"  Max drawdown             {m['max_drawdown_pct']}%")
    print(f"  Max drawdown duration    {m['max_drawdown_days']} days")
    print(f"  Avg trade duration       {m['avg_duration_hours']} hours")
    print(f"  Trades per month         {m['trades_per_month']}")

    print(f"\nSLIPPAGE IMPACT")
    print(SEP)
    print(f"  Total pips lost          {m['slippage_total_pips']}")
    print(f"  Gross expectancy         {m['gross_expectancy_r']} R")
    print(f"  Net expectancy           {m['expectancy_net_r']} R")
    print(f"  Slippage drag            {round(m['gross_expectancy_r'] - m['expectancy_net_r'], 4)} R/trade")


def compare_is_oos(is_m: dict, oos_m: dict) -> None:
    print(f"\n{'=' * 65}")
    print(f"  IN-SAMPLE vs OUT-OF-SAMPLE")
    print(f"{'=' * 65}")
    rows = [
        ("total_trades",     "Total trades"),
        ("win_rate_pct",     "Win rate %"),
        ("expectancy_net_r", "Expectancy (net R)"),
        ("gross_ev",         "Gross EV (R/trade)"),
        ("net_ev",           "Net EV (R/trade)"),
        ("net_r",            "Total net R"),
        ("max_drawdown_pct", "Max drawdown %"),
        ("trades_per_month", "Trades / month"),
    ]
    print(f"  {'Metric':<28} {'In-Sample':>12} {'Out-of-Sample':>14}")
    print(f"  {'-' * 56}")
    for key, label in rows:
        print(f"  {label:<28} {str(is_m.get(key,'n/a')):>12} {str(oos_m.get(key,'n/a')):>14}")

    flags = []
    for key, name in [("win_rate_pct", "Win rate"), ("expectancy_net_r", "Expectancy")]:
        iv, ov = is_m.get(key, 0) or 0, oos_m.get(key, 0) or 0
        if iv > 0 and ov < iv * 0.80:
            flags.append(f"  !! {name}: IS={iv}  OOS={ov}  (>{20}% drop)")
    if flags:
        print(f"\n  *** OVERFITTING WARNING ***")
        for f in flags:
            print(f)
    else:
        print(f"\n  No significant IS->OOS degradation detected.")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_equity_curve(eq: pd.DataFrame, title: str = "Equity Curve",
                      path: str = "equity_curve.png", split_trade: int = None) -> None:
    if eq.empty:
        print("No equity data to plot.")
        return
    x  = np.arange(len(eq))
    b  = eq["balance"].values
    dd = eq["drawdown_pct"].values
    mi = dd.argmin()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(x, b, color="royalblue", lw=1.2, label="Equity")
    ax1.axhline(INITIAL_BALANCE, color="grey", lw=0.7, ls="--", label=f"Start ${INITIAL_BALANCE:,.0f}")
    ax1.scatter([mi], [b[mi]], color="red", s=100, zorder=5,
                label=f"Max DD @ trade {mi+1}: {dd[mi]:.1f}%")
    if split_trade is not None:
        ax1.axvline(split_trade, color="orange", lw=1.2, ls="--", label="IS / OOS split")
    ax1.set_ylabel("Balance (USD)"); ax1.set_title(title)
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.25)

    ax2.fill_between(x, dd, 0, color="red", alpha=0.35)
    ax2.axhline(0, color="grey", lw=0.7)
    ax2.set_xlabel("Trade number"); ax2.set_ylabel("Drawdown %")
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Equity curve saved: {path}")
    plt.show()


def plot_recent_trades(df: pd.DataFrame, trades: list, path: str = "recent_trades.png",
                       days: int = 30) -> None:
    """Candlestick chart of the last `days` days with trade overlays."""
    cutoff = df.index[-1] - pd.Timedelta(days=days)
    df_plot = df[df.index >= cutoff].copy()
    visible = [t for t in trades if t.entry_date >= cutoff]

    if df_plot.empty:
        print("No data in the last month — skipping recent trades chart.")
        return

    fig, ax = plt.subplots(figsize=(22, 9), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    # Draw OHLC candles
    for ts, row in df_plot.iterrows():
        bull = row["Close"] >= row["Open"]
        c_body = "#26a69a" if bull else "#ef5350"
        c_wick = "#26a69a" if bull else "#ef5350"
        ax.plot([ts, ts], [row["Low"], row["High"]], color=c_wick, lw=0.8, alpha=0.7, zorder=1)
        ax.add_patch(plt.Rectangle(
            (mdates.date2num(ts.to_pydatetime()) - 0.055, min(row["Open"], row["Close"])),
            0.11, max(abs(row["Close"] - row["Open"]), 0.00001),
            color=c_body, alpha=0.9, zorder=2,
            transform=ax.transData,
        ))

    # Draw trades
    legend_entries = {}
    for t in visible:
        long = t.direction == "long"
        c_dir  = "#4fa8e8" if long else "#f06292"   # blue / pink

        # Recover original planned stop from tp1 (immune to BE mutations)
        if long:
            orig_stop = t.entry_price - (t.tp1_price - t.entry_price) / 2.0
        else:
            orig_stop = t.entry_price + (t.entry_price - t.tp1_price) / 2.0

        exit_dt = t.exit_date if t.exit_date is not None else df_plot.index[-1]
        win = (t.net_pips or 0) > 0

        # Entry horizontal line
        h = ax.hlines(t.entry_price, t.entry_date, exit_dt,
                      colors=c_dir, lw=1.5, zorder=4, label="Entry")
        legend_entries.setdefault("Entry", h)

        # Entry arrow marker
        dy = -(t.tp1_price - orig_stop) * 0.3 if long else (t.tp1_price - orig_stop) * 0.3
        ax.annotate("", xy=(t.entry_date, t.entry_price),
                    xytext=(t.entry_date, t.entry_price + dy),
                    arrowprops=dict(arrowstyle="->", color=c_dir, lw=1.8), zorder=5)

        # Stop line (original planned stop, red dashed)
        h2 = ax.hlines(orig_stop, t.entry_date, exit_dt,
                       colors="#ff5252", linestyles="--", lw=1.2, alpha=0.85, zorder=3, label="Stop")
        legend_entries.setdefault("Stop", h2)

        # TP line (green dashed)
        h3 = ax.hlines(t.tp1_price, t.entry_date, exit_dt,
                       colors="#69f0ae", linestyles="--", lw=1.2, alpha=0.85, zorder=3, label="TP")
        legend_entries.setdefault("TP", h3)

        # Exit horizontal line + marker
        if t.exit_date is not None:
            c_exit = "#69f0ae" if win else "#ff5252"
            h4 = ax.hlines(t.exit_price, t.entry_date, t.exit_date,
                           colors=c_exit, linestyles=":", lw=1.2, alpha=0.9, zorder=4, label="Exit")
            legend_entries.setdefault("Exit", h4)
            ax.plot(t.exit_date, t.exit_price,
                    marker="D", color=c_exit, ms=6, zorder=6,
                    markeredgecolor="#0e1117", markeredgewidth=0.5)

    # Axes styling
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.xticks(rotation=40, ha="right")
    ax.yaxis.label.set_color("#aaa")
    ax.xaxis.label.set_color("#aaa")
    ax.set_title(f"EURUSD 4H — Last {days} days  |  {len(visible)} trades shown",
                 color="white", fontsize=12, pad=10)
    ax.grid(True, alpha=0.12, color="#555")

    if legend_entries:
        ax.legend(legend_entries.values(), legend_entries.keys(),
                  facecolor="#1a1d24", edgecolor="#444", labelcolor="white",
                  fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Recent trades chart: {path}")
    plt.close(fig)


# ── Exports ───────────────────────────────────────────────────────────────────

def export_trades_csv(trades: list, path: str = "trade_log.csv") -> None:
    rows = [{
        "trade_id":       t.trade_id,
        "direction":      t.direction,
        "entry_date":     t.entry_date,
        "entry_price":    t.entry_price,
        "stop_price":     t.stop_price,
        "tp1_price":      t.tp1_price,
        "exit_date":      t.exit_date,
        "exit_price":     t.exit_price,
        "exit_reason":    t.exit_reason,
        "gross_pips":     t.gross_pips,
        "net_pips":       t.net_pips,
        "gross_r":        t.gross_r,
        "net_r":          t.net_r,
        "duration_hours": t.duration_hours,
        "lot_size":       t.lot_size,
        "regime":         t.regime,
        "trend_strength": t.trend_strength,
        **t.params,
    } for t in trades]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Trade log: {path}  ({len(rows)} rows)")


def export_equity_csv(eq: pd.DataFrame, path: str = "equity_curve.csv") -> None:
    eq.to_csv(path, index=False)
    print(f"Equity CSV: {path}")


def export_opt_csv(results: pd.DataFrame, path: str = "optimisation_results.csv") -> None:
    results.to_csv(path, index=False)
    print(f"Optimisation results: {path}  ({len(results)} rows)")


# ── Walk-forward helpers ──────────────────────────────────────────────────────

def _split_date(df: pd.DataFrame) -> pd.Timestamp:
    s, e = df.index[0], df.index[-1]
    return s + (e - s) * SPLIT_RATIO


def _split_trades(trades: list, sd: pd.Timestamp):
    return ([t for t in trades if t.entry_date <  sd],
            [t for t in trades if t.entry_date >= sd])


# ── Optimisation ─────────────────────────────────────────────────────────────

def optimize(df_raw: pd.DataFrame, split_dt: pd.Timestamp) -> pd.DataFrame:
    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    total  = len(combos)
    print(f"\n  Total combinations to test: {total}")
    results = []
    t0 = time.time()

    for idx, vals in enumerate(combos):
        c = dict(zip(keys, vals))
        trades = run_trades(
            df_raw,
            pullback_lookback=c["PULLBACK_LOOKBACK"],
            stop_buffer=c["STOP_BUFFER"],
            tp_mode=c["TP_MODE"],
            close_strength=c["CLOSE_STRENGTH"],
        )
        is_t, oos_t = _split_trades(trades, split_dt)
        is_m  = _quick_metrics(is_t)
        oos_m = _quick_metrics(oos_t)
        row = {**c,
               **{f"IS_{k}":  v for k, v in is_m.items()},
               **{f"OOS_{k}": v for k, v in oos_m.items()}}
        results.append(row)

        done = idx + 1
        if done % max(1, total // 20) == 0 or done == total:
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f"  {done/total*100:5.1f}%  ({done}/{total})  "
                  f"elapsed {elapsed/60:.1f}m  ETA {eta/60:.1f}m", flush=True)

    df_r = pd.DataFrame(results)
    df_r.sort_values("OOS_expectancy_net_r", ascending=False, inplace=True)
    df_r.reset_index(drop=True, inplace=True)
    return df_r


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_id  = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = Path("results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  EURUSD 4H ALGORITHMIC BACKTEST")
    print("=" * 55)
    print(f"  Results folder: {out_dir}")

    # 1. Data
    print("\n[1] Loading data...")
    df_raw = get_data()
    split_dt = _split_date(df_raw)
    print(f"    Bars:       {len(df_raw)}")
    print(f"    Range:      {df_raw.index[0].date()} to {df_raw.index[-1].date()}")
    print(f"    Split date: {split_dt.date()}  (IS=70%  OOS=30%)")

    # 2. Run strategy
    print("\n[2] Running trades...")
    all_trades = run_trades(df_raw, close_strength=0.6)
    is_t, oos_t = _split_trades(all_trades, split_dt)
    print(f"    Total: {len(all_trades)}  |  IS: {len(is_t)}  |  OOS: {len(oos_t)}")

    # 3. Metrics
    all_m = compute_metrics(all_trades)
    is_m  = compute_metrics(is_t)
    oos_m = compute_metrics(oos_t)

    print_metrics(all_m, "FULL DATASET")
    print_metrics(is_m,  "IN-SAMPLE  (first 70% by date)")
    print_metrics(oos_m, "OUT-OF-SAMPLE  (last 30% by date)")
    compare_is_oos(is_m, oos_m)

    # 4. Save outputs
    print("\n[4] Saving results...")
    eq_all = build_equity(all_trades)
    split_trade_n = len(is_t)

    if not eq_all.empty:
        print(f"    Start:        ${INITIAL_BALANCE:,.2f}")
        print(f"    End:          ${eq_all['balance'].iloc[-1]:,.2f}")
        print(f"    Max drawdown: {eq_all['drawdown_pct'].min():.2f}%")

    plot_equity_curve(eq_all,
                      title=f"EURUSD 4H — Equity Curve ({run_id})",
                      path=str(out_dir / "equity_curve.png"),
                      split_trade=split_trade_n)
    recent_path = str(out_dir / "recent_trades.png")
    plot_recent_trades(df_raw, all_trades, path=recent_path, days=30)
    import os; os.startfile(recent_path)
    export_trades_csv(all_trades, path=str(out_dir / "trade_log.csv"))
    export_equity_csv(eq_all,     path=str(out_dir / "equity_curve.csv"))

    # Save summary metrics as JSON
    summary = {"run_id": run_id, "all": all_m, "is": is_m, "oos": oos_m}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary JSON: {out_dir / 'summary.json'}")

    print(f"\n{'='*65}")
    print(f"  DONE  —  results saved to: {out_dir}")
    print(f"{'='*65}")
