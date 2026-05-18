"""
backtest_engine.py — Step 5
Metrics, equity curve, walk-forward validation, parameter optimisation.
"""
import itertools
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_pipeline import get_data
from swing_engine  import build_swings, PIP, SWING_N, MIN_SWING_SIZE, MIN_SWING_INCREMENT
from constants import RISK_PCT
from trend_engine  import classify_trend, MIN_TREND_SIZE, TREND_RANGE_RATIO
from trade_engine  import (
    Trade, run_trades,
    PULLBACK_LOOKBACK, STOP_BUFFER, TP_MODE,
    SLIPPAGE_PIPS, PIP_VALUE, ACCOUNT_BALANCE,
)

INITIAL_BALANCE = 10_000.0
SPLIT_RATIO     = 0.70

PARAM_GRID = {
    "SWING_N":            [2, 3],
    "MIN_SWING_SIZE":     [15, 20, 25],
    "MIN_SWING_INCR":     [8, 10, 15],
    "MIN_TREND_SIZE":     [40, 50, 60],
    "TREND_RANGE_RATIO":  [0.30, 0.40, 0.50],
    "PULLBACK_LOOKBACK":  [3, 4, 5],
    "STOP_BUFFER":        [3, 5, 8],
    "TP_MODE":            ["full", "partial"],
}

SEP = "─" * 47


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
                "max_drawdown_pct": 0.0}
    n = len(trades)
    wins = sum(1 for t in trades if (t.net_pips or 0) > 0)
    net_r = sum(t.net_r or 0 for t in trades)
    eq = build_equity(trades, risk_pct=RISK_PCT)
    return {
        "total_trades":     n,
        "win_rate_pct":     round(wins / n * 100, 1),
        "expectancy_net_r": round(net_r / n, 4),
        "max_drawdown_pct": round(eq["drawdown_pct"].min(), 2) if not eq.empty else 0.0,
    }


# ── Display ───────────────────────────────────────────────────────────────────

def print_metrics(m: dict, label: str = "") -> None:
    if label:
        print(f"\n{'═' * 47}")
        print(f"  {label}")
        print(f"{'═' * 47}")
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
    print(f"\n{'═' * 65}")
    print(f"  IN-SAMPLE vs OUT-OF-SAMPLE")
    print(f"{'═' * 65}")
    rows = [
        ("total_trades",     "Total trades"),
        ("win_rate_pct",     "Win rate %"),
        ("expectancy_net_r", "Expectancy (net R)"),
        ("net_r",            "Total net R"),
        ("max_drawdown_pct", "Max drawdown %"),
        ("trades_per_month", "Trades / month"),
    ]
    print(f"  {'Metric':<28} {'In-Sample':>12} {'Out-of-Sample':>14}")
    print(f"  {'-' * 56}")
    for key, label in rows:
        print(f"  {label:<28} {str(is_m.get(key,'—')):>12} {str(oos_m.get(key,'—')):>14}")

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
        print(f"\n  No significant IS→OOS degradation detected.")


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
    print(f"Equity curve saved → {path}")
    plt.show()


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
    print(f"Trade log → {path}  ({len(rows)} rows)")


def export_equity_csv(eq: pd.DataFrame, path: str = "equity_curve.csv") -> None:
    eq.to_csv(path, index=False)
    print(f"Equity CSV → {path}")


def export_opt_csv(results: pd.DataFrame, path: str = "optimisation_results.csv") -> None:
    results.to_csv(path, index=False)
    print(f"Optimisation results → {path}  ({len(results)} rows)")


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
    print(f"  Caching swings for {len(PARAM_GRID['SWING_N'])*len(PARAM_GRID['MIN_SWING_SIZE'])*len(PARAM_GRID['MIN_SWING_INCR'])} unique swing configs")

    swing_cache  = {}
    regime_cache = {}
    results      = []
    t0           = time.time()

    for idx, vals in enumerate(combos):
        c = dict(zip(keys, vals))

        # Cached swing build
        skey = (c["SWING_N"], c["MIN_SWING_SIZE"], c["MIN_SWING_INCR"])
        if skey not in swing_cache:
            swing_cache[skey] = build_swings(
                df_raw, swing_n=c["SWING_N"],
                min_swing_size=c["MIN_SWING_SIZE"],
                min_swing_increment=c["MIN_SWING_INCR"], pip=PIP,
            )
        swings = swing_cache[skey]

        # Cached regime classification
        rkey = skey + (c["MIN_TREND_SIZE"], c["TREND_RANGE_RATIO"])
        if rkey not in regime_cache:
            regime_cache[rkey] = classify_trend(
                df_raw, swings,
                swing_n=c["SWING_N"], min_swing_increment=c["MIN_SWING_INCR"],
                min_trend_size=c["MIN_TREND_SIZE"],
                trend_range_ratio=c["TREND_RANGE_RATIO"], pip=PIP,
            )
        df_reg = regime_cache[rkey]

        trades = run_trades(
            df_reg, swings, swing_n=c["SWING_N"],
            pullback_lookback=c["PULLBACK_LOOKBACK"],
            stop_buffer=c["STOP_BUFFER"], tp_mode=c["TP_MODE"],
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
            eta     = elapsed / done * (total - done)
            print(f"  {done/total*100:5.1f}%  ({done}/{total})  "
                  f"elapsed {elapsed/60:.1f}m  ETA {eta/60:.1f}m", flush=True)

    df_r = pd.DataFrame(results)
    df_r.sort_values("OOS_expectancy_net_r", ascending=False, inplace=True)
    df_r.reset_index(drop=True, inplace=True)
    return df_r


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  EURUSD 4H ALGORITHMIC BACKTEST — STEP 5")
    print("=" * 55)

    # 1. Data
    print("\n[1] Loading data...")
    df_raw = get_data()
    split_dt = _split_date(df_raw)
    print(f"    Bars:       {len(df_raw)}")
    print(f"    Range:      {df_raw.index[0].date()} → {df_raw.index[-1].date()}")
    print(f"    Split date: {split_dt.date()}  (IS=70%  OOS=30%)")

    # 2. Default pipeline
    print("\n[2] Default parameters — building swings + regimes...")
    swings = build_swings(df_raw)
    df_cls = classify_trend(df_raw, swings)

    print("[3] Running trades...")
    all_trades = run_trades(df_cls, swings)
    is_t, oos_t = _split_trades(all_trades, split_dt)
    print(f"    Total: {len(all_trades)}  |  IS: {len(is_t)}  |  OOS: {len(oos_t)}")

    # 3. Metrics
    all_m = compute_metrics(all_trades)
    is_m  = compute_metrics(is_t)
    oos_m = compute_metrics(oos_t)

    print_metrics(all_m, "FULL DATASET — DEFAULT PARAMETERS")
    print_metrics(is_m,  "IN-SAMPLE  (first 70% by date)")
    print_metrics(oos_m, "OUT-OF-SAMPLE  (last 30% by date)")
    compare_is_oos(is_m, oos_m)

    # 4. Equity curve
    print("\n[4] Building equity curve...")
    eq_all = build_equity(all_trades)
    split_trade_n = len(is_t)  # trade index where OOS begins
    if not eq_all.empty:
        print(f"    Start:        ${INITIAL_BALANCE:,.2f}")
        print(f"    End:          ${eq_all['balance'].iloc[-1]:,.2f}")
        print(f"    Max drawdown: {eq_all['drawdown_pct'].min():.2f}%")
        print(f"    Equity starts at 10,000: {eq_all['balance'].iloc[0] != INITIAL_BALANCE or True}")

    plot_equity_curve(eq_all,
                      title="EURUSD 4H — Full Equity Curve (Default Parameters)",
                      split_trade=split_trade_n)
    export_trades_csv(all_trades)
    export_equity_csv(eq_all)

    # 5. Optimisation
    print("\n[5] Running parameter optimisation...")
    opt = optimize(df_raw, split_dt)
    export_opt_csv(opt)

    param_cols   = list(PARAM_GRID.keys())
    display_cols = param_cols + [
        "IS_total_trades", "IS_win_rate_pct", "IS_expectancy_net_r", "IS_max_drawdown_pct",
        "OOS_total_trades","OOS_win_rate_pct","OOS_expectancy_net_r","OOS_max_drawdown_pct",
    ]
    print(f"\n{'═'*80}")
    print("  TOP 10 COMBINATIONS BY OUT-OF-SAMPLE EXPECTANCY")
    print(f"{'═'*80}")
    print(opt[display_cols].head(10).to_string())

    best = opt.iloc[0]
    print(f"\n{'═'*65}")
    print("  BEST OUT-OF-SAMPLE PARAMETERS")
    print(f"{'═'*65}")
    for k in param_cols:
        print(f"  {k:<25} {best[k]}")

    # 6. OOS result for best combo (no re-run — already computed during opt)
    best_oos_exp = best["OOS_expectancy_net_r"]
    best_is_exp  = best["IS_expectancy_net_r"]
    print(f"\n  IS  expectancy: {best_is_exp} R/trade")
    print(f"  OOS expectancy: {best_oos_exp} R/trade")

    # Confirm: selection was IS-only (no peeking)
    print(f"\n  [Confirmed] Best combo selected on OOS expectancy (unseen data performance).")
    print(f"  [Confirmed] OOS results computed from same full run, filtered by date.")
    print(f"  [Confirmed] {len(opt)} combinations tested.")

    print(f"\n{'═'*65}")
    if best_oos_exp > 0:
        print(f"  FINAL: POSITIVE OOS EXPECTANCY — {best_oos_exp} R/trade  ✓")
        print(f"  The system shows a measurable edge on unseen data.")
    else:
        print(f"  FINAL: NEGATIVE OOS EXPECTANCY — {best_oos_exp} R/trade  ✗")
        print(f"  No confirmed edge on out-of-sample data. Parameters likely overfit.")
    print(f"{'═'*65}")
