"""
Continuation signal study — trade WITH momentum, not against it.

Current strategy (mean-reversion): buy when funding_z turns NEGATIVE (crowd bearish)
Continuation hypothesis: buy when funding_z turns POSITIVE (crowd turning bullish) on
an ACTIVE coin (oi_z > 1.0). Cross-tab showed oi_Q3 x fz_Q4 = best cell (+1.59%).

Three variants tested:
  A: funding_z crosses ABOVE +0.5  + oi_z > 1.0
  B: oi_delta3 > 5%  (OI growing 3 days) + funding_z > 0 + oi_z > 1.0
  C: atr_ratio > 1.1 (ATR expanding)    + funding_z > 0 + oi_z > 1.0

Then all three at 4H resolution using daily oi_z as gate.

Run: python -m algo.data.continuation_study
"""

import os
import numpy as np
import pandas as pd
from algo.data.zscore_entry_study import load_all
from algo.data.perp_flow_study import load_perp_oi, load_perp_kline, rank_ic
from algo.data.oi_flow_study import build_full_signals
from algo.data.signal_4h_study import load_4h, build_4h_signals

CACHE_DIR   = os.path.join(os.path.dirname(__file__), "cache", "swing_study")
OI_Z_THRESH = 1.0
MAX_HOLD    = 14    # continuation trades fail faster — shorter hold
MIN_STOP    = 0.05
STOP_BARS   = 5     # stop below 5-bar low (tighter than mean-reversion's 10)
MIN_OBS     = 90

# 4H params
MAX_HOLD_4H = 42    # ~7 days
STOP_BARS_4H = 12  # ~2 days


def section(t):
    print(f"\n{'='*65}\n{t}\n{'='*65}")


# ── Daily simulation ──────────────────────────────────────────────────────────

def simulate_continuation(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """
    variant: "A", "B", or "C"

    A: funding_z crosses above +0.5 + oi_z > 1.0
    B: oi_delta3 > 5% + funding_z > 0 + oi_z > 1.0
    C: atr_ratio > 1.1 + funding_z > 0 + oi_z > 1.0

    Stop: below N-bar low, min MIN_STOP
    Exit: funding_z falls below 0 (momentum dying) OR max hold
    """
    df = df.copy().reset_index(drop=True)
    n  = len(df)

    fz      = df["funding_z"].values
    oiz     = df["oi_z"].values
    atr_r   = df["atr_ratio"].values if "atr_ratio" in df.columns else np.ones(n)
    oi_d3   = df["oi_delta3"].values if "oi_delta3" in df.columns else np.zeros(n)
    op_     = df["open"].values
    lo_     = df["low"].values
    cl_     = df["close"].values

    trades   = []
    in_trade = 0

    for i in range(max(STOP_BARS + 1, 10), n - MAX_HOLD - 2):
        if i < in_trade:
            continue
        if np.isnan(fz[i]) or np.isnan(oiz[i]):
            continue

        # OI gate always required
        if oiz[i] < OI_Z_THRESH:
            continue

        # Variant entry logic
        if variant == "A":
            # funding_z crosses above +0.5
            if not (fz[i] >= 0.5 and fz[i-1] < 0.5):
                continue
        elif variant == "B":
            # OI growing 3 days + funding positive
            if np.isnan(oi_d3[i]):
                continue
            if not (oi_d3[i] > 5.0 and fz[i] > 0.0):
                continue
        elif variant == "C":
            # ATR expanding (breakout) + funding positive
            if np.isnan(atr_r[i]):
                continue
            if not (atr_r[i] > 1.1 and fz[i] > 0.0):
                continue

        entry = op_[min(i + 1, n - 1)]
        if entry <= 0:
            continue

        # Stop below STOP_BARS-bar low
        stop = lo_[max(0, i - STOP_BARS):i + 1].min()
        if np.isnan(stop):
            continue
        risk = entry - stop
        if risk <= 0:
            stop = entry * (1 - MIN_STOP)
            risk = entry - stop
        rp = risk / entry
        if rp < MIN_STOP:
            stop = entry * (1 - MIN_STOP)
            risk = entry * MIN_STOP
            rp   = MIN_STOP
        if rp > 0.30:
            continue

        # Walk forward to exit
        exit_bar    = min(i + MAX_HOLD, n - 1)
        exit_reason = "max_hold"
        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            if cl_[j] <= stop:
                exit_bar    = j
                exit_reason = "stop"
                break
            if not np.isnan(fz[j]) and fz[j] < 0.0:
                exit_bar    = j
                exit_reason = "signal"
                break

        exit_p = cl_[exit_bar]
        hold   = exit_bar - i
        ret    = (exit_p - entry) / entry * 100
        ret_r  = (exit_p - entry) / risk

        trades.append({
            "date":        df["date"].iloc[i],
            "ret_pct":     ret,
            "ret_r":       ret_r,
            "risk_pct":    rp * 100,
            "hold":        hold,
            "exit_reason": exit_reason,
            "oi_z":        oiz[i],
            "fz":          fz[i],
            "atr_ratio":   atr_r[i] if not np.isnan(atr_r[i]) else np.nan,
        })
        in_trade = exit_bar + 1

    return pd.DataFrame(trades)


def run_daily(all_sigs: dict, variant: str) -> pd.DataFrame:
    parts = []
    for sym, df in all_sigs.items():
        t = simulate_continuation(df, variant)
        if not t.empty:
            t["symbol"] = sym
            parts.append(t)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)


def print_summary(trades: pd.DataFrame, label: str):
    if trades.empty or len(trades) < 5:
        print(f"  {label:50} n={len(trades):3d}  insufficient data")
        return
    wr  = (trades["ret_r"] > 0).mean()
    avg = trades["ret_pct"].mean()
    med = trades["ret_pct"].median()
    std = trades["ret_pct"].std()
    shr = avg / std if std > 0 else 0
    print(f"  {label:50} n={len(trades):3d}  wr={wr*100:.0f}%  "
          f"avg={avg:+.2f}%  med={med:+.2f}%  shr={shr:+.3f}")


# ── 4H simulation ─────────────────────────────────────────────────────────────

def simulate_continuation_4h(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    n  = len(df)

    fz        = df["funding_z"].values
    oiz_4h    = df["oi_z_4h"].values if "oi_z_4h" in df.columns else np.zeros(n)
    daily_oiz = df["daily_oi_z"].fillna(0).values
    atr_r     = df["atr_ratio"].values
    atr_s     = df["atr_s"].values
    op_       = df["open"].values
    lo_       = df["low"].values
    cl_       = df["close"].values

    trades   = []
    in_trade = 0

    for i in range(max(STOP_BARS_4H + 1, 20), n - MAX_HOLD_4H - 2):
        if i < in_trade:
            continue
        if np.isnan(fz[i]) or daily_oiz[i] < OI_Z_THRESH:
            continue

        if variant == "A":
            if not (fz[i] >= 0.5 and fz[i-1] < 0.5):
                continue
        elif variant == "B":
            # 4H OI rising (use oi_z_4h as proxy) + funding positive
            if np.isnan(oiz_4h[i]):
                continue
            if not (oiz_4h[i] > 0.5 and fz[i] > 0.0):
                continue
        elif variant == "C":
            if np.isnan(atr_r[i]):
                continue
            if not (atr_r[i] > 1.1 and fz[i] > 0.0):
                continue

        entry = op_[min(i + 1, n - 1)]
        if entry <= 0:
            continue

        stop = lo_[max(0, i - STOP_BARS_4H):i + 1].min()
        if np.isnan(stop) or entry <= stop:
            stop = entry * (1 - MIN_STOP)
        risk = entry - stop
        rp   = risk / entry
        if rp < MIN_STOP:
            stop = entry * (1 - MIN_STOP)
            risk = entry * MIN_STOP
            rp   = MIN_STOP
        if rp > 0.30:
            continue

        exit_bar    = min(i + MAX_HOLD_4H, n - 1)
        exit_reason = "max_hold"
        for j in range(i + 1, min(i + MAX_HOLD_4H + 1, n)):
            if cl_[j] <= stop:
                exit_bar    = j
                exit_reason = "stop"
                break
            if not np.isnan(fz[j]) and fz[j] < 0.0:
                exit_bar    = j
                exit_reason = "signal"
                break

        exit_p = cl_[exit_bar]
        hold_b = exit_bar - i
        ret    = (exit_p - entry) / entry * 100
        ret_r  = (exit_p - entry) / risk

        trades.append({
            "time":        df["time"].iloc[i],
            "ret_pct":     ret,
            "ret_r":       ret_r,
            "risk_pct":    rp * 100,
            "hold_bars":   hold_b,
            "hold_days":   hold_b / 6,
            "exit_reason": exit_reason,
            "daily_oi_z":  daily_oiz[i],
            "funding_z":   fz[i],
        })
        in_trade = exit_bar + 1

    return pd.DataFrame(trades)


def run_4h(all_4h: dict, variant: str) -> pd.DataFrame:
    parts = []
    for sym, df in all_4h.items():
        t = simulate_continuation_4h(df, variant)
        if not t.empty:
            t["symbol"] = sym
            parts.append(t)
    if not parts:
        return pd.DataFrame()
    trades = pd.concat(parts, ignore_index=True)
    trades["time"] = pd.to_datetime(trades["time"])
    return trades.sort_values("time").reset_index(drop=True)


def print_detail(trades: pd.DataFrame, time_col: str = "date"):
    if trades.empty:
        return
    trades = trades.copy()
    trades["_t"] = pd.to_datetime(trades[time_col])
    wr  = (trades["ret_r"] > 0).mean()
    avg = trades["ret_pct"].mean()
    std = trades["ret_pct"].std()
    shr = avg / std if std > 0 else 0
    span = (trades["_t"].max() - trades["_t"].min()).days
    tpy  = len(trades) / (span / 365) if span > 0 else 0

    print(f"  Trades: {len(trades)}  ({tpy:.0f}/yr)  wr={wr*100:.0f}%  "
          f"avg={avg:+.2f}%  med={trades['ret_pct'].median():+.2f}%  shr={shr:+.3f}")
    reasons = trades["exit_reason"].value_counts().to_dict()
    print(f"  Exits: {reasons}")

    # Year breakdown
    trades["year"] = trades["_t"].dt.year
    print(f"  {'Year':>6}  {'n':>4}  {'wr%':>4}  {'avg%':>7}")
    for yr, g in trades.groupby("year"):
        print(f"  {yr:>6}  {len(g):>4}  {(g['ret_r']>0).mean()*100:>3.0f}%  "
              f"{g['ret_pct'].mean():>+6.2f}%")

    # Equity ($5k, 1% risk)
    eq = 5000.0
    for _, row in trades.sort_values("_t").iterrows():
        eq += eq * 0.01 * row["ret_r"]
    years = span / 365 if span > 0 else 1
    cagr  = (eq / 5000) ** (1 / years) - 1
    print(f"  Equity: $5k -> ${eq:,.0f}  CAGR={cagr*100:+.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading daily signals...")
    coins = load_all()

    all_sigs = {}
    for sym, spot_df in sorted(coins.items()):
        oi_df   = load_perp_oi(sym)
        perp_df = load_perp_kline(sym)
        if oi_df.empty or perp_df.empty:
            continue
        sig = build_full_signals(spot_df, oi_df, perp_df)
        if len(sig) >= MIN_OBS:
            all_sigs[sym] = sig

    print(f"Daily signals: {len(all_sigs)} coins")

    # ── Daily IC baseline ─────────────────────────────────────────────────────
    pool_parts = []
    for sym, df in all_sigs.items():
        d = df.copy(); d["symbol"] = sym; pool_parts.append(d)
    pool = pd.concat(pool_parts, ignore_index=True)

    section("0. IC CONTEXT — are continuation signals in the data at all?")
    print(f"\n  {'Signal':14}  {'IC_3d':>8}  {'IC_7d':>8}  {'IC_14d':>8}")
    for sig in ["funding_z", "oi_z", "oi_delta3", "atr_ratio"]:
        if sig not in pool.columns:
            continue
        ics = []
        for h in [3, 7, 14]:
            s = pool[[sig, f"fwd{h}"]].dropna()
            ics.append(rank_ic(s[sig], s[f"fwd{h}"]))
        print(f"  {sig:14}  {ics[0]:>+7.4f}   {ics[1]:>+7.4f}   {ics[2]:>+7.4f}")

    print(f"\n  Note: positive funding_z IC = high funding predicts higher returns = CONTINUATION")
    print(f"  Note: positive oi_z IC = high OI predicts higher returns = CONFIRMED")

    # ── Daily simulations ─────────────────────────────────────────────────────
    section("1. DAILY — three continuation variants")
    print(f"\n  {'Variant':50}  {'n':>4}  {'wr':>4}  {'avg%':>6}  {'med%':>6}  {'shr':>6}")
    print(f"  {'-'*80}")

    daily_results = {}
    for variant, label in [
        ("A", "A: fz cross above +0.5 + oi_z>1.0"),
        ("B", "B: oi_delta3>5% + fz>0 + oi_z>1.0"),
        ("C", "C: atr_ratio>1.1 + fz>0 + oi_z>1.0"),
    ]:
        t = run_daily(all_sigs, variant)
        daily_results[variant] = t
        print_summary(t, label)

    # ── Daily detail for best variants ────────────────────────────────────────
    for variant, label in [("A","A"), ("B","B"), ("C","C")]:
        t = daily_results[variant]
        if t.empty or len(t) < 5:
            continue
        avg = t["ret_pct"].mean()
        std = t["ret_pct"].std()
        shr = avg / std if std > 0 else 0
        if avg > 1.0 or shr > 0.08:   # only detail if it clears a bar
            section(f"DAILY DETAIL — variant {label}")
            print_detail(t, time_col="date")

    # ── Compare to mean-reversion baseline ────────────────────────────────────
    section("2. COMPARISON vs MEAN-REVERSION BASELINE")
    print(f"""
  Mean-reversion (frozen spec):
    n=82  wr=50%  avg=+2.75%  shr=+0.135  trades/yr ~62

  Continuation variants above.
  Verdict:
    If any continuation variant avg >= +1.5% and shr >= +0.10:
      -> worth combining in portfolio_sim (two signal types, uncorrelated)
    If all continuation variants are flat or negative:
      -> small-cap crypto doesn't trend at daily timeframe, mean-reversion only
""")

    # ── 4H signal frames ──────────────────────────────────────────────────────
    section("3. 4H — load cached data and build signal frames")
    print("Loading 4H data...")

    all_4h = {}
    for sym in sorted(all_sigs.keys()):
        spot_4h, perp_4h, oi_4h = load_4h(sym)
        if spot_4h.empty or perp_4h.empty or oi_4h.empty:
            continue
        sig = build_4h_signals(spot_4h, perp_4h, oi_4h, all_sigs[sym])
        if len(sig) >= 90:
            all_4h[sym] = sig

    print(f"4H signal frames: {len(all_4h)} coins")

    if not all_4h:
        print("No 4H data found. Run signal_4h_fetch.py first.")
    else:
        section("4. 4H — continuation variants with daily oi_z gate")
        print(f"\n  {'Variant':50}  {'n':>4}  {'wr':>4}  {'avg%':>6}  {'med%':>6}  {'shr':>6}")
        print(f"  {'-'*80}")

        h4_results = {}
        for variant, label in [
            ("A", "A: 4H fz cross above +0.5 + daily oi_z>1.0"),
            ("B", "B: 4H oi_z_4h>0.5 + fz>0 + daily oi_z>1.0"),
            ("C", "C: 4H atr_ratio>1.1 + fz>0 + daily oi_z>1.0"),
        ]:
            t = run_4h(all_4h, variant)
            h4_results[variant] = t
            print_summary(t, label)

        for variant, label in [("A","A"), ("B","B"), ("C","C")]:
            t = h4_results[variant]
            if t.empty or len(t) < 5:
                continue
            avg = t["ret_pct"].mean()
            std = t["ret_pct"].std()
            shr = avg / std if std > 0 else 0
            if avg > 1.0 or shr > 0.08:
                section(f"4H DETAIL — variant {label}")
                print_detail(t, time_col="time")

        # ── 4H mean-reversion baseline for context ───────────────────────────
        section("5. 4H MEAN-REVERSION BASELINE (existing signal at 4H)")
        from algo.data.signal_4h_study import simulate_4h
        mr_parts = []
        for sym, df in all_4h.items():
            t = simulate_4h(df)
            if not t.empty:
                t["symbol"] = sym
                mr_parts.append(t)
        if mr_parts:
            mr = pd.concat(mr_parts, ignore_index=True)
            mr["time"] = pd.to_datetime(mr["time"])
            print_summary(mr, "4H mean-reversion (fz cross below -0.5 + ATR compressed)")
            print_detail(mr, time_col="time")
        else:
            print("  No 4H mean-reversion trades.")

    section("SUMMARY")
    print("""
  Continuation = trade WITH rising funding/OI momentum (opposite of current)
  Mean-reversion = trade AGAINST crowded sentiment (current strategy)

  If continuation avg >= +1.5% and shr >= +0.10:
    Two signal types fire at different market phases -> more total trades,
    smoother equity curve, less regime dependence.
    Next step: add both to portfolio_sim and measure combined Sharpe.

  If continuation fails:
    Small-cap crypto at daily/4H is mean-reversion dominant.
    Focus: increase mean-reversion trade count via 4H and universe expansion.
""")
    print("Done.")
