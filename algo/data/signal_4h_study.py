"""
4H signal study — same logic as daily frozen spec, applied at 4H resolution.

Gate: daily oi_z > 1.0 (coin must be "active" per its own 90d OI history)
Entry: 4H funding_z crosses below -0.5 AND ATR(3)/ATR(6) < 0.80 (compressed)
Exit: 4H funding_z rises above 0.0 OR 42-bar max hold (~7 days)
Stop: 4H ATR-based, 5% minimum

Hypothesis: same signal, 6x more bars = 3-5x more trades.
Daily oi_z gate prevents firing on dead coins.

Run: python -m algo.data.signal_4h_study
"""

import os
import numpy as np
import pandas as pd
from algo.data.zscore_entry_study import load_all
from algo.data.perp_flow_study import load_perp_oi, load_perp_kline, rank_ic
from algo.data.oi_flow_study import build_full_signals

CACHE_DIR  = os.path.join(os.path.dirname(__file__), "cache", "swing_study")
FZ_WIN     = 90 * 6   # 90 daily bars = 540 4H bars
OI_Z_WIN   = 90       # daily oi_z still computed on daily bars
ATR_S      = 3        # short ATR window (4H bars)
ATR_L      = 6        # long ATR window (4H bars)
FZ_THRESH  = -0.5
ATR_THRESH = 0.80
OI_Z_THRESH = 1.0
EXIT_Z     = 0.0
MAX_HOLD   = 42       # bars (~7 days at 4H)
MIN_STOP   = 0.05
MIN_OBS    = 90       # minimum 4H bars to include a coin


def load_4h(sym: str):
    spot_p = os.path.join(CACHE_DIR, f"{sym}_spot_4h.parquet")
    perp_p = os.path.join(CACHE_DIR, f"{sym}_perp_4h.parquet")
    oi_p   = os.path.join(CACHE_DIR, f"{sym}_perp_oi_4h.parquet")
    if not all(os.path.exists(p) for p in [spot_p, perp_p, oi_p]):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    return (
        pd.read_parquet(spot_p),
        pd.read_parquet(perp_p),
        pd.read_parquet(oi_p),
    )


def build_4h_signals(spot_4h: pd.DataFrame, perp_4h: pd.DataFrame,
                     oi_4h: pd.DataFrame, daily_sig: pd.DataFrame) -> pd.DataFrame:
    """
    Merges 4H bars with daily oi_z (from frozen daily signal frame).
    Computes 4H funding_z, ATR ratio, and forward returns.
    """
    # ── 4H ATR ───────────────────────────────────────────────────────────────
    p = spot_4h.copy().sort_values("time").reset_index(drop=True)
    hi, lo, cl = p["high"].values, p["low"].values, p["close"].values
    prev_cl = np.roll(cl, 1); prev_cl[0] = cl[0]
    tr  = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_cl), np.abs(lo - prev_cl)))
    p["atr_s"] = pd.Series(tr).rolling(ATR_S).mean().values
    p["atr_l"] = pd.Series(tr).rolling(ATR_L).mean().values
    p["atr_ratio"] = p["atr_s"] / p["atr_l"].replace(0, np.nan)

    # ── 4H basis + funding_z ─────────────────────────────────────────────────
    q = perp_4h.copy().sort_values("time").reset_index(drop=True)
    q = q.rename(columns={"close": "perp_close"})

    # Funding rate stored in perp kline turnover slot? No — perp kline has no funding.
    # We compute a proxy: basis pct change as a funding-rate stand-in for 4H.
    # Real funding is 8H settlement — we use basis as the 4H analog.
    merged_pq = pd.merge_asof(
        p.sort_values("time"),
        q[["time","perp_close"]].sort_values("time"),
        on="time", direction="nearest", tolerance=pd.Timedelta("2h")
    )
    merged_pq["basis"] = (merged_pq["perp_close"] - merged_pq["close"]) / merged_pq["close"] * 100

    # basis z-score as funding_z analog
    roll = merged_pq["basis"].rolling(FZ_WIN, min_periods=FZ_WIN // 2)
    merged_pq["funding_z"] = (merged_pq["basis"] - roll.mean()) / roll.std()

    # ── 4H OI z-score ────────────────────────────────────────────────────────
    r = oi_4h.copy().sort_values("time").reset_index(drop=True)
    r = r.rename(columns={"oi": "oi_4h"})
    merged_pq = pd.merge_asof(
        merged_pq.sort_values("time"),
        r[["time","oi_4h"]].sort_values("time"),
        on="time", direction="nearest", tolerance=pd.Timedelta("2h")
    )
    roll_oi = merged_pq["oi_4h"].rolling(OI_Z_WIN * 6, min_periods=OI_Z_WIN * 3)
    merged_pq["oi_z_4h"] = (merged_pq["oi_4h"] - roll_oi.mean()) / roll_oi.std()

    # ── Attach daily oi_z as gate ─────────────────────────────────────────────
    # daily_sig has "date" (tz-naive) and "oi_z"
    daily_sig = daily_sig[["date","oi_z"]].copy()
    daily_sig["date_key"] = pd.to_datetime(daily_sig["date"])
    merged_pq["date_key"] = merged_pq["time"].dt.normalize().dt.tz_localize(None)

    merged_pq = merged_pq.merge(
        daily_sig[["date_key","oi_z"]].rename(columns={"oi_z":"daily_oi_z"}),
        on="date_key", how="left"
    )
    merged_pq["daily_oi_z"] = merged_pq["daily_oi_z"].ffill()

    # ── Forward returns (in bars) ─────────────────────────────────────────────
    cl_arr = merged_pq["close"].values
    for h in [6, 12, 42]:  # ~1d, ~2d, ~7d
        merged_pq[f"fwd{h}"] = (
            pd.Series(cl_arr).shift(-h) / pd.Series(cl_arr) - 1
        ).values * 100

    return merged_pq.reset_index(drop=True)


def simulate_4h(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    n  = len(df)

    fz        = df["funding_z"].values
    atr_ratio = df["atr_ratio"].values
    daily_oiz = df["daily_oi_z"].fillna(0).values
    op_       = df["open"].values
    lo_       = df["low"].values
    cl_       = df["close"].values
    atr_s_    = df["atr_s"].values

    trades   = []
    in_trade = 0

    for i in range(1, n - MAX_HOLD - 2):
        if i < in_trade:
            continue
        # Daily OI gate
        if daily_oiz[i] < OI_Z_THRESH:
            continue
        # Signal: funding_z crosses below threshold
        if not (fz[i] < FZ_THRESH and fz[i-1] >= FZ_THRESH):
            continue
        # ATR compressed
        if np.isnan(atr_ratio[i]) or atr_ratio[i] >= ATR_THRESH:
            continue

        entry = op_[min(i+1, n-1)]
        if entry <= 0:
            continue
        stop  = lo_[i] - max(atr_s_[i], entry * MIN_STOP)
        risk  = entry - stop
        if risk <= 0:
            continue
        rp = risk / entry
        if rp > 0.30:
            continue

        # Walk forward to exit
        exit_bar    = i + MAX_HOLD
        exit_reason = "max_hold"
        for j in range(i+1, min(i + MAX_HOLD + 1, n)):
            if cl_[j] <= stop:
                exit_bar    = j
                exit_reason = "stop"
                break
            if not np.isnan(fz[j]) and fz[j] > EXIT_Z:
                exit_bar    = j
                exit_reason = "signal"
                break

        exit_p = cl_[min(exit_bar, n-1)]
        hold   = exit_bar - i
        ret    = (exit_p - entry) / entry * 100
        ret_r  = (exit_p - entry) / risk

        trades.append({
            "bar":         i,
            "time":        df["time"].iloc[i],
            "ret_pct":     ret,
            "ret_r":       ret_r,
            "risk_pct":    rp * 100,
            "hold_bars":   hold,
            "hold_days":   hold / 6,
            "exit_reason": exit_reason,
            "daily_oi_z":  daily_oiz[i],
            "funding_z":   fz[i],
            "atr_ratio":   atr_ratio[i],
        })
        in_trade = exit_bar + 1

    return pd.DataFrame(trades)


def section(t):
    print(f"\n{'='*65}\n{t}\n{'='*65}")


if __name__ == "__main__":
    print("Loading daily signals for oi_z gate...")
    coins = load_all()

    daily_sigs = {}
    for sym, spot_df in sorted(coins.items()):
        oi_df   = load_perp_oi(sym)
        perp_df = load_perp_kline(sym)
        if oi_df.empty or perp_df.empty:
            continue
        sig = build_full_signals(spot_df, oi_df, perp_df)
        if len(sig) >= 90:
            daily_sigs[sym] = sig

    print(f"Daily signals: {len(daily_sigs)} coins")

    print("Building 4H signal frames...")
    all_4h = {}
    for sym in sorted(daily_sigs.keys()):
        spot_4h, perp_4h, oi_4h = load_4h(sym)
        if spot_4h.empty or perp_4h.empty or oi_4h.empty:
            continue
        sig = build_4h_signals(spot_4h, perp_4h, oi_4h, daily_sigs[sym])
        if len(sig) >= MIN_OBS:
            all_4h[sym] = sig

    print(f"4H signal frames: {len(all_4h)} coins")

    if not all_4h:
        print("No 4H data. Run signal_4h_fetch.py first.")
        exit()

    # ── IC analysis ───────────────────────────────────────────────────────────
    pool_parts = []
    for sym, df in all_4h.items():
        d = df.copy(); d["symbol"] = sym; pool_parts.append(d)
    pool = pd.concat(pool_parts, ignore_index=True)
    pool = pool.dropna(subset=["funding_z","fwd42"])

    section("1. IC — 4H SIGNALS vs FORWARD RETURNS")
    print(f"\n  {'Signal':14}  {'IC_6bar':>8}  {'IC_12bar':>9}  {'IC_42bar':>9}")
    print(f"  {'':14}  {'(~1d)':>8}  {'(~2d)':>9}  {'(~7d)':>9}")
    for sig in ["funding_z","oi_z_4h","daily_oi_z","atr_ratio"]:
        if sig not in pool.columns:
            continue
        ics = []
        for h in [6, 12, 42]:
            s = pool[[sig, f"fwd{h}"]].dropna()
            ics.append(rank_ic(s[sig], s[f"fwd{h}"]))
        print(f"  {sig:14}  {ics[0]:>+7.4f}   {ics[1]:>+8.4f}   {ics[2]:>+8.4f}")

    # ── Quintile breakdown ────────────────────────────────────────────────────
    section("2. QUINTILE — funding_z vs 7-day (42-bar) forward return")
    sub = pool[["funding_z","fwd42"]].dropna().copy()
    try:
        sub["Q"] = pd.qcut(sub["funding_z"], q=5, labels=[1,2,3,4,5], duplicates="drop")
        print(f"\n  Q1 = most negative funding (crowded shorts/bearish)  Q5 = most positive")
        print(f"  {'Q':>3}  {'n':>6}  {'mean%':>8}  {'wr%':>6}  {'med%':>8}")
        for q, grp in sub.groupby("Q"):
            print(f"  {q:>3}  {len(grp):>6}  {grp['fwd42'].mean():>+7.2f}%  "
                  f"{(grp['fwd42']>0).mean()*100:>5.0f}%  {grp['fwd42'].median():>+7.2f}%")
    except Exception as e:
        print(f"  Quintile error: {e}")

    # ── Trade simulation ──────────────────────────────────────────────────────
    section("3. SIMULATION — 4H signal with daily oi_z gate")
    all_trades = []
    for sym, df in all_4h.items():
        t = simulate_4h(df)
        if t.empty:
            continue
        t["symbol"] = sym
        all_trades.append(t)

    if not all_trades:
        print("  No trades fired.")
    else:
        trades = pd.concat(all_trades, ignore_index=True)
        trades["time"] = pd.to_datetime(trades["time"])
        trades = trades.sort_values("time").reset_index(drop=True)

        wr  = (trades["ret_r"] > 0).mean()
        avg = trades["ret_pct"].mean()
        med = trades["ret_pct"].median()
        std = trades["ret_pct"].std()
        shr = avg / std if std > 0 else 0

        span_days = (trades["time"].max() - trades["time"].min()).days
        years     = span_days / 365
        tpy       = len(trades) / years if years > 0 else 0

        print(f"\n  Trades: {len(trades)}  ({tpy:.0f}/yr)  wr={wr*100:.0f}%  "
              f"avg={avg:+.2f}%  med={med:+.2f}%  sharpe={shr:+.3f}")
        print(f"  Avg hold: {trades['hold_days'].mean():.1f} days")

        # Compare to daily baseline (82 trades)
        print(f"\n  Daily baseline: 82 trades, avg=+2.75%, wr=50%, Sharpe=+0.135")
        print(f"  4H this run:   {len(trades)} trades, avg={avg:+.2f}%, "
              f"wr={wr*100:.0f}%, Sharpe={shr:+.3f}")
        print(f"  Multiplier vs daily: {len(trades)/82:.1f}x trades")

        # Exit reason breakdown
        reasons = trades["exit_reason"].value_counts()
        print(f"\n  Exit reasons: {dict(reasons)}")

        # Year by year
        trades["year"] = trades["time"].dt.year
        print(f"\n  {'Year':>6}  {'n':>5}  {'wr%':>5}  {'avg%':>7}  {'sharpe':>8}")
        for yr, grp in trades.groupby("year"):
            g_wr  = (grp["ret_r"] > 0).mean()
            g_avg = grp["ret_pct"].mean()
            g_std = grp["ret_pct"].std()
            g_shr = g_avg / g_std if g_std > 0 else 0
            print(f"  {yr:>6}  {len(grp):>5}  {g_wr*100:>4.0f}%  "
                  f"{g_avg:>+6.2f}%  {g_shr:>+7.3f}")

        # Equity curve ($5k, 1% risk)
        section("4. EQUITY — $5k account, 1% risk/trade")
        equity = 5000
        eq_curve = []
        for _, row in trades.iterrows():
            pnl    = equity * 0.01 * row["ret_r"]
            equity += pnl
            eq_curve.append(equity)
        eq_s = pd.Series(eq_curve)
        mdd  = ((eq_s - eq_s.cummax()) / eq_s.cummax()).min() * 100
        total_ret = equity / 5000 - 1
        cagr      = (equity / 5000) ** (1 / years) - 1 if years > 0 else 0

        print(f"\n  Final equity:  ${equity:,.0f}  ({total_ret*100:+.1f}%)")
        print(f"  CAGR:          {cagr*100:+.1f}%  ({years:.1f} years)")
        print(f"  Max drawdown:  {mdd:+.1f}%")

    section("SUMMARY")
    print("""
  4H signal: same funding_z cross + ATR compression, but at 4H resolution.
  Daily oi_z > 1.0 gate prevents firing on dead coins.

  Key questions answered:
  - How many more trades vs daily? (expect 3-5x)
  - Does per-trade edge hold up? (expect some degradation — more noise at 4H)
  - Does daily oi_z gate still add lift?

  If avg >= +1.5% and sharpe >= +0.10: incorporate into portfolio_sim as second
  signal layer. Run python -m algo.data.portfolio_sim after adding 4H trades.
""")
    print("Done.")
