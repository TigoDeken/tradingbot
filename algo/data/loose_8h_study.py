"""
Loosened spec study + 8H signal using real funding rates.

Two questions:
  1. DAILY: does removing ATR compression meaningfully hurt edge?
     ATR compression was never tested without oi_z. May be redundant.

  2. 4H bars + REAL funding settlements (8H cycle):
     Previous 4H study failed because it used basis z-score as proxy.
     Funding settles every 8H (00:00, 08:00, 16:00 UTC).
     4H bars land exactly on those timestamps every 2 bars.
     => assign real funding rates to matching 4H bars, forward-fill intra-period.
     This is the actual signal at sub-daily resolution, not a proxy.

Loosening means: keep what's proven (oi_z gate, funding_z cross, signal exit),
drop what isn't (ATR compression). Also test oi_z at 0.5 threshold.

Run: python -m algo.data.loose_8h_study
"""

import os
import numpy as np
import pandas as pd
from algo.data.zscore_entry_study import load_all, load_funding, CACHE_DIR
from algo.data.perp_flow_study import load_perp_oi, load_perp_kline, rank_ic
from algo.data.oi_flow_study import build_full_signals

# ── Parameters ────────────────────────────────────────────────────────────────
FZ_THRESH    = -0.5
EXIT_Z       = 0.0
OI_Z_WIN     = 90       # daily
FZ_WIN_DAILY = 90       # daily funding z window
FZ_WIN_4H    = 90 * 3   # 270 settlements = 90 days of 8H funding
MIN_STOP     = 0.05
STOP_DAYS    = 10       # daily
STOP_BARS_4H = 20       # ~3.3 days of 4H bars
MAX_HOLD_D   = 21       # daily max hold
MAX_HOLD_4H  = 42       # ~7 days of 4H bars
MIN_OBS      = 90


def section(t):
    print(f"\n{'='*65}\n{t}\n{'='*65}")


# ── Daily loosened simulation ─────────────────────────────────────────────────

def simulate_loose(df: pd.DataFrame, use_atr: bool, oi_thresh: float) -> pd.DataFrame:
    n   = len(df)
    fz  = df["funding_z"].values
    oiz = df["oi_z"].values
    atr = df["atr_ratio"].values if "atr_ratio" in df.columns else np.ones(n)
    op_ = df["open"].values
    lo_ = df["low"].values
    cl_ = df["close"].values

    trades, in_trade = [], 0

    for i in range(FZ_WIN_DAILY + 10, n - MAX_HOLD_D - 1):
        if i < in_trade:
            continue
        if np.isnan(fz[i]) or np.isnan(fz[i-1]):
            continue
        if not (fz[i] < FZ_THRESH and fz[i-1] >= FZ_THRESH):
            continue
        if use_atr and (np.isnan(atr[i]) or atr[i] >= 0.80):
            continue
        if np.isnan(oiz[i]) or oiz[i] < oi_thresh:
            continue

        entry = op_[min(i+1, n-1)]
        stop  = lo_[max(0, i-STOP_DAYS):i].min()
        if np.isnan(stop) or entry <= 0:
            continue
        risk = entry - stop
        if risk <= 0:
            continue
        rp = risk / entry
        if rp < MIN_STOP or rp > 0.30:
            continue

        exit_bar, exit_reason = min(i + MAX_HOLD_D, n-1), "max_hold"
        for j in range(i+1, min(i + MAX_HOLD_D + 1, n)):
            if lo_[j] <= stop:
                exit_bar, exit_reason = j, "stop"
                break
            if not np.isnan(fz[j]) and fz[j] > EXIT_Z:
                exit_bar, exit_reason = j, "signal"
                break

        exit_p = cl_[exit_bar]
        hold   = exit_bar - i
        ret    = (exit_p - entry) / entry * 100

        trades.append({
            "date":        df["date"].iloc[i],
            "ret_pct":     ret,
            "ret_r":       ret / (rp * 100),
            "risk_pct":    rp * 100,
            "hold":        hold,
            "exit_reason": exit_reason,
            "oi_z":        oiz[i],
            "fz":          fz[i],
        })
        in_trade = exit_bar + 1

    return pd.DataFrame(trades)


def print_row(label: str, t: pd.DataFrame):
    if t.empty or len(t) < 5:
        print(f"  {label:52} n={len(t):4d}  -")
        return
    wr  = (t["ret_r"] > 0).mean()
    avg = t["ret_pct"].mean()
    std = t["ret_pct"].std()
    shr = avg / std if std > 0 else 0
    print(f"  {label:52} n={len(t):4d}  wr={wr*100:.0f}%  "
          f"avg={avg:+.2f}%  shr={shr:+.3f}")


# ── 4H signal builder with real funding ──────────────────────────────────────

def build_4h_real_funding(sym: str, daily_sig: pd.DataFrame) -> pd.DataFrame:
    """
    Load 4H spot klines + real 8H funding settlements + 4H perp OI.
    Funding rates are assigned to the 4H bar that starts at each settlement
    timestamp (00:00, 08:00, 16:00 UTC) and forward-filled intra-period.
    """
    spot_p = os.path.join(CACHE_DIR, f"{sym}_spot_4h.parquet")
    oi_p   = os.path.join(CACHE_DIR, f"{sym}_perp_oi_4h.parquet")
    if not os.path.exists(spot_p) or not os.path.exists(oi_p):
        return pd.DataFrame()

    spot = pd.read_parquet(spot_p).sort_values("time").reset_index(drop=True)
    oi4h = pd.read_parquet(oi_p).sort_values("time").reset_index(drop=True)

    funding = load_funding(sym)
    if funding.empty:
        return pd.DataFrame()

    funding = funding.copy().sort_values("time")
    # Normalise timestamps to UTC
    if funding["time"].dt.tz is None:
        funding["time"] = funding["time"].dt.tz_localize("UTC")
    if spot["time"].dt.tz is None:
        spot["time"] = spot["time"].dt.tz_localize("UTC")

    # ── ATR on 4H bars ───────────────────────────────────────────────────────
    hi, lo, cl = spot["high"].values, spot["low"].values, spot["close"].values
    prev_cl = np.roll(cl, 1); prev_cl[0] = cl[0]
    tr      = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_cl), np.abs(lo - prev_cl)))
    spot["atr5"]     = pd.Series(tr).rolling(5).mean().values
    spot["atr10"]    = pd.Series(tr).rolling(10).mean().values
    spot["atr_ratio"] = spot["atr5"] / spot["atr10"].replace(0, np.nan)

    # ── Attach funding to 4H bars ─────────────────────────────────────────────
    # Each funding record has a timestamp aligned to 8H boundary.
    # Round spot bar times DOWN to nearest 8H to find which funding period they belong to.
    spot["time_8h"] = spot["time"].dt.floor("8h")
    funding = funding.rename(columns={"time": "time_8h", "funding_rate": "fr_raw"})
    funding["time_8h"] = funding["time_8h"].dt.floor("8h")
    funding = funding.groupby("time_8h")["fr_raw"].last().reset_index()

    spot = spot.merge(funding, on="time_8h", how="left")
    # Forward-fill: 4H bars between settlements inherit prior settlement's rate
    spot["fr_raw"] = spot["fr_raw"].ffill()

    # ── Funding z-score: rolling on actual settlement values ──────────────────
    # Mark settlement bars (those where time aligns with an 8H boundary)
    spot["is_settlement"] = (spot["time"].dt.hour % 8 == 0)
    # Compute rolling stats on settlement rates only, then broadcast back
    fr = spot["fr_raw"].copy()
    roll = fr.rolling(FZ_WIN_4H, min_periods=FZ_WIN_4H // 2)
    spot["funding_z"] = (fr - roll.mean()) / roll.std()

    # ── OI z-score at 4H ────────────────────────────────────────────────────
    if oi4h["time"].dt.tz is None:
        oi4h["time"] = oi4h["time"].dt.tz_localize("UTC")
    oi4h = oi4h.rename(columns={"oi": "oi_4h"})
    spot = pd.merge_asof(
        spot.sort_values("time"),
        oi4h[["time","oi_4h"]].sort_values("time"),
        on="time", direction="nearest", tolerance=pd.Timedelta("4h")
    )
    roll_oi = spot["oi_4h"].rolling(FZ_WIN_4H, min_periods=FZ_WIN_4H // 2)
    spot["oi_z_4h"] = (spot["oi_4h"] - roll_oi.mean()) / roll_oi.std()

    # ── Daily oi_z gate ──────────────────────────────────────────────────────
    d = daily_sig[["date","oi_z"]].copy()
    d["date_key"] = pd.to_datetime(d["date"])
    spot["date_key"] = spot["time"].dt.normalize().dt.tz_localize(None)
    spot = spot.merge(
        d[["date_key","oi_z"]].rename(columns={"oi_z":"daily_oi_z"}),
        on="date_key", how="left"
    )
    spot["daily_oi_z"] = spot["daily_oi_z"].ffill()

    # ── Forward returns ──────────────────────────────────────────────────────
    cl_arr = spot["close"].values
    for h in [6, 12, 42]:
        spot[f"fwd{h}"] = (pd.Series(cl_arr).shift(-h) / pd.Series(cl_arr) - 1).values * 100

    return spot.reset_index(drop=True)


# ── 4H simulation ─────────────────────────────────────────────────────────────

def simulate_4h(df: pd.DataFrame, use_atr: bool, oi_thresh: float) -> pd.DataFrame:
    n          = len(df)
    fz         = df["funding_z"].values
    daily_oiz  = df["daily_oi_z"].fillna(0).values
    atr        = df["atr_ratio"].values
    op_        = df["open"].values
    lo_        = df["low"].values
    cl_        = df["close"].values

    trades, in_trade = [], 0

    for i in range(FZ_WIN_4H + 5, n - MAX_HOLD_4H - 2):
        if i < in_trade:
            continue
        if np.isnan(fz[i]) or np.isnan(fz[i-1]):
            continue
        if not (fz[i] < FZ_THRESH and fz[i-1] >= FZ_THRESH):
            continue
        if use_atr and (np.isnan(atr[i]) or atr[i] >= 0.80):
            continue
        if daily_oiz[i] < oi_thresh:
            continue

        entry = op_[min(i+1, n-1)]
        if entry <= 0:
            continue
        stop  = lo_[max(0, i - STOP_BARS_4H):i+1].min()
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

        exit_bar, exit_reason = min(i + MAX_HOLD_4H, n-1), "max_hold"
        for j in range(i+1, min(i + MAX_HOLD_4H + 1, n)):
            if cl_[j] <= stop:
                exit_bar, exit_reason = j, "stop"
                break
            if not np.isnan(fz[j]) and fz[j] > EXIT_Z:
                exit_bar, exit_reason = j, "signal"
                break

        exit_p   = cl_[exit_bar]
        hold_b   = exit_bar - i
        ret      = (exit_p - entry) / entry * 100
        ret_r    = (exit_p - entry) / risk

        trades.append({
            "time":        df["time"].iloc[i],
            "ret_pct":     ret,
            "ret_r":       ret_r,
            "risk_pct":    rp * 100,
            "hold_bars":   hold_b,
            "hold_days":   hold_b / 6,
            "exit_reason": exit_reason,
            "daily_oi_z":  daily_oiz[i],
            "fz":          fz[i],
        })
        in_trade = exit_bar + 1

    return pd.DataFrame(trades)


def detail(trades: pd.DataFrame, tcol: str = "date"):
    if trades.empty:
        return
    t = trades.copy()
    t["_t"] = pd.to_datetime(t[tcol])
    wr  = (t["ret_r"] > 0).mean()
    avg = t["ret_pct"].mean()
    std = t["ret_pct"].std()
    shr = avg / std if std > 0 else 0
    span = (t["_t"].max() - t["_t"].min()).days
    tpy  = len(t) / (span / 365) if span > 0 else 0

    print(f"\n  {len(t)} trades ({tpy:.0f}/yr)  wr={wr*100:.0f}%  "
          f"avg={avg:+.2f}%  med={t['ret_pct'].median():+.2f}%  shr={shr:+.3f}")
    reasons = t["exit_reason"].value_counts().to_dict()
    print(f"  Exits: {reasons}")

    t["year"] = t["_t"].dt.year
    print(f"  {'Year':>5}  {'n':>4}  {'wr%':>4}  {'avg%':>7}")
    for yr, g in t.groupby("year"):
        print(f"  {yr:>5}  {len(g):>4}  {(g['ret_r']>0).mean()*100:>3.0f}%  "
              f"{g['ret_pct'].mean():>+6.2f}%")

    eq = 5000.0
    for _, row in t.sort_values("_t").iterrows():
        eq += eq * 0.01 * row["ret_r"]
    years = span / 365 if span > 0 else 1
    cagr  = (eq / 5000) ** (1 / years) - 1
    mdd_s = pd.Series([5000.0])
    eq2 = 5000.0
    for _, row in t.sort_values("_t").iterrows():
        eq2 += eq2 * 0.01 * row["ret_r"]
        mdd_s = pd.concat([mdd_s, pd.Series([eq2])], ignore_index=True)
    peak = mdd_s.cummax()
    mdd  = ((mdd_s - peak) / peak).min() * 100
    print(f"  $5k -> ${eq:,.0f}  CAGR={cagr*100:+.1f}%  Max DD={mdd:.1f}%")


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

    # ── Section 1: Daily — loosen ATR requirement ─────────────────────────────
    section("1. DAILY — does ATR compression filter add value?")
    print(f"\n  {'Config':52}  {'n':>4}  {'wr':>4}  {'avg%':>6}  {'shr':>6}")
    print(f"  {'-'*72}")

    daily_variants = {}
    for label, use_atr, thresh in [
        ("FROZEN:  ATR<0.80 + oi_z>1.0",         True,  1.0),
        ("LOOSE A: no ATR  + oi_z>1.0",           False, 1.0),
        ("LOOSE B: no ATR  + oi_z>0.5",           False, 0.5),
        ("LOOSE C: ATR<0.80 + oi_z>0.5",          True,  0.5),
        ("LOOSE D: no ATR  + oi_z>0.0 (all OI)",  False, 0.0),
    ]:
        parts = []
        for sym, df in all_sigs.items():
            t = simulate_loose(df, use_atr=use_atr, oi_thresh=thresh)
            if not t.empty:
                t["symbol"] = sym
                parts.append(t)
        all_t = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        daily_variants[label] = all_t
        print_row(label, all_t)

    # Detail for LOOSE A (no ATR, oi>1.0) if it improves on frozen
    print()
    for lbl in ["LOOSE A: no ATR  + oi_z>1.0", "LOOSE B: no ATR  + oi_z>0.5"]:
        t = daily_variants[lbl]
        if not t.empty and len(t) >= 5:
            avg = t["ret_pct"].mean()
            shr = avg / t["ret_pct"].std() if t["ret_pct"].std() > 0 else 0
            print(f"  --- {lbl} ---")
            detail(t, "date")
            print()

    # ── Section 2: 4H with real funding ───────────────────────────────────────
    section("2. 4H BARS WITH REAL 8H FUNDING RATES")
    print("  Building 4H signal frames with actual funding settlements...")

    all_4h = {}
    missing = []
    for sym in sorted(all_sigs.keys()):
        df4 = build_4h_real_funding(sym, all_sigs[sym])
        if df4.empty or len(df4) < 90:
            missing.append(sym)
            continue
        if df4["funding_z"].notna().sum() < 50:
            missing.append(sym)
            continue
        all_4h[sym] = df4

    print(f"  4H frames built: {len(all_4h)} coins  (skipped: {len(missing)})")

    if all_4h:
        # IC check first
        pool_4h = pd.concat(
            [d.assign(symbol=s) for s, d in all_4h.items()], ignore_index=True
        )
        sub = pool_4h[["funding_z","oi_z_4h","daily_oi_z","fwd6","fwd12","fwd42"]].dropna(
            subset=["funding_z","fwd42"])
        print(f"\n  IC of 4H signals vs forward returns:")
        print(f"  {'Signal':14}  {'IC_6bar':>8}  {'IC_12bar':>9}  {'IC_42bar':>9}")
        print(f"  {'':14}  {'(~1d)':>8}  {'(~2d)':>9}  {'(~7d)':>9}")
        for sig in ["funding_z", "oi_z_4h", "daily_oi_z"]:
            if sig not in sub.columns:
                continue
            ics = []
            for h in [6, 12, 42]:
                s = sub[[sig, f"fwd{h}"]].dropna()
                ics.append(rank_ic(s[sig], s[f"fwd{h}"]))
            print(f"  {sig:14}  {ics[0]:>+7.4f}   {ics[1]:>+8.4f}   {ics[2]:>+8.4f}")

        section("3. 4H SIMULATION — signal variants with real funding")
        print(f"\n  {'Config':52}  {'n':>4}  {'wr':>4}  {'avg%':>6}  {'shr':>6}")
        print(f"  {'-'*72}")

        h4_variants = {}
        for label, use_atr, thresh in [
            ("FROZEN logic at 4H: ATR + oi_z>1.0",    True,  1.0),
            ("4H LOOSE A: no ATR + oi_z>1.0",         False, 1.0),
            ("4H LOOSE B: no ATR + oi_z>0.5",         False, 0.5),
            ("4H LOOSE C: ATR + oi_z>0.5",            True,  0.5),
        ]:
            parts = []
            for sym, df in all_4h.items():
                t = simulate_4h(df, use_atr=use_atr, oi_thresh=thresh)
                if not t.empty:
                    t["symbol"] = sym
                    parts.append(t)
            all_t = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            if not all_t.empty:
                all_t["time"] = pd.to_datetime(all_t["time"])
                all_t = all_t.sort_values("time").reset_index(drop=True)
            h4_variants[label] = all_t
            print_row(label, all_t)

        # Detail for best 4H variants
        print()
        for lbl in ["4H LOOSE A: no ATR + oi_z>1.0", "4H LOOSE B: no ATR + oi_z>0.5"]:
            t = h4_variants[lbl]
            if not t.empty and len(t) >= 5:
                avg = t["ret_pct"].mean()
                if avg > 0.5:
                    print(f"  --- {lbl} ---")
                    detail(t, "time")
                    print()

    section("SUMMARY")
    print("""
  Daily loosening:
    If no-ATR variant has similar avg and shr with more trades: drop ATR filter.
    If no-ATR degrades significantly: keep ATR compression as a useful filter.

  4H with real funding:
    If 4H avg >= +1.5% and shr >= +0.10: real signal at sub-daily resolution.
    If 4H is positive but below daily: still useful for volume (more trades).
    If 4H is negative: the daily signal only works at daily resolution.

  Key comparison:
    Daily frozen: n=82, avg=+2.75%, shr=+0.135
    4H target:    n>=200, avg>=+1.5%, shr>=+0.10
""")
    print("Done.")
