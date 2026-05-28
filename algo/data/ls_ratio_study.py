"""
Long/Short ratio study — directional crowding as a contrarian signal.

Bybit /v5/market/account-ratio returns the % of unique accounts
holding net-long vs net-short perpetual positions per coin.

When retail is overwhelmingly long (high ls_ratio), smart money sits short.
Extreme long crowding precedes corrections. Extreme short crowding precedes squeezes.

Signals:
  ls_ratio  = long_account_pct / short_account_pct
  ls_z      = 90-day rolling z-score of ls_ratio
  Low ls_z (crowded shorts) -> bullish for spot (contrarian long setup)
  High ls_z (crowded longs) -> bearish (crowded, due for correction)

Run: python -m algo.data.ls_ratio_study
"""

import os
import time
import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP
from algo.data.zscore_entry_study import load_all
from algo.data.perp_flow_study import load_perp_oi, load_perp_kline, rank_ic
from algo.data.oi_flow_study import build_full_signals

SESSION   = HTTP(testnet=False)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "swing_study")
LS_Z_WIN  = 90
MIN_OBS   = 90


def fetch_ls_ratio(symbol: str) -> pd.DataFrame:
    try:
        rows, cursor = [], None
        while True:
            p = dict(category="linear", symbol=symbol, period="1d", limit=500)
            if cursor:
                p["cursor"] = cursor
            resp   = SESSION.get_long_short_ratio(**p)
            result = resp.get("result", {})
            raw    = result.get("list", [])
            if not raw:
                break
            rows.extend(raw)
            cursor = result.get("nextPageCursor", "")
            time.sleep(0.05)
            if not cursor or len(raw) < 500:
                break
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["time"]       = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
        df["buy_ratio"]  = df["buyRatio"].astype(float)
        df["sell_ratio"] = df["sellRatio"].astype(float)
        df["ls_ratio"]   = df["buy_ratio"] / df["sell_ratio"].replace(0, np.nan)
        return df[["time","buy_ratio","sell_ratio","ls_ratio"]].sort_values("time").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def load_ls_ratio(symbol: str) -> pd.DataFrame:
    cache = os.path.join(CACHE_DIR, f"{symbol}_ls_ratio.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    df = fetch_ls_ratio(symbol)
    if not df.empty:
        df.to_parquet(cache)
    return df


def section(t):
    print(f"\n{'='*65}\n{t}\n{'='*65}")


if __name__ == "__main__":
    print("Loading spot + perp data...")
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

    print(f"Coins with full data: {len(all_sigs)}")
    print(f"\nFetching L/S ratio data (cached after first run)...")

    merged_parts = []
    no_ls = []

    for sym in sorted(all_sigs.keys()):
        ls_df = load_ls_ratio(sym)
        if ls_df.empty:
            no_ls.append(sym)
            continue

        sig = all_sigs[sym].copy()
        ls  = ls_df.copy()
        ls["date"] = ls["time"].dt.normalize().dt.tz_localize(None)
        ls = ls.groupby("date")[["ls_ratio","buy_ratio"]].last().reset_index()

        merged = sig.merge(ls, on="date", how="inner")
        if len(merged) < MIN_OBS:
            no_ls.append(sym)
            continue

        # Compute ls_z
        roll = merged["ls_ratio"].rolling(LS_Z_WIN)
        merged["ls_z"] = (merged["ls_ratio"] - roll.mean()) / roll.std()

        merged["symbol"] = sym
        merged_parts.append(merged)
        print(f"  {sym:18} {len(merged):4d} obs  "
              f"ls_ratio=[{merged['ls_ratio'].min():.2f},{merged['ls_ratio'].max():.2f}]")

    if not merged_parts:
        print("No coins with L/S data. Exiting.")
        exit()

    pool = pd.concat(merged_parts, ignore_index=True)
    pool = pool.dropna(subset=["ls_z","fwd7"])

    section("1. DATA AVAILABILITY")
    print(f"  Coins with L/S data: {len(merged_parts)}")
    print(f"  Coins without:       {len(no_ls)}")
    print(f"  Total observations:  {len(pool)}")
    print(f"\n  L/S ratio distribution (all coins):")
    ls = pool["ls_ratio"].dropna()
    print(f"    Mean:   {ls.mean():.3f}  (>1 = more longs than shorts)")
    print(f"    Median: {ls.median():.3f}")
    print(f"    p10:    {ls.quantile(.10):.3f}  p90: {ls.quantile(.90):.3f}")

    section("2. IC — L/S ratio signals vs forward returns")
    sigs = ["ls_z", "ls_ratio", "buy_ratio"]
    print(f"\n  {'Signal':12}  {'IC_3d':>8}  {'IC_7d':>8}  {'IC_14d':>8}")
    for sig in sigs:
        if sig not in pool.columns:
            continue
        ics = []
        for h in [3, 7, 14]:
            s = pool[[sig, f"fwd{h}"]].dropna()
            ics.append(rank_ic(s[sig], s[f"fwd{h}"]))
        print(f"  {sig:12}  {ics[0]:>+7.4f}   {ics[1]:>+7.4f}   {ics[2]:>+7.4f}")

    section("3. QUINTILE BREAKDOWN — ls_z vs 7-day forward return")
    sub = pool[["ls_z","fwd7"]].dropna().copy()
    try:
        sub["Q"] = pd.qcut(sub["ls_z"], q=5, labels=[1,2,3,4,5], duplicates="drop")
        print(f"\n  ls_z Q1 = most crowded shorts (bullish?)  Q5 = most crowded longs (bearish?)")
        print(f"  {'Q':>3}  {'n':>5}  {'mean%':>7}  {'wr%':>6}  {'med%':>7}")
        for q, grp in sub.groupby("Q"):
            print(f"  {q:>3}  {len(grp):>5}  {grp['fwd7'].mean():>+6.2f}%  "
                  f"{(grp['fwd7']>0).mean()*100:>5.0f}%  {grp['fwd7'].median():>+6.2f}%")
    except Exception as e:
        print(f"  Quintile error: {e}")

    section("4. COMBINING ls_z WITH EXISTING ENTRY CONDITIONS")
    print("  Does ls_z filter improve the frozen signal?")
    base  = pool[(pool["oi_z"] > 1.0) & (pool["funding_z"] < -0.5)]
    # Low ls_z (crowded shorts) confirms the long setup
    comb  = pool[(pool["oi_z"] > 1.0) & (pool["funding_z"] < -0.5) & (pool["ls_z"] < -0.5)]

    print(f"\n  Base (oi_z>1 + fz<-0.5):           {len(base)} obs")
    print(f"  + ls_z < -0.5 (crowded shorts):    {len(comb)} obs")
    if len(base) > 0:
        print(f"\n  {'Horizon':>10}  {'Combined':>10}  {'Base':>8}  {'Lift':>6}")
        for h in [3, 7, 14]:
            col = f"fwd{h}"
            c = comb[col].dropna()
            b = base[col].dropna()
            if len(c) < 3:
                continue
            lift = c.mean() - b.mean()
            print(f"  {h:>8}d  {c.mean():>+9.2f}%  {b.mean():>+7.2f}%  {lift:>+5.2f}%")

    section("5. PER-COIN IC (time-series check)")
    per_coin_ics = []
    for sym, df in [(s, d) for s, d in zip([p["symbol"].iloc[0] for p in merged_parts],
                                             merged_parts)]:
        s = df[["ls_z","fwd7"]].dropna()
        if len(s) >= 20:
            per_coin_ics.append(rank_ic(s["ls_z"], s["fwd7"]))

    if per_coin_ics:
        ics = pd.Series(per_coin_ics).dropna()
        from scipy import stats as sc
        t_stat, p_val = sc.ttest_1samp(ics, 0)
        stars = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.10 else ""))
        print(f"\n  Per-coin ls_z IC: mean={ics.mean():+.4f}  "
              f"median={ics.median():+.4f}  {(ics>0).mean()*100:.0f}% positive  "
              f"p={p_val:.3f} {stars}")
        print(f"  Interpretation: {'time-series (real effect)' if ics.mean() > 0.01 else 'weak or cross-sectional'}")

    section("SUMMARY")
    print("""
  L/S ratio (what fraction of traders are long vs short):
  - If ls_z IC is negative and significant: crowded longs predict corrections
  - If per-coin IC confirms: it's a real per-coin time-series effect
  - If IC is near zero: the signal doesn't add value beyond what we have

  Trade frequency impact:
  - As a filter ON existing signal: fewer but higher quality trades
  - As a standalone signal: if IC > 0.02, could fire more often than funding_z crossing
""")
    print("Done.")
