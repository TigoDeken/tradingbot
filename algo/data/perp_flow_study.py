"""
Perp flow study — the perpetual futures market leads the spot market.

Hypothesis: Large players enter via perp first (leveraged, sized).
Spot follows. We use three perp signals to detect when shorts are
"trapped" — building, paying carry, and pushing perp below spot —
then measure whether spot rallies over the next 3-14 days.

Signals built from perp data:
  basis      = (perp_close - spot_close) / spot_close * 100  [%]
               negative -> perp below spot -> shorts more aggressive
  oi_delta3  = 3-day % change in open interest
               positive -> new positions being added
  oi_z       = 90-day z-score of OI level
               high -> positioning extreme relative to history
  funding_z  = 90-day z-score of daily funding (already proven)
               negative -> shorts paying carry

"Shorts trapped" = oi growing + funding negative + perp below spot

Run: python -m algo.data.perp_flow_study
"""

import os
import time
import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP
from algo.data.zscore_entry_study import load_all

SESSION   = HTTP(testnet=False)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "swing_study")
OI_Z_WIN  = 90
MIN_OBS   = 90   # minimum daily rows needed per coin
HORIZONS  = [3, 7, 14]


# ── Perp data fetchers ────────────────────────────────────────────────────────

def fetch_perp_oi(symbol: str) -> pd.DataFrame:
    try:
        bars, cursor = [], None
        while True:
            p = dict(category="linear", symbol=symbol, intervalTime="1d", limit=200)
            if cursor:
                p["cursor"] = cursor
            resp   = SESSION.get_open_interest(**p)
            result = resp["result"]
            raw    = result.get("list", [])
            if not raw:
                break
            bars.extend(raw)
            cursor = result.get("nextPageCursor", "")
            time.sleep(0.05)
            if not cursor or len(raw) < 200:
                break
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars)
        df["time"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
        df["oi"]   = df["openInterest"].astype(float)
        return df[["time", "oi"]].sort_values("time").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def load_perp_oi(symbol: str) -> pd.DataFrame:
    cache = os.path.join(CACHE_DIR, f"{symbol}_perp_oi.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    df = fetch_perp_oi(symbol)
    if not df.empty:
        df.to_parquet(cache)
    return df


def fetch_perp_kline(symbol: str) -> pd.DataFrame:
    try:
        bars, end = [], None
        while len(bars) < 1500:
            p = dict(category="linear", symbol=symbol, interval="D", limit=200)
            if end:
                p["end"] = end
            raw = SESSION.get_kline(**p)["result"]["list"]
            if not raw:
                break
            bars.extend(raw)
            end = int(raw[-1][0]) - 1
            time.sleep(0.05)
            if len(raw) < 200:
                break
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars, columns=["ts","open","high","low","close","volume","turnover"])
        df["close"] = df["close"].astype(float)
        df["time"]  = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True)
        return (df[["time","close"]]
                .rename(columns={"close": "perp_close"})
                .sort_values("time")
                .reset_index(drop=True))
    except Exception:
        return pd.DataFrame()


def load_perp_kline(symbol: str) -> pd.DataFrame:
    cache = os.path.join(CACHE_DIR, f"{symbol}_perp_daily.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    df = fetch_perp_kline(symbol)
    if not df.empty:
        df.to_parquet(cache)
    return df


# ── Signal construction ───────────────────────────────────────────────────────

def build_signals(spot_df: pd.DataFrame,
                  oi_df:   pd.DataFrame,
                  perp_df: pd.DataFrame) -> pd.DataFrame:
    spot = spot_df[["time", "close", "funding_z"]].copy()
    spot["date"] = spot["time"].dt.normalize().dt.tz_localize(None)

    oi = oi_df.copy()
    oi["date"] = oi["time"].dt.normalize().dt.tz_localize(None)
    oi = oi.groupby("date")["oi"].last().reset_index()

    perp = perp_df.copy()
    perp["date"] = perp["time"].dt.normalize().dt.tz_localize(None)
    perp = perp.groupby("date")["perp_close"].last().reset_index()

    df = spot.merge(oi, on="date", how="inner").merge(perp, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < MIN_OBS:
        return pd.DataFrame()

    df["basis"]     = (df["perp_close"] - df["close"]) / df["close"] * 100
    df["oi_delta3"] = df["oi"].pct_change(3) * 100

    roll          = df["oi"].rolling(OI_Z_WIN)
    df["oi_z"]    = (df["oi"] - roll.mean()) / roll.std()

    for h in HORIZONS:
        df[f"fwd{h}"] = (df["close"].shift(-h) / df["close"] - 1) * 100

    return df.dropna(subset=["basis", "oi_delta3", "oi_z", "funding_z"]).reset_index(drop=True)


# ── IC helpers ────────────────────────────────────────────────────────────────

def rank_ic(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < 10:
        return np.nan
    return x[mask].rank().corr(y[mask].rank())


def permtest_pvalue(obs: float, x: pd.Series, y: pd.Series, n: int = 2000) -> float:
    rng  = np.random.default_rng(42)
    y_   = y.dropna().values
    x_   = x[y.notna()].values
    null = [rank_ic(pd.Series(x_), pd.Series(rng.permutation(y_))) for _ in range(n)]
    return (np.abs(np.array(null)) >= np.abs(obs)).mean()


def section(t: str):
    print(f"\n{'='*70}")
    print(t)
    print(f"{'='*70}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading spot + funding data...")
    coins = load_all()

    print(f"\nFetching perp data for {len(coins)} coins (cached after first run)...")
    all_signals = {}
    no_perp     = []

    for sym, spot_df in sorted(coins.items()):
        oi_df   = load_perp_oi(sym)
        perp_df = load_perp_kline(sym)
        if oi_df.empty or perp_df.empty:
            no_perp.append(sym)
            continue
        sig = build_signals(spot_df, oi_df, perp_df)
        if len(sig) >= MIN_OBS:
            all_signals[sym] = sig
            print(f"  {sym:18} {len(sig):4d} obs  "
                  f"basis=[{sig['basis'].min():.2f}%, {sig['basis'].max():.2f}%]  "
                  f"oi_range=[{sig['oi'].min():.0f}, {sig['oi'].max():.0f}]")
        else:
            no_perp.append(sym)

    section("1. DATA AVAILABILITY")
    print(f"  Coins with complete perp + spot + funding: {len(all_signals)}")
    print(f"  Coins without perp data or insufficient history: {len(no_perp)}")
    if no_perp:
        print(f"  Missing: {', '.join(sorted(no_perp)[:15])}{'...' if len(no_perp) > 15 else ''}")

    if len(all_signals) < 5:
        print("\nNot enough coins with perp data. Exiting.")
        exit()

    pool_parts = []
    for sym, df in all_signals.items():
        d = df.copy()
        d["symbol"] = sym
        pool_parts.append(d)
    pool = pd.concat(pool_parts, ignore_index=True)

    print(f"\n  Total pooled observations: {len(pool)}")
    print(f"  Date range: {pool['date'].min().date()} to {pool['date'].max().date()}")

    # ── 2. Signal distributions ───────────────────────────────────────────────
    section("2. SIGNAL DISTRIBUTIONS")
    print(f"\n  {'Signal':12}  {'mean':>8}  {'std':>7}  {'p10':>8}  {'p25':>8}  {'med':>8}  {'p75':>8}  {'p90':>8}")
    for sig in ["basis", "oi_delta3", "oi_z", "funding_z"]:
        s = pool[sig].dropna()
        print(f"  {sig:12}  {s.mean():>+7.3f}%  {s.std():>6.3f}  "
              f"{s.quantile(.10):>+7.3f}  {s.quantile(.25):>+7.3f}  "
              f"{s.median():>+7.3f}  {s.quantile(.75):>+7.3f}  {s.quantile(.90):>+7.3f}")

    # ── 3. IC per signal ──────────────────────────────────────────────────────
    section("3. INFORMATION COEFFICIENT — each signal vs forward spot return")
    print(f"\n  {'Signal':12}  {'IC_3d':>8}  {'IC_7d':>8}  {'IC_14d':>8}  {'p-val(7d)':>10}")
    print(f"  {'-'*58}")

    ic_results = {}
    for sig in ["basis", "oi_delta3", "oi_z", "funding_z"]:
        row = []
        for h in HORIZONS:
            sub  = pool[[sig, f"fwd{h}"]].dropna()
            obs  = rank_ic(sub[sig], sub[f"fwd{h}"])
            row.append(obs)
        pval = permtest_pvalue(row[1], pool[sig], pool["fwd7"])
        ic_results[sig] = {"ic3": row[0], "ic7": row[1], "ic14": row[2], "pval7": pval}
        stars = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.10 else ""))
        print(f"  {sig:12}  {row[0]:>+7.4f}   {row[1]:>+7.4f}   {row[2]:>+7.4f}   "
              f"{pval:>8.3f} {stars}")

    print(f"\n  IC guide: |IC| > 0.05 = strong signal  |IC| > 0.02 = tradeable")
    print(f"  p-value: probability IC is due to chance (< 0.05 = significant)")

    # ── 4. Quintile analysis ──────────────────────────────────────────────────
    section("4. QUINTILE BREAKDOWN — 7-day forward return by signal quintile")
    print(f"  (Q1 = lowest signal value, Q5 = highest)")

    for sig in ["basis", "oi_delta3", "oi_z", "funding_z"]:
        sub = pool[[sig, "fwd7"]].dropna().copy()
        try:
            sub["Q"] = pd.qcut(sub[sig], q=5, labels=[1,2,3,4,5], duplicates="drop")
        except Exception:
            continue
        print(f"\n  {sig}:  Q1-Q5 spread = "
              f"{sub[sub['Q']==1]['fwd7'].mean() - sub[sub['Q']==5]['fwd7'].mean():+.2f}%")
        print(f"  {'Q':>3}  {'n':>4}  {'mean%':>7}  {'wr%':>6}  {'med%':>7}")
        for q, grp in sub.groupby("Q"):
            print(f"  {q:>3}  {len(grp):>4}  {grp['fwd7'].mean():>+6.2f}%  "
                  f"{(grp['fwd7']>0).mean()*100:>5.0f}%  {grp['fwd7'].median():>+6.2f}%")

    # ── 5. "Shorts trapped" combination ──────────────────────────────────────
    section("5. 'SHORTS TRAPPED' — all three conditions simultaneously")
    print("""
  Conditions:
    oi_delta3 > 2%    (OI grew >2% in 3 days -> new shorts being added)
    funding_z < -0.5  (shorts paying negative funding)
    basis < -0.1%     (perp trading below spot -> shorts more aggressive in perp)
    """)

    pool["oi_growing"]    = pool["oi_delta3"] > 2.0
    pool["funding_neg"]   = pool["funding_z"] < -0.5
    pool["perp_discount"] = pool["basis"] < -0.1
    pool["trapped"]       = pool["oi_growing"] & pool["funding_neg"] & pool["perp_discount"]

    trapped  = pool[pool["trapped"]]
    baseline = pool[~pool["trapped"]]

    print(f"  'Trapped' days: {len(trapped)} ({len(trapped)/len(pool)*100:.1f}% of all observations)")
    print(f"  Baseline days:  {len(baseline)}")
    print(f"  Coins with at least one 'trapped' day: "
          f"{pool[pool['trapped']]['symbol'].nunique()}")

    print(f"\n  {'Horizon':>10}  {'Trapped mean':>13}  {'Trapped wr':>11}  "
          f"{'Baseline mean':>14}  {'Baseline wr':>12}  {'Edge':>7}")
    print(f"  {'-'*75}")
    for h in HORIZONS:
        t_col = f"fwd{h}"
        t_ret = trapped[t_col].dropna()
        b_ret = baseline[t_col].dropna()
        if len(t_ret) < 5:
            continue
        edge = t_ret.mean() - b_ret.mean()
        print(f"  {h:>8}d  {t_ret.mean():>+11.2f}%  "
              f"{(t_ret>0).mean()*100:>10.0f}%  "
              f"{b_ret.mean():>+12.2f}%  "
              f"{(b_ret>0).mean()*100:>11.0f}%  "
              f"{edge:>+6.2f}%")

    # ── 6. Combination IC ─────────────────────────────────────────────────────
    section("6. COMBINED SIGNAL — weighted rank score")
    print("""
  Score = rank(funding_z) + rank(basis) + rank(-oi_delta3)
  (All signals flipped so that low = favorable condition)
  Low score = funding calm + perp at discount + OI not growing = quiet setup
  High score = crowd excited + perp premium + OI building = crowded
  """)

    pool["combo_score"] = (
        pool["funding_z"].rank() +
        pool["basis"].rank() +
        (-pool["oi_delta3"]).rank()
    )

    for h in HORIZONS:
        obs  = rank_ic(pool["combo_score"], pool[f"fwd{h}"])
        pval = permtest_pvalue(obs, pool["combo_score"], pool[f"fwd{h}"])
        stars = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.10 else ""))
        print(f"  Combined score IC at {h:>2}d: {obs:>+.4f}  p={pval:.3f} {stars}")

    # Quintile of combined score
    sub = pool[["combo_score", "fwd7"]].dropna().copy()
    try:
        sub["Q"] = pd.qcut(sub["combo_score"], q=5, labels=[1,2,3,4,5], duplicates="drop")
        print(f"\n  Combined score quintiles (7-day forward return):")
        print(f"  {'Q':>3}  {'n':>4}  {'mean%':>7}  {'wr%':>6}  {'med%':>7}  {'interpretation'}")
        labels = {1:"most favorable", 2:"", 3:"neutral", 4:"", 5:"most crowded/risky"}
        for q, grp in sub.groupby("Q"):
            print(f"  {q:>3}  {len(grp):>4}  {grp['fwd7'].mean():>+6.2f}%  "
                  f"{(grp['fwd7']>0).mean()*100:>5.0f}%  {grp['fwd7'].median():>+6.2f}%  "
                  f"{labels.get(q, '')}")
    except Exception:
        pass

    # ── 7. Summary ────────────────────────────────────────────────────────────
    section("7. WHAT THIS TELLS US")

    best_sig = max(ic_results, key=lambda k: abs(ic_results[k]["ic7"]))
    best_ic  = ic_results[best_sig]["ic7"]
    best_p   = ic_results[best_sig]["pval7"]

    print(f"""
  Strongest single signal: {best_sig}  IC(7d)={best_ic:+.4f}  p={best_p:.3f}

  Does the perp market lead spot?
    If basis IC is significantly negative: yes, perp shorts predict spot rallies.
    If oi_delta3 IC is significant: OI buildup direction matters.
    If combined score IC > individual: signals are complementary.

  The "shorts trapped" setup:
    Is it rare? (few % of days = high-conviction filter)
    Is the edge large? (compare trapped mean vs baseline mean)

  If any of these signals produce IC > 0.05 with p < 0.05:
    We have a new, non-obvious signal from the STRUCTURE of the market,
    not from an indicator threshold.
    """)

    print("Done.")
