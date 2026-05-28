"""
Funding z-score universe study.

Questions answered:
  1. How many coins are in Q1/Q2 (calm funding) at any given time?
  2. How long does a coin stay in Q1/Q2 once it enters?
  3. Are z-scores correlated across coins? (do they all move together?)
  4. What fraction of time is capital deployed?

This tells us whether running the full universe in parallel is viable,
and how much of our capital we can realistically deploy at any time.

Run: python -m algo.data.funding_universe_study
"""

import os
import time
import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP

SESSION   = HTTP(testnet=False)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "swing_study")

MIN_VOL_USD = 500_000
MAX_VOL_USD = 10_000_000
FZ_WINDOW   = 90      # days rolling window for z-score
CALM_THRESH = -0.25   # z-score below this = Q1/Q2 (calm, crowd absent)
MIN_BARS    = 120     # minimum daily bars needed


# ── Universe ──────────────────────────────────────────────────────────────────

def get_universe() -> list[str]:
    resp = SESSION.get_tickers(category="spot")
    out  = []
    for t in resp["result"]["list"]:
        if not t["symbol"].endswith("USDT"):
            continue
        try:
            vol = float(t["turnover24h"])
        except (KeyError, ValueError, TypeError):
            continue
        if MIN_VOL_USD <= vol <= MAX_VOL_USD:
            out.append(t["symbol"])
    return sorted(out)


# ── Funding fetch ─────────────────────────────────────────────────────────────

def fetch_funding(symbol: str) -> pd.DataFrame:
    try:
        bars, end = [], None
        while True:
            p = dict(category="linear", symbol=symbol, limit=200)
            if end:
                p["endTime"] = end
            raw = SESSION.get_funding_rate_history(**p)["result"]["list"]
            if not raw:
                break
            bars.extend(raw)
            end = int(raw[-1]["fundingRateTimestamp"]) - 1
            time.sleep(0.05)
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars)
        df["time"] = pd.to_datetime(df["fundingRateTimestamp"].astype(int), unit="ms", utc=True)
        df["funding_rate"] = df["fundingRate"].astype(float) * 100
        return df[["time", "funding_rate"]].sort_values("time").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def load_funding(symbol: str) -> pd.DataFrame:
    cache = os.path.join(CACHE_DIR, f"{symbol}_funding.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    df = fetch_funding(symbol)
    if not df.empty:
        df.to_parquet(cache)
    return df


# ── Z-score computation ───────────────────────────────────────────────────────

def funding_zscore_daily(funding_df: pd.DataFrame) -> pd.Series:
    """Aggregate funding to daily, compute 90-day rolling z-score."""
    funding_df = funding_df.copy()
    funding_df["date"] = funding_df["time"].dt.normalize()
    daily = funding_df.groupby("date")["funding_rate"].sum()
    roll  = daily.rolling(FZ_WINDOW)
    zscore = (daily - roll.mean()) / roll.std()
    return zscore.dropna()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("Fetching universe...")
    symbols = get_universe()
    print(f"Found {len(symbols)} coins in ${MIN_VOL_USD/1e6:.1f}M-${MAX_VOL_USD/1e6:.0f}M daily volume tier\n")

    print("Loading funding data (fetching missing)...")
    zscore_series = {}

    for symbol in symbols:
        funding_df = load_funding(symbol)
        if funding_df.empty:
            continue
        zs = funding_zscore_daily(funding_df)
        if len(zs) < MIN_BARS:
            continue
        zscore_series[symbol] = zs
        print(f"  {symbol:18} {len(zs)} days  z-score range: "
              f"[{zs.min():.2f}, {zs.max():.2f}]  mean={zs.mean():.2f}")

    print(f"\n{len(zscore_series)} coins with sufficient funding history\n")

    if len(zscore_series) < 3:
        print("Not enough coins. Exiting.")
        exit()

    # Align all z-scores to common daily index
    df_z = pd.DataFrame(zscore_series)
    df_z = df_z.sort_index().dropna(how="all")

    # ── 1. How many coins are calm at any given time? ─────────────────────────
    calm = (df_z < CALM_THRESH)          # True = coin is in Q1/Q2 (calm)
    n_calm = calm.sum(axis=1)            # count per day
    n_coins_present = df_z.notna().sum(axis=1)  # coins with data that day

    print(f"{'='*70}")
    print(f"HOW MANY COINS ARE IN Q1/Q2 (z-score < {CALM_THRESH}) AT ANY TIME?")
    print(f"{'='*70}")
    print(f"  Based on {len(zscore_series)} coins  |  "
          f"{df_z.index[0].date()} to {df_z.index[-1].date()}")
    print(f"\n  Distribution of concurrent calm coins:")
    print(f"    Min:    {n_calm.min():.0f}")
    print(f"    p10:    {n_calm.quantile(.10):.0f}")
    print(f"    p25:    {n_calm.quantile(.25):.0f}")
    print(f"    Median: {n_calm.median():.0f}")
    print(f"    p75:    {n_calm.quantile(.75):.0f}")
    print(f"    p90:    {n_calm.quantile(.90):.0f}")
    print(f"    Max:    {n_calm.max():.0f}")
    print(f"\n  Average concurrent calm coins: {n_calm.mean():.1f} "
          f"out of {n_coins_present.mean():.1f} with data")
    print(f"  Average fraction of universe that is calm: "
          f"{(n_calm / n_coins_present.replace(0, np.nan)).mean()*100:.1f}%")

    # Breakdown by bucket
    print(f"\n  Frequency of concurrent calm coin count:")
    buckets = [(0,0,"Zero (no positions)"), (1,2,"1-2 coins"), (3,5,"3-5 coins"),
               (6,10,"6-10 coins"), (11,99,"11+ coins")]
    for lo, hi, label in buckets:
        mask  = (n_calm >= lo) & (n_calm <= hi)
        days  = mask.sum()
        pct   = days / len(n_calm) * 100
        print(f"    {label:25} {days:4d} days  ({pct:.1f}% of time)")

    # ── 2. How long does a coin stay calm once it enters? ────────────────────
    print(f"\n{'='*70}")
    print("HOW LONG DOES A COIN STAY IN Q1/Q2 ONCE IT ENTERS?")
    print(f"{'='*70}")
    run_lengths = []
    for symbol in zscore_series:
        col = calm[symbol].dropna().astype(int)
        in_run = 0
        for val in col:
            if val == 1:
                in_run += 1
            elif in_run > 0:
                run_lengths.append(in_run)
                in_run = 0
        if in_run > 0:
            run_lengths.append(in_run)

    if run_lengths:
        runs = pd.Series(run_lengths)
        print(f"  Based on {len(runs)} calm periods across all coins")
        print(f"    Min:    {runs.min():.0f} days")
        print(f"    p25:    {runs.quantile(.25):.0f} days")
        print(f"    Median: {runs.median():.0f} days")
        print(f"    p75:    {runs.quantile(.75):.0f} days")
        print(f"    p90:    {runs.quantile(.90):.0f} days")
        print(f"    Max:    {runs.max():.0f} days")
        print(f"  Periods <= 3 days (too short to trade): "
              f"{(runs <= 3).sum()} ({(runs <= 3).mean()*100:.0f}%)")
        print(f"  Periods >= 7 days (tradeable):          "
              f"{(runs >= 7).sum()} ({(runs >= 7).mean()*100:.0f}%)")

    # ── 3. Are z-scores correlated? (do coins move together?) ────────────────
    print(f"\n{'='*70}")
    print("ARE Z-SCORES CORRELATED ACROSS COINS?")
    print(f"(If yes: positions cluster, capital not diversified)")
    print(f"{'='*70}")
    corr = df_z.corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    flat  = upper.stack()
    print(f"  Pairwise z-score correlations across {len(zscore_series)} coins:")
    print(f"    Mean:   {flat.mean():.3f}")
    print(f"    Median: {flat.median():.3f}")
    print(f"    p25:    {flat.quantile(.25):.3f}")
    print(f"    p75:    {flat.quantile(.75):.3f}")
    print(f"    Fraction with correlation > 0.5: {(flat > 0.5).mean()*100:.1f}%")
    print(f"    Fraction with correlation > 0.7: {(flat > 0.7).mean()*100:.1f}%")
    if flat.mean() > 0.5:
        print(f"\n  WARNING: high correlation means positions are NOT independent.")
        print(f"  When one coin enters Q1/Q2, most others likely do too.")
        print(f"  Diversification benefit is limited.")
    else:
        print(f"\n  Low-moderate correlation: positions are reasonably independent.")
        print(f"  Running multiple coins adds real diversification.")

    # ── 4. Capital deployment ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CAPITAL DEPLOYMENT")
    print(f"(If each position uses equal allocation across N concurrent coins)")
    print(f"{'='*70}")
    avg_concurrent = n_calm.mean()
    for alloc_pct in [5, 10, 20]:
        deployed = min(avg_concurrent * alloc_pct, 100)
        print(f"  {alloc_pct}% per position x {avg_concurrent:.1f} avg concurrent = "
              f"{deployed:.0f}% capital deployed on average")

    print(f"\n  Days with 0 positions (capital fully idle): "
          f"{(n_calm == 0).sum()} ({(n_calm == 0).mean()*100:.1f}% of time)")

    # ── 5. Entries per year per coin ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TRADE FREQUENCY")
    print(f"(New calm entries per coin per year = potential new positions)")
    print(f"{'='*70}")
    entries_per_coin = []
    for symbol in zscore_series:
        col    = calm[symbol].dropna().astype(int)
        days   = len(col)
        # new entry = transition from not-calm to calm
        new_entries = ((col == 1) & (col.shift(1) == 0)).sum()
        per_year    = new_entries / days * 365
        entries_per_coin.append(per_year)
        print(f"  {symbol:18} {new_entries:3d} entries  {per_year:.1f}/yr")

    ep = pd.Series(entries_per_coin)
    print(f"\n  Across universe: median {ep.median():.1f} entries/coin/yr  "
          f"mean {ep.mean():.1f}")
    print(f"  With {len(zscore_series)} coins: ~{ep.mean()*len(zscore_series):.0f} "
          f"total new positions per year")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"""
  Universe: {len(zscore_series)} coins with funding data in the $500K-$10M tier
  Average concurrent calm coins: {n_calm.mean():.1f}
  Median calm run length: {pd.Series(run_lengths).median() if run_lengths else 0:.0f} days
  Z-score correlation (mean): {flat.mean():.2f}

  This tells you:
  - How many positions you'll hold at once (sizing guide)
  - Whether coins enter Q1/Q2 independently or together
  - How often you get new entry opportunities
    """)

    print("Done.")
