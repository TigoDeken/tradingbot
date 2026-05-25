"""
Quick comparison: OI 72h change vs OI 168h change as signals.
Loads existing caches — no API calls.

Run: python -m algo.data.oi_168h_test
"""

import os
import pandas as pd
from scipy import stats
import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
HORIZONS  = [4, 24, 168]
N_PERMS   = 500

SYMBOLS = {
    "ETHUSDT":  "$200M tier",
    "SOLUSDT":  "$50M tier",
    "ADAUSDT":  "$10M tier",
    "UNIUSDT":  "$1M tier",
    "ALGOUSDT": "$300K tier",
}


def spearman_ic(signal, fwd):
    ic, pval = stats.spearmanr(signal, fwd)
    return float(ic), float(pval)


def permutation_test(signal, fwd, n=N_PERMS):
    real_ic, _ = spearman_ic(signal, fwd)
    rng = np.random.default_rng(42)
    count = sum(
        abs(spearman_ic(rng.permutation(signal), fwd)[0]) >= abs(real_ic)
        for _ in range(n)
    )
    return count / n


def analyse(df, signal):
    sub = df.dropna(subset=[signal]).copy()
    sub["q"] = pd.qcut(sub[signal].rank(method="first"), 5, labels=["Q1","Q2","Q3","Q4","Q5"])

    q_rets = {}
    for q in ["Q1","Q2","Q3","Q4","Q5"]:
        grp = sub[sub["q"] == q]
        q_rets[q] = {h: grp[f"fwd_{h}h"].mean() for h in HORIZONS}

    print(f"    {'':5}" + "".join(f"  {h:>4}h_ret    WR" for h in HORIZONS))
    for q in ["Q1","Q2","Q3","Q4","Q5"]:
        grp = sub[sub["q"] == q]
        row = f"    {q:5}"
        for h in HORIZONS:
            ret = q_rets[q][h]
            wr  = (grp[f"fwd_{h}h"] > 0).mean() * 100
            row += f"  {ret:+.2f}%  {wr:.0f}%"
        print(row)

    print(f"\n    Q5-Q1 spread:")
    for h in HORIZONS:
        spread = q_rets["Q5"][h] - q_rets["Q1"][h]
        flag   = "  <<" if abs(spread) >= 0.4 else ""
        print(f"      {h:>4}h: {spread:+.3f}%{flag}")

    print(f"\n    Spearman IC:")
    for h in HORIZONS:
        ic, pval = spearman_ic(sub[signal].values, sub[f"fwd_{h}h"].values)
        p_perm   = permutation_test(sub[signal].values, sub[f"fwd_{h}h"].values)
        sig = "  **" if p_perm < 0.05 else ""
        print(f"      {h:>4}h: IC={ic:+.4f}  p(perm)={p_perm:.3f}{sig}")

    mid = len(sub) // 2
    print(f"\n    Consistency (Q5-Q1, 24h fwd):")
    for name, half in [("First half ", sub.iloc[:mid]), ("Second half", sub.iloc[mid:])]:
        half = half.copy()
        half["q"] = pd.qcut(half[signal].rank(method="first"), 5, labels=["Q1","Q2","Q3","Q4","Q5"])
        q5 = half[half["q"] == "Q5"]["fwd_24h"].mean()
        q1 = half[half["q"] == "Q1"]["fwd_24h"].mean()
        yr0, yr1 = half.index[0].year, half.index[-1].year
        print(f"      {name} ({yr0}-{yr1}): {q5-q1:+.3f}%")


def run(symbol, tier):
    cache = os.path.join(CACHE_DIR, f"{symbol.lower()}_1h_vtier.parquet")
    if not os.path.exists(cache):
        print(f"  No cache found — skipping")
        return

    df = pd.read_parquet(cache)
    df["oi_chg_168h"] = df["oi"].pct_change(168) * 100
    df = df.dropna(subset=["oi_chg_168h"])

    print(f"\n{'#'*60}")
    print(f"  {symbol}  ({tier})  —  {len(df)} bars")
    print(f"{'#'*60}")

    for col, label in [("oi_chg_72h", "OI 72h change"), ("oi_chg_168h", "OI 168h change")]:
        print(f"\n  -- {label} --")
        analyse(df, col)


if __name__ == "__main__":
    for symbol, tier in SYMBOLS.items():
        run(symbol, tier)
    print("\nDone.")
