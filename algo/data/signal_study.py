"""
Signal study: funding rate, OI, and volume as predictors of BTC 1H forward returns.

Tests each signal for:
  - Quintile return table (Q1 to Q5 vs forward return)
  - Spearman IC: does the signal consistently rank future returns?
  - Permutation test: is the IC larger than random chance?
  - Consistency: does the effect hold in both halves of the data?

On incremental value / avoiding parameter fitting:
  - These tests measure raw predictive power of the signal in isolation.
  - To prove a signal adds value to a trading system, the correct method is:
      1. Build a baseline entry rule (no signal filter).
      2. Pre-commit a single filter threshold (e.g. "only trade when OI Q1").
      3. Test baseline vs filtered strategy on HELD-OUT data only.
      4. The signal passes only if the improvement survives out-of-sample.
  - This file establishes whether signals are worth testing as filters at all.

Run: python -m algo.data.signal_study
"""

import os
import time
import numpy as np
import pandas as pd
from scipy import stats
from pybit.unified_trading import HTTP

SESSION    = HTTP(testnet=False)
SYMBOL     = "BTCUSDT"
CACHE      = os.path.join(os.path.dirname(__file__), "cache", "btc_1h_study.parquet")
HORIZONS   = [4, 24, 168]
MIN_EDGE   = 0.40   # minimum Q5-Q1 spread (%) to flag as potentially useful
N_PERMS    = 1000   # permutation test iterations
COST_RT    = 0.20   # round-trip cost %


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_ohlcv() -> pd.DataFrame:
    print("  Fetching 1H OHLCV...")
    bars, end = [], None
    while True:
        p = dict(category="spot", symbol=SYMBOL, interval="60", limit=200)
        if end:
            p["end"] = end
        raw = SESSION.get_kline(**p)["result"]["list"]
        if not raw:
            break
        bars.extend(raw)
        end = int(raw[-1][0]) - 1
        time.sleep(0.05)
    df = pd.DataFrame(bars, columns=["ts","open","high","low","close","volume","turnover"])
    df = df.astype({c: float for c in ["open","high","low","close","volume","turnover"]})
    df["ts"] = df["ts"].astype(int)
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.sort_values("time").reset_index(drop=True)


def fetch_funding() -> pd.DataFrame:
    print("  Fetching funding rate...")
    bars, end = [], None
    while True:
        p = dict(category="linear", symbol=SYMBOL, limit=200)
        if end:
            p["endTime"] = end
        raw = SESSION.get_funding_rate_history(**p)["result"]["list"]
        if not raw:
            break
        bars.extend(raw)
        end = int(raw[-1]["fundingRateTimestamp"]) - 1
        time.sleep(0.05)
    df = pd.DataFrame(bars)
    df["time"] = pd.to_datetime(df["fundingRateTimestamp"].astype(int), unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float) * 100
    return df[["time","funding_rate"]].sort_values("time").reset_index(drop=True)


def fetch_oi() -> pd.DataFrame:
    print("  Fetching open interest...")
    bars, end = [], None
    while True:
        p = dict(category="linear", symbol=SYMBOL, intervalTime="1h", limit=200)
        if end:
            p["endTime"] = end
        raw = SESSION.get_open_interest(**p)["result"]["list"]
        if not raw:
            break
        bars.extend(raw)
        end = int(raw[-1]["timestamp"]) - 1
        time.sleep(0.05)
    df = pd.DataFrame(bars)
    df["time"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    df["oi"] = df["openInterest"].astype(float)
    return df[["time","oi"]].sort_values("time").reset_index(drop=True)


# ── Build dataset ─────────────────────────────────────────────────────────────

def build(ohlcv: pd.DataFrame, funding: pd.DataFrame, oi: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv.set_index("time")

    df["funding_rate"] = funding.set_index("time")["funding_rate"].reindex(df.index, method="ffill")
    df["oi"]           = oi.set_index("time")["oi"].reindex(df.index, method="ffill")

    df["rel_vol"]        = df["volume"] / df["volume"].rolling(20).mean()
    fr_roll              = df["funding_rate"].rolling(2160)
    df["funding_zscore"] = (df["funding_rate"] - fr_roll.mean()) / fr_roll.std()
    df["oi_chg_24h"]     = df["oi"].pct_change(24) * 100

    df["above_200d"] = df["close"] > df["close"].rolling(4800).mean()

    for h in HORIZONS:
        df[f"fwd_{h}h"] = (df["close"].shift(-h) / df["close"] - 1) * 100

    required = ["rel_vol","funding_zscore","oi_chg_24h"] + [f"fwd_{h}h" for h in HORIZONS]
    return df.dropna(subset=required)


# ── Statistical tests ─────────────────────────────────────────────────────────

def spearman_ic(signal: np.ndarray, fwd: np.ndarray) -> tuple[float, float]:
    """Spearman rank correlation between signal and forward return, with p-value."""
    ic, pval = stats.spearmanr(signal, fwd)
    return float(ic), float(pval)


def permutation_test(signal: np.ndarray, fwd: np.ndarray, n: int = N_PERMS) -> float:
    """
    Permute the signal randomly n times.
    Returns the fraction of permutations with |IC| >= real |IC|.
    A low p-value means the real IC is unlikely to occur by chance.
    """
    real_ic, _ = spearman_ic(signal, fwd)
    rng = np.random.default_rng(42)
    count = sum(
        abs(spearman_ic(rng.permutation(signal), fwd)[0]) >= abs(real_ic)
        for _ in range(n)
    )
    return count / n


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse_signal(df: pd.DataFrame, signal: str, label: str):
    print(f"\n{'='*66}")
    print(f"SIGNAL: {label}")
    print(f"{'='*66}")

    for regime_label, mask in [
        ("ALL",       pd.Series(True, index=df.index)),
        ("BULL only", df["above_200d"]),
        ("BEAR only", ~df["above_200d"]),
    ]:
        sub = df[mask].copy()
        if len(sub) < 300:
            continue

        sub["q"] = pd.qcut(
            sub[signal].rank(method="first"), 5,
            labels=["Q1","Q2","Q3","Q4","Q5"]
        )

        print(f"\n  {regime_label}  ({len(sub)} bars)")

        # ── Quintile table ───────────────────────────────────────────────────
        header = f"  {'':5}" + "".join(f"  {h:>4}h_ret    WR" for h in HORIZONS) + "      N"
        print(header)

        q_rets = {}
        for q in ["Q1","Q2","Q3","Q4","Q5"]:
            grp = sub[sub["q"] == q]
            row = f"  {q:5}"
            q_rets[q] = {}
            for h in HORIZONS:
                col = f"fwd_{h}h"
                ret = grp[col].mean()
                wr  = (grp[col] > 0).mean() * 100
                q_rets[q][h] = ret
                row += f"  {ret:+.2f}%  {wr:.0f}%"
            row += f"  {len(grp):6}"
            print(row)

        # ── Q5-Q1 spread (bug-fixed) ─────────────────────────────────────────
        print(f"\n  Q5-Q1 spread (edge above cost threshold = {MIN_EDGE}%):")
        for h in HORIZONS:
            spread = q_rets["Q5"][h] - q_rets["Q1"][h]
            flag   = "  <<" if abs(spread) >= MIN_EDGE else ""
            print(f"    {h:>4}h: {spread:+.3f}%{flag}")

        # ── Spearman IC + permutation test ───────────────────────────────────
        sig_vals = sub[signal].values
        print(f"\n  Spearman IC (signal vs forward return):")
        for h in HORIZONS:
            fwd_vals = sub[f"fwd_{h}h"].values
            ic, pval = spearman_ic(sig_vals, fwd_vals)
            p_perm   = permutation_test(sig_vals, fwd_vals)
            sig_flag = "  ** significant" if p_perm < 0.05 else ""
            print(f"    {h:>4}h: IC={ic:+.4f}  p(parametric)={pval:.3f}  "
                  f"p(permutation)={p_perm:.3f}{sig_flag}")

    # ── Consistency: does effect hold in both halves? ─────────────────────────
    mid = len(df) // 2
    print(f"\n  Consistency check (Q5-Q1 spread, 24h forward):")
    for name, half in [("First half ", df.iloc[:mid]), ("Second half", df.iloc[mid:])]:
        half = half.copy()
        half["q"] = pd.qcut(
            half[signal].rank(method="first"), 5,
            labels=["Q1","Q2","Q3","Q4","Q5"]
        )
        q5 = half[half["q"] == "Q5"]["fwd_24h"].mean()
        q1 = half[half["q"] == "Q1"]["fwd_24h"].mean()
        yr0, yr1 = half.index[0].year, half.index[-1].year
        print(f"    {name} ({yr0}-{yr1}): Q5-Q1 = {q5 - q1:+.3f}%")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)

    if os.path.exists(CACHE):
        print(f"Loading cached dataset...")
        df = pd.read_parquet(CACHE)
    else:
        print("Fetching data (this takes a few minutes)...")
        df = build(fetch_ohlcv(), fetch_funding(), fetch_oi())
        df.to_parquet(CACHE)
        print(f"Cached.")

    print(f"\nDataset: {len(df)} rows  |  "
          f"{df.index[0].date()} to {df.index[-1].date()}  |  "
          f"Bull: {df['above_200d'].sum()}  Bear: {(~df['above_200d']).sum()}")

    for col, label in [
        ("rel_vol",        "Relative Volume (vol / 20h rolling mean)"),
        ("funding_rate",   "Funding Rate raw (%)"),
        ("funding_zscore", "Funding Rate Z-Score (90d window)"),
        ("oi_chg_24h",     "Open Interest 24h change (%)"),
    ]:
        analyse_signal(df, col, label)

    print(f"\n\n{'='*66}")
    print("HOW TO USE THESE RESULTS")
    print(f"{'='*66}")
    print("""
  A signal passes this study if:
    1. Q5-Q1 spread > {MIN_EDGE}% at a relevant horizon (raw predictive power)
    2. Permutation p-value < 0.05 (not explainable by chance)
    3. Effect holds in both halves of the data (consistency)

  Passing this study means: worth testing as a FILTER on a price-based entry.
  It does NOT mean: trade this signal directly.

  The correct incremental value test (next step):
    - Build a baseline entry rule (e.g. momentum crossover)
    - Measure baseline Sharpe, win rate, avg trade return on held-out data
    - Apply this signal as a go/no-go filter with ONE pre-committed threshold
    - Measure improvement on the SAME held-out data
    - Only accept if improvement is material and consistent
    - Never re-optimise the threshold after seeing held-out results
    """.format(MIN_EDGE=MIN_EDGE))

    print("Done.")
