"""
Signal study for multiple symbols beyond BTC.
Same methodology as signal_study.py — run per-symbol, compare results.

Run: python -m algo.data.multi_signal_study
"""

import os
import time
import numpy as np
import pandas as pd
from scipy import stats
from pybit.unified_trading import HTTP

SESSION   = HTTP(testnet=False)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
HORIZONS  = [4, 24, 168]
MIN_EDGE  = 0.40
N_PERMS   = 1000

SYMBOLS = {
    "AVAXUSDT": "mid-cap",
    "LINKUSDT": "mid-cap",
    "ATOMUSDT": "small-cap",
    "NEARUSDT": "small-cap",
}


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str) -> pd.DataFrame:
    print(f"  [{symbol}] Fetching 1H OHLCV...")
    bars, end = [], None
    while True:
        p = dict(category="spot", symbol=symbol, interval="60", limit=200)
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


def fetch_funding(symbol: str) -> pd.DataFrame:
    print(f"  [{symbol}] Fetching funding rate...")
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
    df = pd.DataFrame(bars)
    df["time"] = pd.to_datetime(df["fundingRateTimestamp"].astype(int), unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float) * 100
    return df[["time","funding_rate"]].sort_values("time").reset_index(drop=True)


def fetch_oi(symbol: str) -> pd.DataFrame:
    print(f"  [{symbol}] Fetching open interest...")
    bars, end = [], None
    while True:
        p = dict(category="linear", symbol=symbol, intervalTime="1h", limit=200)
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


# ── Build ─────────────────────────────────────────────────────────────────────

def build(ohlcv, funding, oi) -> pd.DataFrame:
    df = ohlcv.set_index("time")
    df["funding_rate"] = funding.set_index("time")["funding_rate"].reindex(df.index, method="ffill")
    df["oi"]           = oi.set_index("time")["oi"].reindex(df.index, method="ffill")

    df["rel_vol"]        = df["volume"] / df["volume"].rolling(20).mean()
    fr_roll              = df["funding_rate"].rolling(2160)
    df["funding_zscore"] = (df["funding_rate"] - fr_roll.mean()) / fr_roll.std()
    df["oi_chg_24h"]     = df["oi"].pct_change(24) * 100
    df["above_200d"]     = df["close"] > df["close"].rolling(4800).mean()

    for h in HORIZONS:
        df[f"fwd_{h}h"] = (df["close"].shift(-h) / df["close"] - 1) * 100

    required = ["rel_vol","funding_zscore","oi_chg_24h"] + [f"fwd_{h}h" for h in HORIZONS]
    return df.dropna(subset=required)


# ── Stats ─────────────────────────────────────────────────────────────────────

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


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyse_signal(df, signal, label):
    print(f"\n  -- {label} --")

    for regime_label, mask in [
        ("ALL",       pd.Series(True, index=df.index)),
        ("BULL only", df["above_200d"]),
        ("BEAR only", ~df["above_200d"]),
    ]:
        sub = df[mask].copy()
        if len(sub) < 200:
            continue

        sub["q"] = pd.qcut(
            sub[signal].rank(method="first"), 5,
            labels=["Q1","Q2","Q3","Q4","Q5"]
        )

        print(f"\n    {regime_label}  ({len(sub)} bars)")
        header = f"    {'':5}" + "".join(f"  {h:>4}h_ret    WR" for h in HORIZONS) + "      N"
        print(header)

        q_rets = {}
        for q in ["Q1","Q2","Q3","Q4","Q5"]:
            grp = sub[sub["q"] == q]
            row = f"    {q:5}"
            q_rets[q] = {}
            for h in HORIZONS:
                col = f"fwd_{h}h"
                ret = grp[col].mean()
                wr  = (grp[col] > 0).mean() * 100
                q_rets[q][h] = ret
                row += f"  {ret:+.2f}%  {wr:.0f}%"
            row += f"  {len(grp):6}"
            print(row)

        print(f"\n    Q5-Q1 spread (edge threshold = {MIN_EDGE}%):")
        for h in HORIZONS:
            spread = q_rets["Q5"][h] - q_rets["Q1"][h]
            flag   = "  <<" if abs(spread) >= MIN_EDGE else ""
            print(f"      {h:>4}h: {spread:+.3f}%{flag}")

        sig_vals = sub[signal].values
        print(f"\n    Spearman IC:")
        for h in HORIZONS:
            fwd_vals = sub[f"fwd_{h}h"].values
            ic, pval = spearman_ic(sig_vals, fwd_vals)
            p_perm   = permutation_test(sig_vals, fwd_vals)
            sig_flag = "  ** significant" if p_perm < 0.05 else ""
            print(f"      {h:>4}h: IC={ic:+.4f}  p(param)={pval:.3f}  "
                  f"p(perm)={p_perm:.3f}{sig_flag}")

    mid = len(df) // 2
    print(f"\n    Consistency (Q5-Q1 spread, 24h forward):")
    for name, half in [("First half ", df.iloc[:mid]), ("Second half", df.iloc[mid:])]:
        half = half.copy()
        half["q"] = pd.qcut(
            half[signal].rank(method="first"), 5,
            labels=["Q1","Q2","Q3","Q4","Q5"]
        )
        q5 = half[half["q"] == "Q5"]["fwd_24h"].mean()
        q1 = half[half["q"] == "Q1"]["fwd_24h"].mean()
        yr0, yr1 = half.index[0].year, half.index[-1].year
        print(f"      {name} ({yr0}-{yr1}): Q5-Q1 = {q5 - q1:+.3f}%")


def run_symbol(symbol: str, cap_label: str):
    cache = os.path.join(CACHE_DIR, f"{symbol.lower()}_1h_study.parquet")

    print(f"\n{'#'*70}")
    print(f"  {symbol}  ({cap_label})")
    print(f"{'#'*70}")

    if os.path.exists(cache):
        print(f"  Loading cache...")
        df = pd.read_parquet(cache)
    else:
        df = build(fetch_ohlcv(symbol), fetch_funding(symbol), fetch_oi(symbol))
        df.to_parquet(cache)
        print(f"  Cached.")

    print(f"  Dataset: {len(df)} rows  |  "
          f"{df.index[0].date()} to {df.index[-1].date()}  |  "
          f"Bull: {df['above_200d'].sum()}  Bear: {(~df['above_200d']).sum()}")

    for col, label in [
        ("oi_chg_24h",     "OI 24h change (%)"),
        ("funding_rate",   "Funding Rate raw (%)"),
        ("funding_zscore", "Funding Rate Z-Score (90d)"),
        ("rel_vol",        "Relative Volume"),
    ]:
        analyse_signal(df, col, label)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)
    for symbol, cap in SYMBOLS.items():
        run_symbol(symbol, cap)
    print("\n\nDone.")
