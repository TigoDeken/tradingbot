"""
4H data fetch — spot klines + perp klines + perp OI at 4-hour resolution.

Fetches for all 78 coins already in our daily universe.
Cached to parquet. Run once, then signal_4h_study.py reads from cache.

Run: python -m algo.data.signal_4h_fetch
"""

import os
import time
import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP
from algo.data.zscore_entry_study import load_all

SESSION   = HTTP(testnet=False)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "swing_study")
LIMIT     = 1000   # max rows per request
INTERVAL  = "240"  # 4 hours in minutes


def _fetch_kline(category: str, symbol: str, interval: str) -> pd.DataFrame:
    rows, end_time = [], None
    while True:
        p = dict(category=category, symbol=symbol, interval=interval, limit=LIMIT)
        if end_time:
            p["end"] = end_time
        try:
            resp = SESSION.get_kline(**p)
            raw  = resp.get("result", {}).get("list", [])
        except Exception:
            break
        if not raw:
            break
        rows.extend(raw)
        earliest = int(raw[-1][0])
        if len(raw) < LIMIT:
            break
        end_time = earliest - 1
        time.sleep(0.05)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume","turnover"])
    df["time"]  = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True)
    for c in ["open","high","low","close","volume","turnover"]:
        df[c] = df[c].astype(float)
    return df[["time","open","high","low","close","volume"]].sort_values("time").reset_index(drop=True)


def _fetch_oi_4h(symbol: str) -> pd.DataFrame:
    rows, cursor = [], None
    while True:
        p = dict(category="linear", symbol=symbol, intervalTime="4h", limit=200)
        if cursor:
            p["cursor"] = cursor
        try:
            resp   = SESSION.get_open_interest(**p)
            result = resp.get("result", {})
            raw    = result.get("list", [])
        except Exception:
            break
        if not raw:
            break
        rows.extend(raw)
        cursor = result.get("nextPageCursor", "")
        time.sleep(0.05)
        if not cursor or len(raw) < 200:
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    df["oi"]   = df["openInterest"].astype(float)
    return df[["time","oi"]].sort_values("time").reset_index(drop=True)


def fetch_perp_funding_4h(symbol: str) -> pd.DataFrame:
    """Funding rate is only published every 8h — resample to 4H by forward-fill."""
    rows, cursor = [], None
    while True:
        p = dict(category="linear", symbol=symbol, limit=200)
        if cursor:
            p["cursor"] = cursor
        try:
            resp   = SESSION.get_funding_rate_history(**p)
            result = resp.get("result", {})
            raw    = result.get("list", [])
        except Exception:
            break
        if not raw:
            break
        rows.extend(raw)
        cursor = result.get("nextPageCursor", "")
        time.sleep(0.05)
        if not cursor or len(raw) < 200:
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["time"]         = pd.to_datetime(df["fundingRateTimestamp"].astype(int), unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    return df[["time","funding_rate"]].sort_values("time").reset_index(drop=True)


def fetch_and_cache(sym: str, force: bool = False):
    spot_path  = os.path.join(CACHE_DIR, f"{sym}_spot_4h.parquet")
    perp_path  = os.path.join(CACHE_DIR, f"{sym}_perp_4h.parquet")
    oi_path    = os.path.join(CACHE_DIR, f"{sym}_perp_oi_4h.parquet")

    missing = not os.path.exists(spot_path) or not os.path.exists(perp_path) or not os.path.exists(oi_path)
    if not missing and not force:
        return True

    perp_sym = sym  # e.g. BTCUSDT — same symbol on linear perps

    spot = _fetch_kline("spot",   sym,      INTERVAL)
    perp = _fetch_kline("linear", perp_sym, INTERVAL)
    oi   = _fetch_oi_4h(perp_sym)

    if spot.empty or perp.empty or oi.empty:
        return False

    spot.to_parquet(spot_path)
    perp.to_parquet(perp_path)
    oi.to_parquet(oi_path)
    return True


if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)
    coins = load_all()
    syms  = sorted(coins.keys())
    print(f"Fetching 4H data for {len(syms)} coins...")

    ok, fail = [], []
    for i, sym in enumerate(syms, 1):
        success = fetch_and_cache(sym)
        if success:
            ok.append(sym)
            print(f"  [{i:2d}/{len(syms)}] {sym:18} OK")
        else:
            fail.append(sym)
            print(f"  [{i:2d}/{len(syms)}] {sym:18} SKIP (no perp data)")

    print(f"\nDone: {len(ok)} cached, {len(fail)} skipped.")
    if fail:
        print(f"Skipped: {', '.join(fail)}")
