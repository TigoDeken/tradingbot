"""
Universe scanner — refreshes cache and computes current signal state for all coins.

Call scan() to get a ranked list of signal states for all 78 coins.
Results are written to live_state.json so the dashboard can read them without
running the scanner again.
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from algo.data.zscore_entry_study import load_all, load_daily, load_funding, build_coin_df, CACHE_DIR
from algo.data.perp_flow_study import load_perp_oi, load_perp_kline, fetch_perp_oi, fetch_perp_kline
from algo.engine.signals import compute_signals
from algo.engine.state import load_state, save_state

import os
from pybit.unified_trading import HTTP

SESSION = HTTP(testnet=False)


def _refresh_spot(symbol: str):
    """Append any new daily bars to the spot cache."""
    import time as _time
    cache = os.path.join(CACHE_DIR, f"{symbol}_daily.parquet")
    if not os.path.exists(cache):
        return load_daily(symbol)

    df = pd.read_parquet(cache)
    last_ts = int(df["ts"].max())

    bars = []
    end = None
    while True:
        p = dict(category="spot", symbol=symbol, interval="D", limit=200)
        if end:
            p["end"] = end
        raw = SESSION.get_kline(**p)["result"]["list"]
        if not raw:
            break
        new = [b for b in raw if int(b[0]) > last_ts]
        bars.extend(new)
        if len(new) < len(raw):
            break
        end = int(raw[-1][0]) - 1
        _time.sleep(0.05)

    if bars:
        new_df = pd.DataFrame(bars, columns=["ts","open","high","low","close","volume","turnover"])
        new_df = new_df.astype({c: float for c in ["open","high","low","close","volume","turnover"]})
        new_df["ts"]   = new_df["ts"].astype(int)
        new_df["time"] = pd.to_datetime(new_df["ts"], unit="ms", utc=True)
        df = pd.concat([df, new_df], ignore_index=True).drop_duplicates("ts").sort_values("ts")
        df.to_parquet(cache)
    return df


def _refresh_funding(symbol: str):
    """Append any new funding rates to the funding cache."""
    import time as _time
    cache = os.path.join(CACHE_DIR, f"{symbol}_funding.parquet")
    bars, end = [], None

    existing = pd.DataFrame()
    last_ts  = 0
    if os.path.exists(cache):
        existing = pd.read_parquet(cache)
        if not existing.empty and "time" in existing.columns:
            t = existing["time"]
            if hasattr(t.dtype, "tz") and t.dt.tz is not None:
                last_ts = int(t.max().timestamp() * 1000)
            else:
                last_ts = int(pd.to_datetime(t.max()).timestamp() * 1000)

    while True:
        p = dict(category="linear", symbol=symbol, limit=200)
        if end:
            p["endTime"] = end
        raw = SESSION.get_funding_rate_history(**p)["result"]["list"]
        if not raw:
            break
        new = [r for r in raw if int(r["fundingRateTimestamp"]) > last_ts]
        bars.extend(new)
        if len(new) < len(raw):
            break
        end = int(raw[-1]["fundingRateTimestamp"]) - 1
        _time.sleep(0.05)

    if bars:
        df = pd.DataFrame(bars)
        df["time"]         = pd.to_datetime(df["fundingRateTimestamp"].astype(int), unit="ms", utc=True)
        df["funding_rate"] = df["fundingRate"].astype(float) * 100
        df = df[["time","funding_rate"]]
        if not existing.empty:
            df = pd.concat([existing, df], ignore_index=True).drop_duplicates("time").sort_values("time")
        df.to_parquet(cache)
        return df
    return existing


def _refresh_perp_oi(symbol: str):
    """Append new perp OI bars to the perp_oi cache."""
    cache = os.path.join(CACHE_DIR, f"{symbol}_perp_oi.parquet")
    existing = pd.DataFrame()
    last_ts  = 0
    if os.path.exists(cache):
        existing = pd.read_parquet(cache)
        if not existing.empty:
            ts_col = existing["time"]
            if hasattr(ts_col.dtype, "tz") and ts_col.dt.tz is not None:
                last_ts = int(ts_col.max().timestamp() * 1000)
            else:
                last_ts = int(pd.to_datetime(ts_col.max()).timestamp() * 1000)

    bars, cursor = [], None
    while True:
        p = dict(category="linear", symbol=symbol, intervalTime="1d", limit=200)
        if cursor:
            p["cursor"] = cursor
        resp   = SESSION.get_open_interest(**p)
        result = resp.get("result", {})
        raw    = result.get("list", [])
        if not raw:
            break
        new = [r for r in raw if int(r["timestamp"]) > last_ts]
        bars.extend(new)
        cursor = result.get("nextPageCursor", "")
        time.sleep(0.05)
        if not cursor or len(new) < len(raw):
            break

    if bars:
        df = pd.DataFrame(bars)
        df["time"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
        df["oi"]   = df["openInterest"].astype(float)
        df = df[["time","oi"]]
        if not existing.empty:
            df = pd.concat([existing, df], ignore_index=True).drop_duplicates("time").sort_values("time")
        df.to_parquet(cache)
        return df
    return existing


def _refresh_perp_kline(symbol: str):
    """Append new perp klines (for basis/perp_close) to the cache."""
    cache = os.path.join(CACHE_DIR, f"{symbol}_perp_daily.parquet")
    if not os.path.exists(cache):
        df = fetch_perp_kline(symbol)
        if not df.empty:
            df.to_parquet(cache)
        return df

    existing = pd.read_parquet(cache)
    last_ts  = 0
    if not existing.empty and "time" in existing.columns:
        t = existing["time"]
        if hasattr(t.dtype, "tz") and t.dt.tz is not None:
            last_ts = int(t.max().timestamp() * 1000)
        else:
            last_ts = int(pd.to_datetime(t.max()).timestamp() * 1000)

    bars, end = [], None
    while True:
        p = dict(category="linear", symbol=symbol, interval="D", limit=200)
        if end:
            p["end"] = end
        raw = SESSION.get_kline(**p)["result"]["list"]
        if not raw:
            break
        new = [b for b in raw if int(b[0]) > last_ts]
        bars.extend(new)
        if len(new) < len(raw):
            break
        end = int(raw[-1][0]) - 1
        time.sleep(0.05)

    if bars:
        df = pd.DataFrame(bars, columns=["ts","open","high","low","close","volume","turnover"])
        df = df.astype({c: float for c in ["open","high","low","close","volume","turnover"]})
        df["time"]       = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True)
        df["perp_close"] = df["close"]
        df = df[["time","perp_close"]]
        if not existing.empty:
            df = pd.concat([existing, df], ignore_index=True).drop_duplicates("time").sort_values("time")
        df.to_parquet(cache)
        return df
    return existing


def scan(refresh: bool = True) -> list[dict]:
    """
    Scan all coins. Returns list of signal dicts sorted by entry readiness.
    If refresh=True, fetches latest bars first (takes ~2 min for 78 coins).
    """
    print("Loading coin list...")
    coins_raw = load_all()
    symbols   = sorted(coins_raw.keys())
    print(f"Scanning {len(symbols)} coins (refresh={refresh})...")

    results = []
    for i, sym in enumerate(symbols, 1):
        try:
            if refresh:
                spot_raw = _refresh_spot(sym)
                fund_raw = _refresh_funding(sym)
                oi_df    = _refresh_perp_oi(sym)
                perp_df  = _refresh_perp_kline(sym)
            else:
                spot_raw = load_daily(sym)
                fund_raw = load_funding(sym)
                oi_df    = load_perp_oi(sym)
                perp_df  = load_perp_kline(sym)

            if spot_raw.empty or fund_raw.empty or oi_df.empty or perp_df.empty:
                continue

            spot_df = build_coin_df(spot_raw, fund_raw)
            sig     = compute_signals(spot_df, oi_df, perp_df)
            sig["symbol"] = sym

            results.append(sig)
            flag = " <<< ENTRY" if sig["entry_signal"] else (" EXIT" if sig["exit_signal"] else "")
            if i % 10 == 0 or sig["entry_signal"] or sig["exit_signal"]:
                fz  = f"{sig['funding_z']:+.2f}" if not np.isnan(sig.get("funding_z", np.nan) or np.nan) else "n/a"
                oiz = f"{sig['oi_z']:+.2f}"      if not np.isnan(sig.get("oi_z",     np.nan) or np.nan) else "n/a"
                atr = f"{sig['atr_ratio']:.2f}"  if not np.isnan(sig.get("atr_ratio", np.nan) or np.nan) else "n/a"
                print(f"  [{i:2d}/{len(symbols)}] {sym:18} fz={fz}  oiz={oiz}  atr={atr}{flag}")

        except Exception as e:
            print(f"  [{i:2d}/{len(symbols)}] {sym:18} ERROR: {e}")
            continue

    # Sort: entry signals first, then by conditions_met desc, then by fz asc
    results.sort(key=lambda r: (
        -int(r.get("entry_signal", False)),
        -r.get("conditions_met", 0),
        r.get("funding_z", 0) or 0,
    ))

    # Save to state
    state = load_state()
    state["scan_results"] = results
    state["last_scan"]    = datetime.now(timezone.utc).isoformat()
    save_state(state)

    print(f"\nScan complete. {sum(r['entry_signal'] for r in results)} entry signals.")
    return results
