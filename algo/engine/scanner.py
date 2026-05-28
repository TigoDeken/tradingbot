"""
Universe scanner — refreshes data cache and computes signal state for all coins.

Call scan() to get a ranked list of signal states for all coins.
Results are written to live_state.json so the dashboard reads them directly.
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from algo.data.zscore_entry_study import load_all, load_daily, load_funding, build_coin_df, CACHE_DIR
from algo.data.perp_flow_study    import load_perp_oi, load_perp_kline, fetch_perp_oi, fetch_perp_kline
from algo.engine.signals          import compute_signals
from algo.engine.state            import load_state, save_state
from algo.live.logger             import get_logger

import os
from pybit.unified_trading import HTTP

SESSION = HTTP(testnet=False)
log     = get_logger("scanner")


def _refresh_spot(symbol: str):
    """Append any new daily bars to the spot cache."""
    cache = os.path.join(CACHE_DIR, f"{symbol}_daily.parquet")
    if not os.path.exists(cache):
        return load_daily(symbol)

    df      = pd.read_parquet(cache)
    last_ts = int(df["ts"].max())
    bars    = []
    end     = None

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
        time.sleep(0.05)

    if bars:
        cols   = ["ts","open","high","low","close","volume","turnover"]
        new_df = pd.DataFrame(bars, columns=cols)
        new_df = new_df.astype({c: float for c in cols[1:]})
        new_df["ts"]   = new_df["ts"].astype(int)
        new_df["time"] = pd.to_datetime(new_df["ts"], unit="ms", utc=True)
        df = pd.concat([df, new_df], ignore_index=True).drop_duplicates("ts").sort_values("ts")
        df.to_parquet(cache)
    return df


def _refresh_funding(symbol: str):
    """Append any new funding settlements to the funding cache."""
    cache    = os.path.join(CACHE_DIR, f"{symbol}_funding.parquet")
    existing = pd.DataFrame()
    last_ts  = 0

    if os.path.exists(cache):
        existing = pd.read_parquet(cache)
        if not existing.empty and "time" in existing.columns:
            t = existing["time"]
            last_ts = int(t.max().timestamp() * 1000) if t.dt.tz is not None \
                      else int(pd.to_datetime(t.max()).timestamp() * 1000)

    bars, end = [], None
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
        time.sleep(0.05)

    if bars:
        df = pd.DataFrame(bars)
        df["time"]         = pd.to_datetime(df["fundingRateTimestamp"].astype(int), unit="ms", utc=True)
        df["funding_rate"] = df["fundingRate"].astype(float) * 100
        df = df[["time", "funding_rate"]]
        if not existing.empty:
            df = pd.concat([existing, df], ignore_index=True).drop_duplicates("time").sort_values("time")
        df.to_parquet(cache)
        return df
    return existing


def _refresh_perp_oi(symbol: str):
    """Append new perp OI bars to the OI cache."""
    cache    = os.path.join(CACHE_DIR, f"{symbol}_perp_oi.parquet")
    existing = pd.DataFrame()
    last_ts  = 0

    if os.path.exists(cache):
        existing = pd.read_parquet(cache)
        if not existing.empty:
            ts_col  = existing["time"]
            last_ts = int(ts_col.max().timestamp() * 1000) if ts_col.dt.tz is not None \
                      else int(pd.to_datetime(ts_col.max()).timestamp() * 1000)

    bars, cursor = [], None
    while True:
        p = dict(category="linear", symbol=symbol, intervalTime="1d", limit=200)
        if cursor:
            p["cursor"] = cursor
        result = SESSION.get_open_interest(**p).get("result", {})
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
        df = df[["time", "oi"]]
        if not existing.empty:
            df = pd.concat([existing, df], ignore_index=True).drop_duplicates("time").sort_values("time")
        df.to_parquet(cache)
        return df
    return existing


def _refresh_perp_kline(symbol: str):
    """Append new perp klines to the perp cache."""
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
        last_ts = int(t.max().timestamp() * 1000) if t.dt.tz is not None \
                  else int(pd.to_datetime(t.max()).timestamp() * 1000)

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
        cols = ["ts","open","high","low","close","volume","turnover"]
        df   = pd.DataFrame(bars, columns=cols)
        df   = df.astype({c: float for c in cols[1:]})
        df["time"]       = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True)
        df["perp_close"] = df["close"]
        df = df[["time", "perp_close"]]
        if not existing.empty:
            df = pd.concat([existing, df], ignore_index=True).drop_duplicates("time").sort_values("time")
        df.to_parquet(cache)
        return df
    return existing


def scan(refresh: bool = True) -> list[dict]:
    """
    Scan all coins. Returns signal dicts sorted by entry readiness.
    If refresh=True, fetches the latest bars first (takes ~2 min for 80 coins).
    """
    log.info("Loading coin list...")
    coins_raw = load_all()
    symbols   = sorted(coins_raw.keys())
    log.info("Scanning %d coins (refresh=%s)...", len(symbols), refresh)

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

            if sig["entry_signal"] or sig["exit_signal"] or i % 10 == 0:
                fz  = f"{sig['funding_z']:+.2f}" if sig.get("funding_z") is not None \
                      and not np.isnan(sig["funding_z"]) else "n/a"
                oiz = f"{sig['oi_z']:+.2f}"      if sig.get("oi_z") is not None \
                      and not np.isnan(sig["oi_z"]) else "n/a"
                atr = f"{sig['atr_ratio']:.2f}"  if sig.get("atr_ratio") is not None \
                      and not np.isnan(sig["atr_ratio"]) else "n/a"
                flag = " <<< ENTRY" if sig["entry_signal"] else (" EXIT" if sig["exit_signal"] else "")
                log.info("[%2d/%d] %-18s  fz=%s  oiz=%s  atr=%s%s",
                         i, len(symbols), sym, fz, oiz, atr, flag)

        except Exception as e:
            log.warning("[%2d/%d] %-18s  ERROR: %s", i, len(symbols), sym, e)

    # Sort: entry signals first, then by conditions_met desc, then fz asc (most negative = best)
    results.sort(key=lambda r: (
        -int(r.get("entry_signal", False)),
        -r.get("conditions_met", 0),
        r.get("funding_z", 0) or 0,
    ))

    state               = load_state()
    state["scan_results"] = results
    state["last_scan"]    = datetime.now(timezone.utc).isoformat()
    save_state(state)

    n_entry = sum(r["entry_signal"] for r in results)
    log.info("Scan complete. %d entry signals.", n_entry)
    return results
