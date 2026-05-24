import sys
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

TF_MAP = {
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1":  mt5.TIMEFRAME_H1,
}

SESSIONS_HOURS = {
    "asia":    (0, 8),
    "london":  (8, 13),
    "overlap": (13, 16),
    "ny":      (16, 21),
}


def connect():
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        return False
    return True


def disconnect():
    mt5.shutdown()


def _resolve_symbol(primary, fallbacks):
    for sym in [primary] + (fallbacks or []):
        info = mt5.symbol_info(sym)
        if info is not None:
            mt5.symbol_select(sym, True)
            return sym
    return None


def fetch_ohlcv(symbol, timeframe_str, n_bars=200_000, fallbacks=None):
    tf = TF_MAP.get(timeframe_str)
    if tf is None:
        raise ValueError(f"Unknown timeframe: {timeframe_str}")

    actual_sym = _resolve_symbol(symbol, fallbacks)
    if actual_sym is None:
        raise RuntimeError(
            f"Symbol '{symbol}' and fallbacks {fallbacks} not found in Market Watch"
        )

    rates = mt5.copy_rates_from_pos(actual_sym, tf, 0, n_bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"No data for {actual_sym}/{timeframe_str}: {mt5.last_error()}"
        )

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    df = df.rename(
        columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "tick_volume": "Volume",
        }
    )
    df = df[["Open", "High", "Low", "Close", "Volume"]].sort_index()

    days = (df.index[-1] - df.index[0]).days / 365.25
    print(f"  Symbol   : {actual_sym}")
    print(f"  TF       : {timeframe_str}")
    print(f"  Range    : {df.index[0].date()} to {df.index[-1].date()} ({days:.1f} yr)")
    print(f"  Bars     : {len(df):,}")

    hours = df.index.hour
    for sess, (s, e) in SESSIONS_HOURS.items():
        n = int(((hours >= s) & (hours < e)).sum())
        print(f"  {sess:8s}: {n:,} bars  ({n/len(df)*100:.1f}%)")

    return df, actual_sym
