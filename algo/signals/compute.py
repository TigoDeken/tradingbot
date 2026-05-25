import time
import pandas as pd
from pybit.unified_trading import HTTP

_SESSION = HTTP(testnet=False)


def _fetch_oi(symbol: str, n: int) -> pd.DataFrame:
    bars, end = [], None
    while len(bars) < n:
        p = dict(category="linear", symbol=symbol, intervalTime="1h", limit=min(200, n - len(bars)))
        if end:
            p["endTime"] = end
        raw = _SESSION.get_open_interest(**p)["result"]["list"]
        if not raw:
            break
        bars.extend(raw)
        end = int(raw[-1]["timestamp"]) - 1
        time.sleep(0.05)
    df = pd.DataFrame(bars[:n])
    df["time"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    df["oi"] = df["openInterest"].astype(float)
    return df[["time", "oi"]].sort_values("time").reset_index(drop=True)


def _fetch_ohlcv(symbol: str, n: int) -> pd.DataFrame:
    bars, end = [], None
    while len(bars) < n:
        p = dict(category="spot", symbol=symbol, interval="60", limit=min(200, n - len(bars)))
        if end:
            p["end"] = end
        raw = _SESSION.get_kline(**p)["result"]["list"]
        if not raw:
            break
        bars.extend(raw)
        end = int(raw[-1][0]) - 1
        time.sleep(0.05)
    df = pd.DataFrame(bars[:n], columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
    df["close"] = df["close"].astype(float)
    df["ts"] = df["ts"].astype(int)
    return df.sort_values("ts").reset_index(drop=True)


def _to_quintile(value: float, history: pd.Series) -> tuple[int, str]:
    pct = (history < value).mean()
    q = min(5, int(pct * 5) + 1)
    return q, f"Q{q}"


def compute_oi_signal(symbol: str, lookback: int = 500) -> dict:
    """OI 24h change value and quintile vs last `lookback` bars."""
    df = _fetch_oi(symbol, lookback + 24)
    df["oi_chg_24h"] = df["oi"].pct_change(24) * 100
    df = df.dropna(subset=["oi_chg_24h"])
    current = df["oi_chg_24h"].iloc[-1]
    q_num, q_label = _to_quintile(current, df["oi_chg_24h"].iloc[:-1])
    return {"oi_chg_24h": current, "oi_quintile_num": q_num, "oi_quintile": q_label}


def compute_regime(symbol: str) -> str:
    """BULL if close > 200d MA (4800 1H bars), else BEAR."""
    df = _fetch_ohlcv(symbol, 4800)
    if len(df) < 4800:
        return "UNKNOWN"
    ma = df["close"].rolling(4800).mean().iloc[-1]
    return "BULL" if df["close"].iloc[-1] > ma else "BEAR"


def compute_funding(symbol: str) -> float:
    """Latest funding rate in %."""
    raw = _SESSION.get_funding_rate_history(category="linear", symbol=symbol, limit=1)["result"]["list"]
    if not raw:
        return 0.0
    return float(raw[0]["fundingRate"]) * 100
