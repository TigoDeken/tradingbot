"""
Fetch historical OHLCV from Bybit and assess data quality.
Run: python -m algo.data.fetch_and_check
"""

import time
import pandas as pd
from pybit.unified_trading import HTTP

# Public market data — always mainnet, no auth required
SESSION = HTTP(testnet=False)

INTERVAL    = "5"
TARGET_BARS = 50000


def fetch_candles(symbol: str, interval: str, n_bars: int) -> pd.DataFrame:
    bars = []
    end_time = None

    while len(bars) < n_bars:
        params = dict(category="linear", symbol=symbol, interval=interval, limit=200)
        if end_time:
            params["end"] = end_time

        resp = SESSION.get_kline(**params)
        raw = resp["result"]["list"]
        if not raw:
            break

        bars.extend(raw)
        end_time = int(raw[-1][0]) - 1
        time.sleep(0.1)

    df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
    df = df.astype({"ts": int, "open": float, "high": float, "low": float,
                    "close": float, "volume": float, "turnover": float})
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    return df.iloc[-n_bars:].reset_index(drop=True)


def check(df: pd.DataFrame, symbol: str, interval_minutes: int):
    print(f"\n{'='*60}")
    print(f"{symbol}  {interval_minutes}m  |  {len(df)} bars  "
          f"|  {df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()}")

    vol = df["volume"]
    print(f"\n  Volume percentiles:")
    for p in [50, 75, 90, 95, 99]:
        print(f"    p{p:2d}: {vol.quantile(p/100):.4f}")
    print(f"    max: {vol.max():.2f}")
    print(f"    zero-vol: {(vol == 0).sum()} ({(vol == 0).mean()*100:.1f}%)")

    low_vol = df[vol < 1.0]
    print(f"\n  Candles with volume < 1.0: {len(low_vol)} ({len(low_vol)/len(df)*100:.1f}%)")
    if not low_vol.empty:
        print(f"    Price range: {low_vol['close'].min():.2f} - {low_vol['close'].max():.2f}")
        print(f"    Date range:  {low_vol['time'].iloc[0].date()} - {low_vol['time'].iloc[-1].date()}")

    # OHLC sanity
    bad = df[(df["high"] < df["low"]) |
             (df["open"] > df["high"]) | (df["open"] < df["low"]) |
             (df["close"] > df["high"]) | (df["close"] < df["low"])]
    print(f"  Malformed OHLC: {len(bad)}")

    # Gaps
    expected = pd.Timedelta(minutes=interval_minutes)
    deltas = df["time"].diff().dropna()
    gaps = deltas[deltas > expected * 1.5]
    print(f"  Gaps (>{interval_minutes * 1.5:.0f}m): {len(gaps)}")
    if not gaps.empty:
        for idx in gaps.index[:5]:
            print(f"    {df['time'].iloc[idx-1]}  ->  {df['time'].iloc[idx]}  ({deltas.iloc[idx-1]})")

    # Monthly coverage
    df["month"] = df["time"].dt.tz_localize(None).dt.to_period("M")
    expected_per_month = int(30 * 24 * 60 / interval_minutes)
    print(f"\n  Monthly coverage:")
    for month, count in df.groupby("month").size().items():
        pct = count / expected_per_month * 100
        flag = " << incomplete" if pct < 90 else ""
        print(f"    {month}: {count} bars ({pct:.0f}%){flag}")

    print(f"{'='*60}")


if __name__ == "__main__":
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        print(f"\nFetching {TARGET_BARS} x {INTERVAL}m candles for {symbol}...")
        df = fetch_candles(symbol, INTERVAL, TARGET_BARS)
        check(df, symbol, int(INTERVAL))
