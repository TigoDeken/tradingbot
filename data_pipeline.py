import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone


SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H4
NUM_BARS = 5000


def connect() -> bool:
    if not mt5.initialize():
        print(f"MT5 initialize failed: {mt5.last_error()}")
        return False
    print(f"MT5 connected — build {mt5.version()}")
    return True


def disconnect() -> None:
    mt5.shutdown()


def fetch_ohlc(symbol: str = SYMBOL, timeframe: int = TIMEFRAME, bars: int = NUM_BARS) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        raise RuntimeError(f"Failed to fetch data for {symbol}: {mt5.last_error()}")

    df = pd.DataFrame(rates)[["time", "open", "high", "low", "close"]]
    df.rename(columns={"time": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], unit="s", utc=True)
    df.set_index("Date", inplace=True)
    df = df.astype({"Open": float, "High": float, "Low": float, "Close": float})
    df.sort_index(inplace=True)
    return df


def get_data(symbol: str = SYMBOL, timeframe: int = TIMEFRAME, bars: int = NUM_BARS) -> pd.DataFrame:
    if not connect():
        raise ConnectionError("Could not connect to MT5")
    try:
        df = fetch_ohlc(symbol, timeframe, bars)
    finally:
        disconnect()
    return df


if __name__ == "__main__":
    df = get_data()
    print(f"\nFetched {len(df)} bars of {SYMBOL} 4H data")
    print(f"Range: {df.index[0]} → {df.index[-1]}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"Dtypes:\n{df.dtypes}")
    print(f"\nFirst 3 rows:\n{df.head(3)}")
    print(f"\nLast 3 rows:\n{df.tail(3)}")
    print(f"\nMissing values: {df.isnull().sum().sum()}")
