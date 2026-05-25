"""
bybit_data_pipeline.py
Fetch OHLCV data from Bybit V5 REST API for linear perpetuals (USDT-margined).

Produces the same DataFrame format used throughout the system:
  index:   DatetimeIndex UTC
  columns: Open  High  Low  Close  Volume
"""
import time

import pandas as pd
import requests

BYBIT_BASE_URL    = "https://api.bybit.com"
BYBIT_TESTNET_URL = "https://api-testnet.bybit.com"
MAX_PER_REQUEST   = 200   # Bybit kline limit per call

# MT5-style → Bybit interval string
TF_MAP = {
    "M1": "1",  "M3": "3",  "M5": "5",  "M15": "15", "M30": "30",
    "H1": "60", "H4": "240", "D1": "D",
    # pass-through for native Bybit strings
    "1": "1", "3": "3", "5": "5", "15": "15", "30": "30",
    "60": "60", "120": "120", "240": "240", "D": "D",
}


def _base(testnet: bool) -> str:
    return BYBIT_TESTNET_URL if testnet else BYBIT_BASE_URL


def fetch_ohlc(symbol: str, interval: str = "5", bars: int = 2000,
               testnet: bool = False) -> pd.DataFrame:
    """
    Fetch completed OHLCV bars for a Bybit linear perpetual.
    Paginates automatically for requests larger than 200 bars.

    Args:
        symbol:   e.g. "SOLUSDT"
        interval: Bybit interval string or MT5-style ("M5", "H1")
        bars:     number of completed bars to return (current incomplete bar excluded)
        testnet:  use Bybit testnet endpoint

    Returns:
        DataFrame with UTC DatetimeIndex and columns Open/High/Low/Close/Volume
    """
    bybit_iv  = TF_MAP.get(interval, interval)
    url       = f"{_base(testnet)}/v5/market/kline"
    all_rows: list = []
    end_ms: int | None = None
    remaining = bars + 1  # +1 so we can drop the current (incomplete) bar

    while remaining > 0:
        limit  = min(remaining, MAX_PER_REQUEST)
        params = {"category": "linear", "symbol": symbol,
                  "interval": bybit_iv, "limit": limit}
        if end_ms is not None:
            params["end"] = end_ms

        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        body = r.json()

        if body.get("retCode") != 0:
            raise RuntimeError(
                f"Bybit kline error [{symbol}]: {body.get('retMsg')} "
                f"(retCode={body.get('retCode')})"
            )

        candles: list = body["result"]["list"]
        if not candles:
            break

        all_rows.extend(candles)
        remaining -= len(candles)

        # Bybit returns newest-first; oldest is last element
        end_ms = int(candles[-1][0]) - 1  # paginate earlier

        if len(candles) < limit:
            break

        time.sleep(0.05)  # rate-limit courtesy

    if not all_rows:
        raise RuntimeError(f"No kline data returned for {symbol} interval={interval}")

    # Each row: [startTime_ms, open, high, low, close, volume, turnover]
    df = pd.DataFrame(
        all_rows,
        columns=["ts", "Open", "High", "Low", "Close", "Volume", "Turnover"],
    )
    df["Date"] = pd.to_datetime(df["ts"].astype(float), unit="ms", utc=True)
    df = df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df.sort_index(inplace=True)
    df = df.iloc[:-1]   # drop latest incomplete bar

    return df.tail(bars)


def get_symbol_info(symbol: str, testnet: bool = False) -> dict:
    """
    Return instrument specification from Bybit:
      min_qty, qty_step, tick_size, max_qty, status
    """
    url  = f"{_base(testnet)}/v5/market/instruments-info"
    r    = requests.get(url, params={"category": "linear", "symbol": symbol}, timeout=10)
    r.raise_for_status()
    body = r.json()

    if body.get("retCode") != 0:
        raise RuntimeError(f"Bybit instruments-info error: {body.get('retMsg')}")

    items = body["result"]["list"]
    if not items:
        raise ValueError(f"Symbol {symbol!r} not found on Bybit linear")

    item  = items[0]
    lot   = item.get("lotSizeFilter", {})
    price = item.get("priceFilter", {})
    return {
        "symbol":    symbol,
        "status":    item.get("status", ""),
        "min_qty":   float(lot.get("minOrderQty",  "0.01")),
        "max_qty":   float(lot.get("maxOrderQty",  "1000000")),
        "qty_step":  float(lot.get("qtyStep",       "0.01")),
        "tick_size": float(price.get("tickSize",    "0.01")),
    }
