"""
backtest/data_loader.py
Historical OHLCV data loader for backtesting.

Responsibilities:
- Fetch historical kline data from Bybit V5 REST API with full pagination
- Cache downloaded data to data/cache/ as Parquet files (keyed by symbol + interval)
- Serve cached data when available, refresh when stale or forced
- Return a clean pandas DataFrame (UTC DatetimeIndex, Open/High/Low/Close/Volume)
"""

# TODO: implement DataLoader class
