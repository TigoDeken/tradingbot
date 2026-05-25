"""
live/exchange.py
Bybit exchange adapter for live trading.

Responsibilities:
- Authenticated REST session via pybit (BYBIT_API_KEY / BYBIT_API_SECRET)
- Fetch real-time OHLCV data from Bybit V5 kline endpoint with pagination
- Fetch instrument info (qty_step, tick_size, min_qty)
- Query USDT wallet balance
- Query open positions
- Place and cancel market orders with native server-side SL/TP
- Set leverage per symbol
"""

# TODO: implement BybitExchange class
