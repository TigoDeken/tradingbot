"""
backtest/engine.py
Core backtesting engine.

Responsibilities:
- Iterate over historical OHLCV bars in chronological order
- Feed each bar to the strategy for signal generation (no lookahead)
- Hand signals to the simulated execution layer (fill at next bar open)
- Manage open trades: check stop and TP bar-by-bar with conservative tie-break
- Collect completed Trade records and pass to metrics module
"""

# TODO: implement BacktestEngine class
