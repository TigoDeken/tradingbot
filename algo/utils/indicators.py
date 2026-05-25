"""
utils/indicators.py
Technical indicator functions used by both live and backtest strategy modules.

Responsibilities:
- ATR (Average True Range) — EWM and rolling variants
- ATR median over a rolling time window (for volatility regime filter)
- N-bar rolling min/max (for breakout level detection)
- Bar structure scoring (clean up / clean down / dirty classification)
- All functions operate on pandas Series/DataFrames with no lookahead
"""

# TODO: implement indicator functions
