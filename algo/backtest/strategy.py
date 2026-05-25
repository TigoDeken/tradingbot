"""
backtest/strategy.py
Strategy signal logic for backtesting.

Responsibilities:
- Implement the N-bar close breakout signal (no lookahead via shift(1))
- Compute ATR, ATR median, breakout depth, and bar structure features
- Return a pending signal dict when conditions are met, or None
- Accept overrideable parameters (n_bars, atr_mult, depth_thresh, tp_rr)
- Keep logic identical to live/strategy.py to avoid train/live divergence
"""

# TODO: implement NbarBreakoutStrategy class
