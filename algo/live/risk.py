"""
live/risk.py
Risk management and position sizing.

Responsibilities:
- Compute position size (qty in base currency) from risk config and stop distance
- Round quantity to symbol qty_step
- Enforce max open position limit
- Track session drawdown and trigger circuit breaker when limit exceeded
- Reset session starting balance at configurable intervals
"""

# TODO: implement RiskManager class
