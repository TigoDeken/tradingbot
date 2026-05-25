"""
live/execution.py
Order execution and position lifecycle management.

Responsibilities:
- Place market entry orders with SL/TP via exchange adapter
- Poll open positions each candle to detect exchange-side closes (SL/TP hit)
- Sync local state file after each position change
- Retry failed orders up to ORDER_RETRY_LIMIT with exponential back-off
- Emit structured log entries for every order event
"""

# TODO: implement ExecutionHandler class
