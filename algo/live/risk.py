"""
Position sizing and risk management.

1% of current equity risked per trade.
Max 5 concurrent positions.
Circuit breaker: halt engine if session drawdown > 10%.
"""

import math
from algo.live.logger import get_logger

log = get_logger("risk")

RISK_FRAC      = 0.01   # fraction of equity risked per trade
MAX_POSITIONS  = 5
MAX_SESSION_DD = 0.10   # halt if session equity drops 10%


class RiskManager:
    def __init__(self, session_start_equity: float):
        self.session_start = session_start_equity
        self._halted = False

    def size_position(self, equity: float, entry: float, stop: float) -> float:
        """
        Return position size in base currency (e.g. BTC, not USDT).
        risk_usdt = equity * RISK_FRAC
        qty = risk_usdt / (entry - stop)
        """
        if entry <= stop:
            raise ValueError(f"Entry {entry} must be above stop {stop}")
        risk_usdt = equity * RISK_FRAC
        qty = risk_usdt / (entry - stop)
        log.info("Size: equity=%.2f  risk_usdt=%.2f  entry=%.6f  stop=%.6f  qty=%.6f",
                 equity, risk_usdt, entry, stop, qty)
        return qty

    def check_circuit_breaker(self, current_equity: float) -> bool:
        """Returns True if trading should halt."""
        dd = (self.session_start - current_equity) / self.session_start
        if dd >= MAX_SESSION_DD:
            if not self._halted:
                log.error("CIRCUIT BREAKER: session drawdown %.1f%% >= %.1f%%",
                          dd * 100, MAX_SESSION_DD * 100)
            self._halted = True
        return self._halted

    def can_open(self, open_positions: list) -> bool:
        if self._halted:
            log.warning("Engine halted — circuit breaker active")
            return False
        if len(open_positions) >= MAX_POSITIONS:
            log.info("Max positions (%d) reached", MAX_POSITIONS)
            return False
        return True

    def reset_session(self, new_equity: float):
        self.session_start = new_equity
        self._halted = False
        log.info("Session reset. Start equity=%.2f", new_equity)
