"""
Position sizing and risk management.

All parameters loaded from algo/config/config.json at instantiation.
Defaults: 1% risk per trade, max 5 positions, 10% circuit breaker.
"""

import json
import os
from algo.live.logger import get_logger

log = get_logger("risk")

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.json")


def _load_risk_config() -> dict:
    defaults = {"risk_frac": 0.01, "max_positions": 5, "max_session_dd": 0.10}
    try:
        with open(_CONFIG_PATH) as f:
            cfg = json.load(f)
        s = cfg.get("strategy", {})
        e = cfg.get("engine", {})
        return {
            "risk_frac":      float(s.get("risk_frac",           defaults["risk_frac"])),
            "max_positions":  int(  s.get("max_positions",        defaults["max_positions"])),
            "max_session_dd": float(e.get("max_session_drawdown", defaults["max_session_dd"])),
        }
    except Exception as ex:
        log.warning("Could not load risk config: %s — using defaults", ex)
        return defaults


class RiskManager:
    def __init__(self, session_start_equity: float):
        cfg = _load_risk_config()
        self.session_start  = session_start_equity
        self.risk_frac      = cfg["risk_frac"]
        self.max_positions  = cfg["max_positions"]
        self.max_session_dd = cfg["max_session_dd"]
        self._halted        = False
        log.info(
            "RiskManager: risk=%.1f%%  max_pos=%d  circuit_breaker=%.1f%%",
            self.risk_frac * 100, self.max_positions, self.max_session_dd * 100,
        )

    def size_position(self, equity: float, entry: float, stop: float) -> float:
        """Return position size in base currency (qty = risk_usdt / (entry - stop))."""
        if entry <= stop:
            raise ValueError(f"Entry {entry} must be above stop {stop}")
        risk_usdt = equity * self.risk_frac
        qty = risk_usdt / (entry - stop)
        log.info(
            "Size: equity=%.2f  risk_usdt=%.2f  entry=%.6f  stop=%.6f  qty=%.6f",
            equity, risk_usdt, entry, stop, qty,
        )
        return qty

    def check_circuit_breaker(self, current_equity: float) -> bool:
        """Returns True if engine should halt (session drawdown >= threshold)."""
        if self.session_start <= 0:
            return False
        dd = (self.session_start - current_equity) / self.session_start
        if dd >= self.max_session_dd:
            if not self._halted:
                log.error(
                    "CIRCUIT BREAKER: session drawdown %.1f%% >= %.1f%%",
                    dd * 100, self.max_session_dd * 100,
                )
            self._halted = True
        return self._halted

    def can_open(self, open_positions: list) -> bool:
        if self._halted:
            log.warning("Engine halted — circuit breaker active")
            return False
        if len(open_positions) >= self.max_positions:
            log.info("Max positions (%d) reached", self.max_positions)
            return False
        return True

    def reset_session(self, new_equity: float):
        self.session_start = new_equity
        self._halted = False
        log.info("Session reset. Start equity=%.2f", new_equity)
