"""
Live trading engine — run after each 8H funding settlement.

Flow:
  1. Scan:  refresh cache, compute signals for all coins (public data only)
  2. Connect: verify API keys and get current equity
  3. Sync:  detect stops hit by exchange; re-place disappeared stops
  4. Exit:  close positions where funding_z has risen above EXIT_Z
  5. Enter: open positions for coins with a fresh entry signal
  6. Save:  persist state

Schedule: 00:05 / 08:05 / 16:05 UTC (5 min after each funding settlement)
Run:      python -m algo.live.main
"""

import json
import os
import sys
from datetime import datetime, timezone

from algo.live.logger    import get_logger, setup_logger
from algo.live.exchange  import BybitExchange
from algo.live.risk      import RiskManager
from algo.live.execution import ExecutionHandler
from algo.engine.scanner import scan
from algo.engine.state   import load_state, save_state

setup_logger()
log = get_logger("main")

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.json")


def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def run():
    log.info("=" * 60)
    log.info("Engine starting — %s UTC", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))

    cfg       = _load_config()
    strat_cfg = cfg.get("strategy", {})
    max_stop_pct = float(strat_cfg.get("max_stop_pct", 0.30))

    # ── Step 1: Scan ─────────────────────────────────────────────────────────
    log.info("Scanning universe...")
    results       = scan(refresh=True)
    entry_signals = [r for r in results if r.get("entry_signal")]
    exit_signals  = [r for r in results if r.get("exit_signal")]
    log.info(
        "Entry signals: %d  Exit signals: %d  Watching: %d",
        len(entry_signals), len(exit_signals),
        sum(1 for r in results if r.get("conditions_met", 0) >= 2),
    )

    # ── Step 2: Account connection ───────────────────────────────────────────
    try:
        ex     = BybitExchange()
        equity = ex.get_usdt_balance()
        if equity <= 0:
            raise ValueError("Balance is 0 — account may be empty or keys lack permission")
        trading_enabled = True
        log.info("Account connected. Equity: %.2f USDT", equity)
    except Exception as e:
        log.warning("Account unavailable (%s) — scan-only mode", e)
        log.info("Scan complete. Add API keys to algo/.env to enable trading.")
        log.info("=" * 60)
        return

    state   = load_state()
    risk    = RiskManager(session_start_equity=equity)
    handler = ExecutionHandler(ex)

    log.info("Open positions: %d", len(state["positions"]))

    # ── Step 3: Sync ─────────────────────────────────────────────────────────
    stopped = handler.sync_positions()
    if stopped:
        log.info("Stops hit / positions synced: %s", stopped)
        state  = load_state()
        equity = ex.get_usdt_balance()

    # ── Step 4: Exit ─────────────────────────────────────────────────────────
    for pos in list(state["positions"]):
        sym = pos["symbol"]
        sig = next((r for r in results if r["symbol"] == sym), None)
        if sig and sig.get("exit_signal"):
            log.info("Exit signal for %s (fz=%.2f)", sym, sig.get("funding_z", 0))
            if handler.close_position(sym, reason="signal"):
                equity = ex.get_usdt_balance()

    # ── Step 5: Enter ─────────────────────────────────────────────────────────
    state = load_state()
    for sig in entry_signals:
        sym = sig["symbol"]

        if any(p["symbol"] == sym for p in state["positions"]):
            log.info("Already in %s — skip", sym)
            continue

        if not risk.can_open(state["positions"]):
            log.info("No more position slots")
            break

        if risk.check_circuit_breaker(equity):
            log.error("Circuit breaker active — halting entries")
            break

        entry = sig["close"]
        stop  = sig["stop"]

        if entry <= stop:
            log.warning("Invalid stop for %s: entry=%.6f stop=%.6f — skip", sym, entry, stop)
            continue

        stop_dist = (entry - stop) / entry
        if stop_dist > max_stop_pct:
            log.warning("%s stop too wide: %.1f%% > %.1f%% max — skip",
                        sym, stop_dist * 100, max_stop_pct * 100)
            continue

        try:
            qty = risk.size_position(equity, entry, stop)
        except Exception as e:
            log.warning("size_position failed for %s: %s", sym, e)
            continue

        # Hard cap: single position cannot exceed 30% of account
        if qty * entry > equity * 0.30:
            log.warning("%s position value %.2f > 30%% of equity %.2f — skip",
                        sym, qty * entry, equity)
            continue

        log.info(
            "ENTRY: %s  entry~%.6f  stop=%.6f  qty=%.6f  stop_dist=%.1f%%  fz=%.2f  oiz=%.2f",
            sym, entry, stop, qty, stop_dist * 100,
            sig.get("funding_z", 0), sig.get("oi_z", 0),
        )

        if handler.open_position(sym, entry, stop, qty, meta=sig):
            equity = ex.get_usdt_balance()
            state  = load_state()

    # ── Step 6: Save ─────────────────────────────────────────────────────────
    state           = load_state()
    equity          = ex.get_usdt_balance()
    state["equity"] = equity
    save_state(state)

    log.info("Run complete. Equity=%.2f  Open=%d", equity, len(state["positions"]))
    log.info("=" * 60)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        log.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        log.exception("Unhandled exception: %s", e)
        sys.exit(1)
