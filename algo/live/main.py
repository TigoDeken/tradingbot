"""
Live trading engine — run once per day after the daily close.

Flow:
  1. Sync: check if any stops were hit since last run
  2. Exit: close positions where funding_z has risen above 0
  3. Scan: refresh data cache, compute signals for all 78 coins
  4. Enter: open positions for coins with an active entry signal
  5. Save state

Run: python -m algo.live.main
Schedule: daily at 00:05 UTC (5 min after daily bar closes)
"""

import sys
from algo.live.logger    import get_logger, setup_logger
from algo.live.exchange  import BybitExchange
from algo.live.risk      import RiskManager
from algo.live.execution import ExecutionHandler
from algo.engine.scanner import scan
from algo.engine.state   import load_state, save_state

setup_logger()
log = get_logger("main")


def run():
    log.info("=" * 60)
    log.info("Engine starting")

    ex      = BybitExchange()
    state   = load_state()
    equity  = ex.get_usdt_balance()
    risk    = RiskManager(session_start_equity=equity)
    handler = ExecutionHandler(ex)

    log.info("Equity: %.2f USDT  Open positions: %d",
             equity, len(state["positions"]))

    # ── Step 1: Sync — detect stops hit by exchange ───────────────────────────
    stopped = handler.sync_positions()
    if stopped:
        log.info("Stops hit: %s", stopped)
        state  = load_state()
        equity = ex.get_usdt_balance()

    # ── Step 2: Scan — refresh data and compute signals ───────────────────────
    log.info("Scanning universe...")
    results = scan(refresh=True)
    entry_signals = [r for r in results if r.get("entry_signal")]
    exit_signals  = [r for r in results if r.get("exit_signal")]

    log.info("Entry signals: %d  Exit signals: %d",
             len(entry_signals), len(exit_signals))

    # ── Step 3: Exit — close positions where signal reversed ──────────────────
    for pos in list(state["positions"]):
        sym = pos["symbol"]
        sig = next((r for r in results if r["symbol"] == sym), None)
        if sig and sig.get("exit_signal"):
            log.info("Exit signal for %s (fz=%.2f)", sym, sig.get("funding_z", 0))
            handler.close_position(sym, reason="signal")
            equity = ex.get_usdt_balance()

    # ── Step 4: Enter — open new positions ────────────────────────────────────
    state = load_state()
    for sig in entry_signals:
        sym = sig["symbol"]

        # Skip if already in this position
        if any(p["symbol"] == sym for p in state["positions"]):
            log.info("Already in %s — skip", sym)
            continue

        if not risk.can_open(state["positions"]):
            log.info("No more position slots")
            break

        if risk.check_circuit_breaker(equity):
            log.error("Circuit breaker active — halting entries")
            break

        entry = sig["close"]  # use last close as proxy; live entry will be next open
        stop  = sig["stop"]

        if entry <= stop:
            log.warning("Invalid stop for %s: entry=%.6f stop=%.6f", sym, entry, stop)
            continue

        try:
            qty = risk.size_position(equity, entry, stop)
        except Exception as e:
            log.warning("size_position failed for %s: %s", sym, e)
            continue

        # Sanity: position value < 30% of equity
        pos_value = qty * entry
        if pos_value > equity * 0.30:
            log.warning("%s position value %.2f > 30%% of equity — skip", sym, pos_value)
            continue

        log.info("ENTRY: %s  entry~%.6f  stop=%.6f  qty=%.6f  fz=%.2f  oiz=%.2f",
                 sym, entry, stop, qty,
                 sig.get("funding_z", 0), sig.get("oi_z", 0))

        success = handler.open_position(sym, entry, stop, qty, meta=sig)
        if success:
            equity = ex.get_usdt_balance()
            state  = load_state()

    # ── Step 5: Save final state ──────────────────────────────────────────────
    state   = load_state()
    equity  = ex.get_usdt_balance()
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
