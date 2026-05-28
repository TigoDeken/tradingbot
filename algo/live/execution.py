"""
Order execution and position lifecycle.

open_position:  market buy → place stop-loss → save to state
close_position: cancel stop → market sell → log trade
sync_positions: detect stops hit by exchange; re-place stop if it disappeared
"""

import time
from datetime import datetime, timezone
from algo.live.exchange import BybitExchange
from algo.live.logger   import get_logger
from algo.engine.state  import (
    load_state, save_state, add_position, remove_position,
    get_position, log_trade,
)

log        = get_logger("execution")
ORDER_RETRY = 3


class ExecutionHandler:
    def __init__(self, exchange: BybitExchange):
        self.ex = exchange

    # ── Open ─────────────────────────────────────────────────────────────────

    def open_position(self, symbol: str, entry: float, stop: float,
                      qty: float, meta: dict | None = None) -> bool:
        """Market buy + stop-loss order + persist to state. Returns True on success."""
        meta = meta or {}

        # 1. Market buy (with retry)
        buy_order_id = None
        for attempt in range(1, ORDER_RETRY + 1):
            try:
                result       = self.ex.place_market_buy(symbol, qty)
                buy_order_id = result.get("orderId")
                break
            except Exception as e:
                log.warning("Buy attempt %d/%d for %s: %s", attempt, ORDER_RETRY, symbol, e)
                if attempt == ORDER_RETRY:
                    log.error("Buy abandoned for %s after %d attempts", symbol, ORDER_RETRY)
                    return False
                time.sleep(2 ** attempt)

        # 2. Get actual fill price
        fill_price = entry  # fallback
        if buy_order_id:
            time.sleep(1)  # allow fill to settle
            actual = self.ex.get_order_fill_price(symbol, buy_order_id)
            if actual:
                fill_price = actual
                log.info("Fill price confirmed: %s @ %.6f", symbol, fill_price)
        if fill_price == entry:
            try:
                fill_price = self.ex.get_ticker(symbol)["last"]
            except Exception:
                pass

        # 3. Stop-loss order (with retry)
        stop_order_id = None
        for attempt in range(1, ORDER_RETRY + 1):
            try:
                stop_result   = self.ex.place_stop_sell(symbol, qty, stop)
                stop_order_id = stop_result.get("orderId")
                break
            except Exception as e:
                log.warning("Stop attempt %d/%d for %s: %s", attempt, ORDER_RETRY, symbol, e)
                if attempt == ORDER_RETRY:
                    log.error("CRITICAL: Could not place stop for %s — POSITION UNPROTECTED", symbol)
                time.sleep(2 ** attempt)

        # 4. Persist
        pos = {
            "symbol":        symbol,
            "entry_price":   fill_price,
            "stop":          stop,
            "qty":           qty,
            "entry_date":    datetime.now(timezone.utc).isoformat(),
            "stop_order_id": stop_order_id,
            "entry_fz":      meta.get("funding_z"),
            "entry_oiz":     meta.get("oi_z"),
            "entry_atr":     meta.get("atr_ratio"),
        }
        state = load_state()
        state = add_position(state, pos)
        save_state(state)

        if stop_order_id:
            log.info("OPENED %s  entry=%.6f  stop=%.6f  qty=%.6f  stop_order=%s",
                     symbol, fill_price, stop, qty, stop_order_id)
        else:
            log.error("OPENED (UNPROTECTED) %s  entry=%.6f  stop=%.6f  qty=%.6f",
                      symbol, fill_price, stop, qty)
        return True

    # ── Close ─────────────────────────────────────────────────────────────────

    def close_position(self, symbol: str, reason: str = "signal") -> bool:
        """Cancel stop order + market sell + log trade. Returns True on success."""
        state = load_state()
        pos   = get_position(state, symbol)
        if not pos:
            log.warning("close_position: no open position for %s", symbol)
            return False

        # Cancel stop order first
        if pos.get("stop_order_id"):
            self.ex.cancel_order(symbol, pos["stop_order_id"])
            time.sleep(0.5)

        # Market sell (with retry)
        qty = pos["qty"]
        for attempt in range(1, ORDER_RETRY + 1):
            try:
                self.ex.place_market_sell(symbol, qty)
                break
            except Exception as e:
                log.warning("Sell attempt %d/%d for %s: %s", attempt, ORDER_RETRY, symbol, e)
                if attempt == ORDER_RETRY:
                    log.error("Sell abandoned for %s — position may still be open", symbol)
                    return False
                time.sleep(2 ** attempt)

        # Get exit price
        exit_price = pos["entry_price"]
        try:
            exit_price = self.ex.get_ticker(symbol)["last"]
        except Exception:
            pass

        entry = float(pos["entry_price"])
        ret   = (exit_price - entry) / entry * 100 if entry else 0
        pnl   = (exit_price - entry) * float(pos["qty"])

        log_trade({
            "symbol":      symbol,
            "entry_date":  pos.get("entry_date"),
            "entry_price": entry,
            "exit_price":  exit_price,
            "qty":         pos["qty"],
            "stop":        pos.get("stop"),
            "ret_pct":     round(ret, 4),
            "pnl_usdt":    round(pnl, 2),
            "exit_reason": reason,
            "entry_fz":    pos.get("entry_fz"),
            "entry_oiz":   pos.get("entry_oiz"),
            "entry_atr":   pos.get("entry_atr"),
        })

        state = remove_position(state, symbol)
        save_state(state)
        log.info("CLOSED %s  exit=%.6f  ret=%.2f%%  pnl=%.2f  reason=%s",
                 symbol, exit_price, ret, pnl, reason)
        return True

    # ── Sync ─────────────────────────────────────────────────────────────────

    def sync_positions(self) -> list[str]:
        """
        Detect positions closed by the exchange (stop triggered or manual close).
        Also re-places a stop if it has gone missing.
        Returns list of symbols that were stopped out.
        """
        state   = load_state()
        stopped = []
        changed = False

        for pos in list(state["positions"]):
            symbol        = pos["symbol"]
            stop_order_id = pos.get("stop_order_id")

            try:
                coin_bal         = self.ex.get_coin_balance(symbol)
                position_gone    = coin_bal < float(pos["qty"]) * 0.05

                if not stop_order_id:
                    log.warning("UNPROTECTED: %s has no stop_order_id — checking balance", symbol)
                    if position_gone:
                        log.info("Position for %s is gone (balance=%.6f) — recording as closed", symbol, coin_bal)
                        self._record_close(pos, pos.get("stop", pos["entry_price"]), "stop", state)
                        stopped.append(symbol)
                        changed = True
                    continue

                # Check if stop order is still open
                open_stops  = self.ex.get_open_stop_orders(symbol)
                open_ids    = {o["orderId"] for o in open_stops}
                stop_active = stop_order_id in open_ids

                if not stop_active and position_gone:
                    # Stop was triggered and position is closed
                    log.info("STOP HIT confirmed: %s (balance=%.6f)", symbol, coin_bal)
                    self._record_close(pos, pos.get("stop", pos["entry_price"]), "stop", state)
                    stopped.append(symbol)
                    changed = True

                elif not stop_active and not position_gone:
                    # Stop order disappeared but we still hold the coin — re-place it
                    log.error("Stop order gone for %s but coins remain — re-placing stop", symbol)
                    try:
                        result = self.ex.place_stop_sell(symbol, pos["qty"], float(pos["stop"]))
                        new_id = result.get("orderId")
                        pos["stop_order_id"] = new_id
                        log.info("Re-placed stop for %s: order %s", symbol, new_id)
                        changed = True
                    except Exception as e:
                        log.error("Could not re-place stop for %s: %s — UNPROTECTED", symbol, e)

            except Exception as e:
                log.warning("sync_positions error for %s: %s", symbol, e)

        if changed:
            save_state(state)
        return stopped

    # ── Internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _record_close(pos: dict, exit_price: float, reason: str, state: dict):
        """Log the trade and remove from state in-place."""
        entry = float(pos["entry_price"])
        qty   = float(pos["qty"])
        ret   = (exit_price - entry) / entry * 100 if entry else 0
        pnl   = (exit_price - entry) * qty

        log_trade({
            "symbol":      pos["symbol"],
            "entry_date":  pos.get("entry_date"),
            "entry_price": entry,
            "exit_price":  exit_price,
            "qty":         qty,
            "stop":        pos.get("stop"),
            "ret_pct":     round(ret, 4),
            "pnl_usdt":    round(pnl, 2),
            "exit_reason": reason,
            "entry_fz":    pos.get("entry_fz"),
            "entry_oiz":   pos.get("entry_oiz"),
            "entry_atr":   pos.get("entry_atr"),
        })
        state["positions"] = [p for p in state["positions"] if p["symbol"] != pos["symbol"]]
