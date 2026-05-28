"""
Order execution and position lifecycle.

open_position:  market buy + place stop-loss order
close_position: cancel stop + market sell
sync_positions: check if any stop orders were filled by the exchange
"""

import time
from datetime import datetime, timezone
from algo.live.exchange import BybitExchange
from algo.live.logger  import get_logger
from algo.engine.state import (
    load_state, save_state, add_position, remove_position,
    get_position, log_trade
)

log = get_logger("execution")
ORDER_RETRY = 3


class ExecutionHandler:
    def __init__(self, exchange: BybitExchange):
        self.ex = exchange

    def open_position(self, symbol: str, entry: float, stop: float,
                      qty: float, meta: dict | None = None) -> bool:
        """
        1. Place market buy
        2. Place stop-loss sell order
        3. Save position to state
        """
        meta = meta or {}
        for attempt in range(1, ORDER_RETRY + 1):
            try:
                buy_result = self.ex.place_market_buy(symbol, qty)
                break
            except Exception as e:
                log.warning("Buy attempt %d failed for %s: %s", attempt, symbol, e)
                if attempt == ORDER_RETRY:
                    log.error("Buy abandoned for %s after %d attempts", symbol, ORDER_RETRY)
                    return False
                time.sleep(2 ** attempt)

        # Small pause to let the order settle
        time.sleep(1)

        # Try to get fill price from order result
        fill_price = entry  # fallback
        try:
            ticker = self.ex.get_ticker(symbol)
            fill_price = ticker["last"]
        except Exception:
            pass

        # Place stop
        stop_order_id = None
        for attempt in range(1, ORDER_RETRY + 1):
            try:
                stop_result   = self.ex.place_stop_sell(symbol, qty, stop)
                stop_order_id = stop_result.get("orderId")
                break
            except Exception as e:
                log.warning("Stop attempt %d failed for %s: %s", attempt, symbol, e)
                if attempt == ORDER_RETRY:
                    log.error("CRITICAL: Could not place stop for %s — POSITION UNPROTECTED", symbol)
                    break
                time.sleep(2 ** attempt)

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
        log.info("OPENED %s  entry=%.6f  stop=%.6f  qty=%.6f  stop_order=%s",
                 symbol, fill_price, stop, qty, stop_order_id)
        return True

    def close_position(self, symbol: str, reason: str = "signal") -> bool:
        """Cancel stop order and market sell."""
        state = load_state()
        pos   = get_position(state, symbol)
        if not pos:
            log.warning("close_position: no open position for %s", symbol)
            return False

        # Cancel the stop order
        if pos.get("stop_order_id"):
            self.ex.cancel_order(symbol, pos["stop_order_id"])
            time.sleep(0.5)

        # Market sell
        qty = pos["qty"]
        for attempt in range(1, ORDER_RETRY + 1):
            try:
                self.ex.place_market_sell(symbol, qty)
                break
            except Exception as e:
                log.warning("Sell attempt %d failed for %s: %s", attempt, symbol, e)
                if attempt == ORDER_RETRY:
                    log.error("Sell abandoned for %s", symbol)
                    return False
                time.sleep(2 ** attempt)

        # Get exit price
        exit_price = pos["entry_price"]
        try:
            exit_price = self.ex.get_ticker(symbol)["last"]
        except Exception:
            pass

        entry = pos["entry_price"]
        ret   = (exit_price - entry) / entry * 100
        pnl   = (exit_price - entry) * qty

        log_trade({
            "symbol":       symbol,
            "entry_date":   pos.get("entry_date"),
            "entry_price":  entry,
            "exit_price":   exit_price,
            "qty":          qty,
            "stop":         pos.get("stop"),
            "ret_pct":      round(ret, 4),
            "pnl_usdt":     round(pnl, 2),
            "exit_reason":  reason,
            "entry_fz":     pos.get("entry_fz"),
            "entry_oiz":    pos.get("entry_oiz"),
            "entry_atr":    pos.get("entry_atr"),
        })

        state = remove_position(state, symbol)
        save_state(state)
        log.info("CLOSED %s  exit=%.6f  ret=%.2f%%  pnl=%.2f  reason=%s",
                 symbol, exit_price, ret, pnl, reason)
        return True

    def sync_positions(self) -> list[str]:
        """
        Check if any stop orders were filled by the exchange (stop hit).
        Returns list of symbols where stop was hit.
        """
        state   = load_state()
        stopped = []

        for pos in list(state["positions"]):
            symbol = pos["symbol"]
            try:
                # Check if coin balance is near zero (stop was hit and position is closed)
                coin_bal = self.ex.get_coin_balance(symbol)
                if coin_bal < pos["qty"] * 0.01:  # less than 1% remaining = closed
                    log.info("STOP HIT detected for %s (coin balance %.6f)", symbol, coin_bal)
                    entry  = pos["entry_price"]
                    stop   = pos["stop"]
                    pnl    = (stop - entry) * pos["qty"]
                    ret    = (stop - entry) / entry * 100

                    log_trade({
                        "symbol":      symbol,
                        "entry_date":  pos.get("entry_date"),
                        "entry_price": entry,
                        "exit_price":  stop,
                        "qty":         pos["qty"],
                        "stop":        stop,
                        "ret_pct":     round(ret, 4),
                        "pnl_usdt":    round(pnl, 2),
                        "exit_reason": "stop",
                        "entry_fz":    pos.get("entry_fz"),
                        "entry_oiz":   pos.get("entry_oiz"),
                        "entry_atr":   pos.get("entry_atr"),
                    })
                    state = remove_position(state, symbol)
                    stopped.append(symbol)
            except Exception as e:
                log.warning("sync_positions error for %s: %s", symbol, e)

        if stopped:
            save_state(state)
        return stopped
