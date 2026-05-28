"""
Bybit exchange adapter — spot margin trading.

category="spot" for all order calls.
Borrow is automatic in Unified Trading Account when leverage > 1x.
"""

import os
import time
import math
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
from algo.live.logger import get_logger

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
log = get_logger("exchange")


class BybitExchange:
    def __init__(self):
        self._session = HTTP(
            testnet=False,
            api_key=os.getenv("BYBIT_API_KEY"),
            api_secret=os.getenv("BYBIT_API_SECRET"),
        )
        self._instruments: dict = {}

    # ── Market data ──────────────────────────────────────────────────────────

    def get_ticker(self, symbol: str) -> dict:
        r = self._session.get_tickers(category="spot", symbol=symbol)
        t = r["result"]["list"][0]
        return {
            "last": float(t["lastPrice"]),
            "bid":  float(t["bid1Price"]),
            "ask":  float(t["ask1Price"]),
        }

    def get_instrument_info(self, symbol: str) -> dict:
        if symbol in self._instruments:
            return self._instruments[symbol]
        r    = self._session.get_instruments_info(category="spot", symbol=symbol)
        info = r["result"]["list"][0]
        lot  = info["lotSizeFilter"]
        pri  = info["priceFilter"]
        out  = {
            "qty_step":  float(lot["basePrecision"]),
            "min_qty":   float(lot["minOrderQty"]),
            "max_qty":   float(lot["maxOrderQty"]),
            "tick_size": float(pri["tickSize"]),
        }
        self._instruments[symbol] = out
        return out

    # ── Account ──────────────────────────────────────────────────────────────

    def get_usdt_balance(self) -> float:
        r     = self._session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        coins = r["result"]["list"][0]["coin"]
        for c in coins:
            if c["coin"] == "USDT":
                return float(c["availableToWithdraw"] or c["walletBalance"] or 0)
        return 0.0

    def get_coin_balance(self, symbol: str) -> float:
        coin  = symbol.replace("USDT", "")
        r     = self._session.get_wallet_balance(accountType="UNIFIED", coin=coin)
        items = r["result"]["list"][0]["coin"]
        for c in items:
            if c["coin"] == coin:
                return float(c["walletBalance"] or 0)
        return 0.0

    # ── Orders ───────────────────────────────────────────────────────────────

    def place_market_buy(self, symbol: str, qty: float) -> dict:
        info = self.get_instrument_info(symbol)
        qty  = self._round_qty(qty, info["qty_step"])
        if qty < info["min_qty"]:
            raise ValueError(f"{symbol}: qty {qty} < min_qty {info['min_qty']}")
        log.info("BUY  %s qty=%.6f", symbol, qty)
        r = self._session.place_order(
            category="spot", symbol=symbol,
            side="Buy", orderType="Market", qty=str(qty),
        )
        return r["result"]

    def place_stop_sell(self, symbol: str, qty: float, stop_price: float) -> dict:
        """Conditional market-sell that triggers when price falls to stop_price."""
        info       = self.get_instrument_info(symbol)
        qty        = self._round_qty(qty, info["qty_step"])
        stop_price = self._round_price(stop_price, info["tick_size"])
        log.info("STOP %s qty=%.6f stop=%.6f", symbol, qty, stop_price)
        r = self._session.place_order(
            category="spot", symbol=symbol,
            side="Sell", orderType="Market",
            qty=str(qty),
            triggerPrice=str(stop_price),
            triggerDirection=2,      # trigger when price falls below triggerPrice
            orderFilter="StopOrder",
        )
        return r["result"]

    def place_market_sell(self, symbol: str, qty: float) -> dict:
        info = self.get_instrument_info(symbol)
        qty  = self._round_qty(qty, info["qty_step"])
        log.info("SELL %s qty=%.6f", symbol, qty)
        r = self._session.place_order(
            category="spot", symbol=symbol,
            side="Sell", orderType="Market", qty=str(qty),
        )
        return r["result"]

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        try:
            self._session.cancel_order(category="spot", symbol=symbol, orderId=order_id)
            log.info("CANCELLED order %s on %s", order_id, symbol)
            return True
        except Exception as e:
            log.warning("Cancel failed %s %s: %s", symbol, order_id, e)
            return False

    def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        """Fetch regular open orders."""
        p = dict(category="spot", openOnly=1)
        if symbol:
            p["symbol"] = symbol
        r = self._session.get_open_orders(**p)
        return r["result"]["list"]

    def get_open_stop_orders(self, symbol: str | None = None) -> list[dict]:
        """Fetch open conditional (stop) orders."""
        p = dict(category="spot", openOnly=1, orderFilter="StopOrder")
        if symbol:
            p["symbol"] = symbol
        r = self._session.get_open_orders(**p)
        return r["result"]["list"]

    def get_order_fill_price(self, symbol: str, order_id: str) -> float | None:
        """Return average fill price for a completed order, or None if unavailable."""
        try:
            r      = self._session.get_order_history(category="spot", symbol=symbol, orderId=order_id)
            orders = r["result"]["list"]
            if orders:
                avg = float(orders[0].get("avgPrice") or 0)
                return avg if avg > 0 else None
        except Exception as e:
            log.warning("get_order_fill_price %s %s: %s", symbol, order_id, e)
        return None

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _round_qty(qty: float, step: float) -> float:
        if step <= 0:
            return qty
        decimals = max(0, -int(math.floor(math.log10(step))))
        return round(math.floor(qty / step) * step, decimals)

    @staticmethod
    def _round_price(price: float, tick: float) -> float:
        if tick <= 0:
            return price
        decimals = max(0, -int(math.floor(math.log10(tick))))
        return round(round(price / tick) * tick, decimals)
