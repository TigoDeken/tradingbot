"""
bybit_execution.py
Bybit V5 authenticated order execution for linear perpetuals (USDT-margined).

Required env vars:
    BYBIT_API_KEY    — Bybit API key
    BYBIT_API_SECRET — Bybit API secret

Optional:
    BYBIT_TESTNET=1  — force testnet (config.json testnet flag takes precedence)
"""
import logging
import os

from pybit.unified_trading import HTTP


def _check(resp: dict, ctx: str) -> None:
    if resp.get("retCode") != 0:
        raise RuntimeError(
            f"Bybit error [{ctx}]: {resp.get('retMsg')} (retCode={resp.get('retCode')})"
        )


class BybitExecutor:
    """
    Thin wrapper around pybit HTTP for linear perpetual trading.
    All order quantities are in base currency (e.g. SOL, AVAX).
    All prices are in USDT.
    """

    def __init__(self, testnet: bool = True, logger: logging.Logger | None = None):
        self.testnet = testnet
        self.log     = logger or logging.getLogger(__name__)

        api_key    = os.environ.get("BYBIT_API_KEY", "")
        api_secret = os.environ.get("BYBIT_API_SECRET", "")
        if not api_key or not api_secret:
            raise EnvironmentError(
                "BYBIT_API_KEY and BYBIT_API_SECRET must be set for live trading."
            )

        self.session = HTTP(testnet=testnet, api_key=api_key, api_secret=api_secret)
        self.log.info(f"BybitExecutor ready (testnet={testnet})")

    # ── Account ───────────────────────────────────────────────────────────────

    def get_balance(self) -> float:
        """Return available USDT in the Unified Trading Account."""
        resp = self.session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        _check(resp, "get_balance")
        for c in resp["result"]["list"][0]["coin"]:
            if c["coin"] == "USDT":
                return float(c.get("availableToWithdraw") or c["walletBalance"])
        return 0.0

    # ── Positions ─────────────────────────────────────────────────────────────

    def get_positions(self, symbol: str = "") -> list[dict]:
        """Return open position dicts (size > 0) for USDT linear perps."""
        params = {"category": "linear", "settleCoin": "USDT"}
        if symbol:
            params["symbol"] = symbol
        resp = self.session.get_positions(**params)
        _check(resp, "get_positions")
        return [p for p in resp["result"]["list"] if float(p.get("size", 0)) > 0]

    # ── Orders ────────────────────────────────────────────────────────────────

    def place_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        reduce_only: bool = False,
    ) -> dict:
        """
        Place a market order on Bybit linear perps.

        Args:
            symbol:      e.g. "SOLUSDT"
            side:        "Buy" (long) or "Sell" (short / close-long)
            qty:         base-currency quantity (e.g. 5.0 SOL)
            stop_loss:   SL trigger price — Bybit manages execution server-side
            take_profit: TP trigger price — Bybit manages execution server-side
            reduce_only: True for close-only orders

        Returns:
            Bybit order result dict (contains orderId, etc.)
        """
        kwargs: dict = {
            "category":    "linear",
            "symbol":      symbol,
            "side":        side,
            "orderType":   "Market",
            "qty":         str(qty),
            "timeInForce": "IOC",
        }
        if stop_loss is not None:
            kwargs["stopLoss"]    = str(stop_loss)
            kwargs["slTriggerBy"] = "LastPrice"
        if take_profit is not None:
            kwargs["takeProfit"]  = str(take_profit)
            kwargs["tpTriggerBy"] = "LastPrice"
        if reduce_only:
            kwargs["reduceOnly"] = True

        resp = self.session.place_order(**kwargs)
        _check(resp, f"place_market_order {side} {qty} {symbol}")
        oid = resp["result"]["orderId"]
        self.log.info(f"[{symbol}] Market {side} {qty} — orderId={oid}")
        return resp["result"]

    def close_position(self, symbol: str) -> None:
        """Close the entire open position for a symbol."""
        positions = self.get_positions(symbol)
        if not positions:
            self.log.info(f"[{symbol}] No open position to close.")
            return
        pos        = positions[0]
        size       = float(pos["size"])
        close_side = "Sell" if pos["side"] == "Buy" else "Buy"
        self.place_market_order(symbol, close_side, size, reduce_only=True)
        self.log.info(f"[{symbol}] Closed {pos['side']} position ({size} contracts).")

    def cancel_all_orders(self, symbol: str) -> None:
        """Cancel all open orders for a symbol."""
        resp = self.session.cancel_all_orders(category="linear", symbol=symbol)
        if resp.get("retCode") != 0:
            self.log.warning(f"[{symbol}] cancel_all_orders: {resp.get('retMsg')}")
        else:
            self.log.info(f"[{symbol}] All open orders cancelled.")

    def set_leverage(self, symbol: str, leverage: int = 1) -> None:
        """Set leverage for a symbol. 1x = no leverage (safest default)."""
        resp = self.session.set_leverage(
            category="linear", symbol=symbol,
            buyLeverage=str(leverage), sellLeverage=str(leverage),
        )
        code = resp.get("retCode")
        if code not in (0, 110043):   # 110043 = already at requested leverage
            self.log.warning(f"[{symbol}] set_leverage {leverage}x failed: {resp.get('retMsg')}")
        else:
            self.log.info(f"[{symbol}] Leverage confirmed at {leverage}x")
