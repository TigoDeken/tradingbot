"""
live_trader.py
Live and paper-mode execution engine for Bybit linear perpetuals (USDT-margined).
Runs on every M5 candle close (or configurable timeframe).
Supports multiple symbols simultaneously via per-symbol state files.

Usage:
    python live_trader.py            # normal run
    python live_trader.py --once     # process one candle then exit (testing)
    python live_trader.py --test-cb  # force circuit-breaker trigger and exit (testing)

Environment variables (live mode only):
    BYBIT_API_KEY    — Bybit API key
    BYBIT_API_SECRET — Bybit API secret
"""
import json
import logging
import math
import sys
import threading
import time
import traceback
from pathlib import Path

import pandas as pd
import requests

from bybit_data_pipeline import fetch_ohlc, get_symbol_info as bybit_get_symbol_info

# ── Constants ─────────────────────────────────────────────────────────────────
CONFIG_PATH       = Path("config.json")
LEGACY_STATE_PATH = Path("state.json")
LOG_PATH          = Path("trading_log.txt")

RECONNECT_INTERVAL = 60
ORDER_RETRY_LIMIT  = 3

_TF_BYBIT = {
    "M1": "1", "M3": "3", "M5": "5", "M15": "15", "M30": "30",
    "H1": "60", "H4": "240", "D1": "D",
}
_TF_MINUTES = {
    "M1": 1, "M3": 3, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440,
}


# ── Exceptions ────────────────────────────────────────────────────────────────

class TradingHalt(Exception):
    pass

class OrderSkip(Exception):
    """Entry order failed — skip this symbol this candle, do not halt."""
    pass


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("live_trader")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        fmt="%(asctime)s UTC [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logging.Formatter.converter = time.gmtime
    return logger


# ── Config ────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.json not found at {CONFIG_PATH.absolute()}")
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    if "symbol" in cfg and "symbols" not in cfg:
        cfg["symbols"] = [cfg["symbol"]]
    required = ["symbols", "timeframe", "paper_mode", "max_drawdown_pct"]
    missing  = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"config.json missing keys: {missing}")
    return cfg


# ── Symbol info ───────────────────────────────────────────────────────────────

def get_symbol_info(symbol: str, testnet: bool = False,
                    logger: logging.Logger | None = None) -> tuple:
    """
    Return (pip, pip_value, qty_step) for a Bybit USDT linear perpetual.

    For USDT-margined perps the accounting is simply:
      P&L = (exit_price - entry_price) * qty_coins * direction
    so we model: pip=1.0, pip_value=1.0, and track position size in coins.
    qty_step is the minimum order-size increment from Bybit's instrument spec.
    """
    try:
        info = bybit_get_symbol_info(symbol, testnet=testnet)
        return 1.0, 1.0, info["qty_step"]
    except Exception as e:
        if logger:
            logger.warning(f"[{symbol}] Could not fetch Bybit symbol info: {e} — using defaults")
        return 1.0, 1.0, 0.01


# ── State I/O ─────────────────────────────────────────────────────────────────

_DEFAULT_STATE: dict = {
    "circuit_breaker_active":   False,
    "circuit_breaker_reason":   None,
    "session_starting_balance": None,
    "current_balance":          None,
    "drawdown_pct":             0.0,
    "open_trade":               None,
    "pending_limit":            None,
    "last_updated":             None,
}


def _state_path(symbol: str) -> Path:
    return Path(f"state_{symbol}.json")


def load_state(symbol: str) -> dict:
    path = _state_path(symbol)
    if path.exists():
        with open(path) as f:
            saved = json.load(f)
        clean = {k: v for k, v in saved.items() if k in _DEFAULT_STATE}
        return {**_DEFAULT_STATE, **clean}
    return dict(_DEFAULT_STATE)


def save_state(state: dict, symbol: str) -> None:
    state["last_updated"] = pd.Timestamp.now(tz="UTC").isoformat()
    path = _state_path(symbol)
    tmp  = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    tmp.replace(path)


def _migrate_legacy_state(symbols: list, logger: logging.Logger) -> None:
    if not LEGACY_STATE_PATH.exists():
        return
    first  = symbols[0]
    target = _state_path(first)
    if not target.exists():
        LEGACY_STATE_PATH.rename(target)
        logger.info(f"Migrated state.json -> {target}")
    else:
        LEGACY_STATE_PATH.unlink()
        logger.info("Removed legacy state.json (per-symbol files already exist)")


# ── API reachability ──────────────────────────────────────────────────────────

def _api_reachable(testnet: bool = False) -> bool:
    base = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
    try:
        r = requests.get(f"{base}/v5/market/time", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def ensure_api_reachable(testnet: bool, logger: logging.Logger) -> None:
    if _api_reachable(testnet):
        return
    attempt = 0
    while True:
        attempt += 1
        logger.warning(f"Bybit API unreachable. Retry {attempt}...")
        time.sleep(RECONNECT_INTERVAL)
        if _api_reachable(testnet):
            logger.info("Bybit API reachable.")
            return


# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch_data(config: dict, logger: logging.Logger) -> pd.DataFrame:
    symbol  = config["symbol"]
    tf      = _TF_BYBIT.get(config["timeframe"], config["timeframe"])
    bars    = int(config.get("lookback_bars", 5000))
    testnet = config.get("testnet", False)
    df      = fetch_ohlc(symbol=symbol, interval=tf, bars=bars, testnet=testnet)
    logger.debug(f"[{symbol}] Fetched {len(df)} bars. Last: {df.index[-1]}")
    return df


# ── Circuit breaker ───────────────────────────────────────────────────────────

def _close_all_paper(state: dict, symbol: str, logger: logging.Logger) -> None:
    if state.get("open_trade"):
        logger.warning(f"[{symbol}] [CB] Closing paper position.")
        state["open_trade"] = None
    if state.get("pending_limit"):
        logger.warning(f"[{symbol}] [CB] Cancelling pending paper order.")
        state["pending_limit"] = None


def _close_all_live(symbol: str, executor, logger: logging.Logger) -> None:
    try:
        executor.cancel_all_orders(symbol)
        executor.close_position(symbol)
    except Exception as e:
        logger.error(f"[{symbol}] Error closing positions at circuit break: {e}")


def trigger_circuit_breaker(config: dict, state: dict, symbol: str,
                             reason: str, executor,
                             logger: logging.Logger) -> None:
    logger.critical(f"[{symbol}] CIRCUIT BREAKER: {reason}")
    state["circuit_breaker_active"] = True
    state["circuit_breaker_reason"] = reason
    if config["paper_mode"]:
        _close_all_paper(state, symbol, logger)
    else:
        _close_all_live(symbol, executor, logger)
        state["open_trade"]    = None
        state["pending_limit"] = None
    save_state(state, symbol)
    logger.critical(
        f"[{symbol}] Trading halted. Edit state_{symbol}.json and set "
        f"circuit_breaker_active=false to resume."
    )


def check_circuit_breaker(config: dict, state: dict, symbol: str,
                           executor, logger: logging.Logger) -> bool:
    if state["circuit_breaker_active"]:
        return True
    start = state.get("session_starting_balance")
    cur   = state.get("current_balance")
    if start and cur and start > 0:
        dd_pct = (start - cur) / start * 100
        state["drawdown_pct"] = round(dd_pct, 3)
        if dd_pct >= config["max_drawdown_pct"]:
            trigger_circuit_breaker(
                config, state, symbol,
                f"Drawdown {dd_pct:.2f}% exceeded limit {config['max_drawdown_pct']}%",
                executor, logger,
            )
            return True
    return False


# ── Account helpers ───────────────────────────────────────────────────────────

def get_balance(config: dict, state: dict, executor=None) -> float:
    if config["paper_mode"]:
        return float(state.get("current_balance") or 10_000.0)
    if executor is not None:
        return executor.get_balance()
    return float(state.get("current_balance") or 0.0)


def compute_lot(config: dict, balance: float, entry: float, stop: float,
                pip: float, pip_value: float, qty_step: float = 0.01) -> float:
    """
    Compute position size in base-currency units (e.g. SOL).

    With pip=1, pip_value=1 (USDT perps):
      qty = risk_usd / stop_distance_usd
    Result is rounded down to the nearest qty_step.
    """
    if config.get("risk_mode") == "fixed":
        risk = float(config.get("fixed_risk_amount", 0.0))
    else:
        risk = balance * float(config.get("risk_per_trade", 0.01))

    stop_dist = abs(entry - stop) / pip
    if stop_dist <= 0 or pip_value <= 0:
        return 0.0

    lot_raw = risk / (stop_dist * pip_value)
    lot     = math.floor(lot_raw / qty_step) * qty_step
    lot     = round(lot, 8)
    if 0 < lot < qty_step:
        lot = qty_step
    return lot


# ── Paper-mode helpers ────────────────────────────────────────────────────────

def paper_market_entry(state: dict, symbol: str, direction: str, entry: float,
                        stop: float, tp1: float, lot: float,
                        logger: logging.Logger) -> None:
    state["open_trade"] = {
        "direction":  direction,
        "entry":      entry,
        "stop":       stop,
        "tp1":        tp1,
        "lot":        lot,
        "entry_time": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    logger.info(
        f"[{symbol}] [PAPER] Market {direction.upper()} {lot:.4f} @ {entry:.4f}  "
        f"SL={stop:.4f}  TP={tp1:.4f}"
    )


def paper_close_trade(config: dict, state: dict, symbol: str, exit_price: float,
                       reason: str, pip: float, pip_value: float,
                       logger: logging.Logger) -> None:
    t    = state["open_trade"]
    sign = 1 if t["direction"] == "long" else -1

    gross_per_unit = sign * (exit_price - t["entry"]) / pip
    slip_per_unit  = float(config.get("slippage_pips", 0.1))
    net_per_unit   = gross_per_unit - slip_per_unit
    pnl            = net_per_unit * pip_value * t["lot"]

    # Commission: Bybit taker 0.055% × 2 sides round trip
    comm_pct = float(config.get("commission_pct", 0.00055)) * 2
    avg_px   = (exit_price + t["entry"]) / 2
    pnl     -= avg_px * t["lot"] * comm_pct

    state["current_balance"] = round((state.get("current_balance") or 0.0) + pnl, 2)
    state["open_trade"]      = None
    logger.info(
        f"[{symbol}] [PAPER] Closed ({reason}): exit={exit_price:.4f}  "
        f"P&L=${pnl:.2f}  balance=${state['current_balance']:.2f}"
    )


# ── Live-mode order helpers ───────────────────────────────────────────────────

def live_market_entry(config: dict, state: dict, symbol: str,
                       direction: str, entry: float, stop: float, tp1: float,
                       lot: float, executor, logger: logging.Logger) -> None:
    side = "Buy" if direction == "long" else "Sell"
    try:
        result = executor.place_market_order(
            symbol=symbol, side=side, qty=lot,
            stop_loss=stop, take_profit=tp1,
        )
    except Exception as e:
        raise OrderSkip(str(e))

    state["open_trade"] = {
        "direction":  direction,
        "entry":      entry,
        "stop":       stop,
        "tp1":        tp1,
        "lot":        lot,
        "order_id":   result.get("orderId"),
        "entry_time": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    logger.info(
        f"[{symbol}] [LIVE] Market {direction.upper()} {lot} @ ~{entry:.4f}  "
        f"SL={stop:.4f}  TP={tp1:.4f}  orderId={result.get('orderId')}"
    )


# ── Session check ─────────────────────────────────────────────────────────────

def in_session(config: dict, ts: pd.Timestamp) -> bool:
    # Crypto trades 24/7 — disable session filter if session_filter=false
    if not config.get("session_filter", True):
        return True
    from regime_filter import SESSION_WINDOWS
    sym      = config.get("symbol", "")
    sess_cfg = config.get("instrument_sessions", {}).get(sym, {"london": True, "ny": True})
    h        = ts.hour
    for name, enabled in sess_cfg.items():
        if enabled and name in SESSION_WINDOWS:
            start, end = SESSION_WINDOWS[name]
            if start <= h < end:
                return True
    return False


# ── Trade management ──────────────────────────────────────────────────────────

def _manage_paper_trade(config: dict, state: dict, symbol: str,
                         last_bar: pd.Series, pip: float, pip_value: float,
                         logger: logging.Logger) -> None:
    t         = state["open_trade"]
    h, l      = float(last_bar["High"]), float(last_bar["Low"])
    direction = t["direction"]
    long      = direction == "long"

    stop_dist = (l - t["stop"]) / pip if long else (t["stop"] - h) / pip
    tp_dist   = (t["tp1"] - h)  / pip if long else (l - t["tp1"]) / pip
    logger.info(
        f"[{symbol}] [PAPER] Managing {direction.upper()}  "
        f"entry={t['entry']:.4f}  stop={t['stop']:.4f}  tp={t['tp1']:.4f}  "
        f"H={h:.4f}  L={l:.4f}  "
        f"dist-stop={stop_dist:.2f}  dist-tp={tp_dist:.2f}"
    )

    # Conservative tie-break: check stop before TP
    if (long and l <= t["stop"]) or (not long and h >= t["stop"]):
        paper_close_trade(config, state, symbol, t["stop"], "stop_hit",
                          pip, pip_value, logger)
        return

    if (long and h >= t["tp1"]) or (not long and l <= t["tp1"]):
        paper_close_trade(config, state, symbol, t["tp1"], "tp1_hit",
                          pip, pip_value, logger)


def _manage_live_trade(config: dict, state: dict, symbol: str,
                        executor, logger: logging.Logger) -> None:
    """
    For live Bybit trades SL/TP are managed server-side.
    Poll positions to detect when the exchange has closed the trade.
    """
    try:
        positions = executor.get_positions(symbol)
    except Exception as e:
        logger.error(f"[{symbol}] Could not fetch positions: {e}")
        return

    t = state.get("open_trade")
    if not positions:
        if t:
            logger.info(f"[{symbol}] Position closed by exchange (SL/TP). Clearing state.")
            state["open_trade"]    = None
            state["pending_limit"] = None
            try:
                bal = executor.get_balance()
                state["current_balance"] = bal
                logger.info(f"[{symbol}] Balance after close: ${bal:.2f}")
            except Exception as e:
                logger.warning(f"[{symbol}] Could not refresh balance: {e}")
    else:
        if t is None:
            pos       = positions[0]
            direction = "long" if pos["side"] == "Buy" else "short"
            state["open_trade"] = {
                "direction": direction,
                "entry":     float(pos.get("avgPrice", 0)),
                "stop":      float(pos.get("stopLoss", 0)),
                "tp1":       float(pos.get("takeProfit", 0)),
                "lot":       float(pos.get("size", 0)),
            }
            logger.warning(f"[{symbol}] Reconciled position from Bybit: {state['open_trade']}")


# ── Entry logic ───────────────────────────────────────────────────────────────

def check_entry(config: dict, state: dict, df: pd.DataFrame, symbol: str,
                pip: float, pip_value: float, qty_step: float,
                executor, logger: logging.Logger) -> None:
    if len(df) < 1:
        return

    bar_time = df.index[-1]
    if not in_session(config, bar_time):
        logger.info(f"[{symbol}] No entry — outside session ({bar_time.hour:02d}:00 UTC).")
        return

    curr = df.iloc[-1]
    logger.info(
        f"[{symbol}] Evaluating bar: O={curr['Open']:.4f}  H={curr['High']:.4f}"
        f"  L={curr['Low']:.4f}  C={curr['Close']:.4f}"
    )

    tp_rr     = float(config.get("tp_rr", 2.0))
    all_hours = not config.get("session_filter", True)

    from nbar_breakout_engine import NBAR_SYMBOLS, compute_live_signal
    if symbol in NBAR_SYMBOLS:
        sig = compute_live_signal(df, symbol, all_hours=all_hours)
        if sig is None:
            logger.info(f"[{symbol}] No nbar signal.")
            return

        direction  = sig["direction"]
        atr_val    = sig["atr"]
        entry      = float(curr["Close"])
        stop_mult  = float(config.get("stop_atr_mult", 1.0))
        risk_dist  = stop_mult * atr_val
        stop       = entry - risk_dist if direction == "long" else entry + risk_dist
        tp1        = entry + tp_rr * risk_dist if direction == "long" else entry - tp_rr * risk_dist

        if risk_dist <= 0:
            logger.warning(f"[{symbol}] Zero stop distance — skipping.")
            return

        balance = get_balance(config, state, executor)
        lot     = compute_lot(config, balance, entry, stop, pip, pip_value, qty_step)
        if lot <= 0:
            logger.info(
                f"[{symbol}] Lot=0 "
                f"(balance=${balance:.0f}  stop_dist={risk_dist:.4f})."
            )
            return

        logger.info(
            f"[{symbol}] NBAR {direction.upper()}: "
            f"entry={entry:.4f}  stop={stop:.4f}  tp={tp1:.4f}  "
            f"ATR={atr_val:.4f}  lot={lot:.4f}"
        )
        if config["paper_mode"]:
            paper_market_entry(state, symbol, direction, entry, stop, tp1, lot, logger)
        else:
            live_market_entry(config, state, symbol, direction, entry, stop, tp1,
                              lot, executor, logger)
        return

    logger.info(f"[{symbol}] No entry strategy configured.")


# ── Main candle handler ───────────────────────────────────────────────────────

def run_candle(config: dict, state: dict, df: pd.DataFrame, symbol: str,
               pip: float, pip_value: float, qty_step: float,
               executor, logger: logging.Logger) -> None:
    last_bar = df.iloc[-1]
    bar_time = df.index[-1]

    logger.info(
        f"[{symbol}] Candle {bar_time}  close={last_bar['Close']:.4f}  "
        f"qty_step={qty_step}"
    )

    if check_circuit_breaker(config, state, symbol, executor, logger):
        return

    if state["session_starting_balance"] is None:
        bal = get_balance(config, state, executor)
        state["session_starting_balance"] = bal
        state["current_balance"]          = bal
        logger.info(f"[{symbol}] Session starting balance: ${bal:.2f}")

    if state.get("open_trade"):
        if config["paper_mode"]:
            _manage_paper_trade(config, state, symbol, last_bar, pip, pip_value, logger)
        else:
            _manage_live_trade(config, state, symbol, executor, logger)

    if not state.get("open_trade") and not state.get("pending_limit"):
        check_entry(config, state, df, symbol, pip, pip_value, qty_step,
                    executor, logger)

    save_state(state, symbol)
    logger.info(
        f"[{symbol}] State saved — balance=${state.get('current_balance', '?')}  "
        f"open={bool(state.get('open_trade'))}"
    )


# ── Timing ────────────────────────────────────────────────────────────────────

def wait_for_next_candle(last_bar_times: dict, tf_minutes: int,
                          logger: logging.Logger) -> None:
    if not last_bar_times:
        time.sleep(60)
        return
    earliest_next = min(
        t + pd.Timedelta(minutes=tf_minutes, seconds=30)
        for t in last_bar_times.values()
    )
    now        = pd.Timestamp.now(tz="UTC")
    sleep_secs = (earliest_next - now).total_seconds()
    if sleep_secs > 0:
        logger.info(
            f"Sleeping {sleep_secs:.0f}s until next candle "
            f"({earliest_next.strftime('%Y-%m-%d %H:%M UTC')})"
        )
        time.sleep(sleep_secs)
    logger.info("Waking up — polling for new candle...")


# ── Live mode confirmation ────────────────────────────────────────────────────

def _live_mode_confirmation(config: dict, executor, logger: logging.Logger) -> None:
    balance_str = "N/A"
    try:
        bal = executor.get_balance()
        balance_str = f"${bal:,.2f} USDT"
    except Exception:
        pass

    testnet     = config.get("testnet", True)
    symbols_str = ", ".join(config.get("symbols", [config.get("symbol", "?")]))
    sep    = "=" * 60
    banner = (
        f"\n{sep}\n"
        f"  LIVE TRADING MODE ACTIVE\n"
        f"  Exchange:        Bybit {'TESTNET' if testnet else 'MAINNET'}\n"
        f"  Symbols:         {symbols_str}\n"
        f"  Risk per trade:  {config.get('risk_per_trade', 0) * 100:.1f}%\n"
        f"  Fixed risk:      ${config.get('fixed_risk_amount', 0):.2f} USDT\n"
        f"  Max drawdown:    {config['max_drawdown_pct']}%\n"
        f"  Circuit breaker: ACTIVE\n"
        f"  Balance:         {balance_str}\n"
        + sep
    )
    print(banner)

    if "--live-confirmed" in sys.argv:
        logger.info("Live mode confirmed via --live-confirmed flag.")
        print("\nConfirmed. Starting live trading engine...\n")
        return

    logger.critical("LIVE TRADING MODE ACTIVE — awaiting terminal confirmation")
    answer: list[str] = []
    confirmed = threading.Event()

    def _read() -> None:
        try:
            answer.append(
                input("\nType 'yes' and press Enter to confirm (30s timeout): ").strip().lower()
            )
        except Exception:
            pass
        confirmed.set()

    threading.Thread(target=_read, daemon=True).start()
    confirmed.wait(timeout=30)

    if not answer or answer[0] != "yes":
        logger.critical("Confirmation not received — shutting down safely.")
        print("\nConfirmation not received. Shutting down safely.\n")
        sys.exit(1)

    logger.info("Live mode confirmed by operator.")
    print("\nConfirmed. Starting live trading engine...\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    once    = "--once"    in sys.argv
    test_cb = "--test-cb" in sys.argv

    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("Live Trader starting up — Bybit crypto edition")
    logger.info("=" * 60)

    config  = load_config()
    symbols = config["symbols"]
    testnet = config.get("testnet", True)
    mode    = "PAPER" if config["paper_mode"] else "LIVE"

    logger.info(
        f"Mode: {mode}  Symbols: {symbols}  TF: {config['timeframe']}  "
        f"Bybit {'testnet' if testnet else 'mainnet'}"
    )

    _migrate_legacy_state(symbols, logger)

    for sym in symbols:
        state = load_state(sym)
        if state.get("circuit_breaker_active"):
            logger.critical(
                f"STARTUP BLOCKED — [{sym}] circuit breaker active. "
                f"Reason: {state.get('circuit_breaker_reason')}. "
                f"Edit state_{sym}.json and set circuit_breaker_active=false to resume."
            )
            sys.exit(1)

    # Executor only needed for live mode
    executor = None
    if not config["paper_mode"]:
        from bybit_execution import BybitExecutor
        executor = BybitExecutor(testnet=testnet, logger=logger)

    if test_cb:
        sym   = symbols[0]
        state = load_state(sym)
        logger.info(f"TEST MODE: Forcing circuit breaker on {sym}...")
        if state["session_starting_balance"] is None:
            state["session_starting_balance"] = 10_000.0
            state["current_balance"]          = 10_000.0
        breached = state["session_starting_balance"] * (
            1 - (config["max_drawdown_pct"] + 1) / 100
        )
        state["current_balance"] = round(breached, 2)
        save_state(state, sym)
        triggered = check_circuit_breaker(config, state, sym, executor, logger)
        logger.info(f"Circuit breaker triggered: {triggered}")
        sys.exit(0)

    if not config["paper_mode"]:
        _live_mode_confirmation(config, executor, logger)
        for sym in symbols:
            try:
                executor.set_leverage(sym, leverage=1)
            except Exception as e:
                logger.warning(f"[{sym}] Could not set leverage: {e}")

    ensure_api_reachable(testnet, logger)

    sym_info: dict[str, tuple] = {}
    for sym in symbols:
        pip, pip_value, qty_step = get_symbol_info(sym, testnet=testnet, logger=logger)
        sym_info[sym] = (pip, pip_value, qty_step)
        logger.info(f"[{sym}] pip={pip}  pip_value={pip_value}  qty_step={qty_step}")

    tf_minutes    = _TF_MINUTES.get(config["timeframe"], 5)
    last_bar_times: dict[str, pd.Timestamp] = {}

    while True:
        try:
            processed_any = False

            for sym in list(sym_info.keys()):
                pip, pip_value, qty_step = sym_info[sym]
                sym_cfg = {**config, "symbol": sym}
                state   = load_state(sym)

                try:
                    df = fetch_data(sym_cfg, logger)
                except Exception as e:
                    logger.error(f"[{sym}] Failed to fetch data: {e}")
                    continue

                current_last = df.index[-1]
                prev_last    = last_bar_times.get(sym)

                if prev_last is None or current_last > prev_last:
                    last_bar_times[sym] = current_last
                    try:
                        run_candle(sym_cfg, state, df, sym, pip, pip_value, qty_step,
                                   executor, logger)
                    except OrderSkip as e:
                        logger.error(f"[{sym}] Order failed — skipping candle: {e}")
                    processed_any = True
                else:
                    logger.debug(f"[{sym}] No new candle. Last: {prev_last}")

            if once and processed_any:
                logger.info("--once flag: exiting after processing all symbols.")
                break
            elif once and not processed_any:
                for sym in list(sym_info.keys()):
                    pip, pip_value, qty_step = sym_info[sym]
                    sym_cfg = {**config, "symbol": sym}
                    state   = load_state(sym)
                    df      = fetch_data(sym_cfg, logger)
                    last_bar_times[sym] = df.index[-1]
                    try:
                        run_candle(sym_cfg, state, df, sym, pip, pip_value, qty_step,
                                   executor, logger)
                    except OrderSkip as e:
                        logger.error(f"[{sym}] Order failed — skipping candle: {e}")
                logger.info("--once flag: exiting.")
                break

            wait_for_next_candle(last_bar_times, tf_minutes, logger)

        except TradingHalt as e:
            logger.critical(f"TRADING HALT: {e}")
            sys.exit(1)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt — shutting down gracefully.")
            sys.exit(0)

        except Exception:
            logger.error(f"Unexpected exception:\n{traceback.format_exc()}")
            logger.error("Pausing 60s before retry.")
            time.sleep(60)


if __name__ == "__main__":
    cfg = load_config()
    if not cfg.get("paper_mode") and "--live-confirmed" not in sys.argv:
        print("\nLive mode detected. Add --live-confirmed flag to acknowledge real money trading.")
        print("Or set paper_mode=true in config.json for safe testing.\n")
        sys.exit(1)
    main()
