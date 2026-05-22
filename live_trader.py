"""
live_trader.py
Live and paper-mode execution engine. Runs on every 4H candle close.
Supports multiple symbols simultaneously via per-symbol state files.

Usage:
    python live_trader.py            # normal run
    python live_trader.py --once     # process one candle immediately, then exit (testing)
    python live_trader.py --test-cb  # force circuit breaker trigger and exit (testing)
"""
import json
import logging
import math
import sys
import threading
import time
import traceback
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from data_pipeline import connect as dp_connect, disconnect as dp_disconnect, fetch_ohlc
from constants     import MAGIC
from trade_engine  import _bar_setup, MIN_STOP_PIPS

# ── Constants ─────────────────────────────────────────────────────────────────
CONFIG_PATH         = Path("config.json")
LEGACY_STATE_PATH   = Path("state.json")   # old single-symbol state file
LOG_PATH            = Path("trading_log.txt")

RECONNECT_INTERVAL  = 60
ORDER_RETRY_LIMIT   = 3
POLL_INTERVAL       = 30


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
    # Support both old single-symbol ("symbol") and new multi-symbol ("symbols") config
    if "symbol" in cfg and "symbols" not in cfg:
        cfg["symbols"] = [cfg["symbol"]]
    required = ["symbols", "timeframe", "session_start_utc", "session_end_utc",
                "risk_per_trade", "tp_mode", "pullback_lookback", "stop_buffer",
                "paper_mode", "max_drawdown_pct"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"config.json missing keys: {missing}")
    return cfg


# ── Symbol info ───────────────────────────────────────────────────────────────

def get_symbol_info(symbol: str) -> tuple:
    """Return (pip, pip_value_per_lot) from MT5. Returns (None, None) if unavailable."""
    info = mt5.symbol_info(symbol)
    if info is None:
        return None, None
    # All standard forex pairs: pip = 10 × point (works for both 3-digit JPY and 5-digit EUR pairs)
    pip = info.point * 10
    pip_value = info.trade_tick_value * (pip / info.trade_tick_size)
    return round(pip, 6), round(pip_value, 4)


# ── State I/O ─────────────────────────────────────────────────────────────────

_DEFAULT_STATE: dict = {
    "circuit_breaker_active":  False,
    "circuit_breaker_reason":  None,
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
    """Move old state.json → state_EURUSD.json (or first symbol) on first multi-symbol run."""
    if not LEGACY_STATE_PATH.exists():
        return
    first = symbols[0]
    target = _state_path(first)
    if not target.exists():
        LEGACY_STATE_PATH.rename(target)
        logger.info(f"Migrated state.json -> {target}")
    else:
        LEGACY_STATE_PATH.unlink()
        logger.info("Removed legacy state.json (per-symbol files already exist)")


# ── MT5 connection ────────────────────────────────────────────────────────────

def _is_connected() -> bool:
    try:
        info = mt5.terminal_info()
        return info is not None and info.connected
    except Exception:
        return False


def ensure_connected(logger: logging.Logger) -> None:
    if _is_connected():
        return
    attempt = 0
    while True:
        attempt += 1
        logger.warning(f"MT5 disconnected. Reconnect attempt {attempt}...")
        if dp_connect():
            logger.info("MT5 reconnected successfully.")
            return
        logger.warning(f"Reconnect failed. Waiting {RECONNECT_INTERVAL}s...")
        time.sleep(RECONNECT_INTERVAL)


# ── Data fetch ────────────────────────────────────────────────────────────────

_TF_MAP = {
    "H4": mt5.TIMEFRAME_H4, "H1": mt5.TIMEFRAME_H1,
    "D1": mt5.TIMEFRAME_D1, "M15": mt5.TIMEFRAME_M15,
}


def fetch_data(config: dict, logger: logging.Logger) -> pd.DataFrame:
    ensure_connected(logger)
    tf = _TF_MAP.get(config["timeframe"], mt5.TIMEFRAME_H4)
    df = fetch_ohlc(symbol=config["symbol"], timeframe=tf, bars=5000)
    logger.debug(f"[{config['symbol']}] Fetched {len(df)} bars. Last: {df.index[-1]}")
    return df


# ── Circuit breaker ───────────────────────────────────────────────────────────

def _close_all_paper(state: dict, symbol: str, logger: logging.Logger) -> None:
    if state.get("open_trade"):
        logger.warning(f"[{symbol}] [CB] Closing paper position.")
        state["open_trade"] = None
    if state.get("pending_limit"):
        logger.warning(f"[{symbol}] [CB] Cancelling pending paper limit.")
        state["pending_limit"] = None


def _close_all_live(config: dict, state: dict, symbol: str, logger: logging.Logger) -> None:
    sym = config["symbol"]
    for p in (mt5.positions_get(symbol=sym) or []):
        if p.magic == MAGIC:
            _market_close_live(p, logger)
    for o in (mt5.orders_get(symbol=sym) or []):
        if o.magic == MAGIC:
            mt5.order_cancel(o.ticket)
            logger.info(f"[{symbol}] [CB] Cancelled order {o.ticket}")
    state["open_trade"]    = None
    state["pending_limit"] = None


def trigger_circuit_breaker(config: dict, state: dict, symbol: str,
                             reason: str, logger: logging.Logger) -> None:
    logger.critical(f"[{symbol}] CIRCUIT BREAKER: {reason}")
    state["circuit_breaker_active"] = True
    state["circuit_breaker_reason"] = reason
    if config["paper_mode"]:
        _close_all_paper(state, symbol, logger)
    else:
        try:
            _close_all_live(config, state, symbol, logger)
        except Exception as e:
            logger.error(f"[{symbol}] Error closing positions at CB: {e}")
    save_state(state, symbol)
    logger.critical(f"[{symbol}] Trading halted. Set circuit_breaker_active=false in state_{symbol}.json to resume.")


def check_circuit_breaker(config: dict, state: dict, symbol: str,
                           logger: logging.Logger) -> bool:
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
                logger,
            )
            return True
    return False


# ── Account helpers ───────────────────────────────────────────────────────────

def get_balance(config: dict, state: dict) -> float:
    if config["paper_mode"]:
        return float(state.get("current_balance") or 10_000.0)
    info = mt5.account_info()
    return float(info.balance) if info else float(state.get("current_balance") or 0)


def compute_lot(config: dict, balance: float, entry: float, stop: float,
                pip: float, pip_value: float) -> float:
    risk      = balance * config["risk_per_trade"]
    stop_pips = abs(entry - stop) / pip
    if stop_pips <= 0 or pip_value <= 0:
        return 0.0
    lot = math.floor(risk / (stop_pips * pip_value) * 100) / 100
    if 0 < lot < 0.01:
        lot = 0.01
    return lot


# ── Paper-mode order helpers ──────────────────────────────────────────────────

def paper_place_limit(state: dict, symbol: str, direction: str, entry: float,
                       stop: float, tp1: float, lot: float,
                       logger: logging.Logger) -> None:
    state["pending_limit"] = {
        "direction": direction, "price": entry,
        "stop": stop, "tp1": tp1, "lot": lot,
        "placed_time": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    logger.info(
        f"[{symbol}] [PAPER] Limit {direction.upper()} {lot} lots @ {entry:.5f}  "
        f"SL={stop:.5f}  TP={tp1:.5f}"
    )


def paper_cancel_limit(state: dict, symbol: str, reason: str,
                        logger: logging.Logger) -> None:
    if state.get("pending_limit"):
        logger.info(f"[{symbol}] [PAPER] Limit cancelled — {reason}")
        state["pending_limit"] = None


def paper_open_trade(state: dict, symbol: str, limit: dict, fill_price: float,
                      logger: logging.Logger) -> None:
    state["open_trade"] = {
        "direction":  limit["direction"],
        "entry":      fill_price,
        "stop":       limit["stop"],
        "tp1":        limit["tp1"],
        "lot":        limit["lot"],
        "tp1_hit":    False,
        "trail_stop": None,
        "entry_time": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    state["pending_limit"] = None
    logger.info(
        f"[{symbol}] [PAPER] Position opened: {limit['direction'].upper()} @ {fill_price:.5f}  "
        f"SL={limit['stop']:.5f}  TP={limit['tp1']:.5f}"
    )


def paper_close_trade(config: dict, state: dict, symbol: str, exit_price: float,
                       reason: str, pip: float, pip_value: float,
                       logger: logging.Logger) -> None:
    t    = state["open_trade"]
    sign = 1 if t["direction"] == "long" else -1
    if t["tp1_hit"]:
        trail_pips = sign * (exit_price - t["entry"]) / pip
        gross = 0.5 * (sign * (t["tp1"] - t["entry"]) / pip) + 0.5 * trail_pips
    else:
        gross = sign * (exit_price - t["entry"]) / pip
    net = gross - 3.0
    pnl = net * pip_value * t["lot"]
    state["current_balance"] = round((state["current_balance"] or 0) + pnl, 2)
    state["open_trade"] = None
    logger.info(
        f"[{symbol}] [PAPER] Trade closed ({reason}): exit={exit_price:.5f}  "
        f"gross={gross:.1f}pip  net={net:.1f}pip  P&L=${pnl:.2f}  "
        f"balance=${state['current_balance']:.2f}"
    )


# ── Live-mode order helpers ───────────────────────────────────────────────────

def _send_order(request: dict, logger: logging.Logger,
                description: str) -> mt5.OrderSendResult:
    for attempt in range(1, ORDER_RETRY_LIMIT + 1):
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"{description} — ticket {result.order}")
            return result
        logger.error(
            f"{description} attempt {attempt}/{ORDER_RETRY_LIMIT} failed: "
            f"retcode={getattr(result, 'retcode', None)}  "
            f"comment={getattr(result, 'comment', None)}"
        )
        if getattr(result, 'retcode', None) == mt5.TRADE_RETCODE_TRADE_DISABLED:
            sym = request.get("symbol", "")
            available = [s.name for s in (mt5.symbols_get() or []) if sym.rstrip("#. ") in s.name]
            logger.error(
                f"retcode 10017 = Trade Disabled — symbol name is likely wrong. "
                f"Configured: '{sym}'  Available in MT5: {available}"
            )
        time.sleep(1)
    raise TradingHalt(f"{description} failed after {ORDER_RETRY_LIMIT} attempts")


def live_place_limit(config: dict, state: dict, direction: str, entry: float,
                      stop: float, tp1: float, lot: float,
                      logger: logging.Logger) -> None:
    sym        = config["symbol"]
    order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == "long" else mt5.ORDER_TYPE_SELL_LIMIT
    tp_price   = tp1 if config["tp_mode"] == "full" else 0.0
    request = {
        "action":       mt5.TRADE_ACTION_PENDING,
        "symbol":       sym,
        "volume":       lot,
        "type":         order_type,
        "price":        entry,
        "sl":           stop,
        "tp":           tp_price,
        "deviation":    10,
        "magic":        MAGIC,
        "comment":      "LiveTrader",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }
    try:
        result = _send_order(request, logger, f"[{sym}] Place {direction.upper()} limit @ {entry:.5f}")
    except TradingHalt as e:
        raise OrderSkip(str(e))
    state["pending_limit"] = {
        "direction": direction, "price": entry,
        "stop": stop, "tp1": tp1, "lot": lot, "ticket": result.order,
        "placed_time": pd.Timestamp.now(tz="UTC").isoformat(),
    }


def live_cancel_limits(config: dict, logger: logging.Logger) -> None:
    sym = config["symbol"]
    for o in (mt5.orders_get(symbol=sym) or []):
        if o.magic == MAGIC:
            mt5.order_cancel(o.ticket)
            logger.info(f"[{sym}] Cancelled pending order {o.ticket}")


def _market_close_live(position: mt5.TradePosition, logger: logging.Logger,
                        volume: float = None, comment: str = "") -> None:
    vol  = volume or position.volume
    side = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY
    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       position.symbol,
        "volume":       vol,
        "type":         side,
        "position":     position.ticket,
        "deviation":    20,
        "magic":        MAGIC,
        "comment":      comment or "LiveTrader close",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    _send_order(request, logger, f"Close position {position.ticket}")


def live_modify_sl(config: dict, position: mt5.TradePosition, new_sl: float,
                    logger: logging.Logger) -> None:
    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "symbol":   config["symbol"],
        "position": position.ticket,
        "sl":       new_sl,
        "tp":       position.tp,
    }
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"[{config['symbol']}] Stop modified to {new_sl:.5f} on ticket {position.ticket}")
    else:
        logger.warning(f"[{config['symbol']}] Stop modify failed: {getattr(result, 'comment', result)}")


# ── Session check ─────────────────────────────────────────────────────────────

def in_session(config: dict, ts: pd.Timestamp) -> bool:
    return config["session_start_utc"] <= ts.hour < config["session_end_utc"]


# ── Trail stop helper ─────────────────────────────────────────────────────────

def _new_trail_stop(config: dict, direction: str, df: pd.DataFrame,
                    current_trail: float, pip: float) -> float:
    stop_buf  = config.get("stop_buffer", 5)
    prev_bar  = df.iloc[-2]
    if direction == "long":
        candidate = float(prev_bar["Low"]) - stop_buf * pip
        if current_trail is None or candidate > current_trail:
            return candidate
        return current_trail
    else:
        candidate = float(prev_bar["High"]) + stop_buf * pip
        if current_trail is None or candidate < current_trail:
            return candidate
        return current_trail


# ── Per-candle trade management ───────────────────────────────────────────────

def _manage_paper_trade(config: dict, state: dict, symbol: str,
                         last_bar: pd.Series, df: pd.DataFrame,
                         pip: float, pip_value: float,
                         logger: logging.Logger) -> None:
    t         = state["open_trade"]
    h, l, c   = float(last_bar["High"]), float(last_bar["Low"]), float(last_bar["Close"])
    direction = t["direction"]
    long      = direction == "long"

    stop_dist = (l - t["stop"]) / pip if long else (t["stop"] - h) / pip
    tp_dist   = (t["tp1"] - h)  / pip if long else (l - t["tp1"]) / pip
    logger.info(
        f"[{symbol}] [PAPER] Managing {direction.upper()}  "
        f"entry={t['entry']:.5f}  stop={t['stop']:.5f}  tp={t['tp1']:.5f}"
        f"\n  Bar H={h:.5f}  L={l:.5f}  C={c:.5f}"
        f"\n  Distance to stop: {stop_dist:.1f}pip  Distance to TP: {tp_dist:.1f}pip"
    )

    if t["tp1_hit"]:
        new_ts = _new_trail_stop(config, direction, df, t["trail_stop"], pip)
        if new_ts != t["trail_stop"]:
            t["trail_stop"] = new_ts
            logger.info(f"[{symbol}] [PAPER] Trail stop moved to {new_ts:.5f}")
        ts_level = t["trail_stop"] or t["entry"]
        if (long and c < ts_level) or (not long and c > ts_level):
            paper_close_trade(config, state, symbol, c, "trail_exit", pip, pip_value, logger)
        return

    if (long and l <= t["stop"]) or (not long and h >= t["stop"]):
        paper_close_trade(config, state, symbol, t["stop"], "stop_hit", pip, pip_value, logger)
        return

    tp1 = t["tp1"]
    if (long and h >= tp1) or (not long and l <= tp1):
        if config["tp_mode"] == "full":
            paper_close_trade(config, state, symbol, tp1, "tp1_hit", pip, pip_value, logger)
        else:
            t["tp1_hit"]    = True
            t["trail_stop"] = t["entry"]
            logger.info(
                f"[{symbol}] [PAPER] TP1 hit @ {tp1:.5f}. 50% closed. "
                f"Stop moved to breakeven {t['entry']:.5f}"
            )
            state["current_balance"] = round(
                (state["current_balance"] or 0)
                + 0.5 * (abs(tp1 - t["entry"]) / pip - 3) * pip_value * t["lot"], 2
            )


def _manage_live_trade(config: dict, state: dict, symbol: str,
                        last_bar: pd.Series, df: pd.DataFrame,
                        pip: float, logger: logging.Logger) -> None:
    positions = mt5.positions_get(symbol=config["symbol"]) or []
    our_pos   = next((p for p in positions if p.magic == MAGIC), None)
    t         = state.get("open_trade")

    if our_pos is None:
        if t:
            logger.info(f"[{symbol}] Position closed by MT5. Clearing state.")
            state["open_trade"]    = None
            state["pending_limit"] = None
            bal = mt5.account_info()
            if bal:
                state["current_balance"] = float(bal.balance)
        return

    if t is None:
        direction = "long" if our_pos.type == 0 else "short"
        state["open_trade"] = {
            "direction": direction, "entry": our_pos.price_open,
            "stop": our_pos.sl, "tp1": our_pos.tp, "lot": our_pos.volume,
            "tp1_hit": False, "trail_stop": None,
        }
        logger.warning(f"[{symbol}] Reconciled open position #{our_pos.ticket} from MT5.")
        t = state["open_trade"]

    long = t["direction"] == "long"
    h, l = float(last_bar["High"]), float(last_bar["Low"])
    tp1  = t["tp1"]

    if not t["tp1_hit"] and config["tp_mode"] == "partial":
        if (long and h >= tp1) or (not long and l <= tp1):
            logger.info(f"[{symbol}] TP1 reached. Closing 50% of position {our_pos.ticket}.")
            _market_close_live(our_pos, logger, volume=round(our_pos.volume / 2, 2), comment="TP1 partial")
            live_modify_sl(config, our_pos, our_pos.price_open, logger)
            t["tp1_hit"]    = True
            t["trail_stop"] = our_pos.price_open
            state["current_balance"] = float(mt5.account_info().balance)

    if t["tp1_hit"]:
        new_ts = _new_trail_stop(config, t["direction"], df, t["trail_stop"], pip)
        if new_ts != t["trail_stop"]:
            live_modify_sl(config, our_pos, new_ts, logger)
            t["trail_stop"] = new_ts


def _check_paper_pending(config: dict, state: dict, symbol: str,
                          last_bar: pd.Series, pip: float, pip_value: float,
                          logger: logging.Logger) -> None:
    lim  = state["pending_limit"]

    # Expiry check: cancel if N bars have passed without fill
    expiry_bars = config.get("pending_order_expiry_bars", 1)
    if lim.get("placed_time"):
        placed      = pd.Timestamp(lim["placed_time"])
        current_bar = pd.Timestamp(last_bar.name)
        elapsed     = (current_bar - placed) / pd.Timedelta(hours=4)
        if elapsed >= expiry_bars:
            logger.info(
                f"[{symbol}] [PAPER] Pending limit expired after {elapsed:.1f} bars — cancelling"
            )
            paper_cancel_limit(state, symbol, "expired", logger)
            return

    h, l = float(last_bar["High"]), float(last_bar["Low"])
    dist = (l - lim["price"]) / pip if lim["direction"] == "long" else (lim["price"] - h) / pip
    logger.info(
        f"[{symbol}] [PAPER] Pending {lim['direction'].upper()} limit @ {lim['price']:.5f}  "
        f"Bar H={h:.5f}  L={l:.5f}  "
        f"{'filled' if dist <= 0 else f'not filled ({dist:.1f}pip away)'}"
    )
    filled = (lim["direction"] == "long"  and l <= lim["price"]) or \
             (lim["direction"] == "short" and h >= lim["price"])
    if not filled:
        return
    paper_open_trade(state, symbol, lim, lim["price"], logger)
    t  = state["open_trade"]
    lo = lim["direction"] == "long"
    if (lo and l <= t["stop"]) or (not lo and h >= t["stop"]):
        paper_close_trade(config, state, symbol, t["stop"], "stop_hit_same_bar", pip, pip_value, logger)
    elif (lo and h >= t["tp1"]) or (not lo and l <= t["tp1"]):
        if config["tp_mode"] == "full":
            paper_close_trade(config, state, symbol, t["tp1"], "tp1_hit_same_bar", pip, pip_value, logger)
        else:
            t["tp1_hit"]    = True
            t["trail_stop"] = t["entry"]


def _check_live_pending(config: dict, state: dict, symbol: str,
                         last_bar: pd.Series, logger: logging.Logger) -> None:
    lim = state.get("pending_limit", {})

    # Expiry check: cancel GTC order if N bars have passed without fill
    expiry_bars = config.get("pending_order_expiry_bars", 1)
    if lim.get("placed_time"):
        placed      = pd.Timestamp(lim["placed_time"])
        current_bar = pd.Timestamp(last_bar.name)
        elapsed     = (current_bar - placed) / pd.Timedelta(hours=4)
        if elapsed >= expiry_bars:
            logger.info(
                f"[{symbol}] Pending limit expired after {elapsed:.1f} bars — cancelling"
            )
            live_cancel_limits(config, logger)
            state["pending_limit"] = None
            return

    positions = mt5.positions_get(symbol=config["symbol"]) or []
    our_pos   = next((p for p in positions if p.magic == MAGIC), None)
    if our_pos:
        logger.info(f"[{symbol}] Limit order filled. Position #{our_pos.ticket} opened.")
        lim = state.get("pending_limit", {})
        state["open_trade"] = {
            "direction": "long" if our_pos.type == 0 else "short",
            "entry": our_pos.price_open, "stop": our_pos.sl,
            "tp1": lim.get("tp1", our_pos.tp), "lot": our_pos.volume,
            "tp1_hit": False, "trail_stop": None, "ticket": our_pos.ticket,
        }
        state["pending_limit"] = None
        state["current_balance"] = float(mt5.account_info().balance)


# ── Entry logic ───────────────────────────────────────────────────────────────

def check_entry(config: dict, state: dict, df: pd.DataFrame, symbol: str,
                pip: float, pip_value: float, logger: logging.Logger) -> None:
    if len(df) < 3:
        return

    bar_time = df.index[-1]
    if not in_session(config, bar_time):
        logger.info(
            f"[{symbol}] No entry — outside session "
            f"({bar_time.hour:02d}:00 UTC, window {config['session_start_utc']}-{config['session_end_utc']})"
        )
        return

    prev2 = df.iloc[-3]
    prev1 = df.iloc[-2]
    curr  = df.iloc[-1]

    logger.info(
        f"[{symbol}] Evaluating bars:"
        f"\n  prev2  O={prev2['Open']:.5f}  H={prev2['High']:.5f}  L={prev2['Low']:.5f}  C={prev2['Close']:.5f}"
        f"\n  prev1  O={prev1['Open']:.5f}  H={prev1['High']:.5f}  L={prev1['Low']:.5f}  C={prev1['Close']:.5f}"
        f"\n  curr   O={curr['Open']:.5f}  H={curr['High']:.5f}  L={curr['Low']:.5f}  C={curr['Close']:.5f}"
    )

    p1_took_high = prev1["High"] > prev2["High"]
    p1_took_low  = prev1["Low"]  < prev2["Low"]
    c_took_high  = curr["High"]  > prev1["High"]
    c_took_low   = curr["Low"]   < prev1["Low"]
    bar_range    = curr["High"] - curr["Low"]
    close_pos    = ((curr["Close"] - curr["Low"]) / bar_range) if bar_range > 0 else 0.5
    close_strength = config.get("close_strength", 0.6)

    logger.info(
        f"[{symbol}] Bar conditions:"
        f"\n  prev1 broke high={p1_took_high}  prev1 broke low={p1_took_low}"
        f"\n  curr  broke high={c_took_high}   curr  broke low={c_took_low}"
        f"\n  close_pos={close_pos:.2f}  "
        f"(need >={close_strength:.2f} long / <={(1-close_strength):.2f} short)"
    )

    long_struct  = p1_took_high and not p1_took_low and c_took_high and not c_took_low
    short_struct = p1_took_low  and not p1_took_high and c_took_low and not c_took_high

    if long_struct:
        if close_pos < close_strength:
            logger.info(f"[{symbol}] No entry — LONG structure valid but close too weak ({close_pos:.2f} < {close_strength:.2f})")
    elif short_struct:
        if close_pos > (1.0 - close_strength):
            logger.info(f"[{symbol}] No entry — SHORT structure valid but close too weak ({close_pos:.2f} > {1-close_strength:.2f})")
    else:
        reasons = []
        if not (p1_took_high and not p1_took_low):
            reasons.append("prev1 not clean up-break")
        if not (p1_took_low and not p1_took_high):
            reasons.append("prev1 not clean down-break")
        logger.info(f"[{symbol}] No entry — no clean consecutive breakout ({', '.join(reasons)})")

    lb      = config.get("pullback_lookback", 4)
    factor  = config.get("pullback_factor", 1.0)
    pb_pips = float(np.median([
        (df.iloc[j]["High"] - df.iloc[j]["Low"]) / pip
        for j in range(max(0, len(df) - lb - 1), len(df) - 1)
    ])) * factor
    min_stop = config.get("min_stop_pips", MIN_STOP_PIPS)
    logger.debug(f"[{symbol}] Pullback: {pb_pips:.1f}pip (factor={factor})  min_stop: {min_stop:.0f}pip")

    result = _bar_setup(prev2, prev1, curr, pb_pips,
                        config.get("stop_buffer", 5), pip, min_stop,
                        close_strength=close_strength)
    if result is None:
        logger.info(f"[{symbol}] No entry — bar geometry invalid (pullback offset exceeded swing range)")
        return

    direction, entry, stop, tp1 = result
    stop_pips = abs(entry - stop) / pip
    risk_r    = abs(tp1 - entry) / abs(entry - stop)

    logger.info(
        f"[{symbol}] SIGNAL: {direction.upper()}  "
        f"entry={entry:.5f}  stop={stop:.5f}  tp={tp1:.5f}  "
        f"stop_dist={stop_pips:.1f}pip  R:R=1:{risk_r:.1f}"
    )

    balance = get_balance(config, state)
    lot     = compute_lot(config, balance, entry, stop, pip, pip_value)
    if lot <= 0:
        logger.info(f"[{symbol}] No entry — lot size zero (stop={stop_pips:.1f}pip  balance=${balance:.2f})")
        return

    logger.info(
        f"[{symbol}] Placing order: {direction.upper()} {lot} lots  "
        f"risk=${balance * config['risk_per_trade']:.2f} ({config['risk_per_trade']*100:.1f}%)"
    )
    if config["paper_mode"]:
        paper_cancel_limit(state, symbol, "new signal", logger)
        paper_place_limit(state, symbol, direction, entry, stop, tp1, lot, logger)
    else:
        live_cancel_limits(config, logger)
        live_place_limit(config, state, direction, entry, stop, tp1, lot, logger)


# ── Main candle handler ───────────────────────────────────────────────────────

def run_candle(config: dict, state: dict, df: pd.DataFrame, symbol: str,
               pip: float, pip_value: float, logger: logging.Logger) -> None:
    ensure_connected(logger)

    last_bar = df.iloc[-1]
    bar_time = df.index[-1]

    logger.info(f"[{symbol}] Candle {bar_time}  close={last_bar['Close']:.5f}  pip={pip}  pip_val={pip_value:.2f}")

    if check_circuit_breaker(config, state, symbol, logger):
        return

    if state["session_starting_balance"] is None:
        bal = get_balance(config, state)
        state["session_starting_balance"] = bal
        state["current_balance"]          = bal
        logger.info(f"[{symbol}] Session starting balance: ${bal:.2f}")

    if state.get("open_trade"):
        if config["paper_mode"]:
            _manage_paper_trade(config, state, symbol, last_bar, df, pip, pip_value, logger)
        else:
            _manage_live_trade(config, state, symbol, last_bar, df, pip, logger)

    elif state.get("pending_limit"):
        if config["paper_mode"]:
            _check_paper_pending(config, state, symbol, last_bar, pip, pip_value, logger)
        else:
            _check_live_pending(config, state, symbol, last_bar, logger)

    if not state.get("open_trade") and not state.get("pending_limit"):
        check_entry(config, state, df, symbol, pip, pip_value, logger)

    save_state(state, symbol)
    logger.info(
        f"[{symbol}] State saved — balance=${state.get('current_balance', '?')}  "
        f"open={bool(state.get('open_trade'))}  pending={bool(state.get('pending_limit'))}"
    )


# ── Timing ────────────────────────────────────────────────────────────────────

def wait_for_next_candle(last_bar_times: dict, logger: logging.Logger) -> None:
    if not last_bar_times:
        time.sleep(60)
        return
    earliest_next = min(t + pd.Timedelta(hours=4, seconds=30) for t in last_bar_times.values())
    now        = pd.Timestamp.now(tz="UTC")
    sleep_secs = (earliest_next - now).total_seconds()
    if sleep_secs > 0:
        logger.info(
            f"Sleeping {sleep_secs/3600:.2f}h until next candle close "
            f"({earliest_next.strftime('%Y-%m-%d %H:%M UTC')})"
        )
        time.sleep(sleep_secs)
    logger.info("Waking up — polling for new candle...")


# ── Live mode confirmation ────────────────────────────────────────────────────

def _live_mode_confirmation(config: dict, logger: logging.Logger) -> None:
    account_num = "N/A"
    try:
        if _is_connected():
            info = mt5.account_info()
            if info:
                account_num = str(info.login)
    except Exception:
        pass

    symbols_str = ", ".join(config.get("symbols", [config.get("symbol", "?")]))
    sep    = "=" * 60
    banner = (
        f"\n{sep}\n"
        f"  LIVE TRADING MODE ACTIVE\n"
        f"  Symbols:         {symbols_str}\n"
        f"  Risk per trade:  {config['risk_per_trade'] * 100:.1f}%\n"
        f"  Max drawdown:    {config['max_drawdown_pct']}%\n"
        f"  Circuit breaker: ACTIVE\n"
        f"  MT5 account:     {account_num}\n"
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
            answer.append(input("\nType 'yes' and press Enter to confirm (30s timeout): ").strip().lower())
        except Exception:
            pass
        confirmed.set()

    reader = threading.Thread(target=_read, daemon=True)
    reader.start()
    confirmed.wait(timeout=30)

    if not answer or answer[0] != "yes":
        logger.critical("Confirmation not received — shutting down safely.")
        print("\nConfirmation not received. Shutting down safely.\n")
        try:
            dp_disconnect()
        except Exception:
            pass
        sys.exit(1)

    logger.info("Live mode confirmed by operator.")
    print("\nConfirmed. Starting live trading engine...\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    once    = "--once"    in sys.argv
    test_cb = "--test-cb" in sys.argv

    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("Live Trader starting up")
    logger.info("=" * 60)

    config  = load_config()
    symbols = config["symbols"]

    mode_str = "PAPER" if config["paper_mode"] else "LIVE"
    logger.info(f"Mode: {mode_str}  Symbols: {symbols}  TF: {config['timeframe']}")

    # Migrate old single-symbol state.json if present
    _migrate_legacy_state(symbols, logger)

    # Check all symbol circuit breakers before starting
    for sym in symbols:
        state = load_state(sym)
        if state.get("circuit_breaker_active"):
            logger.critical(
                f"STARTUP BLOCKED — [{sym}] circuit breaker active. "
                f"Reason: {state.get('circuit_breaker_reason')}. "
                f"Edit state_{sym}.json and set circuit_breaker_active=false to resume."
            )
            sys.exit(1)

    if test_cb:
        sym   = symbols[0]
        state = load_state(sym)
        logger.info(f"TEST MODE: Forcing circuit breaker on {sym}...")
        if state["session_starting_balance"] is None:
            state["session_starting_balance"] = 10_000.0
            state["current_balance"]          = 10_000.0
        breached = state["session_starting_balance"] * (1 - (config["max_drawdown_pct"] + 1) / 100)
        state["current_balance"] = round(breached, 2)
        save_state(state, sym)
        triggered = check_circuit_breaker(config, state, sym, logger)
        logger.info(f"Circuit breaker triggered: {triggered}")
        sys.exit(0)

    if not config["paper_mode"]:
        _live_mode_confirmation(config, logger)

    if not dp_connect():
        logger.error("Initial MT5 connection failed. Retrying...")
        ensure_connected(logger)

    # Fetch pip/pip_value for each symbol once at startup
    sym_info: dict[str, tuple] = {}
    for sym in symbols:
        pip, pip_value = get_symbol_info(sym)
        if pip is None:
            logger.warning(f"[{sym}] Not available in MT5 Market Watch — skipping. Add it to Market Watch to enable.")
        else:
            sym_info[sym] = (pip, pip_value)
            logger.info(f"[{sym}] pip={pip}  pip_value={pip_value:.4f}")

    if not sym_info:
        logger.critical("No symbols available in MT5. Add them to Market Watch and restart.")
        sys.exit(1)

    last_bar_times: dict[str, pd.Timestamp] = {}

    while True:
        try:
            processed_any = False

            for sym in list(sym_info.keys()):
                pip, pip_value = sym_info[sym]
                if pip_value <= 0:
                    pip, pip_value = get_symbol_info(sym)
                    if pip and pip_value > 0:
                        sym_info[sym] = (pip, pip_value)
                        logger.info(f"[{sym}] pip_value refreshed: {pip_value:.4f}")
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
                        run_candle(sym_cfg, state, df, sym, pip, pip_value, logger)
                    except OrderSkip as e:
                        logger.error(f"[{sym}] Order placement failed — skipping this candle: {e}")
                    processed_any = True
                else:
                    logger.debug(f"[{sym}] No new candle. Last: {prev_last}")

            if once and processed_any:
                logger.info("--once flag: exiting after processing all symbols.")
                break
            elif once and not processed_any:
                # Force one pass even if no new candle (for testing)
                for sym in list(sym_info.keys()):
                    pip, pip_value = sym_info[sym]
                    sym_cfg = {**config, "symbol": sym}
                    state   = load_state(sym)
                    df = fetch_data(sym_cfg, logger)
                    last_bar_times[sym] = df.index[-1]
                    try:
                        run_candle(sym_cfg, state, df, sym, pip, pip_value, logger)
                    except OrderSkip as e:
                        logger.error(f"[{sym}] Order placement failed — skipping this candle: {e}")
                logger.info("--once flag: exiting.")
                break

            wait_for_next_candle(last_bar_times, logger)

        except TradingHalt as e:
            logger.critical(f"TRADING HALT: {e}")
            sys.exit(1)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt — shutting down gracefully.")
            dp_disconnect()
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
