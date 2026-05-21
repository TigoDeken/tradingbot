"""
live_trader.py
Live and paper-mode execution engine. Runs on every 4H candle close.

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
from datetime import timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from data_pipeline import connect as dp_connect, disconnect as dp_disconnect, fetch_ohlc
from constants     import PIP, MAGIC, MAX_LIMIT_BARS
from trade_engine  import _bar_setup, PIP_VALUE, MIN_STOP_PIPS

# ── Constants ─────────────────────────────────────────────────────────────────
CONFIG_PATH  = Path("config.json")
STATE_PATH   = Path("state.json")
LOG_PATH     = Path("trading_log.txt")

RECONNECT_INTERVAL  = 60      # seconds between MT5 reconnect attempts
ORDER_RETRY_LIMIT   = 3       # max order_send retries before halting
POLL_INTERVAL       = 30      # seconds between candle polls after expected close

# ── Exceptions ────────────────────────────────────────────────────────────────

class TradingHalt(Exception):
    """Raised when the engine must stop immediately and require manual restart."""


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
    required = [
        "symbol", "timeframe", "session_start_utc", "session_end_utc",
        "risk_per_trade", "tp_mode",
        "pullback_lookback", "stop_buffer", "paper_mode", "max_drawdown_pct",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"config.json missing keys: {missing}")
    return cfg


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


def load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            saved = json.load(f)
        # Drop any stale keys from old state files
        clean = {k: v for k, v in saved.items() if k in _DEFAULT_STATE}
        return {**_DEFAULT_STATE, **clean}
    return dict(_DEFAULT_STATE)


def save_state(state: dict) -> None:
    state["last_updated"] = pd.Timestamp.now(tz="UTC").isoformat()
    tmp = STATE_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    tmp.replace(STATE_PATH)


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
        logger.warning(f"MT5 disconnected. Reconnect attempt {attempt}…")
        if dp_connect():
            logger.info("MT5 reconnected successfully.")
            return
        logger.warning(f"Reconnect failed. Waiting {RECONNECT_INTERVAL}s…")
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
    logger.debug(f"Fetched {len(df)} bars. Last: {df.index[-1]}")
    return df


# ── Circuit breaker ───────────────────────────────────────────────────────────

def _close_all_paper(state: dict, logger: logging.Logger) -> None:
    if state.get("open_trade"):
        logger.warning("[CB] Closing paper position at circuit breaker trigger.")
        state["open_trade"] = None
    if state.get("pending_limit"):
        logger.warning("[CB] Cancelling pending paper limit at circuit breaker trigger.")
        state["pending_limit"] = None


def _close_all_live(config: dict, state: dict, logger: logging.Logger) -> None:
    sym = config["symbol"]
    for p in (mt5.positions_get(symbol=sym) or []):
        if p.magic == MAGIC:
            _market_close_live(p, logger)
    for o in (mt5.orders_get(symbol=sym) or []):
        if o.magic == MAGIC:
            mt5.order_cancel(o.ticket)
            logger.info(f"[CB] Cancelled pending order {o.ticket}")
    state["open_trade"]    = None
    state["pending_limit"] = None


def trigger_circuit_breaker(config: dict, state: dict, reason: str,
                             logger: logging.Logger) -> None:
    logger.critical(f"CIRCUIT BREAKER TRIGGERED: {reason}")
    state["circuit_breaker_active"] = True
    state["circuit_breaker_reason"] = reason
    if config["paper_mode"]:
        _close_all_paper(state, logger)
    else:
        try:
            _close_all_live(config, state, logger)
        except Exception as e:
            logger.error(f"Error closing live positions at CB: {e}")
    save_state(state)
    logger.critical("Trading halted. Edit state.json and set circuit_breaker_active=false to resume.")


def check_circuit_breaker(config: dict, state: dict,
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
                config, state,
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


def compute_lot(config: dict, balance: float, entry: float, stop: float) -> float:
    risk      = balance * config["risk_per_trade"]
    stop_pips = abs(entry - stop) / PIP
    if stop_pips <= 0:
        return 0.0
    lot = math.floor(risk / (stop_pips * PIP_VALUE) * 100) / 100
    if 0 < lot < 0.01:
        lot = 0.01
    return lot


# ── Paper-mode order helpers ──────────────────────────────────────────────────

def paper_place_limit(state: dict, direction: str, entry: float, stop: float,
                       tp1: float, lot: float, logger: logging.Logger) -> None:
    state["pending_limit"] = {
        "direction": direction, "price": entry,
        "stop": stop, "tp1": tp1, "lot": lot,
        "placed_time": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    logger.info(
        f"[PAPER] Limit {direction.upper()} {lot} lots @ {entry:.5f}  "
        f"SL={stop:.5f}  TP={tp1:.5f}"
    )


def paper_cancel_limit(state: dict, reason: str, logger: logging.Logger) -> None:
    if state.get("pending_limit"):
        logger.info(f"[PAPER] Limit cancelled — {reason}")
        state["pending_limit"] = None


def paper_open_trade(state: dict, limit: dict, fill_price: float,
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
        f"[PAPER] Position opened: {limit['direction'].upper()} @ {fill_price:.5f}  "
        f"SL={limit['stop']:.5f}  TP={limit['tp1']:.5f}"
    )


def paper_close_trade(config: dict, state: dict, exit_price: float,
                       reason: str, logger: logging.Logger) -> None:
    t    = state["open_trade"]
    sign = 1 if t["direction"] == "long" else -1
    if t["tp1_hit"]:
        trail_pips = sign * (exit_price - t["entry"]) / PIP
        gross = 0.5 * (sign * (t["tp1"] - t["entry"]) / PIP) + 0.5 * trail_pips
    else:
        gross = sign * (exit_price - t["entry"]) / PIP
    net = gross - 3.0   # 3-pip slippage model
    pnl = net * PIP_VALUE * t["lot"]
    state["current_balance"] = round((state["current_balance"] or 0) + pnl, 2)
    state["open_trade"] = None
    logger.info(
        f"[PAPER] Trade closed ({reason}): exit={exit_price:.5f}  "
        f"gross={gross:.1f}pip  net={net:.1f}pip  P&L=${pnl:.2f}  "
        f"balance=${state['current_balance']:.2f}"
    )


# ── Live-mode order helpers ───────────────────────────────────────────────────

def _send_order(request: dict, logger: logging.Logger, description: str) -> mt5.OrderSendResult:
    for attempt in range(1, ORDER_RETRY_LIMIT + 1):
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"{description} — ticket {result.order}")
            return result
        logger.error(
            f"{description} attempt {attempt}/{ORDER_RETRY_LIMIT} failed: "
            f"retcode={getattr(result, 'retcode', None)} comment={getattr(result, 'comment', None)}"
        )
        time.sleep(1)
    raise TradingHalt(f"{description} failed after {ORDER_RETRY_LIMIT} attempts")


def live_place_limit(config: dict, state: dict, direction: str, entry: float,
                      stop: float, tp1: float, lot: float, logger: logging.Logger) -> None:
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
    result = _send_order(request, logger, f"Place {direction.upper()} limit @ {entry:.5f}")
    state["pending_limit"] = {
        "direction": direction, "price": entry,
        "stop": stop, "tp1": tp1, "lot": lot, "ticket": result.order,
        "placed_time": pd.Timestamp.now(tz="UTC").isoformat(),
    }


def live_cancel_limits(config: dict, logger: logging.Logger) -> None:
    for o in (mt5.orders_get(symbol=config["symbol"]) or []):
        if o.magic == MAGIC:
            mt5.order_cancel(o.ticket)
            logger.info(f"Cancelled pending order {o.ticket}")


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
    _send_order(request, logger, f"Close position {position.ticket} vol={vol}")


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
        logger.info(f"Stop modified to {new_sl:.5f} on ticket {position.ticket}")
    else:
        logger.warning(f"Stop modify failed: {getattr(result, 'comment', result)}")


# ── In-session check ──────────────────────────────────────────────────────────

def in_session(config: dict, ts: pd.Timestamp) -> bool:
    return config["session_start_utc"] <= ts.hour < config["session_end_utc"]


# ── Trail stop helper ─────────────────────────────────────────────────────────

def _new_trail_stop(config: dict, direction: str, df: pd.DataFrame,
                    current_trail: float | None) -> float:
    """Return updated trail stop based on previous bar extreme. Never moves against trade."""
    stop_buf  = config.get("stop_buffer", 5)
    prev_bar  = df.iloc[-2]
    if direction == "long":
        candidate = float(prev_bar["Low"]) - stop_buf * PIP
        if current_trail is None or candidate > current_trail:
            return candidate
        return current_trail
    else:
        candidate = float(prev_bar["High"]) + stop_buf * PIP
        if current_trail is None or candidate < current_trail:
            return candidate
        return current_trail


# ── Per-candle trade management ───────────────────────────────────────────────

def _manage_paper_trade(config: dict, state: dict, last_bar: pd.Series,
                         df: pd.DataFrame, logger: logging.Logger) -> None:
    t         = state["open_trade"]
    h, l, c   = float(last_bar["High"]), float(last_bar["Low"]), float(last_bar["Close"])
    direction = t["direction"]
    long      = direction == "long"

    stop_dist = (l - t["stop"]) / PIP if long else (t["stop"] - h) / PIP
    tp_dist   = (t["tp1"] - h)  / PIP if long else (l - t["tp1"]) / PIP
    logger.info(
        f"[PAPER] Managing {direction.upper()}  entry={t['entry']:.5f}  "
        f"stop={t['stop']:.5f}  tp={t['tp1']:.5f}"
        f"\n  Bar H={h:.5f}  L={l:.5f}  C={c:.5f}"
        f"\n  Distance to stop: {stop_dist:.1f}pip  Distance to TP: {tp_dist:.1f}pip"
    )

    if t["tp1_hit"]:
        # Trail mode — ratchet stop to previous bar extreme, exit on close beyond it
        new_ts = _new_trail_stop(config, direction, df, t["trail_stop"])
        if new_ts != t["trail_stop"]:
            t["trail_stop"] = new_ts
            logger.info(f"[PAPER] Trail stop moved to {new_ts:.5f}")
        ts_level = t["trail_stop"] or t["entry"]
        if (long and c < ts_level) or (not long and c > ts_level):
            paper_close_trade(config, state, c, "trail_exit", logger)
        return

    # Stop takes priority
    if (long and l <= t["stop"]) or (not long and h >= t["stop"]):
        paper_close_trade(config, state, t["stop"], "stop_hit", logger)
        return

    tp1 = t["tp1"]
    if (long and h >= tp1) or (not long and l <= tp1):
        if config["tp_mode"] == "full":
            paper_close_trade(config, state, tp1, "tp1_hit", logger)
        else:
            t["tp1_hit"]    = True
            t["trail_stop"] = t["entry"]
            logger.info(
                f"[PAPER] TP1 hit @ {tp1:.5f}. 50% closed. "
                f"Stop moved to breakeven {t['entry']:.5f}"
            )
            state["current_balance"] = round(
                (state["current_balance"] or 0)
                + 0.5 * (abs(tp1 - t["entry"]) / PIP - 3) * PIP_VALUE * t["lot"], 2
            )


def _manage_live_trade(config: dict, state: dict, last_bar: pd.Series,
                        df: pd.DataFrame, logger: logging.Logger) -> None:
    positions = mt5.positions_get(symbol=config["symbol"]) or []
    our_pos   = next((p for p in positions if p.magic == MAGIC), None)
    t         = state.get("open_trade")

    if our_pos is None:
        if t:
            logger.info("Position was closed by MT5 (stop/TP hit). Clearing state.")
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
        logger.warning(f"Reconciled unknown open position #{our_pos.ticket} from MT5.")
        t = state["open_trade"]

    long  = t["direction"] == "long"
    h, l  = float(last_bar["High"]), float(last_bar["Low"])
    tp1   = t["tp1"]

    if not t["tp1_hit"] and config["tp_mode"] == "partial":
        if (long and h >= tp1) or (not long and l <= tp1):
            logger.info(f"TP1 reached. Closing 50% of position {our_pos.ticket}.")
            _market_close_live(our_pos, logger, volume=round(our_pos.volume / 2, 2), comment="TP1 partial")
            live_modify_sl(config, our_pos, our_pos.price_open, logger)
            t["tp1_hit"]    = True
            t["trail_stop"] = our_pos.price_open
            state["current_balance"] = float(mt5.account_info().balance)

    if t["tp1_hit"]:
        new_ts = _new_trail_stop(config, t["direction"], df, t["trail_stop"])
        if new_ts != t["trail_stop"]:
            live_modify_sl(config, our_pos, new_ts, logger)
            t["trail_stop"] = new_ts


def _check_paper_pending(config: dict, state: dict, last_bar: pd.Series,
                          logger: logging.Logger) -> None:
    lim  = state["pending_limit"]
    h, l = float(last_bar["High"]), float(last_bar["Low"])
    dist = (l - lim["price"]) / PIP if lim["direction"] == "long" else (lim["price"] - h) / PIP
    logger.info(
        f"[PAPER] Pending {lim['direction'].upper()} limit @ {lim['price']:.5f}  "
        f"Bar H={h:.5f}  L={l:.5f}  "
        f"{'filled' if dist <= 0 else f'not filled ({dist:.1f}pip away)'}"
    )
    filled = (lim["direction"] == "long"  and l <= lim["price"]) or \
             (lim["direction"] == "short" and h >= lim["price"])
    if not filled:
        return
    paper_open_trade(state, lim, lim["price"], logger)
    t  = state["open_trade"]
    lo = lim["direction"] == "long"
    if (lo and l <= t["stop"]) or (not lo and h >= t["stop"]):
        paper_close_trade(config, state, t["stop"], "stop_hit_same_bar", logger)
    elif (lo and h >= t["tp1"]) or (not lo and l <= t["tp1"]):
        if config["tp_mode"] == "full":
            paper_close_trade(config, state, t["tp1"], "tp1_hit_same_bar", logger)
        else:
            t["tp1_hit"]    = True
            t["trail_stop"] = t["entry"]


def _check_live_pending(config: dict, state: dict, logger: logging.Logger) -> None:
    positions = mt5.positions_get(symbol=config["symbol"]) or []
    our_pos   = next((p for p in positions if p.magic == MAGIC), None)
    if our_pos:
        logger.info(f"Limit order filled. Position #{our_pos.ticket} opened.")
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

def check_entry(config: dict, state: dict, df: pd.DataFrame,
                logger: logging.Logger) -> None:
    if len(df) < 3:
        return

    bar_time = df.index[-1]
    if not in_session(config, bar_time):
        logger.info(f"No entry — outside session ({bar_time.hour:02d}:00 UTC, window {config['session_start_utc']}-{config['session_end_utc']})")
        return

    prev2 = df.iloc[-3]
    prev1 = df.iloc[-2]
    curr  = df.iloc[-1]

    # Log the three bars being evaluated
    logger.info(
        f"Evaluating bars:"
        f"\n  prev2  O={prev2['Open']:.5f}  H={prev2['High']:.5f}  L={prev2['Low']:.5f}  C={prev2['Close']:.5f}"
        f"\n  prev1  O={prev1['Open']:.5f}  H={prev1['High']:.5f}  L={prev1['Low']:.5f}  C={prev1['Close']:.5f}"
        f"\n  curr   O={curr['Open']:.5f}  H={curr['High']:.5f}  L={curr['Low']:.5f}  C={curr['Close']:.5f}"
    )

    # Bar structure checks
    p1_took_high = prev1["High"] > prev2["High"]
    p1_took_low  = prev1["Low"]  < prev2["Low"]
    c_took_high  = curr["High"]  > prev1["High"]
    c_took_low   = curr["Low"]   < prev1["Low"]
    bar_range    = curr["High"] - curr["Low"]
    close_pos    = ((curr["Close"] - curr["Low"]) / bar_range) if bar_range > 0 else 0.5

    logger.info(
        f"Bar conditions:"
        f"\n  prev1 broke high={p1_took_high}  prev1 broke low={p1_took_low}"
        f"\n  curr  broke high={c_took_high}   curr  broke low={c_took_low}"
        f"\n  close_pos={close_pos:.2f}  (need >={config.get('close_strength', 0.6):.2f} long / <={(1-config.get('close_strength',0.6)):.2f} short)"
    )

    # Diagnose which direction (if any) qualifies
    close_strength = config.get("close_strength", 0.6)
    long_struct  = p1_took_high and not p1_took_low and c_took_high and not c_took_low
    short_struct = p1_took_low  and not p1_took_high and c_took_low and not c_took_high

    if long_struct:
        if close_pos < close_strength:
            logger.info(f"No entry — LONG structure valid but close too weak ({close_pos:.2f} < {close_strength:.2f})")
        else:
            logger.debug("LONG structure passed close_strength check")
    elif short_struct:
        if close_pos > (1.0 - close_strength):
            logger.info(f"No entry — SHORT structure valid but close too weak ({close_pos:.2f} > {1-close_strength:.2f})")
        else:
            logger.debug("SHORT structure passed close_strength check")
    else:
        reasons = []
        if not (p1_took_high and not p1_took_low):
            reasons.append("prev1 not clean up-break")
        if not (p1_took_low and not p1_took_high):
            reasons.append("prev1 not clean down-break")
        logger.info(f"No entry — no clean consecutive breakout ({', '.join(reasons)})")

    lb      = config.get("pullback_lookback", 4)
    pb_pips = float(np.median([
        (df.iloc[j]["High"] - df.iloc[j]["Low"]) / PIP
        for j in range(max(0, len(df) - lb), len(df) - 1)
    ]))
    min_stop       = config.get("min_stop_pips", MIN_STOP_PIPS)
    logger.debug(f"Pullback target: {pb_pips:.1f} pip (median of last {lb} bars)  min_stop: {min_stop:.0f} pip")

    result = _bar_setup(prev2, prev1, curr, pb_pips,
                        config.get("stop_buffer", 5), PIP, min_stop,
                        close_strength=close_strength)
    if result is None:
        return

    direction, entry, stop, tp1 = result
    stop_pips = abs(entry - stop) / PIP
    risk_r    = abs(tp1 - entry) / abs(entry - stop)

    logger.info(
        f"SIGNAL: {direction.upper()}  "
        f"entry={entry:.5f}  stop={stop:.5f}  tp={tp1:.5f}  "
        f"stop_dist={stop_pips:.1f}pip  R:R=1:{risk_r:.1f}"
    )

    balance = get_balance(config, state)
    lot     = compute_lot(config, balance, entry, stop)
    if lot <= 0:
        logger.info(f"No entry — lot size zero (stop={stop_pips:.1f}pip  balance=${balance:.2f})")
        return

    logger.info(
        f"Placing order: {direction.upper()} {lot} lots  "
        f"risk=${balance * config['risk_per_trade']:.2f} ({config['risk_per_trade']*100:.1f}%)"
    )
    if config["paper_mode"]:
        paper_cancel_limit(state, "new signal", logger)
        paper_place_limit(state, direction, entry, stop, tp1, lot, logger)
    else:
        live_cancel_limits(config, logger)
        live_place_limit(config, state, direction, entry, stop, tp1, lot, logger)


# ── Main candle handler ───────────────────────────────────────────────────────

def run_candle(config: dict, state: dict, df: pd.DataFrame,
               logger: logging.Logger) -> None:
    ensure_connected(logger)

    last_bar = df.iloc[-1]
    bar_time = df.index[-1]

    logger.info(f"Candle {bar_time}  close={last_bar['Close']:.5f}")

    if check_circuit_breaker(config, state, logger):
        return

    if state["session_starting_balance"] is None:
        bal = get_balance(config, state)
        state["session_starting_balance"] = bal
        state["current_balance"]          = bal
        logger.info(f"Session starting balance: ${bal:.2f}")

    # Trade management
    if state.get("open_trade"):
        logger.info("Managing open trade…")
        if config["paper_mode"]:
            _manage_paper_trade(config, state, last_bar, df, logger)
        else:
            _manage_live_trade(config, state, last_bar, df, logger)

    elif state.get("pending_limit"):
        if config["paper_mode"]:
            _check_paper_pending(config, state, last_bar, logger)
        else:
            _check_live_pending(config, state, logger)

    # Entry check (only if flat)
    if not state.get("open_trade") and not state.get("pending_limit"):
        check_entry(config, state, df, logger)

    save_state(state)
    logger.info(
        f"State saved — balance=${state.get('current_balance', '—')}  "
        f"open={bool(state.get('open_trade'))}  pending={bool(state.get('pending_limit'))}"
    )


# ── Timing ────────────────────────────────────────────────────────────────────

def wait_for_next_candle(last_bar_time: pd.Timestamp, logger: logging.Logger) -> None:
    target     = last_bar_time + pd.Timedelta(hours=4, seconds=30)
    now        = pd.Timestamp.now(tz="UTC")
    sleep_secs = (target - now).total_seconds()
    if sleep_secs > 0:
        logger.info(
            f"Sleeping {sleep_secs/3600:.2f}h until next candle close "
            f"({target.strftime('%Y-%m-%d %H:%M UTC')})"
        )
        time.sleep(sleep_secs)
    logger.info("Waking up — polling for new candle…")


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

    banner = (
        "\n" + "=" * 60 + "\n"
        "  LIVE TRADING MODE ACTIVE\n"
        f"  Symbol:          {config['symbol']}\n"
        f"  Risk per trade:  {config['risk_per_trade'] * 100:.1f}%\n"
        f"  Max drawdown:    {config['max_drawdown_pct']}%\n"
        "  Circuit breaker: ACTIVE\n"
        f"  MT5 account:     {account_num}\n"
        "=" * 60
    )
    print(banner)
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
        logger.critical("Live mode confirmation not received within 30s — shutting down safely.")
        print("\nConfirmation not received. Shutting down safely.\n")
        try:
            dp_disconnect()
        except Exception:
            pass
        sys.exit(1)

    logger.info("Live mode confirmed by operator.")
    print("\nConfirmed. Starting live trading engine…\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    once    = "--once"    in sys.argv
    test_cb = "--test-cb" in sys.argv

    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("Live Trader starting up")
    logger.info("=" * 60)

    config = load_config()
    state  = load_state()

    mode_str = "PAPER" if config["paper_mode"] else "LIVE"
    logger.info(f"Mode: {mode_str}  Symbol: {config['symbol']}  TF: {config['timeframe']}")

    if state.get("circuit_breaker_active"):
        logger.critical(
            f"STARTUP BLOCKED — circuit breaker is active. "
            f"Reason: {state.get('circuit_breaker_reason')}. "
            f"Edit state.json and set circuit_breaker_active=false to resume."
        )
        sys.exit(1)

    if test_cb:
        logger.info("TEST MODE: Forcing circuit breaker via simulated large loss…")
        if state["session_starting_balance"] is None:
            state["session_starting_balance"] = 10_000.0
            state["current_balance"]          = 10_000.0
        breached_balance = state["session_starting_balance"] * (
            1 - (config["max_drawdown_pct"] + 1) / 100
        )
        state["current_balance"] = round(breached_balance, 2)
        save_state(state)
        triggered = check_circuit_breaker(config, state, logger)
        logger.info(f"Circuit breaker triggered: {triggered}")
        if not triggered:
            logger.error("Circuit breaker DID NOT trigger — check logic!")
        sys.exit(0)

    if not config["paper_mode"]:
        _live_mode_confirmation(config, logger)

    if not dp_connect():
        logger.error("Initial MT5 connection failed. Retrying…")
        ensure_connected(logger)

    last_bar_time = None

    while True:
        try:
            df = fetch_data(config, logger)
            current_last = df.index[-1]

            if last_bar_time is None or current_last > last_bar_time:
                last_bar_time = current_last
                run_candle(config, state, df, logger)
                if once:
                    logger.info("--once flag: exiting after one candle.")
                    break
            else:
                logger.debug(f"No new candle yet. Last: {last_bar_time}")

            wait_for_next_candle(last_bar_time, logger)

        except TradingHalt as e:
            logger.critical(f"TRADING HALT: {e}")
            logger.critical("Manual restart required. Exiting.")
            sys.exit(1)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt — shutting down gracefully.")
            dp_disconnect()
            sys.exit(0)

        except Exception:
            logger.error(f"Unexpected exception:\n{traceback.format_exc()}")
            logger.error("Pausing 60s before retry. Check trading_log.txt.")
            save_state(state)
            time.sleep(60)


if __name__ == "__main__":
    cfg = load_config()
    if not cfg.get("paper_mode") and "--live-confirmed" not in sys.argv:
        print("\nLive mode detected. Add --live-confirmed flag to acknowledge real money trading.")
        print("Or set paper_mode=true in config.json for safe testing.\n")
        sys.exit(1)
    main()
