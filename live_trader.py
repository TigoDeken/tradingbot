"""
live_trader.py — Step 7
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
from swing_engine   import build_swings, PIP
from trend_engine   import classify_trend
from trade_engine   import _compute_setup, _confirmed_at, _pullback_depths, PIP_VALUE
from constants import MAGIC, MAX_LIMIT_BARS

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
    # File handler (always append)
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    # Force UTC timestamps
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
        "risk_per_trade", "tp_mode", "swing_n", "min_swing_size",
        "min_swing_increment", "min_trend_size", "trend_range_ratio",
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
    "open_trade":               None,   # filled when position is open
    "pending_limit":            None,   # filled when limit order is waiting for fill
    "last_regime":              "AMBIGUOUS",
    "last_swing_points":        [],
    "last_entry_zone":          None,
    "last_updated":             None,
}


def load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return {**_DEFAULT_STATE, **json.load(f)}
    return dict(_DEFAULT_STATE)


def save_state(state: dict) -> None:
    state["last_updated"] = pd.Timestamp.now(tz="UTC").isoformat()
    tmp = STATE_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    tmp.replace(STATE_PATH)  # atomic rename — prevents corrupt state.json on crash


# ── MT5 connection ────────────────────────────────────────────────────────────

def _is_connected() -> bool:
    try:
        info = mt5.terminal_info()
        return info is not None and info.connected
    except Exception:
        return False


def ensure_connected(logger: logging.Logger) -> None:
    """Block until MT5 is connected, retrying every RECONNECT_INTERVAL seconds."""
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
    """Fetch OHLC from MT5, reconnecting if necessary."""
    ensure_connected(logger)
    tf = _TF_MAP.get(config["timeframe"], mt5.TIMEFRAME_H4)
    df = fetch_ohlc(symbol=config["symbol"], timeframe=tf, bars=5000)
    logger.debug(f"Fetched {len(df)} bars. Last: {df.index[-1]}")
    return df


# ── Circuit breaker ───────────────────────────────────────────────────────────

def _close_all_paper(state: dict, logger: logging.Logger) -> None:
    if state.get("open_trade"):
        logger.warning("[CB] Closing paper position at circuit breaker trigger.")
        state["open_trade"]    = None
    if state.get("pending_limit"):
        logger.warning("[CB] Cancelling pending paper limit at circuit breaker trigger.")
        state["pending_limit"] = None


def _close_all_live(config: dict, state: dict, logger: logging.Logger) -> None:
    sym = config["symbol"]
    positions = mt5.positions_get(symbol=sym) or []
    for p in positions:
        if p.magic == MAGIC:
            _market_close_live(p, logger)
    orders = mt5.orders_get(symbol=sym) or []
    for o in orders:
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
    """Returns True (and triggers CB) if drawdown limit is breached."""
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
    return math.floor(risk / (stop_pips * PIP_VALUE) * 100) / 100


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
    t = state["open_trade"]
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
    sym = config["symbol"]
    order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == "long" else mt5.ORDER_TYPE_SELL_LIMIT
    tp_price   = tp1 if config["tp_mode"] == "full" else 0.0  # partial: manage TP manually

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
    orders = mt5.orders_get(symbol=config["symbol"]) or []
    for o in orders:
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


# ── Per-candle trade management ───────────────────────────────────────────────

def _manage_paper_trade(config: dict, state: dict, last_bar: pd.Series,
                         swings: pd.DataFrame, df: pd.DataFrame, bar_i: int,
                         logger: logging.Logger) -> None:
    t = state["open_trade"]
    h, l, c = float(last_bar["High"]), float(last_bar["Low"]), float(last_bar["Close"])
    direction = t["direction"]
    long = direction == "long"

    if t["tp1_hit"]:
        # Trail mode — exit on close below trail_stop (long) or above (short)
        confirmed = _confirmed_at(swings, bar_i, config["swing_n"])
        if long:
            lows = confirmed[confirmed["type"] == "low"]
            if not lows.empty:
                latest_l = float(lows.iloc[-1]["price"])
                if t["trail_stop"] is None or latest_l > t["trail_stop"]:
                    t["trail_stop"] = latest_l
                    logger.info(f"[PAPER] Trail stop raised to {latest_l:.5f}")
            ts_level = t["trail_stop"] or t["entry"]
            if c < ts_level:
                paper_close_trade(config, state, c, "trail_exit", logger)
        else:
            highs = confirmed[confirmed["type"] == "high"]
            if not highs.empty:
                latest_h = float(highs.iloc[-1]["price"])
                if t["trail_stop"] is None or latest_h < t["trail_stop"]:
                    t["trail_stop"] = latest_h
                    logger.info(f"[PAPER] Trail stop lowered to {latest_h:.5f}")
            ts_level = t["trail_stop"] or t["entry"]
            if c > ts_level:
                paper_close_trade(config, state, c, "trail_exit", logger)
        return

    # Normal mode — stop takes priority
    if (long and l <= t["stop"]) or (not long and h >= t["stop"]):
        paper_close_trade(config, state, t["stop"], "stop_hit", logger)
        return

    tp1 = t["tp1"]
    if (long and h >= tp1) or (not long and l <= tp1):
        if config["tp_mode"] == "full":
            paper_close_trade(config, state, tp1, "tp1_hit", logger)
        else:
            t["tp1_hit"]    = True
            t["trail_stop"] = t["entry"]  # breakeven
            logger.info(f"[PAPER] TP1 hit @ {tp1:.5f}. 50% closed. Stop moved to breakeven {t['entry']:.5f}")
            state["current_balance"] = round(
                (state["current_balance"] or 0)
                + 0.5 * (abs(tp1 - t["entry"]) / PIP - 3) * PIP_VALUE * t["lot"], 2
            )


def _manage_live_trade(config: dict, state: dict, last_bar: pd.Series,
                        swings: pd.DataFrame, bar_i: int, logger: logging.Logger) -> None:
    """Reconcile live positions with state and apply trail/TP logic."""
    positions = mt5.positions_get(symbol=config["symbol"]) or []
    our_pos   = next((p for p in positions if p.magic == MAGIC), None)
    t = state.get("open_trade")

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
        # MT5 has position we don't know about — reconstruct minimal state
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
        confirmed = _confirmed_at(swings, bar_i, config["swing_n"])
        if long:
            lows = confirmed[confirmed["type"] == "low"]
            if not lows.empty:
                latest_l = float(lows.iloc[-1]["price"])
                if t["trail_stop"] is None or latest_l > t["trail_stop"]:
                    live_modify_sl(config, our_pos, latest_l, logger)
                    t["trail_stop"] = latest_l


def _check_paper_pending(config: dict, state: dict, last_bar: pd.Series,
                          logger: logging.Logger) -> None:
    lim  = state["pending_limit"]
    h, l = float(last_bar["High"]), float(last_bar["Low"])
    filled = (lim["direction"] == "long"  and l <= lim["price"]) or \
             (lim["direction"] == "short" and h >= lim["price"])
    if not filled:
        return
    paper_open_trade(state, lim, lim["price"], logger)
    # Same-bar exit check (stop beats TP)
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
    """Check if our pending limit order was filled — update state if so."""
    positions = mt5.positions_get(symbol=config["symbol"]) or []
    our_pos   = next((p for p in positions if p.magic == MAGIC), None)
    if our_pos:
        logger.info(f"Limit order filled. Position #{our_pos.ticket} opened.")
        lim = state.get("pending_limit", {})
        state["open_trade"] = {
            "direction": "long" if our_pos.type == 0 else "short",
            "entry": our_pos.price_open, "stop": our_pos.sl,
            "tp1": lim.get("tp1", our_pos.tp), "lot": our_pos.volume,
            "tp1_hit": False, "trail_stop": None,
            "ticket": our_pos.ticket,
        }
        state["pending_limit"] = None
        state["current_balance"] = float(mt5.account_info().balance)


# ── Entry logic ───────────────────────────────────────────────────────────────

def check_entry(config: dict, state: dict, df: pd.DataFrame,
                swings: pd.DataFrame, logger: logging.Logger) -> None:
    last_bar  = df.iloc[-1]
    bar_i     = len(df) - 1
    bar_time  = df.index[-1]
    regime    = str(last_bar.get("regime", "AMBIGUOUS"))

    if regime not in ("UPTREND", "DOWNTREND"):
        logger.info(f"No entry — regime: {regime}")
        return
    if not in_session(config, bar_time):
        logger.info(f"No entry — outside session hours ({bar_time.hour:02d}:00 UTC)")
        return

    direction = "long" if regime == "UPTREND" else "short"
    confirmed = _confirmed_at(swings, bar_i, config["swing_n"])
    df_window = df.iloc[max(0, bar_i - 19): bar_i + 1]

    setup = _compute_setup(
        confirmed, direction, df_window,
        config["pullback_lookback"], config["stop_buffer"], PIP,
    )
    if setup is None:
        logger.info("No entry — setup returned None (insufficient structure)")
        state["last_entry_zone"] = None
        return

    entry, stop, tp1 = setup
    state["last_entry_zone"] = entry

    balance = get_balance(config, state)
    lot     = compute_lot(config, balance, entry, stop)
    if lot <= 0:
        logger.info(f"No entry — lot size zero (balance=${balance:.2f}, stop={abs(entry-stop)/PIP:.1f}pip)")
        return

    logger.info(
        f"Entry signal: {direction.upper()}  entry={entry:.5f}  stop={stop:.5f}  "
        f"tp1={tp1:.5f}  lot={lot}  risk={abs(entry-stop)/PIP:.1f}pip"
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

    # Build pipeline
    swings = build_swings(
        df,
        swing_n=config["swing_n"],
        min_swing_size=config["min_swing_size"],
        min_swing_increment=config["min_swing_increment"],
        pip=PIP,
    )
    df = classify_trend(
        df, swings,
        swing_n=config["swing_n"],
        min_swing_increment=config["min_swing_increment"],
        min_trend_size=config["min_trend_size"],
        trend_range_ratio=config["trend_range_ratio"],
        pip=PIP,
    )

    last_bar  = df.iloc[-1]
    bar_i     = len(df) - 1
    bar_time  = df.index[-1]
    regime    = str(last_bar.get("regime", "AMBIGUOUS"))
    strength  = last_bar.get("trend_strength_score")

    # Update state metadata
    confirmed       = _confirmed_at(swings, bar_i, config["swing_n"])
    state["last_regime"] = regime
    state["last_swing_points"] = confirmed.tail(5)[["date", "price", "type"]].to_dict("records")

    logger.info(
        f"Candle {bar_time}  close={last_bar['Close']:.5f}  "
        f"regime={regime}  strength={strength if strength and not (isinstance(strength, float) and np.isnan(strength)) else '—'}"
    )

    # Circuit breaker check
    if check_circuit_breaker(config, state, logger):
        return

    # Init starting balance on first candle
    if state["session_starting_balance"] is None:
        bal = get_balance(config, state)
        state["session_starting_balance"] = bal
        state["current_balance"]          = bal
        logger.info(f"Session starting balance: ${bal:.2f}")

    # ── Trade management ──────────────────────────────────────────────────────
    if state.get("open_trade"):
        logger.info("Managing open trade…")
        if config["paper_mode"]:
            _manage_paper_trade(config, state, last_bar, swings, df, bar_i, logger)
        else:
            _manage_live_trade(config, state, last_bar, swings, bar_i, logger)

    elif state.get("pending_limit"):
        # Check if limit was filled this candle
        if config["paper_mode"]:
            _check_paper_pending(config, state, last_bar, logger)
        else:
            _check_live_pending(config, state, logger)

        # Cancel pending limit if regime changed or outside session
        if state.get("pending_limit"):
            lim = state["pending_limit"]
            lim_direction = lim["direction"]
            if (regime == "UPTREND" and lim_direction != "long") or \
               (regime == "DOWNTREND" and lim_direction != "short") or \
               (regime == "AMBIGUOUS"):
                if config["paper_mode"]:
                    paper_cancel_limit(state, "regime changed", logger)
                else:
                    live_cancel_limits(config, logger)
                    state["pending_limit"] = None

    # ── Entry check ───────────────────────────────────────────────────────────
    if not state.get("open_trade") and not state.get("pending_limit"):
        check_entry(config, state, df, swings, logger)

    save_state(state)
    logger.info(
        f"State saved — balance=${state.get('current_balance', '—')}  "
        f"open={bool(state.get('open_trade'))}  pending={bool(state.get('pending_limit'))}"
    )


# ── Timing ────────────────────────────────────────────────────────────────────

def wait_for_next_candle(last_bar_time: pd.Timestamp, logger: logging.Logger) -> pd.DataFrame:
    """Sleep until 30s after the next expected 4H candle close, then poll until confirmed."""
    target     = last_bar_time + pd.Timedelta(hours=4, seconds=30)
    now        = pd.Timestamp.now(tz="UTC")
    sleep_secs = (target - now).total_seconds()
    if sleep_secs > 0:
        logger.info(f"Sleeping {sleep_secs/3600:.2f}h until next candle close ({target.strftime('%Y-%m-%d %H:%M UTC')})")
        time.sleep(sleep_secs)
    logger.info("Waking up — polling for new candle…")


# ── Live mode confirmation ────────────────────────────────────────────────────

def _live_mode_confirmation(config: dict, logger: logging.Logger) -> None:
    """
    Print and log a live-mode summary, then require the operator to type 'yes'
    within 30 seconds. Exits safely if confirmation is not received in time.
    """
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
        f"  Confirm MT5 account number: {account_num}\n"
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
        logger.critical("Live mode confirmation not received within 30 s — shutting down safely.")
        print("\n⚠  Confirmation not received. Shutting down safely.\n")
        try:
            dp_disconnect()
        except Exception:
            pass
        sys.exit(1)

    logger.info("Live mode confirmed by operator.")
    print("\n✓ Confirmed. Starting live trading engine…\n")


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

    # ── Startup circuit breaker guard ─────────────────────────────────────────
    if state.get("circuit_breaker_active"):
        logger.critical(
            f"STARTUP BLOCKED — circuit breaker is active. "
            f"Reason: {state.get('circuit_breaker_reason')}. "
            f"Edit state.json and set circuit_breaker_active=false to resume."
        )
        sys.exit(1)

    # ── Test mode: force circuit breaker ─────────────────────────────────────
    if test_cb:
        logger.info("TEST MODE: Forcing circuit breaker via simulated large loss…")
        if state["session_starting_balance"] is None:
            state["session_starting_balance"] = 10_000.0
            state["current_balance"]          = 10_000.0
        # Drop balance enough to breach threshold
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

    # ── Live mode confirmation (terminal prompt, 30s timeout) ─────────────────
    if not config["paper_mode"]:
        _live_mode_confirmation(config, logger)

    # ── Connect to MT5 ────────────────────────────────────────────────────────
    if not dp_connect():
        logger.error("Initial MT5 connection failed. Retrying…")
        ensure_connected(logger)

    # ── Main loop ─────────────────────────────────────────────────────────────
    last_bar_time = None

    while True:
        try:
            # Fetch current data
            df = fetch_data(config, logger)

            current_last = df.index[-1]

            if last_bar_time is None or current_last > last_bar_time:
                # New candle — run full logic
                last_bar_time = current_last
                run_candle(config, state, df, logger)

                if once:
                    logger.info("--once flag: exiting after one candle.")
                    break
            else:
                logger.debug(f"No new candle yet. Last: {last_bar_time}")

            # Sleep until next expected candle close
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
    # Guard against accidental live mode
    cfg = load_config()
    if not cfg.get("paper_mode") and "--live-confirmed" not in sys.argv:
        print("\n⚠  Live mode detected. Add --live-confirmed flag to acknowledge real money trading.")
        print("   Or set paper_mode=true in config.json for safe testing.\n")
        sys.exit(1)
    main()
