"""
Persistent state: open positions and trade log.

live_state.json  — open positions + last scan results
trade_log.csv    — completed trade history
"""

import json
import os
import csv
import shutil
import logging
from datetime import datetime, timezone

STATE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "live", "live_state.json")
LOG_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "live", "trade_log.csv")
LOG_FIELDS = ["closed_at", "symbol", "entry_date", "entry_price", "exit_price",
              "qty", "stop", "ret_pct", "pnl_usdt", "exit_reason",
              "entry_fz", "entry_oiz", "entry_atr"]

_EMPTY_STATE = {
    "positions":    [],
    "scan_results": [],
    "last_scan":    None,
    "equity":       None,
}

log = logging.getLogger("state")


def _ensure_dirs():
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH),   exist_ok=True)


def load_state() -> dict:
    _ensure_dirs()
    if not os.path.exists(STATE_PATH):
        return dict(_EMPTY_STATE)
    # Try main file, fall back to backup if corrupted
    for path in (STATE_PATH, STATE_PATH + ".bak"):
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            log.warning("State file corrupted at %s — trying backup", path)
    log.error("All state files corrupted — starting with empty state")
    return dict(_EMPTY_STATE)


def save_state(state: dict):
    """Atomic write: tmp file → rename, with backup of previous state."""
    _ensure_dirs()
    if os.path.exists(STATE_PATH):
        try:
            shutil.copy2(STATE_PATH, STATE_PATH + ".bak")
        except OSError:
            pass
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, STATE_PATH)  # atomic on same filesystem


def add_position(state: dict, pos: dict) -> dict:
    state["positions"] = [p for p in state["positions"] if p["symbol"] != pos["symbol"]]
    pos.setdefault("entry_date", datetime.now(timezone.utc).isoformat())
    state["positions"].append(pos)
    return state


def remove_position(state: dict, symbol: str) -> dict:
    state["positions"] = [p for p in state["positions"] if p["symbol"] != symbol]
    return state


def get_position(state: dict, symbol: str) -> dict | None:
    for p in state["positions"]:
        if p["symbol"] == symbol:
            return p
    return None


def log_trade(trade: dict):
    _ensure_dirs()
    write_header = not os.path.exists(LOG_PATH)
    trade.setdefault("closed_at", datetime.now(timezone.utc).isoformat())
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(trade)


def load_trade_log() -> list:
    _ensure_dirs()
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH) as f:
        return list(csv.DictReader(f))
