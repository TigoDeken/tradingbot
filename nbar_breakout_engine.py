"""
N-bar close-breakout engine (backtest + live signal).

Proven signals (8/8 years gross EV > 0, no-lookahead validated):
  GOLD#      N=8 down-break + ATR > 1.2×med6 + depth > 0.677 ATR + 21-23h UTC → long
  US100Cash# N=8 down-break + ATR > 1.2×med6 + marubozu ≥ 0.80  + 07-12h UTC → long

No-lookahead guarantee: all rolling features use .shift(1). Entry at next bar's open.
Stop = entry − 1×ATR(14 EWM at signal bar). TP = entry + RR × stop-distance.
Conservative tie-break: stop checked before TP within the same bar.
"""

import numpy as np
import pandas as pd
from trade_engine import Trade, _close, _lot_size
from constants import RISK_PCT
from regime_filter import classify_volatility, classify_structure, SESSION_WINDOWS, STRUCTURE_WINDOW

ATR_PERIOD      = 14
ATR_MED_MINBARS = 2000   # min bars for 6-month rolling ATR median

SYMBOL_DEFAULTS: dict[str, dict] = {
    # ── Legacy CFD symbols (MT5) ──────────────────────────────────────────────
    "GOLD#": {
        "pip":          0.1,
        "n_bars":       8,
        "atr_mult":     1.2,
        "depth_thresh": 0.677,
        "require_maru": False,
        "maru_thresh":  0.0,
    },
    "US100Cash#": {
        "pip":          1.0,
        "n_bars":       8,
        "atr_mult":     1.2,
        "depth_thresh": 0.0,
        "require_maru": True,
        "maru_thresh":  0.80,
    },
    # ── Bybit linear perpetuals (USDT-margined) ───────────────────────────────
    # pip=1.0, pip_value=1.0 → lot = risk_usd / stop_distance_usd (qty in coins)
    # Same signal parameters as GOLD# — to be calibrated per symbol over time.
    "SOLUSDT":  {"pip": 1.0, "n_bars": 8, "atr_mult": 1.2, "depth_thresh": 0.677, "require_maru": False, "maru_thresh": 0.0},
    "AVAXUSDT": {"pip": 1.0, "n_bars": 8, "atr_mult": 1.2, "depth_thresh": 0.677, "require_maru": False, "maru_thresh": 0.0},
    "LINKUSDT": {"pip": 1.0, "n_bars": 8, "atr_mult": 1.2, "depth_thresh": 0.677, "require_maru": False, "maru_thresh": 0.0},
    "DOTUSDT":  {"pip": 1.0, "n_bars": 8, "atr_mult": 1.2, "depth_thresh": 0.677, "require_maru": False, "maru_thresh": 0.0},
    "ARBUSDT":  {"pip": 1.0, "n_bars": 8, "atr_mult": 1.2, "depth_thresh": 0.677, "require_maru": False, "maru_thresh": 0.0},
    "OPUSDT":   {"pip": 1.0, "n_bars": 8, "atr_mult": 1.2, "depth_thresh": 0.677, "require_maru": False, "maru_thresh": 0.0},
    "INJUSDT":  {"pip": 1.0, "n_bars": 8, "atr_mult": 1.2, "depth_thresh": 0.677, "require_maru": False, "maru_thresh": 0.0},
    "SUIUSDT":  {"pip": 1.0, "n_bars": 8, "atr_mult": 1.2, "depth_thresh": 0.677, "require_maru": False, "maru_thresh": 0.0},
    "APTUSDT":  {"pip": 1.0, "n_bars": 8, "atr_mult": 1.2, "depth_thresh": 0.677, "require_maru": False, "maru_thresh": 0.0},
    "SEIUSDT":  {"pip": 1.0, "n_bars": 8, "atr_mult": 1.2, "depth_thresh": 0.677, "require_maru": False, "maru_thresh": 0.0},
}

NBAR_SYMBOLS = set(SYMBOL_DEFAULTS.keys())


def _features(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Compute signal features with no lookahead (rolling on shift(1) close)."""
    O, H, L, C = df["Open"], df["High"], df["Low"], df["Close"]

    tr      = pd.concat([(H - L),
                          (H - C.shift(1)).abs(),
                          (L - C.shift(1)).abs()], axis=1).max(axis=1)
    atr     = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    atr_med = atr.rolling("180D", min_periods=ATR_MED_MINBARS).median()

    c_min_n  = C.shift(1).rolling(n).min()
    break_dn = C < c_min_n
    depth_dn = (c_min_n - C).abs() / atr.replace(0, np.nan)

    bar_rng  = (H - L).replace(0, np.nan)
    body_pct = (C - O).abs() / bar_rng
    bar_dn   = (C < O).astype(int)

    return pd.DataFrame(
        {"atr": atr, "atr_med": atr_med,
         "c_min_n": c_min_n,
         "break_dn": break_dn, "depth_dn": depth_dn,
         "body_pct": body_pct, "bar_dn": bar_dn},
        index=df.index,
    )


def _check_signal(row: pd.Series, scfg: dict) -> dict | None:
    """Evaluate signal for one bar. Returns {'direction': 'long', 'atr': float} or None."""
    if pd.isna(row["atr_med"]) or row["atr_med"] <= 0:
        return None
    if not (row["atr"] > scfg["atr_mult"] * row["atr_med"]):
        return None
    if not row["break_dn"]:
        return None
    if scfg["depth_thresh"] > 0 and row["depth_dn"] < scfg["depth_thresh"]:
        return None
    if scfg["require_maru"]:
        if pd.isna(row["body_pct"]) or row["body_pct"] < scfg["maru_thresh"] or row["bar_dn"] != 1:
            return None
    return {"direction": "long", "atr": float(row["atr"])}


def _in_named_sessions(hour: int, active: list) -> bool:
    for name in active:
        start, end = SESSION_WINDOWS[name]
        if start <= hour < end:
            return True
    return False


def run_nbar(
    df:              pd.DataFrame,
    symbol:          str,
    tp_rr:           float = 2.0,
    slippage_pips:   float = 0.0,
    pip:             float | None = None,
    pip_value:       float = 1.0,
    risk_pct:        float = RISK_PCT,
    account_balance: float = 10_000.0,
    commission_rt:   float = 0.0,
    fixed_risk:      float = 100.0,
    sessions:        list | None = None,
    vol_filter:      list | None = None,
    min_cleanness:   float        = 0.0,
    struct_window:   int          = STRUCTURE_WINDOW,
    **overrides,
) -> tuple[list[Trade], dict]:
    """
    Backtest N-bar breakout over a full OHLC DataFrame.
    Returns (trades, stats_dict).
    """
    scfg = {**SYMBOL_DEFAULTS.get(symbol, {}), **overrides}
    if not scfg:
        raise ValueError(f"Unknown symbol '{symbol}' — pass overrides or use a known symbol.")
    if pip is None:
        pip = scfg["pip"]

    n           = int(scfg["n_bars"])
    _sessions   = sessions if sessions else list(SESSION_WINDOWS.keys())
    feat        = _features(df, n)

    _vol_filter = set(vol_filter) if vol_filter else {"low", "mid", "high"}
    vol_class   = classify_volatility(df)
    cleanness   = classify_structure(df, window=struct_window)

    trades:      list[Trade] = []
    open_trades: list[Trade] = []
    pending_sig: dict | None = None
    trade_id = n_signals = n_fills = 0

    for i, (bar_date, row) in enumerate(df.iterrows()):
        o, h, l = row.Open, row.High, row.Low

        # ── 1. Fill pending signal: entry at this bar's open ───────────────────
        if pending_sig is not None and not open_trades:
            sig_atr   = pending_sig["atr"]
            direction = pending_sig["direction"]
            entry     = o
            stop      = entry - sig_atr if direction == "long" else entry + sig_atr
            risk_dist = abs(entry - stop)

            if risk_dist > 0 and not (direction == "long" and entry <= stop):
                tp1 = entry + tp_rr * risk_dist if direction == "long" else entry - tp_rr * risk_dist
                lot = _lot_size(account_balance, entry, stop, pip, pip_value,
                                risk_pct, 2.0, 100.0, fixed_risk)
                if lot > 0:
                    trade_id += 1
                    n_fills  += 1
                    t = Trade(
                        trade_id=trade_id, direction=direction,
                        entry_date=bar_date, entry_price=round(entry, 5),
                        stop_price=round(stop, 5), tp1_price=round(tp1, 5),
                        lot_size=lot, regime="NBAR", trend_strength=None,
                        params={"symbol": symbol, "atr": round(sig_atr, 4),
                                "break_level": round(pending_sig.get("break_level", 0), 4),
                                "vol_class":   pending_sig.get("vol_class"),
                                "cleanness":   pending_sig.get("cleanness")},
                        _fill_bar_idx=i,
                    )
                    open_trades.append(t)
                    # same-bar stop check (conservative tie-break)
                    if direction == "long" and l <= stop:
                        _close(t, bar_date, stop, "stop", pip, slippage_pips, pip_value, commission_rt)
                        trades.append(t)
                        open_trades.remove(t)
            pending_sig = None

        # ── 2. Manage open trades ──────────────────────────────────────────────
        for ot in open_trades[:]:
            if ot.direction == "long":
                if l <= ot.stop_price:
                    _close(ot, bar_date, ot.stop_price, "stop", pip, slippage_pips, pip_value, commission_rt)
                    trades.append(ot); open_trades.remove(ot); continue
                if h >= ot.tp1_price:
                    _close(ot, bar_date, ot.tp1_price, "tp1", pip, slippage_pips, pip_value, commission_rt)
                    trades.append(ot); open_trades.remove(ot); continue
            else:
                if h >= ot.stop_price:
                    _close(ot, bar_date, ot.stop_price, "stop", pip, slippage_pips, pip_value, commission_rt)
                    trades.append(ot); open_trades.remove(ot); continue
                if l <= ot.tp1_price:
                    _close(ot, bar_date, ot.tp1_price, "tp1", pip, slippage_pips, pip_value, commission_rt)
                    trades.append(ot); open_trades.remove(ot); continue

        # ── 3. Check for new signal on this bar ────────────────────────────────
        if open_trades or pending_sig is not None:
            continue
        if not _in_named_sessions(bar_date.hour, _sessions):
            continue
        vc = vol_class.iloc[i]
        if not (pd.isna(vc) or vc in _vol_filter):
            continue
        cs = cleanness.iloc[i]
        if not (pd.isna(cs) or cs >= min_cleanness):
            continue
        sig = _check_signal(feat.iloc[i], scfg)
        if sig:
            pending_sig = {**sig, "break_level": float(feat.iloc[i]["c_min_n"]),
                           "vol_class": vc,
                           "cleanness": round(float(cs), 1) if not pd.isna(cs) else None}
            n_signals  += 1

    # Close any still-open trades at end of data
    for ot in open_trades:
        _close(ot, df.index[-1], df["Close"].iloc[-1], "end_of_data",
               pip, slippage_pips, pip_value, commission_rt)
        trades.append(ot)

    return trades, {"signals": n_signals, "fills": n_fills}


def compute_live_signal(df: pd.DataFrame, symbol: str,
                        sessions: list | None = None,
                        all_hours: bool = False,
                        **overrides) -> dict | None:
    """
    Evaluate the N-bar breakout signal on the last closed bar of `df`.

    Returns {'direction': str, 'atr': float} or None.
    Called by live_trader.py check_entry() on each new candle.

    Args:
        all_hours: if True, skip session-window filtering (for 24/7 crypto).
    """
    scfg = {**SYMBOL_DEFAULTS.get(symbol, {}), **overrides}
    if not scfg:
        return None
    if len(df) < ATR_MED_MINBARS + int(scfg["n_bars"]):
        return None

    if not all_hours:
        _sessions = sessions if sessions else list(SESSION_WINDOWS.keys())
        bar_date  = df.index[-1]
        if not _in_named_sessions(bar_date.hour, _sessions):
            return None

    feat = _features(df, int(scfg["n_bars"]))
    return _check_signal(feat.iloc[-1], scfg)
