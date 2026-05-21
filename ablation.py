"""
Ablation study: remove one entry filter at a time and measure the impact.

Filters tested (in order of likely impact):
  1. Full system (baseline)
  2. No close_strength filter
  3. No session filter (open 24h)
  4. No outside-bar requirement on signal bar (curr can be an outside bar)
  5. No outside-bar requirement on prev1
  6. No min_stop_pips floor
  7. Market entry (enter at signal-bar close, no pullback required)
  8. No same-setup dedup guard
"""

import math
import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field

from data_pipeline import get_data
from constants import PIP, PIP_VALUE, RISK_PCT
from trade_engine import (
    Trade, _lot_size, _close, _in_session,
    PULLBACK_LOOKBACK, STOP_BUFFER, SLIPPAGE_PIPS,
    MIN_STOP_PIPS, MAX_LOT, ACCOUNT_BALANCE,
)
from backtest_engine import compute_metrics, _split_date, _split_trades

# ── Flexible _bar_setup ───────────────────────────────────────────────────────

def _setup(prev2, prev1, curr, pullback_pips, stop_buf, pip, min_stop_pips,
           close_strength=0.6, clean_signal=True, clean_prev1=True):
    p1_took_high = prev1["High"] > prev2["High"]
    p1_took_low  = prev1["Low"]  < prev2["Low"]
    c_took_high  = curr["High"]  > prev1["High"]
    c_took_low   = curr["Low"]   < prev1["Low"]

    bar_range = curr["High"] - curr["Low"]
    close_pos = ((curr["Close"] - curr["Low"]) / bar_range) if bar_range > 0 else 0.5

    # LONG
    prev1_ok_long = p1_took_high and (not p1_took_low if clean_prev1 else True)
    curr_ok_long  = c_took_high  and (not c_took_low  if clean_signal else True)
    if prev1_ok_long and curr_ok_long:
        if close_strength > 0 and close_pos < close_strength:
            pass
        else:
            entry = curr["High"] - pullback_pips * pip
            stop  = prev1["Low"] - stop_buf * pip
            if entry > stop and (entry - stop) / pip >= min_stop_pips:
                tp1 = entry + 2.0 * (entry - stop)
                return "long", round(entry, 5), round(stop, 5), round(tp1, 5)

    # SHORT
    prev1_ok_short = p1_took_low  and (not p1_took_high if clean_prev1 else True)
    curr_ok_short  = c_took_low   and (not c_took_high  if clean_signal else True)
    if prev1_ok_short and curr_ok_short:
        if close_strength > 0 and close_pos > (1.0 - close_strength):
            pass
        else:
            entry = curr["Low"] + pullback_pips * pip
            stop  = prev1["High"] + stop_buf * pip
            if entry < stop and (stop - entry) / pip >= min_stop_pips:
                tp1 = entry - 2.0 * (stop - entry)
                return "short", round(entry, 5), round(stop, 5), round(tp1, 5)

    return None


# ── Flexible runner ───────────────────────────────────────────────────────────

def run(df, session_end=20, close_strength=0.6, clean_signal=True, clean_prev1=True,
        min_stop_pips=MIN_STOP_PIPS, market_entry=False):

    trades, open_trades = [], []
    trade_id = 0

    last_open_bar    = -999

    for i in range(len(df)):
        row      = df.iloc[i]
        bar_date = df.index[i]
        h, l, c  = row["High"], row["Low"], row["Close"]

        any_closed = False
        for ot in open_trades[:]:
            long = ot.direction == "long"
            if long:
                if l <= ot.stop_price:
                    _close(ot, bar_date, ot.stop_price, "stop", PIP, SLIPPAGE_PIPS)
                    trades.append(ot); open_trades.remove(ot); any_closed = True; continue
                if h >= ot.tp1_price:
                    _close(ot, bar_date, ot.tp1_price, "tp1", PIP, SLIPPAGE_PIPS)
                    trades.append(ot); open_trades.remove(ot); any_closed = True; continue
            else:
                if h >= ot.stop_price:
                    _close(ot, bar_date, ot.stop_price, "stop", PIP, SLIPPAGE_PIPS)
                    trades.append(ot); open_trades.remove(ot); any_closed = True; continue
                if l <= ot.tp1_price:
                    _close(ot, bar_date, ot.tp1_price, "tp1", PIP, SLIPPAGE_PIPS)
                    trades.append(ot); open_trades.remove(ot); any_closed = True; continue

        if any_closed and not open_trades:
            continue
        if open_trades:
            continue
        if i < 2 or not _in_session(bar_date, 0, session_end):
            continue

        prev2 = df.iloc[i - 2]
        prev1 = df.iloc[i - 1]
        pb_start = max(0, i - PULLBACK_LOOKBACK)
        pb_pips  = float(np.median([(df.iloc[j]["High"] - df.iloc[j]["Low"]) / PIP
                                     for j in range(pb_start, i)]))

        result = _setup(prev2, prev1, row, pb_pips, STOP_BUFFER, PIP, min_stop_pips,
                        close_strength, clean_signal, clean_prev1)
        if result is None:
            continue

        direction, entry, stop, tp1 = result

        if market_entry:
            # Enter at signal-bar close instead of limit pullback
            entry = round(c, 5)
            if direction == "long":
                tp1 = round(entry + 2.0 * (entry - stop), 5)
            else:
                tp1 = round(entry - 2.0 * (stop - entry), 5)
            filled = True
        else:
            filled = (direction == "long" and l <= entry) or \
                     (direction == "short" and h >= entry)

        if not filled:
            continue

        lot = _lot_size(ACCOUNT_BALANCE, entry, stop, PIP, PIP_VALUE, RISK_PCT,
                        min_stop_pips=min_stop_pips)
        if lot <= 0:
            continue

        last_open_bar = i
        trade_id        += 1
        nt = Trade(
            trade_id=trade_id, direction=direction, entry_date=bar_date,
            entry_price=entry, stop_price=stop, tp1_price=tp1, lot_size=lot,
            regime=direction.upper() + "_SETUP", trend_strength=None,
            params={},
        )
        open_trades.append(nt)

        # Same-bar exit
        if direction == "long":
            if l <= stop:
                _close(nt, bar_date, stop, "stop", PIP, SLIPPAGE_PIPS)
                trades.append(nt); open_trades.remove(nt)
            elif h >= tp1:
                _close(nt, bar_date, tp1, "tp1", PIP, SLIPPAGE_PIPS)
                trades.append(nt); open_trades.remove(nt)
        else:
            if h >= stop:
                _close(nt, bar_date, stop, "stop", PIP, SLIPPAGE_PIPS)
                trades.append(nt); open_trades.remove(nt)
            elif l <= tp1:
                _close(nt, bar_date, tp1, "tp1", PIP, SLIPPAGE_PIPS)
                trades.append(nt); open_trades.remove(nt)

    for ot in open_trades:
        _close(ot, df.index[-1], df["Close"].iloc[-1], "end_of_data", PIP, SLIPPAGE_PIPS)
        trades.append(ot)

    return trades


# ── Configurations ────────────────────────────────────────────────────────────

CONFIGS = [
    ("Full system",                  dict()),
    ("No close_strength",            dict(close_strength=0.0)),
    ("No session filter (24h)",      dict(session_end=24)),
    ("No signal-bar clean breakout", dict(clean_signal=False)),
    ("No prev1 clean breakout",      dict(clean_prev1=False)),
    ("No min_stop_pips",             dict(min_stop_pips=0.0)),
    ("Market entry (no pullback)",   dict(market_entry=True)),

]

BASE = dict(session_end=20, close_strength=0.6, clean_signal=True, clean_prev1=True,
            min_stop_pips=MIN_STOP_PIPS, market_entry=False)


# ── Run ───────────────────────────────────────────────────────────────────────

df = get_data()
split_dt = _split_date(df)

print(f"\n{'Config':<33} {'Trades':>7} {'d_trades':>9} {'WR%':>6} {'Expect':>8} {'Net R':>8} {'MaxDD%':>8} {'T/mo':>6}")
print("-" * 93)

base_n = None
for label, overrides in CONFIGS:
    cfg = {**BASE, **overrides}
    t   = run(df, **cfg)
    m   = compute_metrics(t)

    n    = m["total_trades"]
    wr   = m["win_rate_pct"]
    exp  = m["expectancy_net_r"]
    netr = m["net_r"]
    mdd  = m["max_drawdown_pct"]
    tpm  = m["trades_per_month"]

    if base_n is None:
        base_n = n
        delta  = "—"
    else:
        delta = f"+{n - base_n}" if n >= base_n else str(n - base_n)

    print(f"{label:<33} {n:>7} {delta:>9} {wr:>6.1f} {exp:>8.4f} {netr:>8.1f} {mdd:>8.2f} {tpm:>6.1f}")
