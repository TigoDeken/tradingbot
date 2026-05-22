import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dataclasses import dataclass, field
from typing import Optional

from data_pipeline import get_data
from constants import PIP

# ── Parameters ────────────────────────────────────────────────────────────────
from constants import PIP_VALUE, RISK_PCT  # noqa: F401 (PIP_VALUE re-exported)

PULLBACK_LOOKBACK = 4
STOP_BUFFER       = 5        # pips
TP_MODE           = "full"   # "full" | "partial"
SLIPPAGE_PIPS     = 3
COMMISSION_RT_USD = 7.0      # round-trip commission per lot (raw account)
ACCOUNT_BALANCE   = 10_000.0
MIN_STOP_PIPS     = 10.0     # floor to prevent degenerate lot sizing
MAX_LOT           = 10.0     # hard cap on position size


# ── Trade record ─────────────────────────────────────────────────────────────

@dataclass
class Trade:
    trade_id:            int
    direction:           str            # "long" | "short"
    entry_date:          pd.Timestamp
    entry_price:         float
    stop_price:          float
    tp1_price:           float
    lot_size:            float
    regime:              str
    trend_strength:      Optional[float]
    params:              dict

    # Populated on close
    exit_date:           Optional[pd.Timestamp] = None
    exit_price:          Optional[float]        = None
    exit_reason:         Optional[str]          = None
    gross_pips:          Optional[float]        = None
    net_pips:            Optional[float]        = None
    gross_r:             Optional[float]        = None
    net_r:               Optional[float]        = None
    duration_hours:      Optional[float]        = None

    commission_usd:      Optional[float]        = None
    net_r_pre_commission: Optional[float]       = None

    # Internal partial-mode state (not part of output)
    _tp1_hit:            bool            = field(default=False,  repr=False)
    _tp1_pips:           float           = field(default=0.0,    repr=False)
    _trail_stop:         Optional[float] = field(default=None,   repr=False)
    _consec_fav:         int             = field(default=0,      repr=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _in_session(ts: pd.Timestamp, session_start: int = 7, session_end: int = 20) -> bool:
    """True if ts (UTC) falls within the configured session window."""
    return session_start <= ts.hour < session_end


def _atr(df: pd.DataFrame, i: int, period: int = 14) -> float:
    """Average True Range of the last `period` bars ending at bar i-1."""
    start = max(1, i - period)
    trs = [
        max(df.iloc[j]["High"] - df.iloc[j]["Low"],
            abs(df.iloc[j]["High"] - df.iloc[j - 1]["Close"]),
            abs(df.iloc[j]["Low"]  - df.iloc[j - 1]["Close"]))
        for j in range(start, i)
    ]
    return (sum(trs) / len(trs)) if trs else 0.0


def _grid_levels(price: float, box_size_pips: float, pip: float) -> tuple[float, float]:
    """Return (level_below, level_above) for the box grid containing price."""
    box = box_size_pips * pip
    idx = math.floor(round(price / box, 6))
    return round(idx * box, 5), round((idx + 1) * box, 5)


def _wm_levels(weekly_df: pd.DataFrame, monthly_df: pd.DataFrame,
               bar_time: pd.Timestamp) -> list[float]:
    """Previous week H/L and previous month H/L as candidate levels."""
    levels = []
    pw = weekly_df[weekly_df.index < bar_time]
    if len(pw) >= 1:
        levels += [pw.iloc[-1]["High"], pw.iloc[-1]["Low"]]
    pm = monthly_df[monthly_df.index < bar_time]
    if len(pm) >= 1:
        levels += [pm.iloc[-1]["High"], pm.iloc[-1]["Low"]]
    return sorted(set(round(l, 5) for l in levels))


def _lot_size(balance: float, entry: float, stop: float, pip: float, pip_value: float,
              risk_pct: float = RISK_PCT, min_stop_pips: float = MIN_STOP_PIPS,
              max_lot: float = MAX_LOT) -> float:
    risk      = balance * risk_pct
    stop_pips = abs(entry - stop) / pip
    if stop_pips < min_stop_pips:
        return 0.0
    lot = math.floor(risk / (stop_pips * pip_value) * 100) / 100
    if 0 < lot < 0.01:
        lot = 0.01
    return min(lot, max_lot)


def _close(
    trade:         Trade,
    exit_date:     pd.Timestamp,
    exit_px:       float,
    reason:        str,
    pip:           float,
    slippage:      float,
    pip_value:     float = PIP_VALUE,
    commission_rt: float = COMMISSION_RT_USD,
) -> None:
    sign = 1 if trade.direction == "long" else -1

    if trade._tp1_hit:
        # Partial mode: weight 50 % TP1 + 50 % trail exit
        trail_pips = sign * (exit_px - trade.entry_price) / pip
        gross = 0.5 * trade._tp1_pips + 0.5 * trail_pips
    else:
        gross = sign * (exit_px - trade.entry_price) / pip

    commission_usd  = commission_rt * trade.lot_size
    commission_pips = commission_usd / pip_value
    net       = gross - slippage - commission_pips
    risk_pips = abs(trade.tp1_price - trade.entry_price) / pip / 2.0

    trade.exit_date      = exit_date
    trade.exit_price     = round(exit_px, 5)
    trade.exit_reason    = reason
    trade.gross_pips     = round(gross, 1)
    trade.net_pips       = round(net,   1)
    trade.gross_r        = round(gross / risk_pips, 2) if risk_pips else 0.0
    trade.net_r          = round(net   / risk_pips, 2) if risk_pips else 0.0
    trade.duration_hours      = round((exit_date - trade.entry_date).total_seconds() / 3600, 1)
    trade.commission_usd      = round(commission_usd, 2)
    net_pre                   = gross - slippage
    trade.net_r_pre_commission = round(net_pre / risk_pips, 2) if risk_pips else 0.0


# ── Simulation ────────────────────────────────────────────────────────────────

def run_trades(
    df:               pd.DataFrame,
    pullback_lookback:int   = PULLBACK_LOOKBACK,
    stop_buffer:      float = STOP_BUFFER,
    tp_mode:          str   = TP_MODE,
    slippage_pips:    float = SLIPPAGE_PIPS,
    pip:              float = PIP,
    pip_value:        float = PIP_VALUE,
    risk_pct:         float = RISK_PCT,
    account_balance:  float = ACCOUNT_BALANCE,
    session_start:    int   = 0,
    session_end:      int   = 20,
    min_stop_pips:    float = MIN_STOP_PIPS,
    max_lot:          float = MAX_LOT,
    max_open_lots:        float = 0.0,
    min_pyramid_bars:     int   = 2,
    pyramid_min_profit_r: float = 0.0,
    close_strength:       float = 0.0,
    commission_rt:        float = COMMISSION_RT_USD,
    tp_rr:                float = 1.0,
    fixed_stop_pips:      float = 25.0,
    box_size_pips:        float = 50.0,
    atr_period:           int   = 14,
    atr_max_pips:         float = 0.0,
    stop_atr_mult:        float = 0.0, # 0 = use fixed_stop_pips; >0 = stop = mult * ATR
    consec_bars_exit:     int   = 0,
    strong_bar_exit:      float = 0.0,
    **_kwargs,
) -> tuple[list[Trade], dict]:
    """
    Iterate bar-by-bar and simulate consecutive-breakout entries with stop/TP/trail exits.
    max_open_lots > 0 enables pyramiding: adds fire when a new setup forms in the same
    direction, provided total open lots stay under the cap and min_pyramid_bars have elapsed.
    Each add locks the previous position's stop at breakeven + slippage.
    Returns a list of completed Trade objects.
    """
    trades:           list[Trade]     = []
    open_trades:      list[Trade]     = []
    trade_id          = 0
    last_open_bar:    int             = -999  # bar index when last trade opened

    session_long:  dict | None = None
    session_short: dict | None = None
    signals_total     = 0
    fills_total       = 0
    expirations_total = 0

    params = dict(stop_buffer=stop_buffer, tp_mode=tp_mode,
                  slippage_pips=slippage_pips, box_size_pips=box_size_pips)

    # Pre-compute weekly and monthly OHLC (used for level lookup)
    _agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    weekly_df  = df.resample("W").agg(_agg).dropna()
    monthly_df = df.resample("ME").agg(_agg).dropna()

    for i in range(len(df)):
        row      = df.iloc[i]
        bar_date = df.index[i]
        h, l, c  = row["High"], row["Low"], row["Close"]

        # ── 1. Manage all open trades ────────────────────────────────────────
        any_closed = False
        for ot in open_trades[:]:
            long = ot.direction == "long"

            if ot._tp1_hit:
                # Partial trail: trail stop to previous bar's low/high
                if i >= 1:
                    if long:
                        recent_low = df.iloc[i - 1]["Low"]
                        if ot._trail_stop is None or recent_low > ot._trail_stop:
                            ot._trail_stop = recent_low
                    else:
                        recent_high = df.iloc[i - 1]["High"]
                        if ot._trail_stop is None or recent_high < ot._trail_stop:
                            ot._trail_stop = recent_high
                ts = ot._trail_stop or ot.entry_price
                if (long and c < ts) or (not long and c > ts):
                    _close(ot, bar_date, c, "trail_exit", pip, slippage_pips, pip_value, commission_rt)
                    trades.append(ot); open_trades.remove(ot); any_closed = True; continue

            else:
                # Trail mode: ratchet stop to previous bar's extreme
                if tp_mode == "trail" and i >= 1:
                    prev_bar = df.iloc[i - 1]
                    if long:
                        new_stop = prev_bar["Low"] - stop_buffer * pip
                        if new_stop > ot.stop_price:
                            ot.stop_price = round(new_stop, 5)
                    else:
                        new_stop = prev_bar["High"] + stop_buffer * pip
                        if new_stop < ot.stop_price:
                            ot.stop_price = round(new_stop, 5)

                # Stop takes priority over TP within the same bar
                if long:
                    if l <= ot.stop_price:
                        _close(ot, bar_date, ot.stop_price, "stop", pip, slippage_pips, pip_value, commission_rt)
                        trades.append(ot); open_trades.remove(ot); any_closed = True; continue
                    if h >= ot.tp1_price and tp_mode != "trail":
                        if tp_mode == "full":
                            _close(ot, bar_date, ot.tp1_price, "tp1", pip, slippage_pips, pip_value, commission_rt)
                            trades.append(ot); open_trades.remove(ot); any_closed = True; continue
                        else:
                            ot._tp1_hit   = True
                            ot._tp1_pips  = (ot.tp1_price - ot.entry_price) / pip
                            ot._trail_stop = ot.entry_price
                else:
                    if h >= ot.stop_price:
                        _close(ot, bar_date, ot.stop_price, "stop", pip, slippage_pips, pip_value, commission_rt)
                        trades.append(ot); open_trades.remove(ot); any_closed = True; continue
                    if l <= ot.tp1_price and tp_mode != "trail":
                        if tp_mode == "full":
                            _close(ot, bar_date, ot.tp1_price, "tp1", pip, slippage_pips, pip_value, commission_rt)
                            trades.append(ot); open_trades.remove(ot); any_closed = True; continue
                        else:
                            ot._tp1_hit   = True
                            ot._tp1_pips  = (ot.entry_price - ot.tp1_price) / pip
                            ot._trail_stop = ot.entry_price

                # Consecutive favorable bars exit (close on bar close)
                if consec_bars_exit > 0:
                    bar_bullish = c > row["Open"]
                    fav = bar_bullish if long else not bar_bullish
                    ot._consec_fav = ot._consec_fav + 1 if fav else 0
                    if ot._consec_fav >= consec_bars_exit:
                        _close(ot, bar_date, c, "consec_exit", pip, slippage_pips, pip_value, commission_rt)
                        trades.append(ot); open_trades.remove(ot); any_closed = True; continue

                # Strong single bar exit (close quality >= threshold on a favorable bar)
                if strong_bar_exit > 0:
                    bar_range = h - l
                    if bar_range > 0:
                        if long and c > row["Open"] and (c - l) / bar_range >= strong_bar_exit:
                            _close(ot, bar_date, c, "strong_bar_exit", pip, slippage_pips, pip_value, commission_rt)
                            trades.append(ot); open_trades.remove(ot); any_closed = True; continue
                        elif not long and c < row["Open"] and (h - c) / bar_range >= strong_bar_exit:
                            _close(ot, bar_date, c, "strong_bar_exit", pip, slippage_pips, pip_value, commission_rt)
                            trades.append(ot); open_trades.remove(ot); any_closed = True; continue

        # ── 2. W/M level entry (mirrors live: signal-bar → pending → next-bar fill) ──

        # On trade close: reset pending but stay on this bar (live also re-evaluates same bar)
        if any_closed and not open_trades:
            session_long = session_short = None

        in_sess = _in_session(bar_date, session_start, session_end)
        if not in_sess:
            if session_long or session_short:
                expirations_total += 1
            session_long = session_short = None
            continue

        if open_trades:
            continue

        # Inner helper — defined once, used for both fill and same-bar close checks
        def _open_limit_trade(direction, order):
            nonlocal fills_total, last_open_bar, trade_id
            fills_total += 1; last_open_bar = i; trade_id += 1
            _e, _s, _t1, _lot = order["entry"], order["stop"], order["tp1"], order["lot"]
            t = Trade(trade_id=trade_id, direction=direction,
                      entry_date=bar_date, entry_price=_e,
                      stop_price=_s, tp1_price=_t1, lot_size=_lot,
                      regime="WM_LEVEL", trend_strength=None, params=params.copy())
            open_trades.append(t)
            if direction == "long":
                if l <= _s:
                    _close(t, bar_date, _s, "stop", pip, slippage_pips, pip_value, commission_rt)
                    trades.append(t); open_trades.remove(t)
                elif h >= _t1:
                    _close(t, bar_date, _t1, "tp1", pip, slippage_pips, pip_value, commission_rt)
                    trades.append(t); open_trades.remove(t)
            else:
                if h >= _s:
                    _close(t, bar_date, _s, "stop", pip, slippage_pips, pip_value, commission_rt)
                    trades.append(t); open_trades.remove(t)
                elif l <= _t1:
                    _close(t, bar_date, _t1, "tp1", pip, slippage_pips, pip_value, commission_rt)
                    trades.append(t); open_trades.remove(t)

        # Fill pending placed on previous bar (no same-bar fill — mirrors live)
        if session_long and l <= session_long["entry"]:
            _open_limit_trade("long", session_long)
            session_long = session_short = None
            continue
        if session_short and h >= session_short["entry"]:
            _open_limit_trade("short", session_short)
            session_long = session_short = None
            continue

        # 1-bar expiry: cancel if not filled (mirrors live pending_order_expiry_bars=1)
        if session_long or session_short:
            expirations_total += 1
            session_long = session_short = None

        # New pending: only if this bar already touched a W/M level (mirrors live check_entry)
        if atr_max_pips > 0 and _atr(df, i, atr_period) / pip > atr_max_pips:
            continue
        price       = row["Open"]
        current_atr = _atr(df, i, atr_period)
        stop_d      = (stop_atr_mult * current_atr) if stop_atr_mult > 0 else (fixed_stop_pips * pip)
        levels      = _wm_levels(weekly_df, monthly_df, bar_date)
        below       = [lv for lv in levels if lv < price - stop_d]
        above       = [lv for lv in levels if lv > price + stop_d]
        if not below or not above:
            continue
        lvl_lo, lvl_hi = max(below), min(above)

        if l <= lvl_lo:
            lot_lo = _lot_size(account_balance, lvl_lo, lvl_lo - stop_d,
                               pip, pip_value, risk_pct, min_stop_pips, max_lot)
            if lot_lo > 0:
                signals_total += 1
                session_long = {"entry": lvl_lo, "stop": round(lvl_lo - stop_d, 5),
                                "tp1": round(lvl_lo + tp_rr * stop_d, 5), "lot": lot_lo}
        elif h >= lvl_hi:
            lot_hi = _lot_size(account_balance, lvl_hi, lvl_hi + stop_d,
                               pip, pip_value, risk_pct, min_stop_pips, max_lot)
            if lot_hi > 0:
                signals_total += 1
                session_short = {"entry": lvl_hi, "stop": round(lvl_hi + stop_d, 5),
                                 "tp1": round(lvl_hi - tp_rr * stop_d, 5), "lot": lot_hi}

    # Close any trades still open at end of data
    for ot in open_trades:
        _close(ot, df.index[-1], df["Close"].iloc[-1], "end_of_data", pip, slippage_pips, pip_value, commission_rt)
        trades.append(ot)

    fill_stats = {
        "signals_total":     signals_total,
        "fills_total":       fills_total,
        "expirations_total": expirations_total,
    }
    return trades, fill_stats


# ── Validation ────────────────────────────────────────────────────────────────

def validate_trades(trades: list[Trade]) -> None:
    print(f"\n=== Total trades: {len(trades)} ===")
    if not trades:
        print("No trades generated.")
        return

    print("\n--- First 10 trades ---")
    fields = [
        "trade_id", "direction", "entry_date", "entry_price", "stop_price",
        "tp1_price", "exit_date", "exit_price", "exit_reason",
        "gross_pips", "net_pips", "gross_r", "net_r",
        "duration_hours", "lot_size", "trend_strength", "regime",
    ]
    for t in trades[:10]:
        print()
        for f in fields:
            print(f"  {f:<22} {getattr(t, f)}")

    print("\n--- Sanity checks ---")

    # No overlapping trades
    for i in range(1, len(trades)):
        prev, curr = trades[i - 1], trades[i]
        if prev.exit_date is None or curr.entry_date < prev.exit_date:
            print(f"  !! OVERLAP: trade {prev.trade_id} and {curr.trade_id}")
            break
    else:
        print("  Overlapping trades: NONE ✓")

    # Session hours on entry
    bad_session = [t for t in trades if not _in_session(t.entry_date)]
    print(f"  Entries outside session: {len(bad_session)} {'✓' if not bad_session else '!!'}")

    # Stop distance positive
    bad_stop = [t for t in trades if abs(t.entry_price - t.stop_price) <= 0]
    print(f"  Zero/negative stop distance: {len(bad_stop)} {'✓' if not bad_stop else '!!'}")

    # Lot size > 0
    bad_lot = [t for t in trades if t.lot_size <= 0]
    print(f"  Zero lot size: {len(bad_lot)} {'✓' if not bad_lot else '!!'}")

    # Exit reason populated
    bad_reason = [t for t in trades if not t.exit_reason]
    print(f"  Missing exit reason: {len(bad_reason)} {'✓' if not bad_reason else '!!'}")

    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    print(f"  Exit reasons: {reasons}")


def plot_trades(df: pd.DataFrame, trades: list[Trade], last_n: int = 500) -> None:
    df_plot = df.tail(last_n)
    cutoff  = df_plot.index[0]
    visible = [t for t in trades if t.entry_date >= cutoff]

    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(df_plot.index, df_plot["Close"], color="black", linewidth=0.7, label="Close")

    for t in visible:
        # Entry arrow
        if t.direction == "long":
            ax.annotate("", xy=(t.entry_date, t.entry_price),
                        xytext=(t.entry_date, t.entry_price - 0.0015),
                        arrowprops=dict(arrowstyle="->", color="green", lw=1.8))
        else:
            ax.annotate("", xy=(t.entry_date, t.entry_price),
                        xytext=(t.entry_date, t.entry_price + 0.0015),
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.8))

        # Exit X
        if t.exit_date is not None and t.exit_price is not None:
            win_color = "green" if (t.net_pips or 0) > 0 else "red"
            ax.plot(t.exit_date, t.exit_price, marker="x",
                    color=win_color, markersize=9, markeredgewidth=2, zorder=6)

    # Legend proxies
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="^", color="green",  label="Long entry",  markersize=9, linestyle="None"),
        Line2D([0], [0], marker="v", color="red",    label="Short entry", markersize=9, linestyle="None"),
        Line2D([0], [0], marker="x", color="green",  label="Exit (win)",  markersize=9, linestyle="None"),
        Line2D([0], [0], marker="x", color="red",    label="Exit (loss)", markersize=9, linestyle="None"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")
    ax.set_title(f"EURUSD 4H — Trades  |  {len(visible)} shown of {len(trades)} total  |  TP_MODE={TP_MODE}")
    ax.set_xlabel("Date"); ax.set_ylabel("Price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45, fontsize=7)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("trades.png", dpi=150)
    print("\nPlot saved → trades.png")
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Fetching data...")
    df = get_data()

    print("Running trade simulation...")
    trades, fill_stats = run_trades(
        df,
        account_balance   = ACCOUNT_BALANCE,
        pullback_lookback = PULLBACK_LOOKBACK,
        stop_buffer       = STOP_BUFFER,
        tp_mode           = TP_MODE,
        slippage_pips     = SLIPPAGE_PIPS,
        pip               = PIP,
        pip_value         = PIP_VALUE,
    )
    print(f"Signals: {fill_stats['signals_total']}  Fills: {fill_stats['fills_total']}  "
          f"Expired: {fill_stats['expirations_total']}  "
          f"Fill rate: {round(fill_stats['fills_total']/fill_stats['signals_total']*100, 1) if fill_stats['signals_total'] else 0}%")

    validate_trades(trades)
    plot_trades(df, trades)
