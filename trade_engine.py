import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dataclasses import dataclass, field
from typing import Optional

from data_pipeline import get_data
from swing_engine import build_swings, SWING_N, MIN_SWING_SIZE, MIN_SWING_INCREMENT, PIP
from trend_engine import classify_trend, MIN_TREND_SIZE, TREND_RANGE_RATIO

# ── Parameters ────────────────────────────────────────────────────────────────
from constants import PIP_VALUE, RISK_PCT  # noqa: F401 (PIP_VALUE re-exported)

PULLBACK_LOOKBACK = 4
STOP_BUFFER       = 5        # pips
TP_MODE           = "full"   # "full" | "partial"
SLIPPAGE_PIPS     = 3
ACCOUNT_BALANCE   = 10_000.0


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

    # Internal partial-mode state (not part of output)
    _tp1_hit:            bool            = field(default=False,  repr=False)
    _tp1_pips:           float           = field(default=0.0,    repr=False)
    _trail_stop:         Optional[float] = field(default=None,   repr=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _in_session(ts: pd.Timestamp, session_start: int = 7, session_end: int = 20) -> bool:
    """True if ts (UTC) falls within the configured session window."""
    return session_start <= ts.hour < session_end


def _confirmed_at(swings: pd.DataFrame, bar_i: int, swing_n: int) -> pd.DataFrame:
    """All swings confirmed by bar_i (need swing_n bars after the swing candle)."""
    return swings[swings["bar"] <= bar_i - swing_n]


def _pullback_depths(trend_sw: pd.DataFrame, direction: str, pip: float) -> list[float]:
    """Extract pip distances of each high→low (long) or low→high (short) pair."""
    depths: list[float] = []
    if direction == "long":
        last_h = None
        for _, r in trend_sw.iterrows():
            if r["type"] == "high":
                last_h = r["price"]
            elif r["type"] == "low" and last_h is not None:
                depths.append((last_h - r["price"]) / pip)
                last_h = None
    else:
        last_l = None
        for _, r in trend_sw.iterrows():
            if r["type"] == "low":
                last_l = r["price"]
            elif r["type"] == "high" and last_l is not None:
                depths.append((r["price"] - last_l) / pip)
                last_l = None
    return depths


def _compute_setup(
    confirmed:  pd.DataFrame,
    direction:  str,
    df_window:  pd.DataFrame,   # last 20 bars for fallback range
    lookback:   int,
    stop_buf:   float,          # pips
    pip:        float,
) -> Optional[tuple[float, float, float]]:
    """
    Return (entry, stop, tp1) using median pullback depth, or None if setup
    is invalid (not enough swings, stop crosses entry, etc.).
    """
    highs = confirmed[confirmed["type"] == "high"]
    lows  = confirmed[confirmed["type"] == "low"]
    if len(highs) < 3 or len(lows) < 3:
        return None

    # Trend-defining swings: last 3 highs + last 3 lows
    h3 = highs.tail(3)
    l3 = lows.tail(3)
    oldest_date = min(h3.iloc[0]["date"], l3.iloc[0]["date"])
    trend_sw    = confirmed[confirmed["date"] >= oldest_date]

    depths = _pullback_depths(trend_sw, direction, pip)
    if len(depths) >= 2:
        pb_pips = float(np.median(depths[-lookback:]))
    else:
        # Fallback: median candle range of last 20 bars
        pb_pips = float(np.median((df_window["High"] - df_window["Low"]) / pip))

    if direction == "long":
        swing_h = highs.iloc[-1]["price"]
        swing_l = lows.iloc[-1]["price"]
        entry   = swing_h - pb_pips * pip
        stop    = swing_l - stop_buf * pip
        if entry <= stop:
            return None
        tp1 = entry + 2.0 * (entry - stop)
    else:
        swing_l = lows.iloc[-1]["price"]
        swing_h = highs.iloc[-1]["price"]
        entry   = swing_l + pb_pips * pip
        stop    = swing_h + stop_buf * pip
        if entry >= stop:
            return None
        tp1 = entry - 2.0 * (stop - entry)

    return round(entry, 5), round(stop, 5), round(tp1, 5)


def _lot_size(balance: float, entry: float, stop: float, pip: float, pip_value: float,
              risk_pct: float = RISK_PCT) -> float:
    risk      = balance * risk_pct
    stop_pips = abs(entry - stop) / pip
    if stop_pips == 0:
        return 0.0
    return math.floor(risk / (stop_pips * pip_value) * 100) / 100


def _close(
    trade:     Trade,
    exit_date: pd.Timestamp,
    exit_px:   float,
    reason:    str,
    pip:       float,
    slippage:  float,
) -> None:
    sign = 1 if trade.direction == "long" else -1

    if trade._tp1_hit:
        # Partial mode: weight 50 % TP1 + 50 % trail exit
        trail_pips = sign * (exit_px - trade.entry_price) / pip
        gross = 0.5 * trade._tp1_pips + 0.5 * trail_pips
    else:
        gross = sign * (exit_px - trade.entry_price) / pip

    net       = gross - slippage     # slippage always costs (reduces wins, magnifies losses)
    risk_pips = abs(trade.entry_price - trade.stop_price) / pip

    trade.exit_date     = exit_date
    trade.exit_price    = round(exit_px, 5)
    trade.exit_reason   = reason
    trade.gross_pips    = round(gross, 1)
    trade.net_pips      = round(net,   1)
    trade.gross_r       = round(gross / risk_pips, 2) if risk_pips else 0.0
    trade.net_r         = round(net   / risk_pips, 2) if risk_pips else 0.0
    trade.duration_hours = round((exit_date - trade.entry_date).total_seconds() / 3600, 1)


# ── Simulation ────────────────────────────────────────────────────────────────

def run_trades(
    df:               pd.DataFrame,
    swings:           pd.DataFrame,
    account_balance:  float = ACCOUNT_BALANCE,
    swing_n:          int   = SWING_N,
    pullback_lookback:int   = PULLBACK_LOOKBACK,
    stop_buffer:      float = STOP_BUFFER,
    tp_mode:          str   = TP_MODE,
    slippage_pips:    float = SLIPPAGE_PIPS,
    pip:              float = PIP,
    pip_value:        float = PIP_VALUE,
    risk_pct:         float = RISK_PCT,
    session_start:    int   = 7,
    session_end:      int   = 20,
) -> list[Trade]:
    """
    Iterate bar-by-bar and simulate limit-order entries with stop/TP/trail exits.
    Returns a list of completed Trade objects.
    """
    trades:     list[Trade]       = []
    open_trade: Optional[Trade]   = None
    trade_id    = 0

    params = dict(
        swing_n=swing_n, pullback_lookback=pullback_lookback,
        stop_buffer=stop_buffer, tp_mode=tp_mode,
        slippage_pips=slippage_pips,
    )

    for i in range(len(df)):
        row      = df.iloc[i]
        bar_date = df.index[i]
        h, l, c  = row["High"], row["Low"], row["Close"]

        # ── 1. Manage open trade ─────────────────────────────────────────────
        if open_trade is not None:
            long = open_trade.direction == "long"

            if open_trade._tp1_hit:
                # Partial trail mode: update trailing stop, exit on close
                conf = _confirmed_at(swings, i, swing_n)
                if long:
                    new_lows = conf[conf["type"] == "low"]
                    if not new_lows.empty:
                        latest = new_lows.iloc[-1]["price"]
                        if open_trade._trail_stop is None or latest > open_trade._trail_stop:
                            open_trade._trail_stop = latest
                    ts = open_trade._trail_stop or open_trade.entry_price
                    if c < ts:
                        _close(open_trade, bar_date, c, "trail_exit", pip, slippage_pips)
                        trades.append(open_trade); open_trade = None; continue
                else:
                    new_highs = conf[conf["type"] == "high"]
                    if not new_highs.empty:
                        latest = new_highs.iloc[-1]["price"]
                        if open_trade._trail_stop is None or latest < open_trade._trail_stop:
                            open_trade._trail_stop = latest
                    ts = open_trade._trail_stop or open_trade.entry_price
                    if c > ts:
                        _close(open_trade, bar_date, c, "trail_exit", pip, slippage_pips)
                        trades.append(open_trade); open_trade = None; continue

            else:
                # Normal mode: stop takes priority over TP within same bar
                if long:
                    if l <= open_trade.stop_price:
                        _close(open_trade, bar_date, open_trade.stop_price, "stop", pip, slippage_pips)
                        trades.append(open_trade); open_trade = None; continue
                    if h >= open_trade.tp1_price:
                        if tp_mode == "full":
                            _close(open_trade, bar_date, open_trade.tp1_price, "tp1", pip, slippage_pips)
                            trades.append(open_trade); open_trade = None; continue
                        else:
                            open_trade._tp1_hit   = True
                            open_trade._tp1_pips  = (open_trade.tp1_price - open_trade.entry_price) / pip
                            open_trade._trail_stop = open_trade.entry_price
                else:
                    if h >= open_trade.stop_price:
                        _close(open_trade, bar_date, open_trade.stop_price, "stop", pip, slippage_pips)
                        trades.append(open_trade); open_trade = None; continue
                    if l <= open_trade.tp1_price:
                        if tp_mode == "full":
                            _close(open_trade, bar_date, open_trade.tp1_price, "tp1", pip, slippage_pips)
                            trades.append(open_trade); open_trade = None; continue
                        else:
                            open_trade._tp1_hit   = True
                            open_trade._tp1_pips  = (open_trade.entry_price - open_trade.tp1_price) / pip
                            open_trade._trail_stop = open_trade.entry_price

        # ── 2. Look for entry ────────────────────────────────────────────────
        if open_trade is not None:
            continue

        regime = row.get("regime", "AMBIGUOUS")
        if regime not in ("UPTREND", "DOWNTREND") or not _in_session(bar_date, session_start, session_end):
            continue

        direction = "long" if regime == "UPTREND" else "short"
        conf      = _confirmed_at(swings, i, swing_n)
        df_window = df.iloc[max(0, i - 19):i + 1]

        setup = _compute_setup(conf, direction, df_window, pullback_lookback, stop_buffer, pip)
        if setup is None:
            continue

        entry, stop, tp1 = setup
        lot = _lot_size(account_balance, entry, stop, pip, pip_value, risk_pct)
        if lot <= 0:
            continue

        # Limit order fills when bar reaches entry price
        filled = (direction == "long" and l <= entry) or \
                 (direction == "short" and h >= entry)
        if not filled:
            continue

        trade_id += 1
        open_trade = Trade(
            trade_id        = trade_id,
            direction       = direction,
            entry_date      = bar_date,
            entry_price     = entry,
            stop_price      = stop,
            tp1_price       = tp1,
            lot_size        = lot,
            regime          = regime,
            trend_strength  = row.get("trend_strength_score"),
            params          = params.copy(),
        )

        # Same-bar exit check (stop beats TP if both reachable)
        if direction == "long":
            if l <= stop:
                _close(open_trade, bar_date, stop, "stop", pip, slippage_pips)
                trades.append(open_trade); open_trade = None
            elif h >= tp1:
                if tp_mode == "full":
                    _close(open_trade, bar_date, tp1, "tp1", pip, slippage_pips)
                    trades.append(open_trade); open_trade = None
                else:
                    open_trade._tp1_hit   = True
                    open_trade._tp1_pips  = (tp1 - entry) / pip
                    open_trade._trail_stop = entry
        else:
            if h >= stop:
                _close(open_trade, bar_date, stop, "stop", pip, slippage_pips)
                trades.append(open_trade); open_trade = None
            elif l <= tp1:
                if tp_mode == "full":
                    _close(open_trade, bar_date, tp1, "tp1", pip, slippage_pips)
                    trades.append(open_trade); open_trade = None
                else:
                    open_trade._tp1_hit   = True
                    open_trade._tp1_pips  = (entry - tp1) / pip
                    open_trade._trail_stop = entry

    # Close any trade still open at end of data
    if open_trade is not None:
        _close(open_trade, df.index[-1], df["Close"].iloc[-1], "end_of_data", pip, slippage_pips)
        trades.append(open_trade)

    return trades


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

    print("Detecting swings...")
    swings = build_swings(df, swing_n=SWING_N, min_swing_size=MIN_SWING_SIZE,
                          min_swing_increment=MIN_SWING_INCREMENT, pip=PIP)

    print("Classifying regimes...")
    df = classify_trend(df, swings, swing_n=SWING_N, min_swing_increment=MIN_SWING_INCREMENT,
                        min_trend_size=MIN_TREND_SIZE, trend_range_ratio=TREND_RANGE_RATIO, pip=PIP)

    print("Running trade simulation...")
    trades = run_trades(
        df, swings,
        account_balance   = ACCOUNT_BALANCE,
        swing_n           = SWING_N,
        pullback_lookback = PULLBACK_LOOKBACK,
        stop_buffer       = STOP_BUFFER,
        tp_mode           = TP_MODE,
        slippage_pips     = SLIPPAGE_PIPS,
        pip               = PIP,
        pip_value         = PIP_VALUE,
    )

    validate_trades(trades)
    plot_trades(df, trades)
