"""
Structure filter test — GOLD# M5.

Tests whether classify_structure predicts subsequent price direction.

For every session bar where structure is "up" or "down":
  up   -> long  at close, stop = bar low,  TP = entry + (entry - low)   [1:1]
  down -> short at close, stop = bar high, TP = entry - (high - entry)  [1:1]

Each signal bar is simulated independently (overlapping trades allowed).
This is a filter test, not a strategy backtest — no slippage, no commission.

Also tests a baseline (every session bar, no filter) to show what the filter
changes relative to unfiltered.

Tests window sizes 6 / 12 / 18 / 24 bars.

Usage: cd research_xauusd && python run_structure_test.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.loader  import connect, disconnect, fetch_ohlcv
from regime_filter import classify_structure, SESSION_WINDOWS

SYMBOL    = "GOLD#"
TIMEFRAME = "M5"
N_BARS    = 150_000
OUT_DIR   = "results/structure_test"
WINDOWS   = [6, 12, 18, 24]


# ── Simulation ────────────────────────────────────────────────────────────────

def _simulate(df, signals):
    """
    signals: Series of "long" / "short" / NaN per bar.
    Returns list of result dicts.
    """
    closes = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    idx    = df.index
    n      = len(df)
    records = []

    for i in range(n - 1):
        direction = signals.iloc[i]
        if not isinstance(direction, str):
            continue

        entry = closes[i]

        if direction == "long":
            stop = lows[i]
            dist = entry - stop
            if dist <= 0:
                continue
            tp = entry + dist
        else:
            stop = highs[i]
            dist = stop - entry
            if dist <= 0:
                continue
            tp = entry - dist

        outcome   = None
        bars_held = 0
        for j in range(i + 1, n):
            bars_held += 1
            h, l = highs[j], lows[j]
            if direction == "long":
                if l <= stop:
                    outcome = "loss"; break
                if h >= tp:
                    outcome = "win";  break
            else:
                if h >= stop:
                    outcome = "loss"; break
                if l <= tp:
                    outcome = "win";  break

        if outcome is None:
            continue

        hour    = idx[i].hour
        session = next(
            (name for name, (s, e) in SESSION_WINDOWS.items() if s <= hour < e),
            None
        )
        records.append({
            "timestamp": idx[i],
            "session":   session,
            "direction": direction,
            "dist":      round(dist, 4),
            "outcome":   outcome,
            "bars_held": bars_held,
        })

    return records


# ── Reporting ─────────────────────────────────────────────────────────────────

def _row(label, records):
    if not records:
        return f"  {label:<32}  {'—':>5}  {'—':>6}  {'—':>7}"
    n    = len(records)
    wins = sum(1 for r in records if r["outcome"] == "win")
    wr   = wins / n
    exp  = 2 * wr - 1
    return (
        f"  {label:<32}  {n:>5}  {wr:>5.1%}  {exp:>+6.3f}R"
    )


def _print_block(label, records):
    print(f"\n  -- {label} --")
    print(f"  {'group':<32}  {'n':>5}  {'WR':>6}  {'E(R)':>7}")
    print(_row("all", records))
    for d in ["long", "short"]:
        print(_row(d, [r for r in records if r["direction"] == d]))
    for s in ["asia", "london", "ny"]:
        print(_row(s, [r for r in records if r["session"] == s]))
    for s in ["asia", "london", "ny"]:
        for d in ["long", "short"]:
            sub = [r for r in records if r["session"] == s and r["direction"] == d]
            if len(sub) >= 30:
                print(_row(f"{s} {d}", sub))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Connecting to MT5...")
    if not connect():
        sys.exit(1)

    try:
        print(f"Fetching {SYMBOL} {TIMEFRAME} ({N_BARS:,} bars)...")
        df, _ = fetch_ohlcv(SYMBOL, TIMEFRAME, N_BARS, fallbacks=["XAUUSD"])
        print(f"  {len(df):,} bars  {df.index[0].date()} - {df.index[-1].date()}")

        hour = df.index.hour
        in_session = pd.Series(False, index=df.index)
        for start, end in SESSION_WINDOWS.values():
            in_session |= (hour >= start) & (hour < end)

        # ── Baseline: every session bar, no filter ────────────────────────────
        print("\n" + "="*60)
        print("BASELINE (no structure filter, all session bars)")
        print("="*60)

        bl = _simulate(df, pd.Series(
            np.where(in_session, "long",  None), index=df.index, dtype=object))
        bs = _simulate(df, pd.Series(
            np.where(in_session, "short", None), index=df.index, dtype=object))

        print(f"\n  {'group':<32}  {'n':>5}  {'WR':>6}  {'E(R)':>7}")
        print(_row("baseline long  (all session bars)", bl))
        print(_row("baseline short (all session bars)", bs))
        for sess in ["asia", "london", "ny"]:
            print(_row(f"baseline long  {sess}", [r for r in bl if r["session"] == sess]))
            print(_row(f"baseline short {sess}", [r for r in bs if r["session"] == sess]))

        # ── Structure filter tests ────────────────────────────────────────────
        for w in WINDOWS:
            print("\n" + "="*60)
            print(f"STRUCTURE FILTER  window = {w} bars ({w * 5}min)")
            print("="*60)

            struct = classify_structure(df, window=w)

            signals = pd.Series(np.nan, index=df.index, dtype=object)
            signals[(struct == "up")   & in_session] = "long"
            signals[(struct == "down") & in_session] = "short"

            records = _simulate(df, signals)
            _print_block(f"window={w}", records)

            # Save trades for default window
            if w == 12:
                pd.DataFrame(records).to_csv(
                    f"{OUT_DIR}/trades_w{w}.csv", index=False)
                print(f"\n  Saved {len(records)} trades -> {OUT_DIR}/trades_w{w}.csv")

    finally:
        disconnect()
        print("\nDone.")


if __name__ == "__main__":
    main()
