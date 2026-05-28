"""
N-bar close breakout study.

Signal:  price closes above the highest close of the last N bars (long)
         price closes below the lowest close of the last N bars (short)
Entry:   open of the next bar (no lookahead)
Stop:    lowest low of the N-bar range (long) or highest high (short)
Risk:    entry price minus stop price (in price units)
Exit:    first of: 2R target hit, stop hit, or max_hold bars (time exit)

Tests N in [3, 5, 7, 10].
Also tests whether requiring ATR compression before the signal improves results.

Run: python -m algo.data.breakout_study
"""

import os
import numpy as np
import pandas as pd

CACHE_DIR   = os.path.join(os.path.dirname(__file__), "cache", "swing_study")
N_VALUES    = [3, 5, 7, 10]
TARGET_R    = 2.0   # take profit at 2R
MAX_HOLD    = 15    # bars before time exit
ATR_SHORT   = 5
ATR_LONG    = 10
COMP_THRESH = 0.80  # ATR(5)/ATR(10) below this = compressed setup


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, n: int) -> pd.Series:
    high, low, prev = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def load_coins() -> dict[str, pd.DataFrame]:
    coins = {}
    for fname in os.listdir(CACHE_DIR):
        if not fname.endswith("_daily.parquet"):
            continue
        symbol = fname.replace("_daily.parquet", "")
        df = pd.read_parquet(os.path.join(CACHE_DIR, fname))
        if len(df) < 60:
            continue
        df = df.reset_index(drop=True)
        coins[symbol] = df
    return coins


# ── Signal generation ─────────────────────────────────────────────────────────

def generate_signals(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    For each bar, check if close breaks above N-bar high (long) or below N-bar low (short).
    Lookback is the N bars strictly before the current bar (no lookahead).
    Returns a DataFrame of signals with entry details.
    """
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    opens  = df["open"].values
    atr_s  = compute_atr(df, ATR_SHORT).values
    atr_l  = compute_atr(df, ATR_LONG).values

    signals = []

    for i in range(n + 1, len(df) - 1):
        # N-bar range strictly before current bar
        window_closes = closes[i - n:i]
        window_highs  = highs[i - n:i]
        window_lows   = lows[i - n:i]

        n_bar_high_close = window_closes.max()
        n_bar_low_close  = window_closes.min()
        n_bar_low_wick   = window_lows.min()
        n_bar_high_wick  = window_highs.max()

        current_close  = closes[i]
        entry_price    = opens[i + 1]   # enter at next bar open

        # ATR compression check
        atr_ratio = atr_s[i] / atr_l[i] if atr_l[i] > 0 else np.nan
        compressed = atr_ratio < COMP_THRESH if not np.isnan(atr_ratio) else False

        # Long signal: close breaks above N-bar high close
        if current_close > n_bar_high_close:
            stop  = n_bar_low_wick
            risk  = entry_price - stop
            if risk > 0:
                signals.append({
                    "bar_idx":    i,
                    "entry_bar":  i + 1,
                    "direction":  "long",
                    "entry":      entry_price,
                    "stop":       stop,
                    "risk":       risk,
                    "atr_ratio":  atr_ratio,
                    "compressed": compressed,
                })

        # Short signal: close breaks below N-bar low close
        elif current_close < n_bar_low_close:
            stop  = n_bar_high_wick
            risk  = stop - entry_price
            if risk > 0:
                signals.append({
                    "bar_idx":    i,
                    "entry_bar":  i + 1,
                    "direction":  "short",
                    "entry":      entry_price,
                    "stop":       stop,
                    "risk":       risk,
                    "atr_ratio":  atr_ratio,
                    "compressed": compressed,
                })

    return pd.DataFrame(signals)


# ── Trade simulation ──────────────────────────────────────────────────────────

def simulate_trade(df: pd.DataFrame, sig: dict, max_hold: int, target_r: float) -> dict:
    """
    Simulate one trade bar by bar after entry.
    Returns outcome in R and exit reason.
    """
    entry   = sig["entry"]
    stop    = sig["stop"]
    risk    = sig["risk"]
    dirn    = sig["direction"]
    start   = sig["entry_bar"]
    target  = entry + risk * target_r if dirn == "long" else entry - risk * target_r

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values

    for j in range(start, min(start + max_hold, len(df))):
        h, l, c = highs[j], lows[j], closes[j]

        if dirn == "long":
            if l <= stop:
                return {"outcome_r": -1.0, "exit": "stop", "hold": j - start}
            if h >= target:
                return {"outcome_r": target_r, "exit": "target", "hold": j - start}
        else:
            if h >= stop:
                return {"outcome_r": -1.0, "exit": "stop", "hold": j - start}
            if l <= target:
                return {"outcome_r": target_r, "exit": "target", "hold": j - start}

    # time exit
    last_close = closes[min(start + max_hold - 1, len(df) - 1)]
    if dirn == "long":
        outcome = (last_close - entry) / risk
    else:
        outcome = (entry - last_close) / risk

    return {"outcome_r": outcome, "exit": "time", "hold": max_hold}


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse(all_trades: pd.DataFrame, label: str):
    if all_trades.empty:
        print(f"  {label}: no trades")
        return

    wins    = all_trades[all_trades["outcome_r"] >= TARGET_R]
    losses  = all_trades[all_trades["outcome_r"] <= -1.0]
    time_ex = all_trades[all_trades["exit"] == "time"]

    win_rate = len(wins) / len(all_trades)
    avg_r    = all_trades["outcome_r"].mean()
    exp_val  = avg_r  # already in R units

    print(f"  {label}")
    print(f"    Trades:      {len(all_trades)}  "
          f"(long: {(all_trades['direction']=='long').sum()},  "
          f"short: {(all_trades['direction']=='short').sum()})")
    print(f"    Win rate:    {win_rate*100:.1f}%  "
          f"(target hit: {len(wins)},  stopped: {len(losses)},  time exit: {len(time_ex)})")
    print(f"    Avg outcome: {avg_r:+.2f}R  per trade")
    print(f"    Expected value: {exp_val:+.2f}R  per trade  "
          f"({'positive edge' if exp_val > 0 else 'NO EDGE'})")
    print(f"    Avg hold:    {all_trades['hold'].mean():.1f} bars")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading cached daily data...")
    coins = load_coins()
    print(f"Loaded {len(coins)} coins: {', '.join(coins)}\n")

    for n in N_VALUES:
        print(f"\n{'='*70}")
        print(f"N = {n} bars  |  target = {TARGET_R}R  |  max hold = {MAX_HOLD} bars")
        print(f"  Stop: lowest wick of N-bar range (long) / highest wick (short)")
        print(f"  Compression filter: ATR({ATR_SHORT})/ATR({ATR_LONG}) < {COMP_THRESH}")
        print(f"{'='*70}")

        all_trades    = []
        comp_trades   = []

        for symbol, df in coins.items():
            sigs = generate_signals(df, n)
            if sigs.empty:
                continue

            trades = []
            for _, sig in sigs.iterrows():
                result = simulate_trade(df, sig.to_dict(), MAX_HOLD, TARGET_R)
                trades.append({**sig.to_dict(), **result})

            if trades:
                t = pd.DataFrame(trades)
                all_trades.append(t)
                comp_trades.append(t[t["compressed"]])

        if not all_trades:
            print("  No trades generated.")
            continue

        combined      = pd.concat(all_trades, ignore_index=True)
        compressed    = pd.concat(comp_trades, ignore_index=True) if comp_trades else pd.DataFrame()

        # Signals per coin per year
        avg_sigs_per_year = len(combined) / len(coins) / (
            sum(len(df) for df in coins.values()) / len(coins) / 365
        )

        print(f"\n  All signals (no filter):")
        analyse(combined, f"N={n} all signals  (~{avg_sigs_per_year:.0f} signals/coin/year)")

        print(f"\n  Compressed setups only (ATR ratio < {COMP_THRESH}):")
        if not compressed.empty:
            analyse(compressed, f"N={n} compressed only  ({len(compressed)} trades, "
                                f"{len(compressed)/len(combined)*100:.0f}% of all)")
        else:
            print(f"  No compressed trades found at this threshold.")

        # Long vs short breakdown
        print(f"\n  Direction breakdown (all signals):")
        for dirn in ["long", "short"]:
            sub = combined[combined["direction"] == dirn]
            if len(sub) == 0:
                continue
            wr  = (sub["outcome_r"] >= TARGET_R).mean()
            avg = sub["outcome_r"].mean()
            print(f"    {dirn.upper():5}: n={len(sub):4d}  win={wr*100:.1f}%  avg={avg:+.2f}R")

    print(f"\n{'='*70}")
    print("INTERPRETATION GUIDE")
    print(f"{'='*70}")
    print(f"""
  Expected value > 0 means: on average, each trade makes money.
  Win rate alone means nothing — a 35% win rate with 2R wins and 1R losses
  has expected value of +0.05R per trade, which is profitable.

  What to look for:
    1. Expected value > 0.10R per trade minimum (after costs)
    2. Enough signals per year to matter (at least 20-30 per coin)
    3. Compressed setups should outperform all signals
    4. Long and short should both show edge (otherwise direction-dependent)

  Bybit taker fee is 0.1% per side = 0.2% round trip.
  If average risk per trade is 8%, fee cost = 0.2/8 = 0.025R per trade.
  Subtract this from expected value to get net expected value.
    """)

    print("Done.")
