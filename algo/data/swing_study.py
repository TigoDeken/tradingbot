"""
Swing anatomy study for small-cap spot pairs on Bybit.

What we measure:
  - All significant price swings (zigzag, min 8% reversal to confirm)
  - For each swing: duration (bars), size (%), size in ATR units, direction
  - Pre-swing ATR compression: ATR(5)/ATR(10) in bars before swing start
  - Does low ATR compression predict larger/better swings?

Run: python -m algo.data.swing_study
"""

import os
import time
import numpy as np
import pandas as pd
from scipy import stats
from pybit.unified_trading import HTTP

SESSION   = HTTP(testnet=False)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "swing_study")

# Universe filter
MIN_VOL_USD = 500_000
MAX_VOL_USD = 10_000_000
N_COINS     = 15

# Swing detection
SWING_THRESH   = 0.08  # reversal needed to confirm a swing is over
MIN_SWING_PCT  = 0.05  # ignore swings smaller than this

# ATR compression
ATR_SHORT          = 5
ATR_LONG           = 10
COMPRESSION_THRESH = 0.70  # ATR_SHORT/ATR_LONG below this = compressed/quiet


# ── Universe ──────────────────────────────────────────────────────────────────

def get_universe() -> list[str]:
    resp     = SESSION.get_tickers(category="spot")
    tickers  = resp["result"]["list"]
    candidates = []
    for t in tickers:
        if not t["symbol"].endswith("USDT"):
            continue
        try:
            vol = float(t["turnover24h"])
        except (KeyError, ValueError, TypeError):
            continue
        if MIN_VOL_USD <= vol <= MAX_VOL_USD:
            candidates.append((t["symbol"], vol))
    candidates.sort(key=lambda x: x[1])
    rng      = np.random.default_rng(42)
    idxs     = rng.choice(len(candidates), size=min(N_COINS, len(candidates)), replace=False)
    return [candidates[i][0] for i in sorted(idxs)]


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_daily(symbol: str, n_bars: int = 730) -> pd.DataFrame:
    bars, end = [], None
    while len(bars) < n_bars:
        params = dict(category="spot", symbol=symbol, interval="D", limit=200)
        if end:
            params["end"] = end
        raw = SESSION.get_kline(**params)["result"]["list"]
        if not raw:
            break
        bars.extend(raw)
        end = int(raw[-1][0]) - 1
        time.sleep(0.1)
        if len(raw) < 200:
            break
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
    df = df.astype({c: float for c in ["open", "high", "low", "close", "volume", "turnover"]})
    df["ts"]   = df["ts"].astype(int)
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    return df.iloc[-n_bars:].reset_index(drop=True)


# ── Indicators ────────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, n: int) -> pd.Series:
    high, low, prev = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


# ── Swing detection (zigzag) ──────────────────────────────────────────────────

def find_swings(df: pd.DataFrame, threshold: float) -> list[dict]:
    """
    Identifies confirmed swing highs and lows.
    A swing high is confirmed when price drops >= threshold from the peak.
    A swing low is confirmed when price rises >= threshold from the trough.
    Returns list of {idx, price, type} dicts in order.
    """
    closes = df["close"].values
    swings = []
    direction    = 1        # 1 = tracking upward (looking for high), -1 = tracking downward
    extreme_idx  = 0
    extreme_price = closes[0]

    for i in range(1, len(closes)):
        price = closes[i]
        if direction == 1:
            if price > extreme_price:
                extreme_idx, extreme_price = i, price
            elif (extreme_price - price) / extreme_price >= threshold:
                swings.append({"idx": extreme_idx, "price": extreme_price, "type": "high"})
                direction = -1
                extreme_idx, extreme_price = i, price
        else:
            if price < extreme_price:
                extreme_idx, extreme_price = i, price
            elif (price - extreme_price) / extreme_price >= threshold:
                swings.append({"idx": extreme_idx, "price": extreme_price, "type": "low"})
                direction = 1
                extreme_idx, extreme_price = i, price

    return swings


# ── Swing anatomy ─────────────────────────────────────────────────────────────

def analyse_swings(df: pd.DataFrame, swings: list[dict], symbol: str) -> pd.DataFrame:
    if len(swings) < 2:
        return pd.DataFrame()

    atr_s   = compute_atr(df, ATR_SHORT)
    atr_l   = compute_atr(df, ATR_LONG)
    atr_rat = atr_s / atr_l

    records = []
    for i in range(len(swings) - 1):
        s, e = swings[i], swings[i + 1]
        duration = e["idx"] - s["idx"]
        if duration < 2:
            continue

        if s["type"] == "low":
            direction = "up"
            swing_pct = (e["price"] - s["price"]) / s["price"]
        else:
            direction = "down"
            swing_pct = (s["price"] - e["price"]) / s["price"]

        if swing_pct < MIN_SWING_PCT:
            continue

        atr_at_start = atr_l.iloc[s["idx"]]
        atr_size     = abs(e["price"] - s["price"]) / atr_at_start if atr_at_start > 0 else np.nan

        # pre-swing compression: mean ATR ratio in 5 bars before swing start
        pre_start    = max(0, s["idx"] - 5)
        pre_comp     = atr_rat.iloc[pre_start:s["idx"]].mean()

        records.append({
            "symbol":       symbol,
            "direction":    direction,
            "start_date":   df["time"].iloc[s["idx"]],
            "end_date":     df["time"].iloc[e["idx"]],
            "duration_bars": duration,
            "swing_pct":    swing_pct * 100,
            "atr_size":     atr_size,
            "pre_compression": pre_comp,
        })

    return pd.DataFrame(records)


# ── Output ────────────────────────────────────────────────────────────────────

def print_results(df: pd.DataFrame):
    n_coins = df["symbol"].nunique()
    print(f"\n{'='*70}")
    print(f"SWING ANATOMY  —  {len(df)} swings  |  {n_coins} coins  |  "
          f"{df['start_date'].min().date()} to {df['end_date'].max().date()}")
    print(f"Zigzag threshold: {SWING_THRESH*100:.0f}% reversal to confirm a swing")
    print(f"{'='*70}")

    for direction in ["up", "down"]:
        sub = df[df["direction"] == direction]
        print(f"\n  {direction.upper()} SWINGS  (n={len(sub)})")
        print(f"    Duration (bars): "
              f"p25={sub['duration_bars'].quantile(.25):.0f}  "
              f"median={sub['duration_bars'].median():.0f}  "
              f"p75={sub['duration_bars'].quantile(.75):.0f}  "
              f"p90={sub['duration_bars'].quantile(.90):.0f}")
        print(f"    Size (%):        "
              f"p25={sub['swing_pct'].quantile(.25):.1f}%  "
              f"median={sub['swing_pct'].median():.1f}%  "
              f"p75={sub['swing_pct'].quantile(.75):.1f}%  "
              f"p90={sub['swing_pct'].quantile(.90):.1f}%")
        print(f"    Size (ATR):      "
              f"p25={sub['atr_size'].quantile(.25):.1f}x  "
              f"median={sub['atr_size'].median():.1f}x  "
              f"p75={sub['atr_size'].quantile(.75):.1f}x  "
              f"p90={sub['atr_size'].quantile(.90):.1f}x")

    print(f"\n{'='*70}")
    print("ATR COMPRESSION vs SWING QUALITY")
    print(f"(pre_compression = ATR({ATR_SHORT})/ATR({ATR_LONG}) in 5 bars before swing start)")
    print(f"Q1 = most compressed/quiet  |  Q5 = least compressed/noisy")
    print(f"{'='*70}")

    for direction in ["up", "down"]:
        sub = df[df["direction"] == direction].dropna(subset=["pre_compression", "atr_size"])
        if len(sub) < 20:
            continue

        sub = sub.copy()
        sub["comp_q"] = pd.qcut(sub["pre_compression"].rank(method="first"), 5,
                                labels=["Q1", "Q2", "Q3", "Q4", "Q5"])

        print(f"\n  {direction.upper()} swings:")
        header = f"  {'Quintile':8}  {'n':>4}  {'median %':>9}  {'median ATR':>10}  {'median days':>11}"
        print(header)
        for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            g = sub[sub["comp_q"] == q]
            print(f"  {q:8}  {len(g):>4}  "
                  f"{g['swing_pct'].median():>8.1f}%  "
                  f"{g['atr_size'].median():>9.1f}x  "
                  f"{g['duration_bars'].median():>10.0f}d")

        ic, pval = stats.spearmanr(sub["pre_compression"], sub["atr_size"])
        flag = "  << significant" if pval < 0.05 else ""
        print(f"\n  Spearman IC (compression vs ATR size): {ic:+.3f}  p={pval:.3f}{flag}")
        print(f"  Interpretation: negative IC = more compressed -> bigger swing (what we want)")

    # compressed vs uncompressed direct comparison
    print(f"\n{'='*70}")
    print(f"COMPRESSED (ratio < {COMPRESSION_THRESH}) vs NORMAL SETUPS")
    print(f"{'='*70}")
    for direction in ["up", "down"]:
        sub = df[df["direction"] == direction].dropna(subset=["pre_compression", "atr_size"])
        if len(sub) < 10:
            continue
        comp   = sub[sub["pre_compression"] < COMPRESSION_THRESH]
        normal = sub[sub["pre_compression"] >= COMPRESSION_THRESH]
        print(f"\n  {direction.upper()}:")
        print(f"    Compressed  (n={len(comp):3d}):  "
              f"median {comp['swing_pct'].median():.1f}%  "
              f"{comp['atr_size'].median():.1f}x ATR  "
              f"{comp['duration_bars'].median():.0f} days")
        print(f"    Normal      (n={len(normal):3d}):  "
              f"median {normal['swing_pct'].median():.1f}%  "
              f"{normal['atr_size'].median():.1f}x ATR  "
              f"{normal['duration_bars'].median():.0f} days")

    print(f"\n{'='*70}")
    print("WHAT TO LOOK FOR IN OUTPUT")
    print(f"{'='*70}")
    print("""
  1. Duration: how many bars does a typical swing last? This sets carry cost expectations.
  2. Size in ATR: if median swing = 5x ATR and stop = 1.5x ATR, natural R:R = ~3R (target / stop).
  3. Compression table: does Q1 produce bigger swings than Q5?
     - If yes: ATR compression is a valid setup condition.
     - If no edge: compression doesn't predict swing quality, drop it.
  4. Spearman IC: negative = compressed setups → bigger swings. p < 0.05 = not random.
  """)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("Fetching universe ($500K–$10M daily volume)...")
    symbols = get_universe()
    print(f"Selected {len(symbols)} coins: {', '.join(symbols)}\n")

    all_swings = []
    for symbol in symbols:
        cache_file = os.path.join(CACHE_DIR, f"{symbol}_daily.parquet")
        if os.path.exists(cache_file):
            df = pd.read_parquet(cache_file)
            print(f"  {symbol}: cached ({len(df)} bars, "
                  f"{df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()})")
        else:
            print(f"  {symbol}: fetching...", end=" ", flush=True)
            df = fetch_daily(symbol)
            if df.empty:
                print("no data, skip")
                continue
            df.to_parquet(cache_file)
            print(f"{len(df)} bars ({df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()})")

        if len(df) < 60:
            print(f"  {symbol}: too few bars, skip")
            continue

        swings    = find_swings(df, SWING_THRESH)
        swing_df  = analyse_swings(df, swings, symbol)
        if not swing_df.empty:
            all_swings.append(swing_df)
            print(f"  {symbol}: {len(swing_df)} swings")

    if not all_swings:
        print("No swings found.")
    else:
        combined = pd.concat(all_swings, ignore_index=True)
        print_results(combined)
        out = os.path.join(CACHE_DIR, "swing_study_output.txt")
        print(f"\nResults also in: {out}")

    print("\nDone.")
