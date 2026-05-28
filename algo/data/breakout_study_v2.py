"""
Enhanced N-bar close breakout study — all proven filters stacked.

Builds on breakout_study.py (N=5 is the proven best window).
Adds three filters, tested incrementally:
  1. Regime (200-day moving average): long only in bull, short only in bear
  2. ATR compression (5/10 ratio < 0.8): only trade quiet setups
  3. Funding z-score (90-day): long when crowd is calm (z<0), short when elevated (z>0)

Signal:  N=5 close breakout above/below the 5-bar close range
Entry:   open of next bar
Stop:    5-bar wick low (long) or wick high (short)
Exit:    2R target, stop, or 15-bar time exit

Funding data fetched from corresponding perpetual symbol.
Coins without a perpetual run without the funding filter.

Run: python -m algo.data.breakout_study_v2
"""

import os
import time
import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP

SESSION   = HTTP(testnet=False)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "swing_study")

N          = 5
TARGET_R   = 2.0
MAX_HOLD   = 15
ATR_SHORT  = 5
ATR_LONG   = 10
COMP_THRESH = 0.80
MA_WINDOW  = 200    # bars (daily = 200 days)
FZ_WINDOW  = 90     # days for funding z-score rolling window


# ── Data loading ──────────────────────────────────────────────────────────────

def load_coins() -> dict[str, pd.DataFrame]:
    coins = {}
    for fname in os.listdir(CACHE_DIR):
        if not fname.endswith("_daily.parquet"):
            continue
        symbol = fname.replace("_daily.parquet", "")
        df = pd.read_parquet(os.path.join(CACHE_DIR, fname))
        if len(df) < MA_WINDOW + 30:
            continue
        coins[symbol] = df.reset_index(drop=True)
    return coins


def fetch_funding(symbol: str) -> pd.DataFrame:
    """Fetch perpetual funding history. Returns empty DataFrame if no perpetual exists."""
    try:
        bars, end = [], None
        while True:
            p = dict(category="linear", symbol=symbol, limit=200)
            if end:
                p["endTime"] = end
            raw = SESSION.get_funding_rate_history(**p)["result"]["list"]
            if not raw:
                break
            bars.extend(raw)
            end = int(raw[-1]["fundingRateTimestamp"]) - 1
            time.sleep(0.05)
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars)
        df["time"] = pd.to_datetime(df["fundingRateTimestamp"].astype(int), unit="ms", utc=True)
        df["funding_rate"] = df["fundingRate"].astype(float) * 100
        return df[["time", "funding_rate"]].sort_values("time").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def load_funding(symbol: str) -> pd.DataFrame:
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_funding.parquet")
    if os.path.exists(cache_file):
        return pd.read_parquet(cache_file)
    df = fetch_funding(symbol)
    if not df.empty:
        df.to_parquet(cache_file)
    return df


# ── Indicators ────────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def attach_indicators(df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 200-day MA regime
    df["ma200"]   = df["close"].rolling(MA_WINDOW).mean()
    df["bull"]    = df["close"] > df["ma200"]

    # ATR compression
    df["atr_s"]   = compute_atr(df, ATR_SHORT)
    df["atr_l"]   = compute_atr(df, ATR_LONG)
    df["atr_ratio"] = df["atr_s"] / df["atr_l"]
    df["compressed"] = df["atr_ratio"] < COMP_THRESH

    # Funding z-score (from perpetual, daily aggregated)
    df["funding_zscore"] = np.nan
    if not funding_df.empty:
        funding_df = funding_df.copy()
        funding_df["date"] = funding_df["time"].dt.date
        daily_funding = funding_df.groupby("date")["funding_rate"].sum().reset_index()
        daily_funding["date"] = pd.to_datetime(daily_funding["date"], utc=True)
        daily_funding = daily_funding.set_index("date")["funding_rate"]

        df["date"] = df["time"].dt.normalize()
        df["daily_funding"] = df["date"].map(daily_funding)
        roll = df["daily_funding"].rolling(FZ_WINDOW)
        df["funding_zscore"] = (df["daily_funding"] - roll.mean()) / roll.std()

    return df


# ── Signal generation ─────────────────────────────────────────────────────────

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    opens  = df["open"].values

    bull         = df["bull"].values
    compressed   = df["compressed"].values
    funding_z    = df["funding_zscore"].values

    signals = []
    for i in range(N + 1, len(df) - 1):
        if not bull[i - 1]:   # skip if MA not yet established
            pass              # still generate signal, regime tracked separately

        window_closes = closes[i - N:i]
        n_high_close  = window_closes.max()
        n_low_close   = window_closes.min()
        n_low_wick    = lows[i - N:i].min()
        n_high_wick   = highs[i - N:i].max()

        cur_close  = closes[i]
        entry      = opens[i + 1]
        fz         = funding_z[i]
        is_bull    = bool(bull[i])
        is_comp    = bool(compressed[i])
        has_fz     = not np.isnan(fz)

        if cur_close > n_high_close:
            risk = entry - n_low_wick
            if risk > 0:
                signals.append({
                    "bar_idx":   i,
                    "entry_bar": i + 1,
                    "direction": "long",
                    "entry":     entry,
                    "stop":      n_low_wick,
                    "risk":      risk,
                    "bull":      is_bull,
                    "compressed": is_comp,
                    "funding_z": fz,
                    "has_funding": has_fz,
                    # filter flags
                    "f_regime":   is_bull,
                    "f_compress": is_comp,
                    "f_funding":  has_fz and fz < 0,    # calm = good for longs
                })

        elif cur_close < n_low_close:
            risk = n_high_wick - entry
            if risk > 0:
                signals.append({
                    "bar_idx":   i,
                    "entry_bar": i + 1,
                    "direction": "short",
                    "entry":     entry,
                    "stop":      n_high_wick,
                    "risk":      risk,
                    "bull":      is_bull,
                    "compressed": is_comp,
                    "funding_z": fz,
                    "has_funding": has_fz,
                    # filter flags
                    "f_regime":   not is_bull,           # short in bear
                    "f_compress": is_comp,
                    "f_funding":  has_fz and fz > 0,    # elevated = good for shorts
                })

    return pd.DataFrame(signals)


# ── Trade simulation ──────────────────────────────────────────────────────────

def simulate_trade(df: pd.DataFrame, sig: dict) -> dict:
    entry  = sig["entry"]
    stop   = sig["stop"]
    risk   = sig["risk"]
    dirn   = sig["direction"]
    start  = sig["entry_bar"]
    target = entry + risk * TARGET_R if dirn == "long" else entry - risk * TARGET_R

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values

    for j in range(start, min(start + MAX_HOLD, len(df))):
        h, l = highs[j], lows[j]
        if dirn == "long":
            if l <= stop:
                return {"outcome_r": -1.0, "exit": "stop",   "hold": j - start}
            if h >= target:
                return {"outcome_r": TARGET_R, "exit": "target", "hold": j - start}
        else:
            if h >= stop:
                return {"outcome_r": -1.0, "exit": "stop",   "hold": j - start}
            if l <= target:
                return {"outcome_r": TARGET_R, "exit": "target", "hold": j - start}

    last = closes[min(start + MAX_HOLD - 1, len(df) - 1)]
    outcome = (last - entry) / risk if dirn == "long" else (entry - last) / risk
    return {"outcome_r": outcome, "exit": "time", "hold": MAX_HOLD}


# ── Analysis ──────────────────────────────────────────────────────────────────

def print_result(label: str, trades: pd.DataFrame):
    if trades.empty:
        print(f"    {label:45}  no trades")
        return
    n      = len(trades)
    wr     = (trades["outcome_r"] >= TARGET_R).mean()
    avg_r  = trades["outcome_r"].mean()
    stops  = (trades["exit"] == "stop").sum()
    tgts   = (trades["exit"] == "target").sum()
    times  = (trades["exit"] == "time").sum()
    edge   = "EDGE" if avg_r > 0.05 else ("flat" if avg_r > 0 else "no edge")
    print(f"    {label:45}  n={n:4d}  win={wr*100:4.1f}%  "
          f"ev={avg_r:+.3f}R  [{edge}]  "
          f"(stop={stops} tgt={tgts} time={times})")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading coins and fetching funding data...")
    coins = load_coins()

    all_signals = []
    for symbol, df in coins.items():
        funding_df = load_funding(symbol)
        has_f      = not funding_df.empty
        print(f"  {symbol:15} {len(df)} bars  funding: {'yes' if has_f else 'no'}")
        df2   = attach_indicators(df, funding_df)
        sigs  = generate_signals(df2)
        if sigs.empty:
            continue
        trades = []
        for _, sig in sigs.iterrows():
            r = simulate_trade(df2, sig.to_dict())
            trades.append({**sig.to_dict(), **r})
        if trades:
            t = pd.DataFrame(trades)
            t["symbol"] = symbol
            all_signals.append(t)

    if not all_signals:
        print("No signals generated.")
        exit()

    df_all = pd.concat(all_signals, ignore_index=True)
    has_funding_mask = df_all["has_funding"]

    print(f"\n{'='*90}")
    print(f"N={N} BREAKOUT — FILTER STACK RESULTS  |  "
          f"{df_all['symbol'].nunique()} coins  |  "
          f"target={TARGET_R}R  max_hold={MAX_HOLD}d")
    print(f"{'='*90}")
    print(f"\n  {'Filter combination':45}  {'trades':>6}  {'win%':>5}  "
          f"{'ev/trade':>8}  {'edge?':>6}  {'exits'}")
    print(f"  {'-'*88}")

    # 1. Baseline: no filter
    print_result("1. No filter (baseline)", df_all)

    # 2. Regime only
    regime_mask = (
        ((df_all["direction"] == "long")  & df_all["f_regime"]) |
        ((df_all["direction"] == "short") & df_all["f_regime"])
    )
    print_result("2. Regime only (long=bull, short=bear)", df_all[regime_mask])

    # 3. Regime + compression
    rc_mask = regime_mask & df_all["f_compress"]
    print_result("3. Regime + compression", df_all[rc_mask])

    # 4. Regime + funding z-score (no compression, for comparison)
    rf_mask = regime_mask & df_all["f_funding"]
    has_rf  = df_all[regime_mask & has_funding_mask]
    print_result("4. Regime + funding z-score", df_all[rf_mask])

    # 5. All three: regime + compression + funding z-score
    rcf_mask = regime_mask & df_all["f_compress"] & df_all["f_funding"]
    print_result("5. Regime + compression + funding z-score", df_all[rcf_mask])

    # Subset to coins WITH funding data for fair comparison
    print(f"\n  -- Same filters, coins WITH funding data only ({has_funding_mask.sum()} signals) --")
    sub = df_all[has_funding_mask]
    regime_sub   = (
        ((sub["direction"] == "long")  & sub["f_regime"]) |
        ((sub["direction"] == "short") & sub["f_regime"])
    )
    print_result("2b. Regime only (funding coins)", sub[regime_sub])
    print_result("3b. Regime + compression (funding coins)", sub[regime_sub & sub["f_compress"]])
    print_result("4b. Regime + funding z-score", sub[regime_sub & sub["f_funding"]])
    print_result("5b. All three (funding coins)", sub[regime_sub & sub["f_compress"] & sub["f_funding"]])

    # Direction breakdown on best filter
    print(f"\n{'='*90}")
    print("DIRECTION BREAKDOWN — best filter combo (regime + compression + funding)")
    print(f"{'='*90}")
    for dirn in ["long", "short"]:
        sub_d = df_all[(df_all["direction"] == dirn) & rcf_mask]
        if sub_d.empty:
            print(f"  {dirn.upper()}: no trades")
            continue
        wr  = (sub_d["outcome_r"] >= TARGET_R).mean()
        avg = sub_d["outcome_r"].mean()
        print(f"  {dirn.upper():5}: n={len(sub_d):3d}  win={wr*100:.1f}%  ev={avg:+.3f}R")

    # Signals per coin per year for best filter
    print(f"\n{'='*90}")
    print("SIGNAL FREQUENCY — regime + compression + funding (per coin per year)")
    print(f"{'='*90}")
    best = df_all[rcf_mask]
    if not best.empty:
        years_per_coin = {
            sym: (df.iloc[-1]["time"] - df.iloc[0]["time"]).days / 365
            for sym, df in coins.items()
        }
        for sym in sorted(best["symbol"].unique()):
            n_trades = len(best[best["symbol"] == sym])
            yrs      = years_per_coin.get(sym, 1)
            per_yr   = n_trades / yrs if yrs > 0 else 0
            ev_sym   = best[best["symbol"] == sym]["outcome_r"].mean()
            print(f"  {sym:15} {n_trades:3d} trades  {per_yr:.1f}/yr  ev={ev_sym:+.3f}R")

    print(f"\n{'='*90}")
    print("INTERPRETATION")
    print(f"{'='*90}")
    print(f"""
  Each filter should improve expected value (ev) vs the baseline.
  If regime alone doesn't help: the signal has no directional awareness.
  If compression helps: quiet setups are genuinely better (we proved this separately).
  If funding z-score helps: the crowd signal adds real directional edge.
  If ev >= +0.10R after all filters: worth building into a full backtest.
  Fee cost approx 0.025R per trade — subtract from ev for net expected value.
    """)

    print("Done.")
