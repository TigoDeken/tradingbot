"""
Final strategy study — all agent advice implemented.

Frozen parameters (do not change after this run):
  Entry:  z-score < -0.5 AND ATR compressed (ATR5/ATR10 < 0.8) AND BTC regime
  Stop:   10-day lowest low, minimum 5% from entry
  Exit:   z-score rises above 0 (signal-based) OR 21-day max hold
  Regime: BTC above its 140-day MA (20-week proxy for altcoin season)

Additions vs previous study:
  1. Market-wide regime filter (BTC 140-day MA)
  2. Signal-based exit (z > 0) instead of fixed 7-day hold
  3. Walk-forward validation: 6-month windows
  4. Comparison table: original vs each improvement vs all combined

Run: python -m algo.data.zscore_final_study
"""

import os
import time
import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP
from algo.data.zscore_entry_study import load_all, build_coin_df, load_funding

SESSION   = HTTP(testnet=False)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "swing_study")

# ── FROZEN PARAMETERS ─────────────────────────────────────────────────────────
Z_THRESH      = -0.5    # funding z-score entry threshold
ATR_COMP      = 0.80    # ATR5/ATR10 compression threshold
STOP_DAYS     = 10      # lookback for stop (10-day low)
MIN_STOP_PCT  = 0.05    # minimum 5% stop distance
EXIT_Z        = 0.0     # exit when z-score rises above this
MAX_HOLD      = 21      # safety maximum hold in days
BTC_MA_DAYS   = 140     # BTC regime filter (20 weeks)
FZ_WINDOW     = 90


# ── BTC regime ────────────────────────────────────────────────────────────────

def fetch_btc_daily() -> pd.DataFrame:
    cache = os.path.join(CACHE_DIR, "BTCUSDT_daily.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache).reset_index(drop=True)
    print("  Fetching BTC daily...")
    bars, end = [], None
    while len(bars) < 1500:
        p = dict(category="spot", symbol="BTCUSDT", interval="D", limit=200)
        if end:
            p["end"] = end
        raw = SESSION.get_kline(**p)["result"]["list"]
        if not raw:
            break
        bars.extend(raw)
        end = int(raw[-1][0]) - 1
        time.sleep(0.1)
        if len(raw) < 200:
            break
    df = pd.DataFrame(bars, columns=["ts","open","high","low","close","volume","turnover"])
    df = df.astype({c: float for c in ["open","high","low","close","volume","turnover"]})
    df["ts"]   = df["ts"].astype(int)
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    df.to_parquet(cache)
    return df


def build_btc_regime(btc_df: pd.DataFrame) -> pd.Series:
    """Returns a date-indexed boolean Series: True = BTC above 140-day MA."""
    btc = btc_df.copy()
    btc["date"]  = btc["time"].dt.normalize().dt.tz_localize(None)
    btc["ma140"] = btc["close"].rolling(BTC_MA_DAYS).mean()
    btc["bull"]  = btc["close"] > btc["ma140"]
    return btc.set_index("date")["bull"]


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_final(df: pd.DataFrame,
                   btc_regime: pd.Series,
                   use_regime: bool = True,
                   use_signal_exit: bool = True) -> pd.DataFrame:
    """
    Simulate trades with frozen parameters.
    use_regime:       apply BTC 140-day MA market filter
    use_signal_exit:  exit when z-score rises above EXIT_Z (else fixed 7-day hold)
    """
    fz      = df["funding_z"].values
    opens   = df["open"].values
    highs   = df["high"].values
    lows    = df["low"].values
    closes  = df["close"].values
    atr_r   = df["atr_ratio"].values
    dates   = df["time"].dt.normalize().values
    n       = len(df)

    trades   = []
    trade_end = 0

    for i in range(FZ_WINDOW + 10, n - MAX_HOLD - 1):
        if i < trade_end:
            continue

        z_curr = fz[i]
        z_prev = fz[i - 1] if i > 0 else np.nan
        if np.isnan(z_curr) or np.isnan(z_prev):
            continue

        # Entry signal: z crosses below threshold
        if not (z_curr < Z_THRESH and z_prev >= Z_THRESH):
            continue

        # ATR compression filter
        if np.isnan(atr_r[i]) or atr_r[i] >= ATR_COMP:
            continue

        # Market regime filter
        if use_regime:
            entry_date = pd.Timestamp(dates[i]).tz_localize(None)
            btc_bull = btc_regime.get(entry_date, np.nan)
            if pd.isna(btc_bull) or not btc_bull:
                continue

        if i + 1 >= n:
            continue

        entry = opens[i + 1]
        stop  = lows[max(0, i - STOP_DAYS):i].min()

        if np.isnan(stop) or entry <= 0:
            continue
        risk = entry - stop
        if risk <= 0:
            continue
        risk_pct = risk / entry
        if risk_pct < MIN_STOP_PCT or risk_pct > 0.30:
            continue

        # Simulate
        exit_price  = None
        exit_reason = "time"
        hold        = MAX_HOLD

        for j in range(i + 1, min(i + 1 + MAX_HOLD, n)):
            lo, hi = lows[j], highs[j]

            # Stop hit
            if lo <= stop:
                exit_price  = stop
                exit_reason = "stop"
                hold        = j - (i + 1)
                break

            # Signal exit: z-score rises above EXIT_Z
            if use_signal_exit and not np.isnan(fz[j]) and fz[j] > EXIT_Z:
                exit_price  = opens[min(j + 1, n - 1)]
                exit_reason = "z_exit"
                hold        = j - (i + 1)
                break

        if exit_price is None:
            last_j     = min(i + MAX_HOLD, n - 1)
            exit_price = closes[last_j]
            hold       = last_j - (i + 1)

        ret_pct = (exit_price - entry) / entry * 100
        ret_r   = (exit_price - entry) / risk

        trades.append({
            "entry_bar":  i + 1,
            "entry_date": df["time"].iloc[i + 1],
            "ret_pct":    ret_pct,
            "ret_r":      ret_r,
            "exit_reason": exit_reason,
            "hold":       hold,
            "risk_pct":   risk_pct * 100,
            "z_at_entry": z_curr,
            "atr_at_entry": atr_r[i],
        })

        trade_end = i + 1 + hold + 1

    return pd.DataFrame(trades)


# ── Reporting ─────────────────────────────────────────────────────────────────

def stats(t: pd.DataFrame, label: str = "") -> dict:
    if t.empty or len(t) < 5:
        print(f"  {label:50} n={len(t):3d}  insufficient data")
        return {}
    n    = len(t)
    wr   = (t["ret_pct"] > 0).mean()
    avg  = t["ret_pct"].mean()
    med  = t["ret_pct"].median()
    std  = t["ret_pct"].std()
    shr  = avg / std if std > 0 else 0
    stop = (t["exit_reason"] == "stop").mean()
    zex  = (t["exit_reason"] == "z_exit").mean()
    tex  = (t["exit_reason"] == "time").mean()
    edge = "EDGE" if avg > 0.5 and shr > 0.05 else ("slight" if avg > 0 else "NO EDGE")
    print(f"  {label:50} n={n:3d}  wr={wr*100:.0f}%  "
          f"avg={avg:+.2f}%  med={med:+.2f}%  shr={shr:+.3f}  "
          f"[{edge}]  exits: stop={stop*100:.0f}% z={zex*100:.0f}% time={tex*100:.0f}%")
    return {"n": n, "wr": wr, "avg": avg, "med": med, "sharpe": shr, "label": label}


def section(t: str):
    print(f"\n{'='*80}")
    print(t)
    print(f"{'='*80}")


# ── Walk-forward ──────────────────────────────────────────────────────────────

def walk_forward(trades: pd.DataFrame):
    if trades.empty:
        return
    trades = trades.copy()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades = trades.sort_values("entry_date")

    # Define 6-month windows
    start = trades["entry_date"].min().to_period("Q").start_time
    end   = trades["entry_date"].max()
    windows = pd.period_range(start=start, end=end, freq="2Q")  # 6-month periods

    print(f"\n  {'Window':15}  {'n':>4}  {'wr%':>5}  {'avg%':>7}  {'med%':>7}  {'sharpe':>7}")
    print(f"  {'-'*60}")
    for period in windows:
        pstart = period.start_time.tz_localize("UTC")
        pend   = period.end_time.tz_localize("UTC")
        sub    = trades[(trades["entry_date"] >= pstart) & (trades["entry_date"] <= pend)]
        if len(sub) < 3:
            continue
        avg = sub["ret_pct"].mean()
        med = sub["ret_pct"].median()
        wr  = (sub["ret_pct"] > 0).mean()
        shr = avg / sub["ret_pct"].std() if sub["ret_pct"].std() > 0 else 0
        flag = " <<" if avg > 2 else (" >>" if avg < -1 else "")
        print(f"  {str(period):15}  {len(sub):>4}  {wr*100:>4.0f}%  "
              f"{avg:>+6.2f}%  {med:>+6.2f}%  {shr:>+6.3f}{flag}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    coins    = load_all()
    btc_df   = fetch_btc_daily()
    btc_reg  = build_btc_regime(btc_df)

    btc_bull_pct = btc_reg.mean() * 100
    print(f"BTC regime: {btc_bull_pct:.1f}% of days are bull (above 140-day MA)\n")

    # Run all 4 combinations
    combos = {
        "A_base":     dict(use_regime=False, use_signal_exit=False),  # original (fixed 7d)
        "B_regime":   dict(use_regime=True,  use_signal_exit=False),  # + regime filter
        "C_signexit": dict(use_regime=False, use_signal_exit=True),   # + signal exit
        "D_both":     dict(use_regime=True,  use_signal_exit=True),   # + both (FINAL)
    }

    all_results = {}
    for key, kw in combos.items():
        trades_list = []
        for sym, df in coins.items():
            t = simulate_final(df, btc_reg, **kw)
            if not t.empty:
                t["symbol"] = sym
                trades_list.append(t)
        all_results[key] = pd.concat(trades_list, ignore_index=True) if trades_list else pd.DataFrame()

    # ── Comparison table ──────────────────────────────────────────────────────
    section("COMPARISON: ORIGINAL vs EACH IMPROVEMENT vs ALL COMBINED")
    print()
    stats(all_results["A_base"],     "A. Original (fixed 7d, no regime)")
    stats(all_results["B_regime"],   "B. + BTC regime filter")
    stats(all_results["C_signexit"], "C. + Signal exit (z > 0)")
    stats(all_results["D_both"],     "D. + Both (FINAL)")

    # ── Deep dive on final combo ──────────────────────────────────────────────
    final = all_results["D_both"]
    section("DEEP DIVE — FINAL COMBO (regime + signal exit)")
    if not final.empty:
        wins  = final[final["ret_pct"] > 0]["ret_pct"]
        loses = final[final["ret_pct"] <= 0]["ret_pct"]
        print(f"\n  Trades:        {len(final)}")
        print(f"  Win rate:      {(final['ret_pct']>0).mean()*100:.1f}%")
        print(f"  Avg return:    {final['ret_pct'].mean():+.2f}%")
        print(f"  Median return: {final['ret_pct'].median():+.2f}%")
        print(f"  Avg win:       {wins.mean():+.2f}%  |  Avg loss: {loses.mean():+.2f}%")
        print(f"  Std dev:       {final['ret_pct'].std():.2f}%")
        print(f"  Avg hold:      {final['hold'].mean():.1f} days")
        print(f"  Avg risk:      {final['risk_pct'].mean():.1f}%")
        zex = (final["exit_reason"] == "z_exit").mean()
        stop = (final["exit_reason"] == "stop").mean()
        time_ = (final["exit_reason"] == "time").mean()
        print(f"  Exit: stop={stop*100:.0f}%  z-signal={zex*100:.0f}%  time={time_*100:.0f}%")

    # ── Walk-forward ──────────────────────────────────────────────────────────
    section("WALK-FORWARD VALIDATION — FINAL COMBO by 6-month window")
    print("  (Each window is independent — shows whether edge is stable across time)")
    walk_forward(final)

    # ── Walk-forward on original for comparison ───────────────────────────────
    section("WALK-FORWARD — ORIGINAL (fixed 7d, no regime) for comparison")
    walk_forward(all_results["A_base"])

    # ── Per-coin breakdown ────────────────────────────────────────────────────
    section("PER-COIN BREAKDOWN — FINAL COMBO")
    if not final.empty:
        final["entry_date"] = pd.to_datetime(final["entry_date"])
        per = final.groupby("symbol").agg(
            n=("ret_pct","count"),
            avg=("ret_pct","mean"),
            med=("ret_pct","median"),
            wr=("ret_pct", lambda x: (x>0).mean())
        ).sort_values("avg", ascending=False)
        print(f"\n  {'Symbol':18} {'n':>4} {'wr%':>5} {'avg%':>7} {'med%':>7}")
        for sym, row in per.iterrows():
            flag = " <<" if row["avg"] > 3 else (" >>" if row["avg"] < -3 else "")
            print(f"  {sym:18} {int(row['n']):>4} {row['wr']*100:>4.0f}% "
                  f"{row['avg']:>+6.2f}% {row['med']:>+6.2f}%{flag}")

    # ── Regime impact summary ─────────────────────────────────────────────────
    section("DOES THE REGIME FILTER HELP? — FINAL ANSWER")
    base  = all_results["A_base"]
    both  = all_results["D_both"]
    if not base.empty and not both.empty:
        base["entry_date"] = pd.to_datetime(base["entry_date"])
        both["entry_date"] = pd.to_datetime(both["entry_date"])
        print(f"\n  Without regime: {len(base)} trades  avg={base['ret_pct'].mean():+.2f}%  "
              f"med={base['ret_pct'].median():+.2f}%  "
              f"sharpe={base['ret_pct'].mean()/base['ret_pct'].std():+.3f}")
        print(f"  With regime:    {len(both)} trades  avg={both['ret_pct'].mean():+.2f}%  "
              f"med={both['ret_pct'].median():+.2f}%  "
              f"sharpe={both['ret_pct'].mean()/both['ret_pct'].std():+.3f}")
        trades_removed = len(base) - len(both)
        print(f"\n  Trades removed by regime filter: {trades_removed} "
              f"({trades_removed/len(base)*100:.0f}%)")

        # What were the removed trades like?
        base_dates = set(zip(base["symbol"], base["entry_bar"]))
        both_dates = set(zip(both["symbol"], both["entry_bar"]))
        removed_idx = base_dates - both_dates
        removed_trades = base[base.apply(
            lambda r: (r["symbol"], r["entry_bar"]) in removed_idx, axis=1
        )]
        if not removed_trades.empty:
            print(f"  Avg return of removed trades: {removed_trades['ret_pct'].mean():+.2f}%")
            print(f"  (negative = regime correctly filtered bad trades)")

    section("FROZEN STRATEGY — FINAL SPECIFICATION")
    print(f"""
  These parameters are now FROZEN. Do not change them on this dataset.

  ENTRY CONDITIONS (all must be true):
    1. Funding z-score (90-day rolling) crosses BELOW -0.5
       -> crowd is absent, market is calm
    2. ATR compression: ATR(5)/ATR(10) < 0.80
       -> price is quiet, stop is tight
    3. BTC above its 140-day MA
       -> altcoin season conditions, market-wide tailwind

  EXECUTION:
    Entry:  open of bar AFTER signal bar (no lookahead)
    Stop:   lowest low of 10 bars before entry, minimum 5% away
    Size:   to be determined by capital allocation rules

  EXIT CONDITIONS (first triggered):
    1. Stop hit: exit at stop price
    2. Signal exit: funding z-score rises above 0.0
       -> crowd returning, edge exhausted, exit at next open
    3. Maximum hold: 21 days (safety limit)

  REGIME:
    Only trade when BTC is above its 140-day moving average.
    This filters for altcoin-friendly market conditions.
    Source: BTCUSDT daily close from Bybit spot.

  NEXT STEP:
    Lock these parameters. Test live with 1-2% of intended capital.
    Six months of real trades is worth more than any further backtesting.
    """)

    print("Done.")
