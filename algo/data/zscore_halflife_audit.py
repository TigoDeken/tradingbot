"""
Investigate the first-half vs second-half deterioration.
Also tests minimum stop at 5%.

Questions:
  1. What time periods are the two halves?
  2. Is deterioration a market regime issue (bull vs bear)?
  3. Are specific coins driving first-half profits?
  4. Year-by-year and quarter-by-quarter breakdown
  5. Stats with stop >= 5%

Run: python -m algo.data.zscore_halflife_audit
"""

import os
import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP
from algo.data.zscore_entry_study import load_all, simulate

BEST = dict(z_thresh=-0.5, hold_days=7, stop_type="low10",
            require_bull=False, require_compressed=True)


def section(title):
    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")


if __name__ == "__main__":
    coins = load_all()

    # Collect trades with dates
    all_trades = []
    for sym, df in coins.items():
        t = simulate(df, **BEST)
        if not t.empty:
            t["symbol"] = sym
            # Attach actual entry date
            t["entry_date"] = t["entry_bar"].apply(
                lambda i: df["time"].iloc[int(i)] if int(i) < len(df) else pd.NaT
            )
            all_trades.append(t)

    trades = pd.concat(all_trades, ignore_index=True)
    trades = trades[trades["risk_pct"] >= 3.0].copy()  # baseline: 3% min
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["year"]    = trades["entry_date"].dt.year
    trades["quarter"] = trades["entry_date"].dt.to_period("Q")
    trades = trades.sort_values("entry_date").reset_index(drop=True)

    mid  = len(trades) // 2
    h1   = trades.iloc[:mid]
    h2   = trades.iloc[mid:]

    # ── 1. What dates are the halves? ─────────────────────────────────────────
    section("1. WHAT TIME PERIODS ARE THE TWO HALVES?")
    print(f"  Total trades (stop >= 3%): {len(trades)}")
    print(f"  First half:  {h1['entry_date'].min().date()} to {h1['entry_date'].max().date()}  (n={len(h1)})")
    print(f"  Second half: {h2['entry_date'].min().date()} to {h2['entry_date'].max().date()}  (n={len(h2)})")
    print(f"\n  First half avg:  {h1['ret_pct'].mean():+.2f}%  wr={( h1['ret_pct']>0).mean()*100:.0f}%")
    print(f"  Second half avg: {h2['ret_pct'].mean():+.2f}%  wr={(h2['ret_pct']>0).mean()*100:.0f}%")

    # ── 2. Year-by-year breakdown ─────────────────────────────────────────────
    section("2. YEAR-BY-YEAR BREAKDOWN")
    print(f"  {'Year':>6}  {'n':>4}  {'wr%':>5}  {'avg%':>7}  {'median%':>8}  {'best':>7}  {'worst':>7}")
    for yr, grp in trades.groupby("year"):
        wr  = (grp["ret_pct"] > 0).mean()
        avg = grp["ret_pct"].mean()
        med = grp["ret_pct"].median()
        print(f"  {yr:>6}  {len(grp):>4}  {wr*100:>4.0f}%  "
              f"{avg:>+6.2f}%  {med:>+7.2f}%  "
              f"{grp['ret_pct'].max():>+6.1f}%  {grp['ret_pct'].min():>+6.1f}%")

    # ── 3. Quarter-by-quarter breakdown ──────────────────────────────────────
    section("3. QUARTER-BY-QUARTER BREAKDOWN")
    print(f"  {'Quarter':>8}  {'n':>4}  {'wr%':>5}  {'avg%':>7}  {'median%':>8}")
    for q, grp in trades.groupby("quarter"):
        if len(grp) < 3:
            continue
        wr  = (grp["ret_pct"] > 0).mean()
        avg = grp["ret_pct"].mean()
        med = grp["ret_pct"].median()
        flag = " <<" if avg > 2 else (" >>" if avg < -2 else "")
        print(f"  {str(q):>8}  {len(grp):>4}  {wr*100:>4.0f}%  "
              f"{avg:>+6.2f}%  {med:>+7.2f}%{flag}")

    # ── 4. Are specific coins driving first-half profits? ────────────────────
    section("4. WHICH COINS DRIVE EACH HALF?")
    print(f"\n  TOP 5 COINS BY CONTRIBUTION — FIRST HALF:")
    h1_by_coin = h1.groupby("symbol").agg(
        n=("ret_pct","count"), avg=("ret_pct","mean"),
        total=("ret_pct","sum")
    ).sort_values("total", ascending=False)
    for sym, row in h1_by_coin.head(5).iterrows():
        print(f"    {sym:18} n={int(row['n'])}  avg={row['avg']:+.2f}%  total={row['total']:+.1f}%")

    print(f"\n  BOTTOM 5 COINS BY CONTRIBUTION — FIRST HALF:")
    for sym, row in h1_by_coin.tail(5).iterrows():
        print(f"    {sym:18} n={int(row['n'])}  avg={row['avg']:+.2f}%  total={row['total']:+.1f}%")

    print(f"\n  TOP 5 COINS BY CONTRIBUTION — SECOND HALF:")
    h2_by_coin = h2.groupby("symbol").agg(
        n=("ret_pct","count"), avg=("ret_pct","mean"),
        total=("ret_pct","sum")
    ).sort_values("total", ascending=False)
    for sym, row in h2_by_coin.head(5).iterrows():
        print(f"    {sym:18} n={int(row['n'])}  avg={row['avg']:+.2f}%  total={row['total']:+.1f}%")

    # ── 5. Is it regime? Check BTC trend at time of trades ───────────────────
    section("5. IS IT A MARKET REGIME ISSUE?")
    print("  (Fraction of trades where coin was above its own 200-day MA)")
    bull_h1 = h1["bull"].mean()
    bull_h2 = h2["bull"].mean()
    print(f"  First half:  {bull_h1*100:.1f}% of trades in bull regime")
    print(f"  Second half: {bull_h2*100:.1f}% of trades in bull regime")

    # Bull vs bear performance across ALL trades
    section("6. BULL vs BEAR REGIME PERFORMANCE (all trades)")
    for regime, mask in [("BULL (above 200d MA)", trades["bull"]),
                         ("BEAR (below 200d MA)", ~trades["bull"])]:
        sub = trades[mask]
        if len(sub) < 5:
            continue
        wr  = (sub["ret_pct"] > 0).mean()
        avg = sub["ret_pct"].mean()
        med = sub["ret_pct"].median()
        print(f"  {regime:25}  n={len(sub):3d}  wr={wr*100:.0f}%  "
              f"avg={avg:+.2f}%  median={med:+.2f}%")

    # ── 6. Stop >= 5% filter ──────────────────────────────────────────────────
    section("7. MINIMUM STOP 5% — DOES IT IMPROVE RESULTS?")
    t5 = trades[trades["risk_pct"] >= 5.0].copy()
    t5 = t5.sort_values("entry_date").reset_index(drop=True)
    mid5 = len(t5) // 2
    h1_5, h2_5 = t5.iloc[:mid5], t5.iloc[mid5:]

    print(f"\n  Trades removed by raising floor from 3% to 5%: "
          f"{len(trades) - len(t5)} ({(1 - len(t5)/len(trades))*100:.0f}%)")
    print(f"\n  All trades (stop >= 5%):")
    print(f"    n={len(t5)}  wr={(t5['ret_pct']>0).mean()*100:.1f}%  "
          f"avg={t5['ret_pct'].mean():+.2f}%  "
          f"median={t5['ret_pct'].median():+.2f}%  "
          f"avg_win={t5[t5['ret_pct']>0]['ret_pct'].mean():+.2f}%  "
          f"avg_loss={t5[t5['ret_pct']<=0]['ret_pct'].mean():+.2f}%")
    print(f"    Stop rate: {(t5['exit_reason']=='stop').mean()*100:.1f}%  "
          f"Time exit: {(t5['exit_reason']=='time').mean()*100:.1f}%")

    print(f"\n  Consistency:")
    print(f"    First half  ({h1_5['entry_date'].min().date()} to {h1_5['entry_date'].max().date()})"
          f"  n={len(h1_5)}  avg={h1_5['ret_pct'].mean():+.2f}%  wr={(h1_5['ret_pct']>0).mean()*100:.0f}%")
    print(f"    Second half ({h2_5['entry_date'].min().date()} to {h2_5['entry_date'].max().date()})"
          f"  n={len(h2_5)}  avg={h2_5['ret_pct'].mean():+.2f}%  wr={(h2_5['ret_pct']>0).mean()*100:.0f}%")

    print(f"\n  Year-by-year (stop >= 5%):")
    print(f"  {'Year':>6}  {'n':>4}  {'wr%':>5}  {'avg%':>7}  {'median%':>8}")
    for yr, grp in t5.groupby("year"):
        wr  = (grp["ret_pct"] > 0).mean()
        avg = grp["ret_pct"].mean()
        med = grp["ret_pct"].median()
        print(f"  {yr:>6}  {len(grp):>4}  {wr*100:>4.0f}%  "
              f"{avg:>+6.2f}%  {med:>+7.2f}%")

    # ── 7. Summary ────────────────────────────────────────────────────────────
    section("SUMMARY")
    print(f"""
  The key question: is the first-half vs second-half gap real deterioration,
  or is it explained by market regime (bull vs bear)?

  If second half coincides with bear market:
    - The signal may only work in bull conditions
    - Applying a regime filter would NOT be curve fitting — it would be
      the mechanistically correct fix (z-score was proven in bull markets)

  If second half has similar regime mix but still underperforms:
    - The signal is genuinely decaying or was partly luck in first half
    - Need to investigate further before trusting it
    """)

    print("Done.")
