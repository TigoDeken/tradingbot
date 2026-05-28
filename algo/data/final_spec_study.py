"""
Final specification study — formalizing the complete signal set.

Builds on:
  - zscore_entry_study: funding z-score crossing below -0.5 + ATR compression
  - oi_flow_study: oi_z > 1.0 triples average return, doubles Sharpe

4-way comparison on same 78 coins (those with perp data):
  A. Original:  z-cross + ATR, fixed 7-day hold, no oi filter
  B. + signal exit (funding_z > 0 exit)
  C. + oi_z > 1.0 filter (fixed 7-day hold)
  D. FINAL: z-cross + ATR + oi_z > 1.0 + signal exit

Run: python -m algo.data.final_spec_study
"""

import os
import numpy as np
import pandas as pd
from algo.data.zscore_entry_study import load_all
from algo.data.perp_flow_study import load_perp_oi, load_perp_kline
from algo.data.oi_flow_study import build_full_signals

# ── FROZEN PARAMETERS ─────────────────────────────────────────────────────────
FZ_THRESH   = -0.5    # funding z-score entry threshold
ATR_THRESH  = 0.80    # ATR5/ATR10 compression
OI_Z_THRESH = 1.0     # OI must be >= 1 std above its 90-day mean
STOP_DAYS   = 10      # 10-day lowest low for stop
MIN_STOP    = 0.05    # 5% minimum stop distance
EXIT_Z      = 0.0     # signal exit when funding_z rises above this
MAX_HOLD    = 21      # hard maximum hold days
FIXED_HOLD  = 7       # hold for baseline comparison (original)
FZ_WIN      = 90
OI_Z_WIN    = 90
MIN_OBS     = 90


def section(t):
    print(f"\n{'='*72}")
    print(t)
    print(f"{'='*72}")


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(df: pd.DataFrame,
             use_oi_filter:    bool  = False,
             oi_thresh:        float = 1.0,
             use_signal_exit:  bool  = True,
             fixed_hold:       int   = 7) -> pd.DataFrame:
    n     = len(df)
    fz    = df["funding_z"].values
    oiz   = df["oi_z"].values    if "oi_z"    in df.columns else np.full(n, np.nan)
    atr   = df["atr_ratio"].values if "atr_ratio" in df.columns else np.ones(n)
    op    = df["open"].values
    lo    = df["low"].values
    cl    = df["close"].values

    trades       = []
    in_trade_end = 0

    for i in range(FZ_WIN + 10, n - MAX_HOLD - 1):
        if i < in_trade_end:
            continue

        z_c = fz[i];  z_p = fz[i-1]
        if np.isnan(z_c) or np.isnan(z_p):
            continue

        # Entry: z crosses below threshold + ATR compressed
        if not (z_c < FZ_THRESH and z_p >= FZ_THRESH):
            continue
        if np.isnan(atr[i]) or atr[i] >= ATR_THRESH:
            continue

        # OI filter
        if use_oi_filter:
            oz = oiz[i]
            if np.isnan(oz) or oz < oi_thresh:
                continue

        entry = op[min(i+1, n-1)]
        stop  = lo[max(0, i-STOP_DAYS):i].min()
        if np.isnan(stop) or entry <= 0:
            continue
        risk = entry - stop
        if risk <= 0:
            continue
        rp = risk / entry
        if rp < MIN_STOP or rp > 0.30:
            continue

        hold_limit  = MAX_HOLD if use_signal_exit else fixed_hold
        exit_price  = None
        exit_reason = "time"
        hold        = hold_limit

        for j in range(i+1, min(i+1+hold_limit+1, n)):
            if lo[j] <= stop:
                exit_price  = stop
                exit_reason = "stop"
                hold        = j - (i+1)
                break
            if use_signal_exit and not np.isnan(fz[j]) and fz[j] > EXIT_Z:
                exit_price  = op[min(j+1, n-1)]
                exit_reason = "z_exit"
                hold        = j - (i+1)
                break

        if exit_price is None:
            last_j     = min(i+hold_limit, n-1)
            exit_price = cl[last_j]
            hold       = last_j - (i+1)

        ret = (exit_price - entry) / entry * 100

        trades.append({
            "date":        df["date"].iloc[i],
            "entry_date":  df["date"].iloc[i+1],
            "ret_pct":     ret,
            "exit_reason": exit_reason,
            "hold":        hold,
            "risk_pct":    rp * 100,
            "oi_z":        oiz[i],
            "fz":          z_c,
            "atr":         atr[i],
        })
        in_trade_end = i + 1 + hold + 1

    return pd.DataFrame(trades)


def print_stats(t: pd.DataFrame, label: str) -> dict:
    if t.empty or len(t) < 5:
        print(f"  {label:58} n={len(t):3d}  insufficient")
        return {}
    wr   = (t["ret_pct"] > 0).mean()
    avg  = t["ret_pct"].mean()
    med  = t["ret_pct"].median()
    std  = t["ret_pct"].std()
    shr  = avg / std if std > 0 else 0
    stop = (t["exit_reason"] == "stop").mean()
    zex  = (t["exit_reason"] == "z_exit").mean()
    qual = "STRONG" if shr > 0.10 else ("EDGE" if avg > 0.5 else ("slight" if avg > 0 else "NO"))
    print(f"  {label:58} n={len(t):3d}  wr={wr*100:.0f}%  "
          f"avg={avg:+.2f}%  shr={shr:+.3f}  [{qual}]  "
          f"stop={stop*100:.0f}% z={zex*100:.0f}%")
    return dict(n=len(t), wr=wr, avg=avg, med=med, shr=shr)


def walk_forward(trades: pd.DataFrame, label: str):
    if trades.empty:
        return
    t = trades.copy()
    t["entry_date"] = pd.to_datetime(t["entry_date"])
    t = t.sort_values("entry_date")
    t["period"] = t["entry_date"].dt.to_period("2Q")
    print(f"\n  {label}:")
    print(f"  {'Period':>10}  {'n':>4}  {'wr%':>5}  {'avg%':>7}  {'med%':>7}  {'sharpe':>7}")
    print(f"  {'-'*52}")
    for p, grp in t.groupby("period"):
        if len(grp) < 3:
            continue
        avg = grp["ret_pct"].mean()
        std = grp["ret_pct"].std()
        shr = avg / std if std > 0 else 0
        flag = " <<" if avg > 2 else (" >>" if avg < -1 else "")
        print(f"  {str(p):>10}  {len(grp):>4}  {(grp['ret_pct']>0).mean()*100:>4.0f}%  "
              f"{avg:>+6.2f}%  {grp['ret_pct'].median():>+6.2f}%  "
              f"{shr:>+6.3f}{flag}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    coins = load_all()

    all_dfs = {}
    no_perp = []
    for sym, spot_df in sorted(coins.items()):
        oi_df   = load_perp_oi(sym)
        perp_df = load_perp_kline(sym)
        if oi_df.empty or perp_df.empty:
            no_perp.append(sym)
            continue
        sig = build_full_signals(spot_df, oi_df, perp_df)
        if len(sig) >= MIN_OBS:
            all_dfs[sym] = sig

    print(f"Coins with perp data: {len(all_dfs)}  (excluded: {len(no_perp)})")

    # Run all 4 combinations
    combos = {
        "A": dict(use_oi_filter=False, use_signal_exit=False),
        "B": dict(use_oi_filter=False, use_signal_exit=True),
        "C": dict(use_oi_filter=True,  oi_thresh=OI_Z_THRESH, use_signal_exit=False),
        "D": dict(use_oi_filter=True,  oi_thresh=OI_Z_THRESH, use_signal_exit=True),
    }

    results = {}
    for key, kw in combos.items():
        parts = []
        for sym, df in all_dfs.items():
            t = simulate(df, **kw)
            if not t.empty:
                t["symbol"] = sym
                parts.append(t)
        results[key] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    # ── Comparison table ──────────────────────────────────────────────────────
    section("COMPARISON: 4 COMBINATIONS")
    print()
    print_stats(results["A"], "A. z-cross + ATR, fixed 7d, no oi filter")
    print_stats(results["B"], "B. + signal exit (z > 0)")
    print_stats(results["C"], "C. + oi_z > 1.0 filter, fixed 7d")
    print_stats(results["D"], "D. FINAL: z-cross + ATR + oi_z>1 + signal exit")

    final = results["D"]

    # ── Deep dive on FINAL ────────────────────────────────────────────────────
    section("DEEP DIVE — FINAL COMBINATION (D)")
    if not final.empty:
        wins  = final[final["ret_pct"] > 0]["ret_pct"]
        loses = final[final["ret_pct"] <= 0]["ret_pct"]
        print(f"""
  Trades:         {len(final)}
  Win rate:       {(final['ret_pct']>0).mean()*100:.1f}%
  Avg return:     {final['ret_pct'].mean():+.2f}% per trade
  Median return:  {final['ret_pct'].median():+.2f}% per trade
  Avg win:        {wins.mean():+.2f}%   |  Avg loss: {loses.mean():+.2f}%
  Sharpe ratio:   {final['ret_pct'].mean()/final['ret_pct'].std():+.3f}
  Avg hold:       {final['hold'].mean():.1f} days
  Avg stop dist:  {final['risk_pct'].mean():.1f}%
  Exits:          stop={( final['exit_reason']=='stop').mean()*100:.0f}%  z-signal={(final['exit_reason']=='z_exit').mean()*100:.0f}%  time={(final['exit_reason']=='time').mean()*100:.0f}%
  Avg oi_z:       {final['oi_z'].mean():+.2f}  (at entry)
  Avg funding_z:  {final['fz'].mean():+.2f}  (at entry)""")

    # ── Walk-forward ──────────────────────────────────────────────────────────
    section("WALK-FORWARD (6-month windows)")
    walk_forward(final, "FINAL (D): z-cross + ATR + oi_z>1 + signal exit")
    walk_forward(results["A"], "Baseline (A): z-cross + ATR, fixed 7d")

    # ── Year-by-year ──────────────────────────────────────────────────────────
    section("YEAR-BY-YEAR — FINAL vs BASELINE")
    print(f"\n  {'Year':>6}  {'A_n':>4}  {'A_avg%':>7}  {'D_n':>4}  {'D_avg%':>7}  {'D_wr%':>6}")
    final["year"]  = pd.to_datetime(final["entry_date"]).dt.year
    base = results["A"].copy()
    base["year"]   = pd.to_datetime(base["entry_date"]).dt.year
    for yr in sorted(set(final["year"].tolist() + base["year"].tolist())):
        a_grp = base[base["year"]==yr]
        d_grp = final[final["year"]==yr]
        print(f"  {yr:>6}  {len(a_grp):>4}  {a_grp['ret_pct'].mean():>+6.2f}%  "
              f"{len(d_grp):>4}  {d_grp['ret_pct'].mean():>+6.2f}%  "
              f"{(d_grp['ret_pct']>0).mean()*100:>5.0f}%")

    # ── Per-coin breakdown ────────────────────────────────────────────────────
    section("PER-COIN BREAKDOWN — FINAL")
    if not final.empty:
        per = final.groupby("symbol").agg(
            n=("ret_pct","count"),
            avg=("ret_pct","mean"),
            wr=("ret_pct", lambda x: (x>0).mean()),
            med=("ret_pct","median"),
        ).sort_values("avg", ascending=False)
        print(f"\n  {'Symbol':18} {'n':>4} {'wr%':>5} {'avg%':>7} {'med%':>7}")
        for sym, row in per.iterrows():
            flag = " <<" if row["avg"] > 5 else (" >>" if row["avg"] < -5 else "")
            print(f"  {sym:18} {int(row['n']):>4} {row['wr']*100:>4.0f}% "
                  f"{row['avg']:>+6.2f}% {row['med']:>+6.2f}%{flag}")

    # ── What trades were filtered out ────────────────────────────────────────
    section("WHAT oi_z FILTER REMOVES")
    base_b  = results["B"]
    removed = base_b[~base_b.apply(
        lambda r: ((r["symbol"], r["date"]) in
                   set(zip(final["symbol"], final["date"]))), axis=1
    )] if not base_b.empty and not final.empty else pd.DataFrame()

    if not removed.empty:
        print(f"""
  B (signal exit, no oi filter): {len(base_b)} trades  avg={base_b['ret_pct'].mean():+.2f}%
  D (signal exit + oi filter):   {len(final)} trades  avg={final['ret_pct'].mean():+.2f}%
  Removed by oi_z > 1.0 filter:  {len(removed)} trades  avg={removed['ret_pct'].mean():+.2f}%
  (negative removed avg = filter is cutting bad trades)""")

    # ── Frozen spec ───────────────────────────────────────────────────────────
    section("FROZEN STRATEGY — FINAL SPECIFICATION v2")
    print(f"""
  ENTRY CONDITIONS (all must be true simultaneously):
    1. Funding z-score (90d rolling) crosses BELOW {FZ_THRESH}
       -> crowd is absent relative to recent history, fresh negative sentiment
    2. ATR compression: ATR(5)/ATR(10) < {ATR_THRESH}
       -> price is quiet, stop distance is tight
    3. OI z-score (90d rolling) > {OI_Z_THRESH}
       -> perp market is actively engaged: OI >= 1 std above its own 90d average
       -> THIS IS THE NEW COMPONENT: only trade "live" coins

  EXECUTION:
    Entry:  open of next bar after signal (no lookahead)
    Stop:   lowest low of {STOP_DAYS} bars before entry, minimum {MIN_STOP*100:.0f}% from entry
    Size:   risk-based (TBD: fixed fraction of capital per trade)

  EXIT CONDITIONS (first triggered):
    1. Stop hit: exit at stop price
    2. Signal exit: funding z-score rises above {EXIT_Z}
       -> crowd returning, edge is exhausted, exit at next open
    3. Maximum hold: {MAX_HOLD} days (safety limit)

  PARAMETERS FROZEN — DO NOT CHANGE:
    FZ_THRESH   = {FZ_THRESH}
    ATR_THRESH  = {ATR_THRESH}
    OI_Z_THRESH = {OI_Z_THRESH}
    STOP_DAYS   = {STOP_DAYS}
    MIN_STOP    = {MIN_STOP}
    EXIT_Z      = {EXIT_Z}
    MAX_HOLD    = {MAX_HOLD}
    FZ_WINDOW   = {FZ_WIN}
    OI_Z_WIN    = {OI_Z_WIN}

  WHY IT WORKS:
    The three conditions select for a specific structural state:
    - The COIN is in an active market phase (lots of leveraged exposure, oi_z high)
    - The CROWD has just turned sour (negative funding, longs leaving)
    - The PRICE hasn't moved yet (ATR compressed)
    This is the "coiled spring": active positioning, sentiment shift, no price reaction yet.
    When it unwinds, it unwinds fast. We exit when the crowd returns (z > 0).

  NEXT STEP:
    These parameters are now frozen on this dataset.
    Deploy with 1-2%% of intended capital.
    Collect 50+ live trades before evaluating.
    """)

    print("Done.")
