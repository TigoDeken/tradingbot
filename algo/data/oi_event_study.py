"""
OI event study — large single-day OI drops as reversal entry signals.

Hypothesis (from Agent 4 analysis):
  When OI drops >15% in one day AND prior funding_z was positive (longs dominant),
  a long liquidation cascade has just occurred. Price overshoots downside.
  Spot recovers over the next 3-7 days.

Also tests OI surges (>10% in one day) as momentum entries.

Uses existing OI + spot cache — no new data needed.

Run: python -m algo.data.oi_event_study
"""

import numpy as np
import pandas as pd
from algo.data.zscore_entry_study import load_all
from algo.data.perp_flow_study import load_perp_oi, load_perp_kline
from algo.data.oi_flow_study import build_full_signals
from algo.data.perp_flow_study import rank_ic

MIN_OBS       = 90
OI_DROP_THRESH = -0.15   # >15% single-day OI drop
OI_SURGE_THRESH = 0.10   # >10% single-day OI surge
FZ_BULL_THRESH  = 0.5    # prior funding was positive (longs dominant)
HORIZONS        = [3, 7, 14]


def section(t):
    print(f"\n{'='*65}\n{t}\n{'='*65}")


if __name__ == "__main__":
    print("Loading data...")
    coins = load_all()

    all_sigs = {}
    for sym, spot_df in sorted(coins.items()):
        oi_df   = load_perp_oi(sym)
        perp_df = load_perp_kline(sym)
        if oi_df.empty or perp_df.empty:
            continue
        sig = build_full_signals(spot_df, oi_df, perp_df)
        if len(sig) >= MIN_OBS:
            all_sigs[sym] = sig

    print(f"Coins with data: {len(all_sigs)}")

    # Pool all observations
    parts = []
    for sym, df in all_sigs.items():
        d = df.copy(); d["symbol"] = sym; parts.append(d)
    pool = pd.concat(parts, ignore_index=True)
    print(f"Total observations: {len(pool)}")

    # Compute daily OI change
    pool = pool.sort_values(["symbol","date"]).copy()
    pool["oi_change_1d"] = pool.groupby("symbol")["oi"].pct_change(1)

    # Lagged funding_z (prior day)
    pool["fz_lag1"] = pool.groupby("symbol")["funding_z"].shift(1)

    # ── IC of raw OI change ───────────────────────────────────────────────────
    section("1. IC OF RAW OI DAILY CHANGE vs FORWARD RETURNS")
    print(f"\n  {'Signal':20}  {'IC_3d':>8}  {'IC_7d':>8}  {'IC_14d':>8}")
    for sig in ["oi_change_1d", "oi_delta3"]:
        ics = []
        for h in HORIZONS:
            s = pool[[sig, f"fwd{h}"]].dropna()
            ics.append(rank_ic(s[sig], s[f"fwd{h}"]))
        print(f"  {sig:20}  {ics[0]:>+7.4f}   {ics[1]:>+7.4f}   {ics[2]:>+7.4f}")

    # ── OI collapse event ─────────────────────────────────────────────────────
    section("2. OI COLLAPSE EVENT (>15% single-day drop + prior longs dominant)")
    pool["oi_collapse"] = (
        (pool["oi_change_1d"] < OI_DROP_THRESH) &
        (pool["fz_lag1"] > FZ_BULL_THRESH)
    )
    pool["oi_collapse_any"] = (pool["oi_change_1d"] < OI_DROP_THRESH)

    for label, mask in [
        (f"OI drop >15% + prior fz>{FZ_BULL_THRESH}", pool["oi_collapse"]),
        ("OI drop >15% (any funding)",                  pool["oi_collapse_any"]),
    ]:
        grp  = pool[mask]
        rest = pool[~mask]
        print(f"\n  {label}:")
        print(f"  Events: {mask.sum()} ({mask.mean()*100:.1f}% of days)  "
              f"Coins affected: {pool[mask]['symbol'].nunique()}")
        print(f"\n  {'Horizon':>10}  {'Event mean':>11}  {'Event wr':>10}  "
              f"{'Baseline':>10}  {'Edge':>7}")
        for h in HORIZONS:
            col = f"fwd{h}"
            ev  = grp[col].dropna()
            bl  = rest[col].dropna()
            if len(ev) < 5:
                continue
            edge = ev.mean() - bl.mean()
            print(f"  {h:>8}d  {ev.mean():>+10.2f}%  {(ev>0).mean()*100:>9.0f}%  "
                  f"{bl.mean():>+9.2f}%  {edge:>+6.2f}%")

    # ── OI surge event ────────────────────────────────────────────────────────
    section("3. OI SURGE EVENT (>10% single-day increase = big player entering)")
    pool["oi_surge"] = (pool["oi_change_1d"] > OI_SURGE_THRESH)
    grp  = pool[pool["oi_surge"]]
    rest = pool[~pool["oi_surge"]]

    print(f"\n  OI surge >10% in one day:")
    print(f"  Events: {pool['oi_surge'].sum()} ({pool['oi_surge'].mean()*100:.1f}% of days)  "
          f"Coins: {pool[pool['oi_surge']]['symbol'].nunique()}")
    print(f"\n  {'Horizon':>10}  {'Event mean':>11}  {'Event wr':>10}  "
          f"{'Baseline':>10}  {'Edge':>7}")
    for h in HORIZONS:
        col = f"fwd{h}"
        ev  = grp[col].dropna()
        bl  = rest[col].dropna()
        if len(ev) < 5:
            continue
        edge = ev.mean() - bl.mean()
        print(f"  {h:>8}d  {ev.mean():>+10.2f}%  {(ev>0).mean()*100:>9.0f}%  "
              f"{bl.mean():>+9.2f}%  {edge:>+6.2f}%")

    # ── Combined with existing signals ────────────────────────────────────────
    section("4. OI COLLAPSE + EXISTING ENTRY CONDITIONS")
    print("  Does OI collapse improve results when added to our entry filter?")
    print("  Entry: oi_z > 1.0 AND funding_z < -0.5 AND oi_collapse (extra)")

    comb = pool[
        (pool["oi_z"] > 1.0) &
        (pool["funding_z"] < -0.5) &
        (pool["oi_collapse_any"])
    ]
    base = pool[
        (pool["oi_z"] > 1.0) &
        (pool["funding_z"] < -0.5)
    ]
    print(f"\n  Base (oi_z>1 + fz<-0.5):        {len(base)} obs")
    print(f"  + OI collapse filter:           {len(comb)} obs ({len(comb)/len(base)*100:.0f}% remain)")
    print(f"\n  {'Horizon':>10}  {'Combined':>10}  {'Base':>8}  {'Lift':>6}")
    for h in HORIZONS:
        col = f"fwd{h}"
        c   = comb[col].dropna()
        b   = base[col].dropna()
        if len(c) < 3:
            continue
        lift = c.mean() - b.mean()
        print(f"  {h:>8}d  {c.mean():>+9.2f}%  {b.mean():>+7.2f}%  {lift:>+5.2f}%")

    # ── Event-based simulation ────────────────────────────────────────────────
    section("5. STANDALONE EVENT SIMULATION: OI collapse -> long entry")
    print("  Entry: OI drops >15% AND prior fz > 0.5")
    print("  Exit: 7-day time exit (events mean-revert quickly or not at all)")
    print("  Stop: low of event day - 5%\n")

    from algo.data.final_spec_study import MIN_STOP, STOP_DAYS, MAX_HOLD

    event_trades = []
    for sym, df in all_sigs.items():
        df = df.copy().reset_index(drop=True)
        if "oi_change_1d" not in df.columns:
            df["oi_change_1d"] = df["oi"].pct_change(1)
        if "fz_lag1" not in df.columns:
            df["fz_lag1"] = df["funding_z"].shift(1)

        n      = len(df)
        fz     = df["funding_z"].values
        fz_lag = df["fz_lag1"].fillna(0).values
        oi_ch  = df["oi_change_1d"].fillna(0).values
        op_    = df["open"].values
        lo_    = df["low"].values
        cl_    = df["close"].values

        in_trade = 0
        for i in range(1, n - 8):
            if i < in_trade:
                continue
            if not (oi_ch[i] < OI_DROP_THRESH and fz_lag[i] > FZ_BULL_THRESH):
                continue

            entry = op_[min(i+1, n-1)]
            stop  = lo_[i] * (1 - MIN_STOP)  # event day low - buffer
            if entry <= 0:
                continue
            risk = entry - stop
            if risk <= 0:
                continue
            rp = risk / entry
            if rp > 0.30:
                continue

            # 7-day time exit
            hold_n = min(7, n - i - 2)
            exit_p = cl_[i + 1 + hold_n]
            ret    = (exit_p - entry) / entry * 100
            ret_r  = (exit_p - entry) / risk

            event_trades.append({
                "symbol":   sym,
                "date":     df["date"].iloc[i],
                "ret_pct":  ret,
                "ret_r":    ret_r,
                "risk_pct": rp * 100,
            })
            in_trade = i + 1 + hold_n + 1

    if event_trades:
        et = pd.DataFrame(event_trades)
        wr  = (et["ret_pct"] > 0).mean()
        avg = et["ret_pct"].mean()
        med = et["ret_pct"].median()
        std = et["ret_pct"].std()
        shr = avg / std if std > 0 else 0
        print(f"  Trades: {len(et)}  wr={wr*100:.0f}%  avg={avg:+.2f}%  "
              f"med={med:+.2f}%  sharpe={shr:+.3f}")

        et["year"] = pd.to_datetime(et["date"]).dt.year
        print(f"\n  Year-by-year:")
        for yr, grp in et.groupby("year"):
            print(f"    {yr}: n={len(grp)}  wr={(grp['ret_pct']>0).mean()*100:.0f}%  "
                  f"avg={grp['ret_pct'].mean():+.2f}%")
    else:
        print("  No event trades found.")

    section("SUMMARY")
    print("""
  OI collapse (>15% drop) combined with prior positive funding:
  - If edge is positive and consistent: add as standalone entry signal
  - If edge is marginal: use as confirmation boost for existing signal
  - If negative or noisy: drop entirely

  OI surge (>10% spike): if positive IC, use as momentum filter
  (only trade our mean-reversion entry when OI is NOT surging)
""")
    print("Done.")
