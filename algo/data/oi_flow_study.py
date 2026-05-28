"""
OI flow study — drilling into the oi_z finding from perp_flow_study.

perp_flow_study found oi_z is the strongest signal (IC=0.039, p<0.001).
High OI relative to a coin's own 90d history = better forward returns.

This study answers:
  1. Cross-tab: oi_z x funding_z quartiles — which combo dominates?
     Hypothesis: HIGH oi_z + LOW funding_z = active coin, shorts trapped
  2. Per-coin IC: time-series effect or just big coins having better returns?
  3. Simulation A: original z-cross signal, filtered by oi_z > 0
  4. Simulation B: new signal — oi_z crosses above 0 AND funding_z < 0
     (market waking up while shorts are dominant)

Run: python -m algo.data.oi_flow_study
"""

import os
import numpy as np
import pandas as pd
from algo.data.zscore_entry_study import load_all
from algo.data.perp_flow_study import load_perp_oi, load_perp_kline, rank_ic

OI_Z_WIN  = 90
FZ_WIN    = 90
MIN_STOP  = 0.05
STOP_DAYS = 10
MAX_HOLD  = 21
EXIT_Z    = 0.0
MIN_OBS   = 90


def section(t):
    print(f"\n{'='*70}")
    print(t)
    print(f"{'='*70}")


# ── Data builder ──────────────────────────────────────────────────────────────

def build_full_signals(spot_df: pd.DataFrame,
                       oi_df:   pd.DataFrame,
                       perp_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["time", "open", "high", "low", "close", "funding_z", "atr_ratio"]
    spot = spot_df[[c for c in cols if c in spot_df.columns]].copy()
    spot["date"] = spot["time"].dt.normalize().dt.tz_localize(None)

    oi = oi_df.copy()
    oi["date"] = oi["time"].dt.normalize().dt.tz_localize(None)
    oi = oi.groupby("date")["oi"].last().reset_index()

    perp = perp_df.copy()
    perp["date"] = perp["time"].dt.normalize().dt.tz_localize(None)
    perp = perp.groupby("date")["perp_close"].last().reset_index()

    df = spot.merge(oi, on="date", how="inner").merge(perp, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < MIN_OBS:
        return pd.DataFrame()

    df["basis"]     = (df["perp_close"] - df["close"]) / df["close"] * 100
    df["oi_delta3"] = df["oi"].pct_change(3) * 100

    roll       = df["oi"].rolling(OI_Z_WIN)
    df["oi_z"] = (df["oi"] - roll.mean()) / roll.std()

    for h in [3, 7, 14]:
        df[f"fwd{h}"] = (df["close"].shift(-h) / df["close"] - 1) * 100

    return df.dropna(subset=["oi_z", "funding_z"]).reset_index(drop=True)


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(df: pd.DataFrame,
             use_oi_filter:  bool  = False,
             oi_thresh:      float = 0.0,
             use_new_signal: bool  = False,
             fz_thresh:      float = -0.5) -> pd.DataFrame:
    """
    Original signal: funding_z crosses below fz_thresh + ATR compressed
    New signal:      oi_z crosses above 0 + funding_z < 0

    use_oi_filter: additionally require oi_z > oi_thresh at entry
    """
    n     = len(df)
    fz    = df["funding_z"].values
    oiz   = df["oi_z"].values
    op    = df["open"].values
    lo    = df["low"].values
    cl    = df["close"].values
    atr_r = df["atr_ratio"].values if "atr_ratio" in df.columns else np.ones(n)

    trades       = []
    in_trade_end = 0

    for i in range(FZ_WIN + 10, n - MAX_HOLD - 1):
        if i < in_trade_end:
            continue

        z_c  = fz[i];   z_p  = fz[i-1]
        oz_c = oiz[i];  oz_p = oiz[i-1]

        if np.isnan(z_c) or np.isnan(z_p):
            continue
        if np.isnan(oz_c) or np.isnan(oz_p):
            continue

        if use_new_signal:
            entry_ok = (oz_c > 0) and (oz_p <= 0) and (z_c < 0)
        else:
            atr_ok   = (not np.isnan(atr_r[i])) and (atr_r[i] < 0.80)
            entry_ok = (z_c < fz_thresh) and (z_p >= fz_thresh) and atr_ok

        if not entry_ok:
            continue

        if use_oi_filter and oz_c < oi_thresh:
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

        exit_price  = None
        exit_reason = "time"
        hold        = MAX_HOLD

        for j in range(i+1, min(i+1+MAX_HOLD, n)):
            if lo[j] <= stop:
                exit_price  = stop
                exit_reason = "stop"
                hold        = j - (i+1)
                break
            if not np.isnan(fz[j]) and fz[j] > EXIT_Z:
                exit_price  = op[min(j+1, n-1)]
                exit_reason = "z_exit"
                hold        = j - (i+1)
                break

        if exit_price is None:
            last_j     = min(i+MAX_HOLD, n-1)
            exit_price = cl[last_j]
            hold       = last_j - (i+1)

        ret = (exit_price - entry) / entry * 100

        trades.append({
            "date":       df["date"].iloc[i],
            "ret_pct":    ret,
            "exit_reason": exit_reason,
            "hold":       hold,
            "risk_pct":   rp * 100,
            "oi_z":       oz_c,
            "fz":         z_c,
        })

        in_trade_end = i + 1 + hold + 1

    return pd.DataFrame(trades)


def print_sim(t: pd.DataFrame, label: str):
    if t.empty or len(t) < 5:
        print(f"  {label:55} n={len(t):3d}  insufficient")
        return
    wr   = (t["ret_pct"] > 0).mean()
    avg  = t["ret_pct"].mean()
    med  = t["ret_pct"].median()
    std  = t["ret_pct"].std()
    shr  = avg / std if std > 0 else 0
    stop = (t["exit_reason"] == "stop").mean()
    zex  = (t["exit_reason"] == "z_exit").mean()
    print(f"  {label:55} n={len(t):3d}  wr={wr*100:.0f}%  "
          f"avg={avg:+.2f}%  med={med:+.2f}%  shr={shr:+.3f}  "
          f"stop={stop*100:.0f}%  z_exit={zex*100:.0f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    coins = load_all()

    all_signals = {}
    for sym, spot_df in sorted(coins.items()):
        oi_df   = load_perp_oi(sym)
        perp_df = load_perp_kline(sym)
        if oi_df.empty or perp_df.empty:
            continue
        sig = build_full_signals(spot_df, oi_df, perp_df)
        if len(sig) >= MIN_OBS:
            all_signals[sym] = sig

    print(f"Coins with full data: {len(all_signals)}")

    parts = []
    for sym, df in all_signals.items():
        d = df.copy(); d["symbol"] = sym; parts.append(d)
    pool = pd.concat(parts, ignore_index=True)
    print(f"Total pooled observations: {len(pool)}\n")

    # ── 1. Cross-tab oi_z x funding_z ─────────────────────────────────────────
    section("1. CROSS-TAB: oi_z QUARTILE x funding_z QUARTILE -> 7d FORWARD RETURN")
    print("  (Each cell: mean 7d fwd return. n in brackets.)")
    print("  Row = oi_z quartile (Q1=lowest OI, Q4=highest OI)")
    print("  Col = funding_z quartile (Q1=most negative, Q4=most positive)")

    sub = pool[["oi_z", "funding_z", "fwd7"]].dropna().copy()
    sub["oi_q"]  = pd.qcut(sub["oi_z"],  q=4, labels=[1,2,3,4], duplicates="drop")
    sub["fz_q"]  = pd.qcut(sub["funding_z"], q=4, labels=[1,2,3,4], duplicates="drop")

    print(f"\n  {'':8}", end="")
    for fq in [1,2,3,4]:
        print(f"  fz_Q{fq}({['neg','lo','hi','pos'][fq-1]})", end="")
    print()

    best_cell = {"avg": -999, "label": ""}
    worst_cell = {"avg": 999, "label": ""}

    for oq in [1,2,3,4]:
        print(f"  oi_Q{oq}  ", end="")
        for fq in [1,2,3,4]:
            cell = sub[(sub["oi_q"]==oq) & (sub["fz_q"]==fq)]["fwd7"]
            if len(cell) < 5:
                print(f"  {'---':>10}", end="")
                continue
            avg = cell.mean()
            flag = "**" if abs(avg) > 1.5 else (" *" if abs(avg) > 0.8 else "  ")
            print(f"  {avg:>+5.2f}%{flag}", end="")
            if avg > best_cell["avg"]:
                best_cell = {"avg": avg, "label": f"oi_Q{oq} x fz_Q{fq}", "n": len(cell)}
            if avg < worst_cell["avg"]:
                worst_cell = {"avg": avg, "label": f"oi_Q{oq} x fz_Q{fq}", "n": len(cell)}
        print()

    print(f"\n  Best cell:  {best_cell['label']:20}  avg={best_cell['avg']:+.2f}%  n={best_cell['n']}")
    print(f"  Worst cell: {worst_cell['label']:20}  avg={worst_cell['avg']:+.2f}%  n={worst_cell['n']}")

    # Also show the 3d and 14d for the best cell
    oq_best = int(best_cell["label"].split("_Q")[1][0])
    fq_best = int(best_cell["label"].split("_Q")[2][0])
    cell_data = sub[(sub["oi_q"]==oq_best) & (sub["fz_q"]==fq_best)]
    print(f"\n  Best cell detail:")
    for h in [3, 7, 14]:
        fwd_col = f"fwd{h}"
        fwd_sub = pool[["oi_z","funding_z",fwd_col]].dropna().copy()
        fwd_sub["oi_q"] = pd.qcut(fwd_sub["oi_z"], q=4, labels=[1,2,3,4], duplicates="drop")
        fwd_sub["fz_q"] = pd.qcut(fwd_sub["funding_z"], q=4, labels=[1,2,3,4], duplicates="drop")
        c = fwd_sub[(fwd_sub["oi_q"]==oq_best) & (fwd_sub["fz_q"]==fq_best)][fwd_col]
        if len(c) > 0:
            print(f"    {h}d: mean={c.mean():+.2f}%  wr={(c>0).mean()*100:.0f}%  med={c.median():+.2f}%  n={len(c)}")

    # ── 2. Per-coin IC ─────────────────────────────────────────────────────────
    section("2. PER-COIN IC — is oi_z a time-series effect or cross-sectional?")
    print("  (Computing IC per coin, then averaging across coins)")
    print("  If per-coin IC is similar to pooled IC: effect is time-series (real)")
    print("  If per-coin IC is near zero: effect is cross-sectional (big coins)")

    per_coin_ics = {"oi_z": [], "funding_z": [], "basis": []}
    for sym, df in all_signals.items():
        for sig in ["oi_z", "funding_z", "basis"]:
            if sig in df.columns:
                ic_val = rank_ic(df[sig].dropna(), df.loc[df[sig].notna(), "fwd7"].dropna()
                                 if len(df[sig].dropna()) == len(df["fwd7"].dropna())
                                 else df["fwd7"])
                # Align properly
                s = df[[sig, "fwd7"]].dropna()
                if len(s) >= 20:
                    ic_val = rank_ic(s[sig], s["fwd7"])
                    per_coin_ics[sig].append(ic_val)

    print(f"\n  {'Signal':12}  {'Mean IC':>10}  {'Median IC':>10}  {'% positive':>11}  {'n coins':>8}")
    for sig, ics in per_coin_ics.items():
        if not ics:
            continue
        ics = pd.Series(ics).dropna()
        from scipy import stats as scipy_stats
        t_stat, p_val = scipy_stats.ttest_1samp(ics, 0)
        stars = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.10 else ""))
        print(f"  {sig:12}  {ics.mean():>+9.4f}  {ics.median():>+9.4f}  "
              f"{(ics>0).mean()*100:>10.0f}%  {len(ics):>7}  {stars}")

    print(f"\n  Pooled IC for reference:")
    print(f"  {'Signal':12}  {'Pooled IC(7d)':>14}")
    for sig in ["oi_z", "funding_z", "basis"]:
        s = pool[[sig, "fwd7"]].dropna()
        print(f"  {sig:12}  {rank_ic(s[sig], s['fwd7']):>+13.4f}")

    # ── 3. Simulation A: original signal + oi_z filter ────────────────────────
    section("3. SIMULATION A — original z-cross signal, does oi_z filter help?")
    print("  Original: funding_z crosses below -0.5 + ATR compressed")
    print("  Filter variations: no filter / oi_z > 0 / oi_z > 0.5 / oi_z > 1.0")

    for label, use_filter, thresh in [
        ("No oi_z filter (baseline)",      False, 0.0),
        ("Filter: oi_z > 0.0  (above avg)",True, 0.0),
        ("Filter: oi_z > 0.5",              True, 0.5),
        ("Filter: oi_z > 1.0  (1 std)",     True, 1.0),
    ]:
        ts_list = []
        for sym, df in all_signals.items():
            t = simulate(df, use_oi_filter=use_filter, oi_thresh=thresh,
                         use_new_signal=False)
            if not t.empty:
                t["symbol"] = sym
                ts_list.append(t)
        all_t = pd.concat(ts_list, ignore_index=True) if ts_list else pd.DataFrame()
        print_sim(all_t, label)

    # ── 4. Simulation B: new oi_z wakeup signal ───────────────────────────────
    section("4. SIMULATION B — new signal: oi_z crosses above 0 + funding_z < 0")
    print("  Entry: OI rises above its 90-day avg (market waking up)")
    print("         AND funding is negative (shorts are dominant)")
    print("  Exit:  funding_z rises above 0 OR 21-day max hold")
    print("  Stop:  10-day low, min 5%\n")

    new_list = []
    for sym, df in all_signals.items():
        t = simulate(df, use_new_signal=True)
        if not t.empty:
            t["symbol"] = sym
            new_list.append(t)
    new_trades = pd.concat(new_list, ignore_index=True) if new_list else pd.DataFrame()
    print_sim(new_trades, "New signal: oi_z wakeup + funding negative")

    if not new_trades.empty:
        print(f"\n  Detail:")
        print(f"    Total trades: {len(new_trades)}")
        print(f"    Win rate: {(new_trades['ret_pct']>0).mean()*100:.1f}%")
        print(f"    Avg return: {new_trades['ret_pct'].mean():+.2f}%")
        print(f"    Median return: {new_trades['ret_pct'].median():+.2f}%")
        print(f"    Avg win: {new_trades[new_trades['ret_pct']>0]['ret_pct'].mean():+.2f}%  "
              f"| Avg loss: {new_trades[new_trades['ret_pct']<=0]['ret_pct'].mean():+.2f}%")
        print(f"    Avg hold: {new_trades['hold'].mean():.1f} days")
        print(f"    Exits: stop={( new_trades['exit_reason']=='stop').mean()*100:.0f}%  "
              f"z_exit={(new_trades['exit_reason']=='z_exit').mean()*100:.0f}%  "
              f"time={(new_trades['exit_reason']=='time').mean()*100:.0f}%")
        print(f"    Avg oi_z at entry: {new_trades['oi_z'].mean():+.2f}")
        print(f"    Avg funding_z at entry: {new_trades['fz'].mean():+.2f}")

        # Year-by-year
        new_trades["year"] = pd.to_datetime(new_trades["date"]).dt.year
        print(f"\n  Year-by-year:")
        print(f"  {'Year':>6}  {'n':>4}  {'wr%':>5}  {'avg%':>7}  {'med%':>7}")
        for yr, grp in new_trades.groupby("year"):
            print(f"  {yr:>6}  {len(grp):>4}  {(grp['ret_pct']>0).mean()*100:>4.0f}%  "
                  f"{grp['ret_pct'].mean():>+6.2f}%  {grp['ret_pct'].median():>+6.2f}%")

        # Walk-forward (6-month windows)
        new_trades["date"] = pd.to_datetime(new_trades["date"])
        new_trades = new_trades.sort_values("date")
        print(f"\n  Walk-forward (6-month windows):")
        print(f"  {'Period':>10}  {'n':>4}  {'wr%':>5}  {'avg%':>7}  {'med%':>7}")
        new_trades["half"] = new_trades["date"].dt.to_period("2Q")
        for p, grp in new_trades.groupby("half"):
            if len(grp) < 3:
                continue
            flag = " <<" if grp['ret_pct'].mean() > 2 else (" >>" if grp['ret_pct'].mean() < -1 else "")
            print(f"  {str(p):>10}  {len(grp):>4}  {(grp['ret_pct']>0).mean()*100:>4.0f}%  "
                  f"{grp['ret_pct'].mean():>+6.2f}%  {grp['ret_pct'].median():>+6.2f}%{flag}")

    # ── 5. What we learned ────────────────────────────────────────────────────
    section("5. WHAT WE LEARNED")
    print("""
  The cross-tab tells us which combination of OI state and funding state
  has the highest forward returns. If the best cell is:
    HIGH oi_z + LOW funding_z: "active coin, shorts trapped" hypothesis confirmed
    HIGH oi_z + HIGH funding_z: momentum hypothesis (trend following)
    LOW oi_z + LOW funding_z:  mean-reversion in dead coins (less useful)

  Per-coin IC tells us:
    If mean per-coin IC is positive and significant: oi_z is genuinely time-series
    (within each coin, OI level predicts its own future returns)
    If per-coin IC is near zero: pooled effect was driven by cross-section
    (just capturing that active coins have better returns on average)

  Simulation A tells us:
    Does requiring oi_z > 0 at entry filter out the bad trades?
    If yes: the edge from our original signal improves with the oi_z filter.

  Simulation B tells us:
    Is the "market waking up" moment (oi_z crossing above 0) a tradeable entry?
    If this produces positive average returns with acceptable win rate:
    we have a completely new signal that doesn't depend on ATR compression.
    """)

    print("Done.")
