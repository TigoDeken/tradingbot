"""
Portfolio simulation — N concurrent positions across 78 coins.
Frozen spec: z-cross + ATR + oi_z > 1.0 + signal exit.

Shows N=1 (sequential) vs N=2,3,5 concurrent.
Builds the infrastructure for when 4H signals and short-side are added.

Run: python -m algo.data.portfolio_sim
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from algo.data.zscore_entry_study import load_all
from algo.data.perp_flow_study import load_perp_oi, load_perp_kline
from algo.data.oi_flow_study import build_full_signals
from algo.data.final_spec_study import simulate

START_EQUITY = 5_000
RISK_FRAC    = 0.01
MIN_OBS      = 90


def collect_trades(all_dfs: dict) -> pd.DataFrame:
    parts = []
    for sym, df in sorted(all_dfs.items()):
        t = simulate(df, use_oi_filter=True, oi_thresh=1.0, use_signal_exit=True)
        if t.empty:
            continue
        t["symbol"]     = sym
        t["entry_date"] = pd.to_datetime(t["entry_date"])
        t["exit_date"]  = t["entry_date"] + t["hold"].apply(lambda h: timedelta(days=int(h)))
        t["ret_r"]      = t["ret_pct"] / t["risk_pct"]
        t["score"]      = t["fz"].abs() * 0.6 + (t["oi_z"] - 1.0) * 0.4
        parts.append(t[["symbol","entry_date","exit_date","ret_r","ret_pct","risk_pct","score","fz","oi_z"]])
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values("entry_date").reset_index(drop=True)


def run_portfolio(trades: pd.DataFrame, max_pos: int) -> tuple:
    equity    = START_EQUITY
    open_pos  = []   # {exit_date, ret_r, risk_$, symbol}
    events    = []   # (date, equity, type, symbol)
    taken     = 0
    skipped   = 0

    for _, row in trades.iterrows():
        # Close expired positions
        still = []
        for p in open_pos:
            if p["exit_date"] <= row["entry_date"]:
                equity    += p["risk_usd"] * p["ret_r"]
                events.append((p["exit_date"], equity, "close", p["symbol"]))
            else:
                still.append(p)
        open_pos = still

        # Open if slot available
        if len(open_pos) < max_pos:
            risk_usd = equity * RISK_FRAC
            open_pos.append({"exit_date": row["exit_date"], "ret_r": row["ret_r"],
                              "risk_usd": risk_usd, "symbol": row["symbol"]})
            events.append((row["entry_date"], equity, "open", row["symbol"]))
            taken += 1
        else:
            skipped += 1

    # Close remaining
    for p in sorted(open_pos, key=lambda x: x["exit_date"]):
        equity += p["risk_usd"] * p["ret_r"]
        events.append((p["exit_date"], equity, "close", p["symbol"]))

    return equity, events, taken, skipped


def max_dd(events):
    closes = [(e[0], e[1]) for e in events if e[2] == "close"]
    if len(closes) < 2:
        return 0
    eq = pd.Series([c[1] for c in closes])
    peak = eq.cummax()
    return ((eq - peak) / peak).min()


def section(t):
    print(f"\n{'='*65}\n{t}\n{'='*65}")


if __name__ == "__main__":
    print("Loading data...")
    coins = load_all()

    all_dfs = {}
    for sym, spot_df in sorted(coins.items()):
        oi_df   = load_perp_oi(sym)
        perp_df = load_perp_kline(sym)
        if oi_df.empty or perp_df.empty:
            continue
        sig = build_full_signals(spot_df, oi_df, perp_df)
        if len(sig) >= MIN_OBS:
            all_dfs[sym] = sig

    print("Collecting all trades...")
    trades = collect_trades(all_dfs)
    print(f"Total signal events: {len(trades)}")

    span_days = (trades["exit_date"].max() - trades["entry_date"].min()).days
    years     = span_days / 365

    # ── Overlap analysis ──────────────────────────────────────────────────────
    section("HOW MUCH DO TRADES OVERLAP?")
    # Count how many positions would be open on each day
    all_dates = pd.date_range(trades["entry_date"].min(), trades["exit_date"].max(), freq="D")
    daily_open = []
    for d in all_dates:
        n = ((trades["entry_date"] <= d) & (trades["exit_date"] > d)).sum()
        daily_open.append(n)
    daily_open = pd.Series(daily_open)
    print(f"""
  Total signal events: {len(trades)}
  Period: {trades['entry_date'].min().date()} to {trades['exit_date'].max().date()}

  Simultaneous open positions (if no limit):
    Max concurrent:    {daily_open.max():.0f}
    Average:           {daily_open.mean():.2f}
    Days with 0 open:  {(daily_open==0).sum()} ({(daily_open==0).mean()*100:.0f}%)
    Days with 1 open:  {(daily_open==1).sum()} ({(daily_open==1).mean()*100:.0f}%)
    Days with 2+ open: {(daily_open>=2).sum()} ({(daily_open>=2).mean()*100:.0f}%)
    Days with 3+ open: {(daily_open>=3).sum()} ({(daily_open>=3).mean()*100:.0f}%)
""")

    # ── N comparison ──────────────────────────────────────────────────────────
    section("EQUITY COMPARISON: N=1 vs N=2,3,5 CONCURRENT POSITIONS")
    print(f"\n  {'N':>4}  {'Taken':>6}  {'Skip':>5}  {'Final $':>9}  "
          f"{'Return':>8}  {'CAGR':>7}  {'Max DD':>8}")
    print(f"  {'-'*60}")

    all_results = {}
    for n in [1, 2, 3, 5]:
        eq, events, taken, skipped = run_portfolio(trades, n)
        ret  = (eq / START_EQUITY - 1) * 100
        cagr = ((eq / START_EQUITY) ** (1 / years) - 1) * 100
        mdd  = max_dd(events) * 100
        all_results[n] = dict(eq=eq, events=events, taken=taken, skipped=skipped,
                               ret=ret, cagr=cagr, mdd=mdd)
        print(f"  {n:>4}  {taken:>6}  {skipped:>5}  ${eq:>8,.0f}  "
              f"{ret:>+7.1f}%  {cagr:>+6.1f}%  {mdd:>+7.1f}%")

    # ── Quarterly equity for N=5 ──────────────────────────────────────────────
    section("QUARTERLY EQUITY — N=5 CONCURRENT")
    events5 = all_results[5]["events"]
    eq_df = pd.DataFrame(events5, columns=["date","equity","type","symbol"])
    eq_df["date"] = pd.to_datetime(eq_df["date"])
    eq_df = eq_df[eq_df["type"] == "close"].sort_values("date")
    eq_df["quarter"] = eq_df["date"].dt.to_period("Q")

    print(f"\n  {'Quarter':>8}  {'Equity':>9}  {'Gain':>8}")
    prev = START_EQUITY
    for q, grp in eq_df.groupby("quarter"):
        end_eq = grp["equity"].iloc[-1]
        print(f"  {str(q):>8}  ${end_eq:>8,.0f}  ${end_eq-prev:>+7,.0f}")
        prev = end_eq

    # ── What does this look like with 4x more trades? ─────────────────────────
    section("PROJECTION: WHAT 4H SIGNAL WOULD DO (4x trades, same edge)")
    print("""
  With 4H bars instead of daily, same signal, each daily bar becomes 6 opportunities.
  Conservative estimate: 3x current signal count = ~250 trades/year.

  Below projects using the SAME trade P&L distribution, re-sampled 3x.
  This is illustrative only — actual 4H backtest will give real numbers.
""")

    rng = np.random.default_rng(42)
    ret_r_dist = trades["ret_r"].values

    for multiplier, label in [(3, "3x trades (4H signal, conservative)"),
                               (5, "5x trades (4H + short side)")]:
        # Bootstrap: resample existing ret_r with replacement
        n_sim = len(trades) * multiplier
        boot_r = rng.choice(ret_r_dist, size=n_sim, replace=True)

        # Simulate equity with compounding, N=5 concurrent
        # For simplicity: sequential (since at 3x we still have avg ~0.5 concurrent)
        eq = START_EQUITY
        for r in boot_r:
            eq = eq * (1 + RISK_FRAC * r)
        ret  = (eq / START_EQUITY - 1) * 100
        cagr = ((eq / START_EQUITY) ** (1 / years) - 1) * 100

        # Max DD estimate via running peak
        eq_curve = [START_EQUITY]
        for r in boot_r:
            eq_curve.append(eq_curve[-1] * (1 + RISK_FRAC * r))
        eq_s = pd.Series(eq_curve)
        mdd  = ((eq_s - eq_s.cummax()) / eq_s.cummax()).min() * 100

        print(f"  {label}:")
        print(f"    Final: ${eq:,.0f}  Return: {ret:+.0f}%  CAGR: {cagr:+.0f}%  Max DD: {mdd:.1f}%\n")

    section("NEXT STEPS TO REALIZE THIS POTENTIAL")
    print("""
  The concurrent positions improvement is modest with 82 trades (low overlap).
  The equity jump comes from MORE trades, not just concurrent slots.

  What unlocks higher trade frequency:
    1. 4H signal study  -> 3-5x more trade opportunities per coin
    2. Short side       -> mirror strategy on dead/oversold coins (if borrow available)
    3. L/S ratio signal -> additional filter that fires more frequently
    4. OI event signal  -> large OI drops as reversal entries

  Run these studies in order:
    python -m algo.data.signal_4h_fetch     (fetch 4H data)
    python -m algo.data.signal_4h_study     (4H signal backtest)
    python -m algo.data.borrow_check        (check short side viability)
    python -m algo.data.oi_event_study      (OI collapse signal)
    python -m algo.data.ls_ratio_study      (long/short ratio signal)
""")
    print("Done.")
