"""
Equity simulation: $5,000 account, 1% risk per trade.
Frozen spec: z-cross + ATR + oi_z > 1.0 + signal exit.

Shows: equity curve, max drawdown, streak stats, CAGR, time to recover.

Run: python -m algo.data.equity_sim
"""

import numpy as np
import pandas as pd
from algo.data.zscore_entry_study import load_all
from algo.data.perp_flow_study import load_perp_oi, load_perp_kline
from algo.data.oi_flow_study import build_full_signals
from algo.data.final_spec_study import simulate

START_EQUITY = 5_000
RISK_FRAC    = 0.01     # 1% of current equity per trade
MIN_OBS      = 90


def max_drawdown(eq: pd.Series):
    peak  = eq.cummax()
    dd    = (eq - peak) / peak
    mdd   = dd.min()
    end   = dd.idxmin()
    start = eq[:end].idxmax()
    # Recovery: first bar after end where equity >= peak at start
    peak_val = eq[start]
    recovery_rows = eq[end:][eq[end:] >= peak_val]
    recovery = recovery_rows.index[0] if len(recovery_rows) else None
    return mdd, start, end, recovery


if __name__ == "__main__":
    print("Loading data and re-running frozen simulation...")
    coins = load_all()

    all_trades = []
    for sym, spot_df in sorted(coins.items()):
        oi_df   = load_perp_oi(sym)
        perp_df = load_perp_kline(sym)
        if oi_df.empty or perp_df.empty:
            continue
        sig = build_full_signals(spot_df, oi_df, perp_df)
        if len(sig) >= MIN_OBS:
            t = simulate(sig, use_oi_filter=True, oi_thresh=1.0, use_signal_exit=True)
            if not t.empty:
                t["symbol"] = sym
                all_trades.append(t)

    trades = pd.concat(all_trades, ignore_index=True)
    trades["date"] = pd.to_datetime(trades["date"])
    trades = trades.sort_values("date").reset_index(drop=True)
    trades["ret_r"] = trades["ret_pct"] / trades["risk_pct"]

    print(f"Total trades: {len(trades)}")
    print(f"Period: {trades['date'].min().date()} to {trades['date'].max().date()}")

    # ── Equity curve ──────────────────────────────────────────────────────────
    equity     = START_EQUITY
    curve      = []
    for _, row in trades.iterrows():
        pnl    = equity * RISK_FRAC * row["ret_r"]
        equity += pnl
        curve.append({
            "idx":     len(curve),
            "date":    row["date"],
            "symbol":  row["symbol"],
            "ret_r":   row["ret_r"],
            "ret_pct": row["ret_pct"],
            "risk_pct":row["risk_pct"],
            "pnl":     pnl,
            "equity":  equity,
        })

    eq = pd.DataFrame(curve)
    eq["date"] = pd.to_datetime(eq["date"])

    # ── Max drawdown ──────────────────────────────────────────────────────────
    mdd, dd_start, dd_end, dd_recovery = max_drawdown(eq["equity"])

    # Time from first trade to last trade (calendar days)
    span_days = (trades["date"].max() - trades["date"].min()).days
    years     = span_days / 365

    total_ret  = equity / START_EQUITY - 1
    cagr       = (equity / START_EQUITY) ** (1 / years) - 1 if years > 0 else 0

    # Per-trade Sharpe (risk-adjusted, using % equity impact)
    eq["ret_eq_pct"] = eq["pnl"] / (eq["equity"] - eq["pnl"]) * 100
    sharpe_per_trade = eq["ret_eq_pct"].mean() / eq["ret_eq_pct"].std()

    # Streaks
    wins_losses = (eq["ret_r"] > 0).astype(int)
    max_win_streak  = max((sum(1 for _ in g) for k, g in
                           __import__("itertools").groupby(wins_losses) if k == 1), default=0)
    max_loss_streak = max((sum(1 for _ in g) for k, g in
                           __import__("itertools").groupby(wins_losses) if k == 0), default=0)

    # ── Output ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EQUITY SIMULATION  |  $5,000 account  |  1% risk/trade")
    print(f"{'='*60}")
    print(f"""
  Starting equity:   ${START_EQUITY:>8,.0f}
  Final equity:      ${equity:>8,.0f}
  Net profit:        ${equity - START_EQUITY:>+8,.0f}
  Total return:      {total_ret*100:>+7.1f}%
  CAGR:              {cagr*100:>+7.1f}%  (annualised, {years:.1f} years of data)

  Trades:            {len(trades)}
  Win rate:          {(eq['ret_r']>0).mean()*100:.1f}%
  Avg R per trade:   {eq['ret_r'].mean():+.3f}R
  Avg $ per trade:   ${eq['pnl'].mean():>+.2f}
  Sharpe (trades):   {sharpe_per_trade:+.3f}

  Max drawdown:      {mdd*100:.1f}%   (${abs(mdd) * eq.loc[dd_start,'equity']:,.0f})
  DD peak:           {eq.loc[dd_start,'date'].date()}
  DD trough:         {eq.loc[dd_end,'date'].date()}
  DD recovery:       {'not yet' if dd_recovery is None else str(eq.loc[dd_recovery,'date'].date())}

  Max win streak:    {max_win_streak} trades in a row
  Max loss streak:   {max_loss_streak} trades in a row
""")

    # ── Equity curve: quarterly snapshots ─────────────────────────────────────
    print(f"{'='*60}")
    print("EQUITY BY QUARTER")
    print(f"{'='*60}")
    eq["quarter"] = eq["date"].dt.to_period("Q")
    print(f"\n  {'Quarter':>8}  {'Equity':>9}  {'Gain':>8}  {'n':>4}  {'wr%':>5}  {'avg_R':>7}")
    prev_eq = START_EQUITY
    for q, grp in eq.groupby("quarter"):
        end_eq = grp["equity"].iloc[-1]
        gain   = end_eq - prev_eq
        wr     = (grp["ret_r"] > 0).mean()
        avg_r  = grp["ret_r"].mean()
        flag   = " <<" if gain > 200 else (" >>" if gain < -100 else "")
        print(f"  {str(q):>8}  ${end_eq:>8,.0f}  ${gain:>+7,.0f}  "
              f"{len(grp):>4}  {wr*100:>4.0f}%  {avg_r:>+6.3f}R{flag}")
        prev_eq = end_eq

    # ── Biggest trades ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TOP 5 BEST AND WORST TRADES")
    print(f"{'='*60}")
    print(f"\n  {'Symbol':18} {'Date':>12} {'R':>7} {'P&L':>8}  {'ret%':>7}  {'stop%':>7}")
    best5  = eq.nlargest(5, "pnl")
    worst5 = eq.nsmallest(5, "pnl")
    print("  Best:")
    for _, r in best5.iterrows():
        print(f"  {r['symbol']:18} {str(r['date'].date()):>12} "
              f"{r['ret_r']:>+6.2f}R  ${r['pnl']:>+7.0f}  "
              f"{r['ret_pct']:>+6.1f}%  {r['risk_pct']:>6.1f}%")
    print("  Worst:")
    for _, r in worst5.iterrows():
        print(f"  {r['symbol']:18} {str(r['date'].date()):>12} "
              f"{r['ret_r']:>+6.2f}R  ${r['pnl']:>+7.0f}  "
              f"{r['ret_pct']:>+6.1f}%  {r['risk_pct']:>6.1f}%")

    # ── Without the MAVIAUSDT outlier ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SENSITIVITY: REMOVE LARGEST OUTLIER")
    print(f"{'='*60}")
    no_outlier = eq[eq["ret_r"] < eq["ret_r"].quantile(0.99)]
    e2 = START_EQUITY
    for _, row in no_outlier.iterrows():
        e2 += e2 * RISK_FRAC * row["ret_r"]
    print(f"\n  With all trades:      ${equity:,.0f}  ({total_ret*100:+.1f}%)")
    print(f"  Remove top 1% (R):    ${e2:,.0f}  ({(e2/START_EQUITY-1)*100:+.1f}%)")
    print(f"  Removed {len(eq) - len(no_outlier)} trade(s) above {eq['ret_r'].quantile(0.99):.1f}R")

    # ── Time in market ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TRADE FREQUENCY & TIME IN MARKET")
    print(f"{'='*60}")
    avg_gap = span_days / len(trades)
    avg_hold_days = trades["hold"].mean()
    time_in_mkt = avg_hold_days / avg_gap * 100
    print(f"""
  Period covered:     {span_days} calendar days
  Total trades:       {len(trades)}
  Avg time between:   {avg_gap:.0f} days
  Avg hold:           {avg_hold_days:.1f} days
  Time in market:     ~{time_in_mkt:.0f}% of days you hold a position
  Capital deployed:   ~{time_in_mkt:.0f}% of the time (rest is idle)
""")

    print("Done.")
