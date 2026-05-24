"""
GOLD# regime study — M5 only.

Signal : N=8 down-break (no ATR filter, no session filter).
Sessions tested : Asia (00-08h), London (08-13h), NY (16-21h).
Regime features : LinReg R² (20 bars), ATR ratio (ATR14 / SMA50).
Directions      : LONG (reversal) and SHORT (continuation).
Horizons        : 30m (6 bars), 1h (12 bars), 3h (36 bars).
Stop grid       : 0.5 / 1.0 / 1.5 ATR.
RR grid         : 1.0 / 1.5 / 2.0 / 3.0.

Usage: cd research_xauusd && python run_regime.py
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from data.loader          import connect, disconnect, fetch_ohlcv
from analysis.walkforward import split_is_oos

from regime.features  import build_regime_features
from regime.signals   import detect_signals, add_session
from regime.sim       import run_simulation
from regime.analysis  import run_analysis, top_conditions
from regime.output    import save_results

SYMBOL    = "GOLD#"
TIMEFRAME = "M5"
N_BARS    = 200_000
OUT_DIR   = "results/regime"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Connecting to MT5...")
    if not connect():
        sys.exit(1)

    try:
        print(f"\nFetching {SYMBOL} {TIMEFRAME}...")
        df_raw, _ = fetch_ohlcv(SYMBOL, TIMEFRAME, N_BARS, fallbacks=["XAUUSD"])

        # Build regime features on full dataset
        print("Building regime features...")
        df_feat = build_regime_features(df_raw)
        df_feat = add_session(df_feat)

        # Drop warmup rows
        df_feat = df_feat.dropna(subset=["pre_atr", "linreg_r2", "atr_ratio"])
        print(f"  Valid bars after warmup: {len(df_feat):,}")

        # IS / OOS split (70/30 chronological)
        n_is      = int(len(df_feat) * 0.70)
        df_is     = df_feat.iloc[:n_is].copy()
        df_oos    = df_feat.iloc[n_is:].copy()
        print(f"  IS : {df_is.index[0].date()} to {df_is.index[-1].date()} ({len(df_is):,} bars)")
        print(f"  OOS: {df_oos.index[0].date()} to {df_oos.index[-1].date()} ({len(df_oos):,} bars)")

        # Detect signals
        sig_is  = detect_signals(df_is)
        sig_oos = detect_signals(df_oos)
        print(f"  IS signals : {sig_is.sum():,}  |  OOS signals: {sig_oos.sum():,}")

        # Simulate trades — IS
        print("Running IS simulation...")
        is_rows  = run_simulation(df_is,  sig_is)
        is_sim   = pd.DataFrame(is_rows)
        print(f"  IS simulation rows: {len(is_sim):,}")

        # Simulate trades — OOS
        print("Running OOS simulation...")
        oos_rows = run_simulation(df_oos, sig_oos)
        oos_sim  = pd.DataFrame(oos_rows)
        print(f"  OOS simulation rows: {len(oos_sim):,}")

        if is_sim.empty:
            print("No IS simulation data — check signal detection.")
            return

        # Analyse IS
        print("Analysing IS regime buckets...")
        is_result, thresholds = run_analysis(is_sim, df_is)
        is_top = top_conditions(is_result)
        print(f"  IS conditions meeting >=4% WR lift: {len(is_top)}")

        # Analyse OOS using IS thresholds
        print("Validating OOS...")
        oos_result, _ = run_analysis(oos_sim, df_oos, is_percentiles=thresholds)
        oos_top = top_conditions(oos_result)

        # Print IS R² lift summary for quick read
        _print_r2_summary(is_result)

        # Save everything
        save_results(
            is_result, oos_result, is_top, oos_top,
            thresholds, len(df_is), len(df_oos), OUT_DIR,
        )

    finally:
        disconnect()
        print("\nDone.")


def _print_r2_summary(result_df):
    """Quick console table: R² buckets at 1h / 1.0 ATR stop / 2.0 RR."""
    sub = result_df[
        (result_df["horizon"]   == "1h")
        & (result_df["stop_mult"] == 1.0)
        & (result_df["rr"]        == 2.0)
        & (result_df["regime_type"].isin(["baseline", "r2"]))
        & (result_df["session"]   == "all")
    ].copy()

    if sub.empty:
        return

    print("\n  --- R² regime lift (1h, 1.0 ATR stop, 2.0 RR) ---")
    print(f"  {'direction':<8} {'bucket':<16} {'n':>6} {'WR%':>7} {'lift%':>7} {'edge_r':>8}")
    for _, r in sub.sort_values(["direction", "regime_value"]).iterrows():
        bucket = r["regime_value"] if r["regime_type"] != "baseline" else "baseline"
        wr     = (r["win_rate"] or 0) * 100
        lift   = (r["wr_lift"]  or 0) * 100
        er     = r["edge_ratio"]
        er_str = f"{er:.2f}" if er is not None and not np.isnan(er) else "—"
        print(f"  {r['direction']:<8} {str(bucket):<16} {int(r['n']):>6}"
              f" {wr:>6.1f}% {lift:>+6.1f}% {er_str:>8}")


if __name__ == "__main__":
    main()
