"""
XAU/USD N-bar range axiom study.
Usage: python main.py
Output: results/ directory
"""

import json
import os
import sys
import numpy as np
import pandas as pd

# ── path bootstrap so sub-packages import correctly ──────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from data.loader       import connect, disconnect, fetch_ohlcv
from data.range_builder import build_ranges
from data.features      import build_features
from axioms.range_width     import run_axiom3
from axioms.position_deciles import run_axiom4
from axioms.time_of_day      import run_axiom6
from axioms.bar_structure    import run_axiom8
from analysis.scoring     import run_scoring
from analysis.walkforward import split_is_oos, validate_condition, compare_is_oos
from reporting.output     import save_all_results

# Approximate M5 bars per year for XAUUSD (23h/day × 12 bars/h × 252 days)
BARS_PER_YEAR = {
    "M5":  69_552,
    "M15": 23_184,
    "H1":   5_796,
}


def build_all_features(df_raw, cfg):
    """Range features + bar structure features on a single slice."""
    df = build_ranges(df_raw, cfg["N_values"], cfg["atr_period"], cfg["normal_spread_pips"])
    df = build_features(df)
    return df


def _bars_per_year_from_data(df):
    """Estimate bars/year from actual date range."""
    days = max((df.index[-1] - df.index[0]).days, 1)
    return len(df) / (days / 365.25)


def _add_oos_outcomes(df_oos, N_values, ext):
    """
    Add per-condition RR and reversion columns to df_oos in-place so that
    validate_condition can look them up by column name.
    """
    atr_safe = df_oos["pre_atr"].replace(0, np.nan)
    for N in N_values:
        bp_col = f"bp_{N}"
        rm_col = f"rm_{N}"
        if bp_col not in df_oos.columns:
            continue
        top_m = df_oos[bp_col] > (1.0 - ext)
        bot_m = df_oos[bp_col] < ext

        rr_top = (df_oos["Close"] - df_oos["fwd_close_10"]) / atr_safe - df_oos["spread_rt_atr"]
        rr_bot = (df_oos["fwd_close_10"] - df_oos["Close"]) / atr_safe - df_oos["spread_rt_atr"]
        rv_top = (df_oos["fwd_min_l_10"] <= df_oos[rm_col]).astype(float)
        rv_bot = (df_oos["fwd_max_h_10"] >= df_oos[rm_col]).astype(float)

        df_oos[f"net_rr_{N}"] = np.where(top_m, rr_top, np.where(bot_m, rr_bot, np.nan))
        df_oos[f"rev_{N}"]    = np.where(top_m, rv_top, np.where(bot_m, rv_bot, np.nan))


def run_walkforward_validation(a3, a4, a6, a8, sc, df_oos, cfg):
    """
    For each confirmed IS condition, validate on OOS data.
    Returns dict: {condition_key: compared_result}
    """
    ext   = cfg["position_extreme_threshold"]
    _add_oos_outcomes(df_oos, cfg["N_values"], ext)
    wf    = {}

    # Axiom 3 top conditions
    for N, bucket, setup in a3.get("top_conditions", []):
        p33, p66 = a3["percentiles"][N]
        bp_col   = f"bp_{N}"
        rw_col   = f"rw_atr_{N}"
        rm_col   = f"rm_{N}"

        def mask_fn(d, N=N, bucket=bucket, setup=setup, p33=p33, p66=p66):
            bp   = d[bp_col]
            rw   = d[rw_col]
            ext_mask = (bp > (1.0 - ext)) if setup == "top" else (bp < ext)
            if bucket == "narrow":
                bkt_mask = rw < p33
            elif bucket == "medium":
                bkt_mask = (rw >= p33) & (rw < p66)
            else:
                bkt_mask = rw >= p66
            return ext_mask & bkt_mask

        is_row  = next((r for r in a3["rows"] if r.get("N") == N and r.get("bucket") == bucket
                        and r.get("setup") == setup), {})
        oos_res = validate_condition(df_oos, mask_fn, f"net_rr_{N}", f"rev_{N}",
                                     cfg.get("min_sample_size", 50) // 2)
        key = f"axiom3_N{N}_{bucket}_{setup}"
        wf[key] = compare_is_oos(is_row, oos_res, cfg)

    # Axiom 6 top conditions
    for N, split_type, bucket_val, setup in a6.get("top_conditions", []):
        bp_col = f"bp_{N}"

        def mask_fn(d, N=N, split_type=split_type, bucket_val=bucket_val, setup=setup):
            bp   = d[bp_col]
            ext_mask = (bp > (1.0 - ext)) if setup == "top" else (bp < ext)
            if split_type == "hour":
                time_mask = d["hour_utc"] == bucket_val
            elif split_type == "session":
                time_mask = d["session"] == bucket_val
            else:
                time_mask = d["dow"] == bucket_val
            return ext_mask & time_mask

        is_row  = next((r for r in a6.get("rows_filtered", [])
                        if r.get("N") == N and r.get("split_type") == split_type
                        and r.get("bucket") == bucket_val and r.get("setup") == setup), {})
        oos_res = validate_condition(df_oos, mask_fn, f"net_rr_{N}", f"rev_{N}",
                                     cfg.get("min_sample_size", 50) // 2)
        key = f"axiom6_N{N}_{split_type}_{bucket_val}_{setup}"
        wf[key] = compare_is_oos(is_row, oos_res, cfg)

    # Axiom 8 direction conditions
    for N, setup_type, bar_dir in a8.get("top_conditions", []):
        bp_col = f"bp_{N}"
        dir_map = {"bearish": -1, "bullish": 1, "doji": 0}
        dir_val = dir_map.get(bar_dir, 0)

        def mask_fn(d, N=N, setup_type=setup_type, dir_val=dir_val):
            bp   = d[bp_col]
            ext_mask = (bp > (1.0 - ext)) if setup_type == "top" else (bp < ext)
            bd_mask  = d.get("bar_direction", pd.Series(0, index=d.index)) == dir_val
            return ext_mask & bd_mask

        is_row  = next((r for r in a8.get("direction_rows", [])
                        if r.get("N") == N and r.get("setup") == setup_type
                        and r.get("bar_dir") == bar_dir), {})
        oos_res = validate_condition(df_oos, mask_fn, f"net_rr_{N}", f"rev_{N}",
                                     cfg.get("min_sample_size", 50) // 2)
        key = f"axiom8_N{N}_{setup_type}_{bar_dir}"
        wf[key] = compare_is_oos(is_row, oos_res, cfg)

    return wf


def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    os.makedirs("results", exist_ok=True)

    print("Connecting to MT5...")
    if not connect():
        sys.exit(1)

    all_results = {}

    try:
        for tf in cfg["timeframes"]:
            print(f"\n{'='*60}")
            print(f"Timeframe: {tf}")
            print(f"{'='*60}")

            df_raw, actual_sym = fetch_ohlcv(
                cfg["symbol"], tf, 200_000,
                cfg.get("symbol_fallbacks", []),
            )

            # Estimate bars/year from actual data
            bpy = _bars_per_year_from_data(df_raw)
            print(f"  bars/year (estimated): {bpy:,.0f}")

            # Build all features on full dataset (warmup is minimal vs. total bars)
            print("  Building features...")
            df_full = build_all_features(df_raw, cfg)

            # Drop warmup rows (NaN pre_atr or range features)
            warmup_col = f"rw_atr_{max(cfg['N_values'])}"
            df_full = df_full.dropna(subset=["pre_atr", warmup_col])

            # Chronological IS / OOS split
            n_is  = int(len(df_full) * cfg["train_test_split"])
            df_is  = df_full.iloc[:n_is].copy()
            df_oos = df_full.iloc[n_is:].copy()
            print(f"  IS : {df_is.index[0].date()} to {df_is.index[-1].date()} ({len(df_is):,} bars)")
            print(f"  OOS: {df_oos.index[0].date()} to {df_oos.index[-1].date()} ({len(df_oos):,} bars)")

            # ── Run axiom analyses on IS ──────────────────────────────────────
            print("  Running Axiom 3 (range width)...")
            a3 = run_axiom3(df_is, cfg)

            print("  Running Axiom 4 (position deciles)...")
            a4 = run_axiom4(df_is, cfg, bars_per_year=bpy)

            print("  Running Axiom 6 (time of day)...")
            a6 = run_axiom6(df_is, cfg, bars_per_year=bpy)

            print("  Running Axiom 8 (bar structure)...")
            a8 = run_axiom8(df_is, cfg)

            print("  Running combined scoring...")
            sc = run_scoring(df_is, cfg, a3_percentiles=a3.get("percentiles"), bars_per_year=bpy)

            # ── Walk-forward validation ───────────────────────────────────────
            print("  Running walk-forward validation...")
            wf = run_walkforward_validation(a3, a4, a6, a8, sc, df_oos, cfg)

            # Print OOS validation results
            n_confirmed_is  = (
                sum(1 for r in a3.get("rows", [])          if r.get("confirmed"))
                + sum(1 for r in a4.get("rows", [])        if r.get("confirmed"))
                + sum(1 for r in a8.get("direction_rows",[])if r.get("confirmed"))
                + sum(1 for r in sc.get("rows", [])        if r.get("confirmed"))
            )

            n_passed_oos = sum(
                1 for v in wf.values()
                if (
                    not np.isnan(v.get("oos_rev_rate", np.nan) or np.nan)
                    and (v.get("oos_rev_rate", 0) or 0) > cfg["edge_reversion_rate_outsample"]
                    and (v.get("oos_mean_rr",  0) or 0) > cfg["edge_min_rr"]
                    and v.get("overfit_flag", "") not in ("LIKELY_OVERFIT", "INSUFFICIENT_OOS")
                )
            )
            print(f"  IS confirmed: {n_confirmed_is}  |  OOS validated: {n_passed_oos}/{len(wf)}")

            if wf:
                print("  Walk-forward summary:")
                for key, v in wf.items():
                    flag = v.get("overfit_flag", "")
                    print(
                        f"    {key}: IS_rev={_fmt(v.get('rev_rate'),'0.3f')}"
                        f" OOS_rev={_fmt(v.get('oos_rev_rate'),'0.3f')}"
                        f" OOS_n={v.get('oos_n','?')}  {flag}"
                    )

            all_results[tf] = {
                "n_is":  len(df_is),
                "n_oos": len(df_oos),
                "axiom3":     a3,
                "axiom4":     a4,
                "axiom6":     a6,
                "axiom8":     a8,
                "scoring":    sc,
                "walkforward": wf,
            }

    finally:
        disconnect()
        print("\nDisconnected from MT5.")

    save_all_results(all_results, cfg)


def _fmt(v, spec):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


if __name__ == "__main__":
    main()
