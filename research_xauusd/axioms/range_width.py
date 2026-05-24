"""
Axiom 3: Does N-bar range width predict reversion probability or RR?

Splits setups at range extremes (bar_position > 0.9 or < 0.1) into
narrow / medium / wide range-width buckets and measures:
  - reversion rate (price reaches range midpoint within 10 bars)
  - continuation rate (price extends beyond the broken extreme)
  - mean net RR (fade trade in ATR units, net of spread)

Also mirrors the analysis without the extreme filter to let data decide
whether wide ranges favour continuation.
"""

import numpy as np
import pandas as pd

from analysis.statistics import bh_correction, summarise_group


def _bucket_label(val, p33, p66):
    if np.isnan(val):
        return np.nan
    if val < p33:
        return "narrow"
    if val < p66:
        return "medium"
    return "wide"


def _net_rr_top(df_sub, N):
    """Short fade at a top setup: (Close - fwd_close_10) / pre_atr - spread_rt_atr."""
    c   = df_sub["Close"]
    f10 = df_sub["fwd_close_10"]
    atr = df_sub["pre_atr"].replace(0, np.nan)
    spr = df_sub["spread_rt_atr"]
    return (c - f10) / atr - spr


def _net_rr_bot(df_sub, N):
    """Long fade at a bottom setup: (fwd_close_10 - Close) / pre_atr - spread_rt_atr."""
    c   = df_sub["Close"]
    f10 = df_sub["fwd_close_10"]
    atr = df_sub["pre_atr"].replace(0, np.nan)
    spr = df_sub["spread_rt_atr"]
    return (f10 - c) / atr - spr


def run_axiom3(df, cfg, is_percentiles=None):
    """
    Run Axiom 3 analysis on df.

    is_percentiles: {N: (p33, p66)} computed on in-sample data.
    If None, computes from df itself (use only for IS pass).

    Returns:
      rows — list of result dicts (one per N × bucket × setup_type)
      top_conditions — list of (N, bucket, setup_type) that pass IS thresholds
      percentiles — {N: (p33, p66)} for passing to OOS validation
    """
    ext   = cfg["position_extreme_threshold"]          # 0.10
    n_min = cfg["min_sample_size"]

    rows  = []
    top_conditions = []
    percentiles    = {}

    for N in cfg["N_values"]:
        rw_col = f"rw_atr_{N}"
        bp_col = f"bp_{N}"
        rm_col = f"rm_{N}"
        rh_col = f"rh_{N}"
        rl_col = f"rl_{N}"

        if rw_col not in df.columns:
            continue

        rw_atr = df[rw_col].dropna()
        if is_percentiles and N in is_percentiles:
            p33, p66 = is_percentiles[N]
        else:
            p33 = rw_atr.quantile(0.33)
            p66 = rw_atr.quantile(0.66)

        percentiles[N] = (p33, p66)

        bucket_col = df[rw_col].map(lambda v: _bucket_label(v, p33, p66))

        # top setup: bp > (1 - ext)
        top_mask = df[bp_col] > (1.0 - ext)
        bot_mask = df[bp_col] < ext

        for setup_type, mask, rr_fn, rev_fn, cont_fn in [
            (
                "top",
                top_mask,
                _net_rr_top,
                lambda s: (s["fwd_min_l_10"] <= s[rm_col]).astype(float),
                lambda s: (s["fwd_max_h_10"] >= s[rh_col]).astype(float),
            ),
            (
                "bot",
                bot_mask,
                _net_rr_bot,
                lambda s: (s["fwd_max_h_10"] >= s[rm_col]).astype(float),
                lambda s: (s["fwd_min_l_10"] <= s[rl_col]).astype(float),
            ),
        ]:
            for bucket in ("narrow", "medium", "wide"):
                sel = df.loc[mask & (bucket_col == bucket)]
                if len(sel) < n_min:
                    rows.append(
                        _empty_row(N, bucket, setup_type, len(sel))
                    )
                    continue

                net_rr  = rr_fn(sel, N)
                rev_flag = rev_fn(sel)
                cont_flag = cont_fn(sel)

                g = summarise_group(net_rr, rev_flag, n_min)
                cont_rate = float(cont_flag.mean()) if len(cont_flag) > 0 else np.nan

                row = {
                    "N": N, "bucket": bucket, "setup": setup_type,
                    **g,
                    "cont_rate": cont_rate,
                }
                rows.append(row)

        # Full (no extreme filter) — does time + width predict continuation?
        for bucket in ("narrow", "medium", "wide"):
            sel = df.loc[bucket_col == bucket]
            if len(sel) < n_min:
                continue
            # Raw next-bar return in ATR units (signed: positive = up move)
            raw_ret = (sel["fwd_close_1"] - sel["Close"]) / sel["pre_atr"].replace(0, np.nan)
            # Use upper half as "continuation at top" proxy
            rows.append({
                "N": N, "bucket": bucket, "setup": "all_bars",
                "n": len(sel),
                "rev_rate": np.nan,
                "mean_rr": float(raw_ret.mean()),
                "std_rr": float(raw_ret.std(ddof=1)),
                "p_binom": np.nan, "p_t": np.nan, "cont_rate": np.nan,
            })

    # BH correction on all p_binom and p_t
    all_p_binom = [r.get("p_binom", np.nan) for r in rows]
    all_p_t     = [r.get("p_t", np.nan)     for r in rows]
    _, corr_binom = bh_correction(all_p_binom, cfg["p_value_threshold"])
    _, corr_t     = bh_correction(all_p_t,     cfg["p_value_threshold"])

    thresh_rev_is  = cfg["edge_reversion_rate_insample"]
    thresh_rr      = cfg["edge_min_rr"]
    thresh_p       = cfg["p_value_threshold"]

    for i, row in enumerate(rows):
        row["p_binom_bh"] = corr_binom[i]
        row["p_t_bh"]     = corr_t[i]
        row["confirmed"]  = (
            not np.isnan(row.get("rev_rate", np.nan))
            and row.get("rev_rate", 0) > thresh_rev_is
            and row.get("mean_rr", 0) > thresh_rr
            and (row.get("p_binom_bh", 1) < thresh_p or row.get("p_t_bh", 1) < thresh_p)
            and row.get("n", 0) >= cfg["min_sample_size"]
        )
        if row.get("confirmed") and row.get("setup") in ("top", "bot"):
            top_conditions.append((row["N"], row["bucket"], row["setup"]))

    return {
        "rows": rows,
        "top_conditions": top_conditions,
        "percentiles": percentiles,
    }


def _empty_row(N, bucket, setup, n):
    return {
        "N": N, "bucket": bucket, "setup": setup, "n": n,
        "rev_rate": np.nan, "mean_rr": np.nan, "std_rr": np.nan,
        "p_binom": np.nan, "p_t": np.nan, "cont_rate": np.nan,
    }
