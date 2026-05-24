"""
Axiom 8: Does bar shape at a range extreme predict reversion?

For bars with bar_position > 0.9 (top) and < 0.1 (bottom):
  - Correlate bar structure features with reversion outcomes
  - Split by bar_direction (bearish vs bullish bar at top/bottom)

Features: upper_wick_ratio, lower_wick_ratio, body_position,
          range_expansion, bar_direction, close_position
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.statistics import bh_correction, binomial_pval, summarise_group


def run_axiom8(df, cfg):
    """
    Returns:
      corr_rows      — list of correlation result dicts
      direction_rows — list of bearish/bullish split result dicts
      top_conditions — list of (N, setup_type, bar_dir, feature_condition)
    """
    ext   = cfg["position_extreme_threshold"]
    n_min = cfg["min_sample_size"]

    corr_rows      = []
    direction_rows = []
    top_conditions = []

    STRUCT_FEATURES = [
        "upper_wick_ratio", "lower_wick_ratio", "body_position",
        "range_expansion",  "bar_direction",    "close_position",
    ]
    OUTCOME_COLS = ["reversion_to_mid", "fwd_direction", "net_rr_10"]

    for N in cfg["N_values"]:
        bp_col = f"bp_{N}"
        rm_col = f"rm_{N}"
        if bp_col not in df.columns:
            continue

        for setup_type, mask, rev_fn, rr_fn in [
            (
                "top",
                df[bp_col] > (1.0 - ext),
                lambda s: (s["fwd_min_l_10"] <= s[rm_col]).astype(float),
                lambda s: (s["Close"] - s["fwd_close_10"]) / s["pre_atr"].replace(0, np.nan) - s["spread_rt_atr"],
            ),
            (
                "bot",
                df[bp_col] < ext,
                lambda s: (s["fwd_max_h_10"] >= s[rm_col]).astype(float),
                lambda s: (s["fwd_close_10"] - s["Close"]) / s["pre_atr"].replace(0, np.nan) - s["spread_rt_atr"],
            ),
        ]:
            sel = df.loc[mask].copy()
            if len(sel) < n_min:
                continue

            sel["reversion_to_mid"] = rev_fn(sel)
            sel["net_rr_10"]        = rr_fn(sel)
            # fwd_direction: +1 if next bar goes in reversion direction
            if setup_type == "top":
                sel["fwd_direction"] = (sel["fwd_close_1"] < sel["Close"]).astype(float)
            else:
                sel["fwd_direction"] = (sel["fwd_close_1"] > sel["Close"]).astype(float)

            # ── Correlations ─────────────────────────────────────────────────
            pvals_corr = []
            for feat in STRUCT_FEATURES:
                if feat not in sel.columns:
                    continue
                for outcome in OUTCOME_COLS:
                    x = sel[feat].values.astype(float)
                    y = sel[outcome].values.astype(float)
                    valid = ~(np.isnan(x) | np.isnan(y))
                    if valid.sum() < 20:
                        corr_rows.append({
                            "N": N, "setup": setup_type, "feature": feat,
                            "outcome": outcome, "n": int(valid.sum()),
                            "spearman_r": np.nan, "p": np.nan,
                        })
                        pvals_corr.append(np.nan)
                        continue

                    r, p = spearmanr(x[valid], y[valid])
                    corr_rows.append({
                        "N": N, "setup": setup_type, "feature": feat,
                        "outcome": outcome, "n": int(valid.sum()),
                        "spearman_r": float(r), "p": float(p),
                    })
                    pvals_corr.append(p)

            # BH on correlations for this (N, setup_type)
            _, corr_bh = bh_correction(pvals_corr, cfg["p_value_threshold"])
            last_n = len(pvals_corr)
            for i, row in enumerate(corr_rows[-last_n:]):
                row["p_bh"] = corr_bh[i]

            # ── Direction split ───────────────────────────────────────────────
            # bar_direction: 1=bullish, -1=bearish, 0=doji
            if "bar_direction" not in sel.columns:
                continue

            for bar_dir, dir_label in [(-1, "bearish"), (1, "bullish"), (0, "doji")]:
                g = sel.loc[sel["bar_direction"] == bar_dir]
                n = len(g)

                if n < 10:
                    direction_rows.append({
                        "N": N, "setup": setup_type, "bar_dir": dir_label,
                        "n": n, "rev_rate": np.nan, "mean_rr": np.nan,
                        "p_binom": np.nan, "note": "INSUFFICIENT_SAMPLE",
                    })
                    continue

                rev  = g["reversion_to_mid"].values
                rr   = g["net_rr_10"].values
                n_rev = int(np.nansum(rev))

                rev_rate = n_rev / n
                mean_rr  = float(np.nanmean(rr))
                p_binom  = binomial_pval(n_rev, n, 0.5)
                note     = "INSUFFICIENT_SAMPLE" if n < n_min else ""

                row = {
                    "N": N, "setup": setup_type, "bar_dir": dir_label,
                    "n": n, "rev_rate": rev_rate, "mean_rr": mean_rr,
                    "p_binom": p_binom, "note": note,
                }
                direction_rows.append(row)

                if (
                    n >= n_min
                    and rev_rate > cfg["edge_reversion_rate_insample"]
                    and mean_rr > cfg["edge_min_rr"]
                    and p_binom < cfg["p_value_threshold"]
                ):
                    top_conditions.append((N, setup_type, dir_label))

    # BH on direction_rows p_binom
    all_pb = [r.get("p_binom", np.nan) for r in direction_rows]
    _, corr_all = bh_correction(all_pb, cfg["p_value_threshold"])
    for i, row in enumerate(direction_rows):
        row["p_binom_bh"] = corr_all[i]
        row["confirmed"]  = (
            row.get("n", 0) >= n_min
            and not np.isnan(row.get("rev_rate", np.nan))
            and row.get("rev_rate", 0) > cfg["edge_reversion_rate_insample"]
            and row.get("mean_rr", 0) > cfg["edge_min_rr"]
            and row.get("p_binom_bh", 1) < cfg["p_value_threshold"]
        )

    return {
        "corr_rows":      corr_rows,
        "direction_rows": direction_rows,
        "top_conditions": top_conditions,
    }
