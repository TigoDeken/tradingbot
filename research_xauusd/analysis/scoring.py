"""
Combined edge scoring (Axiom 3+4+6+8 combined).

score = 0
+1 bar_position > 0.9 or < 0.1
+1 range_width = narrow (< 33rd pct of rw_atr_N)
+1 upper_wick_ratio > 0.4 at top  OR  lower_wick_ratio > 0.4 at bottom
+1 session = london or ny
+1 bearish bar at top  OR  bullish bar at bottom

For score buckets 0-1, 2-3, 4-5:
  n, rev_rate, mean_net_rr, Sharpe, MAE percentiles
"""

import numpy as np
import pandas as pd

from analysis.statistics import bh_correction, sharpe_annualized, binomial_pval, t_test_mean


def run_scoring(df, cfg, a3_percentiles=None, bars_per_year=69552):
    """
    a3_percentiles: {N: (p33, p66)} from axiom3 IS run.
    If None, computed from df.

    Returns: rows — list of result dicts per score bucket.
    """
    N     = cfg["N_for_scoring"]
    bp_col = f"bp_{N}"
    rw_col = f"rw_atr_{N}"
    rm_col = f"rm_{N}"
    ext    = cfg["position_extreme_threshold"]
    n_min  = cfg["min_sample_size"]

    if bp_col not in df.columns:
        return {"rows": [], "top_conditions": []}

    rw = df[rw_col].dropna()
    if a3_percentiles and N in a3_percentiles:
        p33, _ = a3_percentiles[N]
    else:
        p33 = rw.quantile(0.33)

    top_mask = df[bp_col] > (1.0 - ext)
    bot_mask = df[bp_col] < ext
    extreme  = top_mask | bot_mask

    # Condition 1: extreme
    c1 = extreme.astype(int)

    # Condition 2: narrow range
    c2 = (df[rw_col] < p33).astype(int)

    # Condition 3: wick against extreme
    wick_top = (top_mask & (df.get("upper_wick_ratio", pd.Series(0, index=df.index)) > 0.4))
    wick_bot = (bot_mask & (df.get("lower_wick_ratio", pd.Series(0, index=df.index)) > 0.4))
    c3 = (wick_top | wick_bot).astype(int)

    # Condition 4: London or NY session
    c4 = df["session"].isin(["london", "ny"]).astype(int)

    # Condition 5: bar direction against extreme
    bar_bear_top = (top_mask & (df.get("bar_direction", pd.Series(0, index=df.index)) == -1))
    bar_bull_bot = (bot_mask & (df.get("bar_direction", pd.Series(0, index=df.index)) == 1))
    c5 = (bar_bear_top | bar_bull_bot).astype(int)

    score = c1 + c2 + c3 + c4 + c5

    # Only score bars that are at an extreme (c1=1)
    valid_mask = extreme & df["pre_atr"].notna() & df["fwd_close_10"].notna()
    sel = df.loc[valid_mask].copy()
    sel["score"] = score.loc[valid_mask]

    # Outcomes
    sel["rev_top"] = (sel["fwd_min_l_10"] <= sel[rm_col]).astype(float).where(top_mask.loc[valid_mask])
    sel["rev_bot"] = (sel["fwd_max_h_10"] >= sel[rm_col]).astype(float).where(bot_mask.loc[valid_mask])
    sel["rev"]     = sel["rev_top"].combine_first(sel["rev_bot"])

    atr_safe = sel["pre_atr"].replace(0, np.nan)
    sel["net_rr_top"] = (
        (sel["Close"] - sel["fwd_close_10"]) / atr_safe - sel["spread_rt_atr"]
    ).where(top_mask.loc[valid_mask])
    sel["net_rr_bot"] = (
        (sel["fwd_close_10"] - sel["Close"]) / atr_safe - sel["spread_rt_atr"]
    ).where(bot_mask.loc[valid_mask])
    sel["net_rr"] = sel["net_rr_top"].combine_first(sel["net_rr_bot"])

    # MAE (max adverse excursion in 10 bars)
    sel["mae_top"] = (
        (sel["fwd_max_h_10"] - sel["Close"]) / atr_safe
    ).clip(lower=0).where(top_mask.loc[valid_mask])
    sel["mae_bot"] = (
        (sel["Close"] - sel["fwd_min_l_10"]) / atr_safe
    ).clip(lower=0).where(bot_mask.loc[valid_mask])
    sel["mae"] = sel["mae_top"].combine_first(sel["mae_bot"])

    BUCKET_DEFS = [(0, 1, "0-1"), (2, 3, "2-3"), (4, 5, "4-5")]
    rows = []

    for lo, hi, label in BUCKET_DEFS:
        g = sel.loc[(sel["score"] >= lo) & (sel["score"] <= hi)]
        n = len(g)

        rr  = g["net_rr"].dropna()
        rev = g["rev"].dropna()
        mae = g["mae"].dropna()

        n_rev    = int(rev.sum()) if len(rev) else 0
        rev_rate = (n_rev / len(rev)) if len(rev) else np.nan
        mean_rr  = float(rr.mean()) if len(rr) else np.nan
        p_binom  = binomial_pval(n_rev, len(rev), 0.5) if len(rev) >= n_min else np.nan
        _, p_t   = t_test_mean(rr.values, 0.0) if len(rr) >= 5 else (np.nan, np.nan)

        sh = np.nan
        if len(rr) >= 5:
            # approx trades per year for this bucket
            tpy = bars_per_year * (n / max(len(df), 1))
            sh  = sharpe_annualized(rr.values, tpy) if tpy > 0 else np.nan

        mae_pcts = {}
        if len(mae) >= 5:
            for q in [25, 50, 75, 90]:
                mae_pcts[f"mae_p{q}"] = float(np.percentile(mae, q))

        rows.append({
            "score_bucket": label,
            "n": n, "rev_rate": rev_rate, "mean_rr": mean_rr,
            "sharpe": sh, "p_binom": p_binom, "p_t": p_t,
            **mae_pcts,
        })

    # BH on this small set
    all_pb = [r.get("p_binom", np.nan) for r in rows]
    _, corr = bh_correction(all_pb, cfg["p_value_threshold"])
    top_conditions = []
    for i, row in enumerate(rows):
        row["p_binom_bh"] = corr[i]
        row["confirmed"]  = (
            row.get("n", 0) >= n_min
            and not np.isnan(row.get("rev_rate", np.nan))
            and row.get("rev_rate", 0) > cfg["edge_reversion_rate_insample"]
            and row.get("mean_rr", 0) > cfg["edge_min_rr"]
            and row.get("p_binom_bh", 1) < cfg["p_value_threshold"]
        )
        if row.get("confirmed"):
            top_conditions.append(row["score_bucket"])

    return {"rows": rows, "top_conditions": top_conditions}
