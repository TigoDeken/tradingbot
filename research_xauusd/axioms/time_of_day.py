"""
Axiom 6: Does the same range setup produce different outcomes at different times?

Analysis A — with extreme filter (bp > 0.9 or < 0.1) for a chosen N:
  Split by hour (0-23), session, day-of-week.
  Metric: reversion-to-mid rate.

Analysis B — no filter, all bars:
  Mean return and Sharpe per hour-of-day.
"""

import numpy as np
import pandas as pd

from analysis.statistics import bh_correction, sharpe_annualized, binomial_pval


def run_axiom6(df, cfg, bars_per_year=69552):
    """
    Returns:
      rows_filtered   — list of result dicts (hour/session/dow × N)
      rows_raw        — list of result dicts for raw-bar analysis per hour
      top_conditions  — list of (N, split_type, bucket_value) passing IS thresholds
    """
    ext   = cfg["position_extreme_threshold"]
    n_min = cfg["min_sample_size"]
    sessions_cfg = cfg["sessions"]

    rows_filtered  = []
    rows_raw       = []
    top_conditions = []

    def session_label(hour):
        for name, (s, e) in sessions_cfg.items():
            if s <= hour < e:
                return name
        return "other"

    # ── Analysis A: filtered extremes per N ───────────────────────────────────
    for N in cfg["N_values"]:
        bp_col = f"bp_{N}"
        rm_col = f"rm_{N}"
        rh_col = f"rh_{N}"
        rl_col = f"rl_{N}"

        if bp_col not in df.columns:
            continue

        top_mask = df[bp_col] > (1.0 - ext)
        bot_mask = df[bp_col] < ext

        def rev_flag_top(s):
            return (s["fwd_min_l_10"] <= s[rm_col]).astype(float)

        def rev_flag_bot(s):
            return (s["fwd_max_h_10"] >= s[rm_col]).astype(float)

        def net_rr_top(s):
            return (s["Close"] - s["fwd_close_10"]) / s["pre_atr"].replace(0, np.nan) - s["spread_rt_atr"]

        def net_rr_bot(s):
            return (s["fwd_close_10"] - s["Close"]) / s["pre_atr"].replace(0, np.nan) - s["spread_rt_atr"]

        for setup_type, mask, rr_fn, rev_fn in [
            ("top", top_mask, net_rr_top, rev_flag_top),
            ("bot", bot_mask, net_rr_bot, rev_flag_bot),
        ]:
            extremes = df.loc[mask].copy()
            if len(extremes) == 0:
                continue

            extremes["sess_label"] = extremes["hour_utc"].map(session_label)

            for split_type, col, values in [
                ("hour",    "hour_utc",   range(24)),
                ("session", "sess_label", list(sessions_cfg.keys()) + ["other"]),
                ("dow",     "dow",        range(7)),
            ]:
                p_binom_list = []
                bucket_rows  = []

                for val in values:
                    g = extremes.loc[extremes[col] == val]
                    n = len(g)

                    if n == 0:
                        continue

                    note = "INSUFFICIENT_SAMPLE" if n < 30 else ""

                    if n < n_min:
                        bucket_rows.append({
                            "N": N, "setup": setup_type,
                            "split_type": split_type, "bucket": val,
                            "n": n, "rev_rate": np.nan,
                            "mean_rr": np.nan, "p_binom": np.nan, "note": note,
                        })
                        p_binom_list.append(np.nan)
                        continue

                    rev  = rev_fn(g)
                    rr   = rr_fn(g)
                    n_rev = int(rev.sum())

                    rev_rate = n_rev / n
                    mean_rr  = float(rr.mean())
                    p_binom  = binomial_pval(n_rev, n, 0.5)

                    p_binom_list.append(p_binom)
                    bucket_rows.append({
                        "N": N, "setup": setup_type,
                        "split_type": split_type, "bucket": val,
                        "n": n, "rev_rate": rev_rate,
                        "mean_rr": mean_rr, "p_binom": p_binom, "note": note,
                    })

                # BH on this split_type's p-values
                _, corr = bh_correction(p_binom_list, cfg["p_value_threshold"])
                for i, row in enumerate(bucket_rows):
                    row["p_binom_bh"] = corr[i] if i < len(corr) else np.nan
                    row["confirmed"]  = (
                        not np.isnan(row.get("rev_rate", np.nan))
                        and row.get("rev_rate", 0) > cfg["edge_reversion_rate_insample"]
                        and row.get("mean_rr", 0) > cfg["edge_min_rr"]
                        and row.get("p_binom_bh", 1) < cfg["p_value_threshold"]
                        and row.get("n", 0) >= n_min
                    )
                    if row.get("confirmed"):
                        top_conditions.append((N, split_type, row["bucket"], setup_type))

                rows_filtered.extend(bucket_rows)

    # ── Analysis B: raw bars — mean return per hour ───────────────────────────
    valid = df["pre_atr"].notna() & df["fwd_close_1"].notna()
    raw   = df.loc[valid].copy()
    raw["ret1_atr"] = (raw["fwd_close_1"] - raw["Close"]) / raw["pre_atr"].replace(0, np.nan)

    tpy_per_hour = bars_per_year / 24.0
    p_t_raw = []
    raw_bucket_rows = []

    for hr in range(24):
        g = raw.loc[raw["hour_utc"] == hr, "ret1_atr"].dropna()
        n = len(g)
        if n == 0:
            raw_bucket_rows.append({"hour_utc": hr, "n": 0,
                                    "mean_ret_atr": np.nan, "sharpe": np.nan, "p_t": np.nan})
            p_t_raw.append(np.nan)
            continue

        mean_r = float(g.mean())
        std_r  = float(g.std(ddof=1))
        from scipy.stats import ttest_1samp
        _, p_t = ttest_1samp(g.values, 0.0)
        sh     = (mean_r / std_r * np.sqrt(tpy_per_hour)) if std_r > 0 else np.nan

        raw_bucket_rows.append({
            "hour_utc": hr, "n": n,
            "mean_ret_atr": mean_r, "sharpe": sh, "p_t": p_t,
        })
        p_t_raw.append(p_t)

    _, corr_raw = bh_correction(p_t_raw, cfg["p_value_threshold"])
    for i, row in enumerate(raw_bucket_rows):
        row["p_t_bh"] = corr_raw[i]

    rows_raw = raw_bucket_rows

    return {
        "rows_filtered":  rows_filtered,
        "rows_raw":       rows_raw,
        "top_conditions": top_conditions,
    }
