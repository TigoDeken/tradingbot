"""
Regime bucket analysis.

Takes the simulation DataFrame and the full feature DataFrame.
Adds regime columns to each signal row, then groups and computes:
  - Baseline WR (no regime filter)
  - WR per R² tertile
  - WR per ATR-ratio tertile
  - WR per combined 3×3 regime state
  - All split by session

Reports WR lift vs baseline and flags conditions meeting the 4-7% target.
"""

import numpy as np
import pandas as pd


def _tertile_labels(series, prefix, percentiles=None):
    """
    Bin series into low / mid / high tertiles.
    percentiles: (p33, p66) — if None, computed from series itself (IS pass).
    Returns (labels_series, p33, p66).
    """
    if percentiles is None:
        p33 = series.quantile(0.33)
        p66 = series.quantile(0.66)
    else:
        p33, p66 = percentiles

    def _label(v):
        if pd.isna(v):
            return np.nan
        if v < p33:
            return f"{prefix}_low"
        if v < p66:
            return f"{prefix}_mid"
        return f"{prefix}_high"

    return series.map(_label), p33, p66


def _group_stats(grp, spread_rt=0.30):
    """
    Summary stats for a group of simulation rows.
    spread_rt: round-trip spread in ATR units (default 0.30 = 0.15 × 2 / ~1 ATR).
    Expectancy nets out a fixed spread cost from each trade.
    """
    n   = len(grp)
    wins = (grp["outcome"] == 1).sum()
    wr   = wins / n if n > 0 else np.nan

    rr_vals = grp["result_r"].values
    exp     = float(np.nanmean(rr_vals)) if n > 0 else np.nan

    mfe_med   = float(np.nanmedian(grp["mfe_atr"])) if n > 0 else np.nan
    mae_med   = float(np.nanmedian(grp["mae_atr"]))  if n > 0 else np.nan
    edge_med  = float(np.nanmedian(grp["edge_ratio"].dropna())) if n > 0 else np.nan

    return {
        "n": n, "win_rate": wr, "expectancy": exp,
        "mfe_p50": mfe_med, "mae_p50": mae_med, "edge_ratio": edge_med,
    }


def run_analysis(sim_df, feature_df, is_percentiles=None):
    """
    sim_df      : output of sim.run_simulation() as a DataFrame
    feature_df  : full M5 df with pre_atr, linreg_r2, atr_ratio, session columns
    is_percentiles: {
        'r2':  (p33, p66),
        'atr': (p33, p66),
    }  — if None, computed from sim rows (IS pass).

    Returns:
      results_df      : one row per (direction, stop, rr, horizon, regime, session)
      is_percentiles  : thresholds for OOS re-use
    """
    df = sim_df.copy()

    # ── Attach regime features to each signal row ─────────────────────────────
    feat = feature_df[["linreg_r2", "atr_ratio", "session"]].copy()
    feat.index = np.arange(len(feat))     # integer index matching bar_idx

    df["linreg_r2"] = feat["linreg_r2"].iloc[df["bar_idx"].values].values
    df["atr_ratio"] = feat["atr_ratio"].iloc[df["bar_idx"].values].values
    df["session"]   = feat["session"].iloc[df["bar_idx"].values].values

    # ── Compute tertile thresholds (IS-only) ──────────────────────────────────
    r2_pcts  = is_percentiles["r2"]  if is_percentiles and "r2"  in is_percentiles else None
    atr_pcts = is_percentiles["atr"] if is_percentiles and "atr" in is_percentiles else None

    df["r2_bucket"],  r2_p33,  r2_p66  = _tertile_labels(df["linreg_r2"], "r2",  r2_pcts)
    df["atr_bucket"], atr_p33, atr_p66 = _tertile_labels(df["atr_ratio"], "atr", atr_pcts)
    df["regime_9"]    = df["r2_bucket"].astype(str) + " | " + df["atr_bucket"].astype(str)

    thresholds = {
        "r2":  (r2_p33,  r2_p66),
        "atr": (atr_p33, atr_p66),
    }

    rows = []
    key_cols = ["direction", "stop_mult", "rr", "horizon"]

    for keys, grp_outer in df.groupby(key_cols):
        direction, stop_mult, rr, horizon = keys

        # Baseline
        base = _group_stats(grp_outer)
        base.update({
            "direction": direction, "stop_mult": stop_mult,
            "rr": rr, "horizon": horizon,
            "regime_type": "baseline", "regime_value": "all",
            "session": "all",
            "wr_lift": 0.0,
        })
        rows.append(base)

        # By session (baseline per session)
        for sess, sess_grp in grp_outer.groupby("session"):
            s = _group_stats(sess_grp)
            s.update({
                "direction": direction, "stop_mult": stop_mult,
                "rr": rr, "horizon": horizon,
                "regime_type": "session", "regime_value": sess,
                "session": sess,
                "wr_lift": (s["win_rate"] or 0) - (base["win_rate"] or 0),
            })
            rows.append(s)

        # By R² tertile
        for bucket, b_grp in grp_outer.groupby("r2_bucket"):
            s = _group_stats(b_grp)
            s.update({
                "direction": direction, "stop_mult": stop_mult,
                "rr": rr, "horizon": horizon,
                "regime_type": "r2", "regime_value": str(bucket),
                "session": "all",
                "wr_lift": (s["win_rate"] or 0) - (base["win_rate"] or 0),
            })
            rows.append(s)

        # By ATR ratio tertile
        for bucket, b_grp in grp_outer.groupby("atr_bucket"):
            s = _group_stats(b_grp)
            s.update({
                "direction": direction, "stop_mult": stop_mult,
                "rr": rr, "horizon": horizon,
                "regime_type": "atr", "regime_value": str(bucket),
                "session": "all",
                "wr_lift": (s["win_rate"] or 0) - (base["win_rate"] or 0),
            })
            rows.append(s)

        # Combined 9-state + session
        for (regime9, sess), rg_grp in grp_outer.groupby(["regime_9", "session"]):
            s = _group_stats(rg_grp)
            s.update({
                "direction": direction, "stop_mult": stop_mult,
                "rr": rr, "horizon": horizon,
                "regime_type": "r2_x_atr", "regime_value": str(regime9),
                "session": sess,
                "wr_lift": (s["win_rate"] or 0) - (base["win_rate"] or 0),
            })
            rows.append(s)

    result_df = pd.DataFrame(rows)

    # Flag conditions meeting the 4-7% WR lift target with sufficient sample
    result_df["meets_target"] = (
        (result_df["wr_lift"].abs() >= 0.04)
        & (result_df["n"] >= 50)
    )

    return result_df, thresholds


def top_conditions(result_df, n_min=50, lift_min=0.04):
    """Return rows that pass the lift target, sorted by wr_lift descending."""
    mask = (
        (result_df["meets_target"] == True)
        & (result_df["regime_type"] != "baseline")
        & (result_df["n"] >= n_min)
    )
    return result_df.loc[mask].sort_values("wr_lift", ascending=False).reset_index(drop=True)
