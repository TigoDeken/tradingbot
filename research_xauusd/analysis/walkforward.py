import numpy as np
import pandas as pd


def split_is_oos(df, train_ratio=0.70):
    """Chronological split. Returns (df_is, df_oos)."""
    idx = int(len(df) * train_ratio)
    return df.iloc[:idx].copy(), df.iloc[idx:].copy()


def validate_condition(df_oos, mask_fn, rr_col, rev_col, n_min=30):
    """
    Apply mask_fn(df_oos) to select rows, then compute reversion rate
    and mean RR on the out-of-sample slice.

    Returns dict: {n, rev_rate, mean_rr, sharpe_approx}
    """
    mask   = mask_fn(df_oos)
    subset = df_oos.loc[mask].copy()

    rr  = subset[rr_col].dropna().values  if rr_col  in subset.columns else np.array([])
    rev = subset[rev_col].dropna().values if rev_col in subset.columns else np.array([])

    n = min(len(rr), len(rev))
    if n < n_min:
        return {"n": n, "rev_rate": np.nan, "mean_rr": np.nan, "sharpe_approx": np.nan}

    rr  = rr[:n]
    rev = rev[:n]

    rev_rate = float(np.nanmean(rev))
    mean_rr  = float(np.nanmean(rr))
    sd       = float(np.nanstd(rr, ddof=1))
    sharpe   = (mean_rr / sd * np.sqrt(n)) if sd > 0 else np.nan

    return {"n": n, "rev_rate": rev_rate, "mean_rr": mean_rr, "sharpe_approx": sharpe}


def compare_is_oos(is_result, oos_result, cfg):
    """
    Mark a finding as LIKELY_OVERFIT if OOS reversion rate drops > 10pp vs IS,
    or OOS n < cfg['min_sample_size'] * 0.6.
    Returns annotated copy of is_result.
    """
    out = dict(is_result)
    out["oos_n"]        = oos_result.get("n", np.nan)
    out["oos_rev_rate"] = oos_result.get("rev_rate", np.nan)
    out["oos_mean_rr"]  = oos_result.get("mean_rr", np.nan)
    out["oos_sharpe"]   = oos_result.get("sharpe_approx", np.nan)

    is_rr  = is_result.get("rev_rate", np.nan)
    oos_rr = oos_result.get("rev_rate", np.nan)

    if np.isnan(is_rr) or np.isnan(oos_rr):
        out["overfit_flag"] = "INSUFFICIENT_OOS"
    elif (is_rr - oos_rr) > 0.10:
        out["overfit_flag"] = "LIKELY_OVERFIT"
    else:
        out["overfit_flag"] = ""

    return out
