import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def bh_correction(pvals, alpha=0.05):
    """
    Benjamini-Hochberg FDR correction.
    Returns (reject_array, corrected_pvals_array).
    NaN p-values are preserved as NaN in output.
    """
    from statsmodels.stats.multitest import multipletests

    arr = np.asarray(pvals, dtype=float)
    valid = ~np.isnan(arr)

    reject    = np.zeros(len(arr), dtype=bool)
    corrected = np.full(len(arr), np.nan)

    if valid.sum() >= 2:
        r, c, _, _ = multipletests(arr[valid], alpha=alpha, method="fdr_bh")
        reject[valid]    = r
        corrected[valid] = c

    return reject, corrected


def t_test_mean(returns, h0_mean=0.0):
    """One-sample t-test against h0_mean. Returns (statistic, p_value) two-sided."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 5:
        return np.nan, np.nan
    stat, p = scipy_stats.ttest_1samp(arr, h0_mean)
    return float(stat), float(p)


def binomial_pval(n_success, n_total, h0_rate=0.5):
    """
    One-sided binomial test: H0 rate <= h0_rate.
    Returns p-value (alternative = 'greater').
    """
    if n_total == 0:
        return np.nan
    result = scipy_stats.binomtest(int(n_success), int(n_total), h0_rate, alternative="greater")
    return float(result.pvalue)


def sharpe_annualized(returns, bars_per_year):
    """
    Annualised Sharpe.  returns = per-trade returns (in ATR units or any unit).
    trades_per_year estimated from len(returns) / observed_years.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    n   = len(arr)
    if n < 5:
        return np.nan
    sd = np.std(arr, ddof=1)
    if sd == 0:
        return np.nan
    per_trade = np.mean(arr) / sd
    return float(per_trade * np.sqrt(bars_per_year))


def summarise_group(returns_series, rev_flag_series, n_min=50):
    """
    Returns a dict with n, rev_rate, mean_rr, std_rr, p_binom, p_t.
    rev_flag_series: boolean (1 = reverted, 0 = did not).
    """
    ret = np.asarray(returns_series, dtype=float)
    rev = np.asarray(rev_flag_series, dtype=float)

    # align on non-NaN in both
    mask = ~(np.isnan(ret) | np.isnan(rev))
    ret  = ret[mask]
    rev  = rev[mask]
    n    = len(ret)

    if n < n_min:
        return {
            "n": n, "rev_rate": np.nan, "mean_rr": np.nan,
            "std_rr": np.nan, "p_binom": np.nan, "p_t": np.nan,
        }

    n_rev    = int(rev.sum())
    rev_rate = n_rev / n
    mean_rr  = float(np.mean(ret))
    std_rr   = float(np.std(ret, ddof=1))

    _, p_t     = t_test_mean(ret, 0.0)
    p_binom    = binomial_pval(n_rev, n, 0.5)

    return {
        "n": n, "rev_rate": rev_rate, "mean_rr": mean_rr,
        "std_rr": std_rr, "p_binom": p_binom, "p_t": p_t,
    }
