"""
Axiom 4: Is there a monotonic relationship between bar_position(N) and
next-bar return?

Splits bar_position into deciles and reports for each:
  - mean 1-bar return  (ATR units)
  - mean 5-bar return  (ATR units)
  - std dev
  - Sharpe of mechanical fade rule
  - continuation rate (move in direction of extreme)
  - reversion rate (move against extreme)
"""

import numpy as np
import pandas as pd

from analysis.statistics import bh_correction, sharpe_annualized, t_test_mean


def run_axiom4(df, cfg, bars_per_year=69552):
    """
    bars_per_year default = M5 gold (~69 552).  Caller should pass the correct
    value for the active timeframe.

    Returns:
      rows — list of result dicts (N × decile)
      top_conditions — list of (N, decile, direction) passing IS thresholds
    """
    n_min = cfg["min_sample_size"]
    spread_rt = 2.0 * cfg["normal_spread_pips"]  # absolute price units round trip

    rows = []
    top_conditions = []

    for N in cfg["N_values"]:
        bp_col = f"bp_{N}"
        if bp_col not in df.columns:
            continue

        valid = df[bp_col].notna() & df["pre_atr"].notna()
        sub   = df.loc[valid].copy()
        if len(sub) < 100:
            continue

        try:
            sub["decile"] = pd.qcut(sub[bp_col], 10, labels=False, duplicates="drop")
        except ValueError:
            continue

        atr = sub["pre_atr"].replace(0, np.nan)

        # Forward returns in ATR units
        sub["ret1_atr"]  = (sub["fwd_close_1"]  - sub["Close"]) / atr
        sub["ret5_atr"]  = (sub["fwd_close_5"]  - sub["Close"]) / atr
        sub["ret10_atr"] = (sub["fwd_close_10"] - sub["Close"]) / atr
        sub["spread_atr"]= (2.0 * cfg["normal_spread_pips"]) / atr

        p_t_list = []
        for d in range(10):
            g = sub.loc[sub["decile"] == d]
            if len(g) < n_min:
                rows.append(_empty_decile_row(N, d))
                p_t_list.append(np.nan)
                continue

            fade_dir = 1 if d < 5 else -1  # long in bottom half, short in top half

            # Fade rule return: directional + net of spread
            fade_ret = fade_dir * g["ret1_atr"] - g["spread_atr"]

            mean1  = float(g["ret1_atr"].mean())
            mean5  = float(g["ret5_atr"].mean())
            std1   = float(g["ret1_atr"].std(ddof=1))

            _, p_t = t_test_mean(fade_ret.values)
            p_t_list.append(p_t)

            tpy = bars_per_year / 10.0  # trades per year in this decile (approx)
            sh  = sharpe_annualized(fade_ret.values, tpy)

            # At tails: explicit continuation / reversion rates
            cont_rate = np.nan
            rev_rate  = np.nan
            if d == 9:  # near-top: continuation = up, reversion = down
                cont_rate = float((g["ret1_atr"] > 0).mean())
                rev_rate  = float((g["ret1_atr"] < 0).mean())
            elif d == 0:  # near-bottom: continuation = down, reversion = up
                cont_rate = float((g["ret1_atr"] < 0).mean())
                rev_rate  = float((g["ret1_atr"] > 0).mean())

            rows.append({
                "N": N, "decile": d, "n": len(g),
                "mean_ret1_atr": mean1, "mean_ret5_atr": mean5,
                "std_ret1_atr": std1,
                "fade_mean_rr": float(fade_ret.mean()),
                "fade_sharpe": sh,
                "cont_rate": cont_rate, "rev_rate": rev_rate,
                "p_t": p_t,
            })

        # BH correct p_t within this N
        _, corr = bh_correction(p_t_list, cfg["p_value_threshold"])
        n_rows_this_N = sum(1 for r in rows if r.get("N") == N)
        for i, row in enumerate(rows[-len(p_t_list):]):
            row["p_t_bh"] = corr[i]
            row["confirmed"] = (
                not np.isnan(row.get("fade_mean_rr", np.nan))
                and row.get("fade_mean_rr", 0) > cfg["edge_min_rr"]
                and row.get("p_t_bh", 1) < cfg["p_value_threshold"]
                and row.get("n", 0) >= n_min
                and abs(row.get("fade_sharpe", 0) or 0) > 0
            )
            if row.get("confirmed"):
                d     = row["decile"]
                direc = "long" if d < 5 else "short"
                top_conditions.append((N, d, direc))

    return {"rows": rows, "top_conditions": top_conditions}


def _empty_decile_row(N, d):
    return {
        "N": N, "decile": d, "n": 0,
        "mean_ret1_atr": np.nan, "mean_ret5_atr": np.nan,
        "std_ret1_atr": np.nan, "fade_mean_rr": np.nan,
        "fade_sharpe": np.nan, "cont_rate": np.nan,
        "rev_rate": np.nan, "p_t": np.nan,
    }
