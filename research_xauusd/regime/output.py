import os
import datetime
import numpy as np
import pandas as pd


def _fmt(v, spec=".3f"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def save_results(is_result_df, oos_result_df, is_top, oos_top,
                 is_thresholds, n_is, n_oos, out_dir="results/regime"):
    os.makedirs(out_dir, exist_ok=True)

    is_result_df.to_csv(f"{out_dir}/is_full.csv",  index=False)
    oos_result_df.to_csv(f"{out_dir}/oos_full.csv", index=False)

    if not is_top.empty:
        is_top.to_csv(f"{out_dir}/is_top_conditions.csv",  index=False)
    if not oos_top.empty:
        oos_top.to_csv(f"{out_dir}/oos_top_conditions.csv", index=False)

    lines = []
    lines.append(f"# GOLD# Regime Study — M5\n")
    lines.append(f"Generated : {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n")
    lines.append(f"IS bars   : {n_is:,}  |  OOS bars: {n_oos:,}\n\n")

    lines.append("## Regime thresholds (from IS data)\n")
    r2_p33,  r2_p66  = is_thresholds.get("r2",  (None, None))
    atr_p33, atr_p66 = is_thresholds.get("atr", (None, None))
    lines.append(f"  LinReg R²  : low < {_fmt(r2_p33)}  |  mid < {_fmt(r2_p66)}  |  high >= {_fmt(r2_p66)}\n")
    lines.append(f"  ATR ratio  : low < {_fmt(atr_p33)} |  mid < {_fmt(atr_p66)} |  high >= {_fmt(atr_p66)}\n\n")

    # Baseline table
    base = is_result_df[is_result_df["regime_type"] == "baseline"].copy()
    if not base.empty:
        lines.append("## Baseline win rates (no regime filter)\n")
        lines.append(f"  {'direction':<8} {'stop':<6} {'rr':<5} {'horizon':<6}"
                     f" {'n':>6} {'WR%':>7} {'exp_R':>7} {'edge_ratio':>10}\n")
        for _, r in base.iterrows():
            lines.append(
                f"  {r['direction']:<8} {r['stop_mult']:<6.1f} {r['rr']:<5.1f} {r['horizon']:<6}"
                f" {int(r['n']):>6} {_fmt((r['win_rate'] or 0)*100, '.1f'):>7}"
                f" {_fmt(r['expectancy']):>7} {_fmt(r['edge_ratio']):>10}\n"
            )
        lines.append("\n")

    # IS top conditions
    lines.append("## IS conditions meeting >=4% WR lift\n")
    if is_top.empty:
        lines.append("  None found.\n\n")
    else:
        lines.append(f"  {'direction':<8} {'stop':<6} {'rr':<5} {'horizon':<6}"
                     f" {'regime_type':<12} {'regime_value':<28} {'session':<10}"
                     f" {'n':>5} {'WR%':>7} {'lift%':>7} {'exp_R':>7} {'edge_r':>7}\n")
        for _, r in is_top.iterrows():
            lines.append(
                f"  {r['direction']:<8} {r['stop_mult']:<6.1f} {r['rr']:<5.1f} {r['horizon']:<6}"
                f" {str(r['regime_type']):<12} {str(r['regime_value']):<28} {str(r['session']):<10}"
                f" {int(r['n']):>5} {_fmt((r['win_rate'] or 0)*100, '.1f'):>7}"
                f" {_fmt((r['wr_lift'] or 0)*100, '+.1f'):>7}"
                f" {_fmt(r['expectancy']):>7} {_fmt(r['edge_ratio']):>7}\n"
            )
        lines.append("\n")

    # OOS validation of IS top conditions
    lines.append("## OOS validation\n")
    if oos_top.empty:
        lines.append("  No IS conditions survived OOS with >=4% lift.\n\n")
    else:
        lines.append(f"  {'regime_value':<28} {'session':<10} {'dir':<7}"
                     f" {'IS WR%':>8} {'OOS WR%':>9} {'IS lift%':>9} {'OOS lift%':>10} {'flag'}\n")
        for _, r in oos_top.iterrows():
            # Find matching IS row
            mask = (
                (is_result_df["direction"]   == r["direction"])
                & (is_result_df["stop_mult"] == r["stop_mult"])
                & (is_result_df["rr"]        == r["rr"])
                & (is_result_df["horizon"]   == r["horizon"])
                & (is_result_df["regime_type"]  == r["regime_type"])
                & (is_result_df["regime_value"] == r["regime_value"])
                & (is_result_df["session"]   == r["session"])
            )
            is_row = is_result_df.loc[mask]
            is_wr   = is_row["win_rate"].values[0]  if len(is_row) else np.nan
            is_lift = is_row["wr_lift"].values[0]   if len(is_row) else np.nan
            oos_wr  = r["win_rate"]
            oos_lift = r["wr_lift"]
            drop    = (is_wr - oos_wr) if not np.isnan(is_wr) else np.nan
            flag    = "LIKELY_OVERFIT" if (not np.isnan(drop) and drop > 0.10) else ""
            lines.append(
                f"  {str(r['regime_value']):<28} {str(r['session']):<10} {r['direction']:<7}"
                f" {_fmt(is_wr*100,  '.1f'):>8} {_fmt(oos_wr*100,  '.1f'):>9}"
                f" {_fmt(is_lift*100, '+.1f'):>9} {_fmt(oos_lift*100, '+.1f'):>10}  {flag}\n"
            )
        lines.append("\n")

    # R² insight
    lines.append("## R² tertile summary (1h horizon, 1.0 ATR stop, 2.0 RR)\n")
    pivot = is_result_df[
        (is_result_df["horizon"]   == "1h")
        & (is_result_df["stop_mult"] == 1.0)
        & (is_result_df["rr"]        == 2.0)
        & (is_result_df["regime_type"].isin(["baseline", "r2"]))
        & (is_result_df["session"]   == "all")
    ].copy()
    if not pivot.empty:
        lines.append(f"  {'direction':<8} {'bucket':<14} {'n':>6} {'WR%':>7} {'lift%':>7} {'edge_r':>8}\n")
        for _, r in pivot.sort_values(["direction", "regime_value"]).iterrows():
            bucket = r["regime_value"] if r["regime_type"] != "baseline" else "baseline"
            lines.append(
                f"  {r['direction']:<8} {str(bucket):<14}"
                f" {int(r['n']):>6} {_fmt((r['win_rate'] or 0)*100, '.1f'):>7}"
                f" {_fmt((r['wr_lift'] or 0)*100, '+.1f'):>7}"
                f" {_fmt(r['edge_ratio']):>8}\n"
            )

    with open(f"{out_dir}/summary.md", "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"  Saved to {out_dir}/")
