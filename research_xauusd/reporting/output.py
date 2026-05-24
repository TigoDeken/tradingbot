import os
import json
import datetime
import numpy as np
import pandas as pd


def _df(rows):
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _safe_fmt(v, fmt=".4f"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    try:
        return format(v, fmt)
    except (TypeError, ValueError):
        return str(v)


def save_all_results(all_results, cfg, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)

    confirmed_rows = []
    wf_rows        = []
    md_sections    = []

    md_sections.append(f"# XAU/USD N-Bar Range Axiom Study\n")
    md_sections.append(f"Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n")
    md_sections.append(f"Symbol:    {cfg['symbol']}\n")
    md_sections.append(f"N values:  {cfg['N_values']}\n\n")

    # ── Per-timeframe sections ────────────────────────────────────────────────
    for tf, res in all_results.items():
        md_sections.append(f"---\n## Timeframe: {tf}\n")
        md_sections.append(f"In-sample bars:  {res.get('n_is', '?'):,}\n")
        md_sections.append(f"Out-sample bars: {res.get('n_oos', '?'):,}\n\n")

        # Axiom 3
        a3 = res.get("axiom3", {})
        df3 = _df(a3.get("rows", []))
        if not df3.empty:
            df3["timeframe"] = tf
            df3.to_csv(f"{out_dir}/axiom3_{tf}.csv", index=False)
            _append_axiom3_md(md_sections, df3, tf)

        # Axiom 4
        a4 = res.get("axiom4", {})
        df4 = _df(a4.get("rows", []))
        if not df4.empty:
            df4["timeframe"] = tf
            df4.to_csv(f"{out_dir}/axiom4_{tf}.csv", index=False)
            _append_axiom4_md(md_sections, df4, tf)

        # Axiom 6
        a6 = res.get("axiom6", {})
        df6f = _df(a6.get("rows_filtered", []))
        df6r = _df(a6.get("rows_raw", []))
        if not df6f.empty:
            df6f["timeframe"] = tf
            df6f.to_csv(f"{out_dir}/axiom6_filtered_{tf}.csv", index=False)
        if not df6r.empty:
            df6r["timeframe"] = tf
            df6r.to_csv(f"{out_dir}/axiom6_raw_{tf}.csv", index=False)
        _append_axiom6_md(md_sections, df6f, df6r, tf)

        # Axiom 8
        a8 = res.get("axiom8", {})
        df8c = _df(a8.get("corr_rows", []))
        df8d = _df(a8.get("direction_rows", []))
        if not df8c.empty:
            df8c["timeframe"] = tf
            df8c.to_csv(f"{out_dir}/axiom8_corr_{tf}.csv", index=False)
        if not df8d.empty:
            df8d["timeframe"] = tf
            df8d.to_csv(f"{out_dir}/axiom8_direction_{tf}.csv", index=False)
        _append_axiom8_md(md_sections, df8d, tf)

        # Scoring
        sc = res.get("scoring", {})
        dfs = _df(sc.get("rows", []))
        if not dfs.empty:
            dfs["timeframe"] = tf
            dfs.to_csv(f"{out_dir}/scoring_{tf}.csv", index=False)
        _append_scoring_md(md_sections, dfs, tf)

        # Walk-forward
        wf = res.get("walkforward", {})
        for condition_key, wf_row in wf.items():
            wf_row["timeframe"]     = tf
            wf_row["condition_key"] = condition_key
            wf_rows.append(wf_row)
            overfit = wf_row.get("overfit_flag", "")
            if wf_row.get("oos_rev_rate", np.nan) is not np.nan:
                md_sections.append(
                    f"  WF {condition_key}: IS {_safe_fmt(wf_row.get('rev_rate'),'0.3f')}"
                    f" -> OOS {_safe_fmt(wf_row.get('oos_rev_rate'),'0.3f')}"
                    f"  n_oos={wf_row.get('oos_n','?')}  {overfit}\n"
                )

        # Collect confirmed rows
        for source, rows in [
            ("axiom3", a3.get("rows", [])),
            ("axiom4", a4.get("rows", [])),
            ("axiom8_dir", a8.get("direction_rows", [])),
            ("scoring", sc.get("rows", [])),
        ]:
            for r in rows:
                if r.get("confirmed"):
                    r2 = dict(r)
                    r2["timeframe"] = tf
                    r2["source"]    = source
                    confirmed_rows.append(r2)

    # ── Write combined CSVs ───────────────────────────────────────────────────
    if wf_rows:
        pd.DataFrame(wf_rows).to_csv(f"{out_dir}/walk_forward_summary.csv", index=False)

    if confirmed_rows:
        pd.DataFrame(confirmed_rows).to_csv(f"{out_dir}/confirmed_edge.csv", index=False)
        md_sections.append("\n---\n## CONFIRMED EDGE\n")
        for r in confirmed_rows:
            md_sections.append(
                f"  {r['timeframe']} | {r['source']} | "
                f"n={r.get('n','?')} | "
                f"rev={_safe_fmt(r.get('rev_rate'),'0.3f')} | "
                f"rr={_safe_fmt(r.get('mean_rr'),'0.3f')}\n"
            )
    else:
        md_sections.append("\n---\n## No confirmed edge found.\n")

    with open(f"{out_dir}/summary.md", "w", encoding="utf-8") as f:
        f.writelines(md_sections)

    print(f"\nResults saved to ./{out_dir}/")
    print(f"  Confirmed edge conditions: {len(confirmed_rows)}")


# ── Markdown helpers ──────────────────────────────────────────────────────────

def _append_axiom3_md(sections, df, tf):
    sections.append(f"\n### Axiom 3 — Range Width ({tf})\n")
    for N in sorted(df["N"].unique()):
        sub = df[df["N"] == N]
        sections.append(f"N={N}\n")
        sections.append(f"  {'setup':<6} {'bucket':<8} {'n':>5} {'rev%':>7} {'mean_rr':>8} {'cont%':>7} {'p_bh':>8}\n")
        for _, row in sub[sub["setup"].isin(["top", "bot"])].iterrows():
            rev   = _safe_fmt(row.get("rev_rate", np.nan) * 100 if not np.isnan(row.get("rev_rate", np.nan) or np.nan) else np.nan, ".1f")
            cont  = _safe_fmt((row.get("cont_rate", np.nan) or np.nan) * 100 if not np.isnan(row.get("cont_rate", np.nan) or np.nan) else np.nan, ".1f")
            rr    = _safe_fmt(row.get("mean_rr"), ".3f")
            p     = _safe_fmt(row.get("p_binom_bh"), ".4f")
            conf  = " *CONFIRMED*" if row.get("confirmed") else ""
            sections.append(
                f"  {str(row.get('setup','')):<6} {str(row.get('bucket','')):<8}"
                f" {row.get('n',0):>5} {rev:>7} {rr:>8} {cont:>7} {p:>8}{conf}\n"
            )


def _append_axiom4_md(sections, df, tf):
    sections.append(f"\n### Axiom 4 — Position Deciles ({tf})\n")
    for N in sorted(df["N"].unique()):
        sub = df[df["N"] == N]
        sections.append(f"N={N}\n")
        sections.append(f"  {'decile':>7} {'n':>6} {'ret1':>8} {'fade_rr':>8} {'sharpe':>8} {'p_bh':>8}\n")
        for _, row in sub.iterrows():
            sections.append(
                f"  {int(row.get('decile',0)):>7}"
                f" {row.get('n',0):>6}"
                f" {_safe_fmt(row.get('mean_ret1_atr'), '.4f'):>8}"
                f" {_safe_fmt(row.get('fade_mean_rr'), '.4f'):>8}"
                f" {_safe_fmt(row.get('fade_sharpe'), '.2f'):>8}"
                f" {_safe_fmt(row.get('p_t_bh'), '.4f'):>8}\n"
            )


def _append_axiom6_md(sections, df_filt, df_raw, tf):
    sections.append(f"\n### Axiom 6 — Time of Day ({tf})\n")
    if not df_raw.empty:
        sections.append("Raw bars mean return per hour (ATR units):\n")
        sections.append(f"  {'hour':>5} {'n':>7} {'mean_ret':>10} {'sharpe':>8} {'p_bh':>8}\n")
        for _, row in df_raw.iterrows():
            sections.append(
                f"  {int(row.get('hour_utc',0)):>5}"
                f" {row.get('n',0):>7}"
                f" {_safe_fmt(row.get('mean_ret_atr'), '.5f'):>10}"
                f" {_safe_fmt(row.get('sharpe'), '.2f'):>8}"
                f" {_safe_fmt(row.get('p_t_bh'), '.4f'):>8}\n"
            )

    if not df_filt.empty:
        confirmed = df_filt[df_filt.get("confirmed", False) == True] if "confirmed" in df_filt.columns else pd.DataFrame()
        if not confirmed.empty:
            sections.append("Confirmed time windows (extreme filter):\n")
            for _, row in confirmed.iterrows():
                sections.append(
                    f"  N={row.get('N')} {row.get('split_type')}={row.get('bucket')}"
                    f" {row.get('setup')} rev={_safe_fmt(row.get('rev_rate'),'0.3f')}"
                    f" rr={_safe_fmt(row.get('mean_rr'),'0.3f')} n={row.get('n')}\n"
                )


def _append_axiom8_md(sections, df_dir, tf):
    sections.append(f"\n### Axiom 8 — Bar Structure ({tf})\n")
    if df_dir.empty:
        return
    sections.append(f"  {'N':>4} {'setup':<6} {'dir':<8} {'n':>5} {'rev%':>7} {'mean_rr':>8} {'p_bh':>8}\n")
    for _, row in df_dir.iterrows():
        rev  = _safe_fmt((row.get("rev_rate") or np.nan) * 100, ".1f")
        conf = " *CONFIRMED*" if row.get("confirmed") else ""
        sections.append(
            f"  {row.get('N',0):>4} {str(row.get('setup','')):<6}"
            f" {str(row.get('bar_dir','')):<8}"
            f" {row.get('n',0):>5} {rev:>7}"
            f" {_safe_fmt(row.get('mean_rr'), '.3f'):>8}"
            f" {_safe_fmt(row.get('p_binom_bh'), '.4f'):>8}{conf}\n"
        )


def _append_scoring_md(sections, df, tf):
    sections.append(f"\n### Combined Scoring ({tf})\n")
    if df.empty:
        return
    sections.append(f"  {'bucket':<8} {'n':>6} {'rev%':>7} {'mean_rr':>8} {'sharpe':>8} {'p_bh':>8}\n")
    for _, row in df.iterrows():
        rev  = _safe_fmt((row.get("rev_rate") or np.nan) * 100, ".1f")
        conf = " *CONFIRMED*" if row.get("confirmed") else ""
        sections.append(
            f"  {str(row.get('score_bucket','')):<8}"
            f" {row.get('n',0):>6} {rev:>7}"
            f" {_safe_fmt(row.get('mean_rr'), '.3f'):>8}"
            f" {_safe_fmt(row.get('sharpe'), '.2f'):>8}"
            f" {_safe_fmt(row.get('p_binom_bh'), '.4f'):>8}{conf}\n"
        )
