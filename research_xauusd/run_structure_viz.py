"""
Structure filter visualisation — GOLD# M5.

Fetches recent data, computes the 12-bar rolling structure classification,
and plots one London session (07-10h UTC) and one NY session (12-15h UTC).

Per bar the chart shows:
  Background colour  — rolling 12-bar classification
                       green  = "up"
                       red    = "down"
                       grey   = "ranging"
                       none   = NaN (warmup / outside session window)
  Dot row below bars — individual bar type vs previous bar
                       green triangle-up   = clean up
                       red   triangle-down = clean down
                       grey  circle        = dirty (outside / inside / wick rejection)

Usage: cd research_xauusd && python run_structure_viz.py
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.loader   import connect, disconnect, fetch_ohlcv
from regime_filter import classify_structure, SESSION_WINDOWS

SYMBOL    = "GOLD#"
TIMEFRAME = "M5"
N_BARS    = 800          # ~2.8 days — enough warmup + sessions to pick from
OUT_DIR   = "results/structure_viz"
WINDOW    = 12

def _cleanness_color(score):
    """Map 0-100 cleanness score to a background colour. Higher = more opaque green."""
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return None
    alpha = 0.05 + (score / 100) * 0.30   # 0.05 at 0%, 0.35 at 100%
    return f"rgba(38,166,154,{alpha:.2f})"


# ── Per-bar classification (clean_up / clean_down / dirty) ───────────────────

def bar_types(df):
    H, L, C = df["High"], df["Low"], df["Close"]
    prev_H, prev_L = H.shift(1), L.shift(1)
    clean_up   = (H > prev_H) & (L >= prev_L) & (C > prev_H)
    clean_down = (L < prev_L) & (H <= prev_H) & (C < prev_L)
    bt = pd.Series("dirty", index=df.index, dtype=object)
    bt[clean_up]   = "clean_up"
    bt[clean_down] = "clean_down"
    bt.iloc[0]     = np.nan   # first bar has no previous bar
    return bt


# ── Build one chart for a session window ─────────────────────────────────────

def build_chart(df_ctx, struct, bt, session_name, sess_start, sess_end):
    """
    df_ctx  : DataFrame of bars to display (context + session)
    struct  : full classify_structure Series (same index as df_ctx)
    bt      : full bar_types Series (same index as df_ctx)
    """
    BAR_WIDTH = pd.Timedelta(minutes=5)

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_ctx.index,
        open=df_ctx["Open"], high=df_ctx["High"],
        low=df_ctx["Low"],   close=df_ctx["Close"],
        increasing=dict(line=dict(color="#26a69a", width=0.8), fillcolor="#1a3d38"),
        decreasing=dict(line=dict(color="#ef5350", width=0.8), fillcolor="#3d1a1a"),
        showlegend=False, name="price",
    ))

    # Background rectangles per bar — opacity driven by cleanness score
    for ts in df_ctx.index:
        score = struct.loc[ts] if ts in struct.index else None
        color = _cleanness_color(score)
        if not color:
            continue
        fig.add_shape(
            type="rect",
            x0=(ts - BAR_WIDTH * 0.45).isoformat(),
            x1=(ts + BAR_WIDTH * 0.45).isoformat(),
            y0=0, y1=1, yref="paper",
            fillcolor=color, line=dict(width=0), layer="below",
        )
        # Score label above bar
        if not (isinstance(score, float) and np.isnan(score)):
            fig.add_annotation(
                x=ts.isoformat(), y=1.0, yref="paper",
                text=f"{score:.0f}",
                showarrow=False,
                font=dict(size=7, color="rgba(150,220,200,0.7)", family="monospace"),
                yanchor="top", xanchor="center",
            )

    # Session boundary vertical lines
    for ts in df_ctx.index:
        if ts.hour == sess_start and ts.minute == 0:
            fig.add_shape(type="line",
                x0=ts.isoformat(), x1=ts.isoformat(), y0=0, y1=1, yref="paper",
                line=dict(color="#ffffff", width=1, dash="dot"))
            fig.add_annotation(x=ts.isoformat(), y=0.98, yref="paper",
                text=f"{session_name} open", showarrow=False,
                font=dict(color="#fff", size=9), xanchor="left")
        if ts.hour == sess_end and ts.minute == 0:
            fig.add_shape(type="line",
                x0=ts.isoformat(), x1=ts.isoformat(), y0=0, y1=1, yref="paper",
                line=dict(color="#888888", width=1, dash="dot"))
            fig.add_annotation(x=ts.isoformat(), y=0.98, yref="paper",
                text=f"{session_name} close", showarrow=False,
                font=dict(color="#888", size=9), xanchor="left")

    # Per-bar type dots — placed just below the bar lows
    price_range = df_ctx["Low"].min()
    offset      = (df_ctx["High"].max() - df_ctx["Low"].min()) * 0.03

    dot_x   = {"clean_up": [], "clean_down": [], "dirty": []}
    dot_y   = {"clean_up": [], "clean_down": [], "dirty": []}

    for ts, row in df_ctx.iterrows():
        btype = bt.loc[ts]
        if not isinstance(btype, str):
            continue
        dot_x[btype].append(ts)
        dot_y[btype].append(df_ctx.loc[ts, "Low"] - offset)

    dot_cfg = {
        "clean_up":   dict(symbol="triangle-up",   color="#26a69a", size=8,  name="clean up"),
        "clean_down": dict(symbol="triangle-down", color="#ef5350", size=8,  name="clean down"),
        "dirty":      dict(symbol="circle",        color="#666",    size=5,  name="dirty"),
    }
    for btype, cfg in dot_cfg.items():
        if dot_x[btype]:
            fig.add_trace(go.Scatter(
                x=dot_x[btype], y=dot_y[btype],
                mode="markers",
                marker=dict(symbol=cfg["symbol"], color=cfg["color"],
                            size=cfg["size"], line=dict(width=0)),
                name=cfg["name"],
            ))

    # Legend
    fig.add_annotation(
        x=0.01, y=0.97, xref="paper", yref="paper",
        text="background opacity = cleanness score (brighter = cleaner)",
        showarrow=False,
        font=dict(color="rgba(150,220,200,0.7)", size=9, family="monospace"),
        xanchor="left", yanchor="top",
    )

    date_str = df_ctx.index[len(df_ctx)//2].strftime("%Y-%m-%d")
    fig.update_layout(
        title=dict(
            text=f"GOLD# M5 — {session_name}  ({sess_start:02d}:00–{sess_end:02d}:00 UTC)  {date_str}"
                 f"<br><sup>Background: cleanness score 0-100 (rolling {WINDOW} bars, brighter = cleaner)  |  "
                 f"Dots: per-bar type (green=clean up, red=clean down, grey=dirty)</sup>",
            font=dict(size=13, color="white"),
        ),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        xaxis=dict(
            rangeslider_visible=False,
            tickfont=dict(color="#aaa", size=9),
            gridcolor="#1e222d",
            type="date",
        ),
        yaxis=dict(tickfont=dict(color="#aaa", size=9), gridcolor="#1e222d"),
        legend=dict(
            font=dict(color="#ccc", size=9),
            bgcolor="rgba(0,0,0,0)",
            orientation="h", y=-0.08,
        ),
        margin=dict(l=50, r=50, t=80, b=60),
        height=520,
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Connecting to MT5...")
    if not connect():
        sys.exit(1)

    try:
        print(f"Fetching {SYMBOL} {TIMEFRAME} ({N_BARS} bars)...")
        df, _ = fetch_ohlcv(SYMBOL, TIMEFRAME, N_BARS, fallbacks=["XAUUSD"])
        print(f"  {len(df):,} bars  {df.index[0].date()} - {df.index[-1].date()}")

        struct = classify_structure(df, window=WINDOW)
        bt     = bar_types(df)

        for sess_name, (sess_start, sess_end) in SESSION_WINDOWS.items():
            if sess_name not in ("london", "ny"):
                continue

            # Find the most recent complete session in the data
            sess_days = sorted(set(
                ts.date() for ts in df.index
                if sess_start <= ts.hour < sess_end
            ))
            if not sess_days:
                print(f"  No {sess_name} session found in data.")
                continue

            # Pick most recent day that has a full session
            target_day = None
            for day in reversed(sess_days):
                sess_bars = df[
                    (df.index.date == day) &
                    (df.index.hour >= sess_start) &
                    (df.index.hour < sess_end)
                ]
                if len(sess_bars) >= 30:
                    target_day = day
                    break

            if target_day is None:
                print(f"  Could not find a complete {sess_name} session.")
                continue

            # Context: 1h before session start to 30min after session end
            tz        = df.index.tz
            ctx_start = (pd.Timestamp(target_day) + pd.Timedelta(hours=sess_start - 1)).tz_localize(tz)
            ctx_end   = (pd.Timestamp(target_day) + pd.Timedelta(hours=sess_end, minutes=30)).tz_localize(tz)
            df_ctx    = df[(df.index >= ctx_start) & (df.index <= ctx_end)]

            if len(df_ctx) < 10:
                print(f"  Not enough bars for {sess_name} context.")
                continue

            print(f"  Plotting {sess_name}  {target_day}"
                  f"  ({len(df_ctx)} bars displayed)")

            fig      = build_chart(df_ctx, struct, bt, sess_name.capitalize(),
                                   sess_start, sess_end)
            out_path = f"{OUT_DIR}/{sess_name}_{target_day}.html"
            fig.write_html(out_path, include_plotlyjs="cdn")
            print(f"  Saved -> {out_path}")

    finally:
        disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
