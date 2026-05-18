"""
dashboard.py — Step 6
Three-page Streamlit dashboard.
Run:  streamlit run dashboard.py
"""
import re
import json
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timezone

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

st.set_page_config(page_title="EURUSD Algo Trader", page_icon="📈", layout="wide")

# ── Optional pipeline imports ─────────────────────────────────────────────────
PIPELINE_OK = False
_import_err  = ""
try:
    from data_pipeline  import get_data
    from swing_engine   import build_swings, PIP
    from trend_engine   import classify_trend
    from trade_engine   import (_compute_setup, _confirmed_at,
                                 _pullback_depths, run_trades)
    from backtest_engine import build_equity, compute_metrics, _split_date
    PIPELINE_OK = True
except Exception as e:
    _import_err = str(e)

# ── MT5 connection probe ──────────────────────────────────────────────────────

def _mt5_connected() -> bool:
    """Return True only if the MetaTrader5 package is importable and the
    terminal reports an active connection. Never raises."""
    try:
        import MetaTrader5 as _mt5
        info = _mt5.terminal_info()
        return info is not None and bool(info.connected)
    except Exception:
        return False


# ── Paths & constants ─────────────────────────────────────────────────────────
OPT_CSV    = Path("optimisation_results.csv")
TRADE_CSV  = Path("trade_log.csv")
EQUITY_CSV = Path("equity_curve.csv")
STATE_JSON = Path("state.json")
LOG_FILE   = Path("trading_log.txt")

PARAM_COLS = [
    "SWING_N", "MIN_SWING_SIZE", "MIN_SWING_INCR",
    "MIN_TREND_SIZE", "TREND_RANGE_RATIO",
    "PULLBACK_LOOKBACK", "STOP_BUFFER", "TP_MODE",
]


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data
def load_opt():
    return pd.read_csv(OPT_CSV) if OPT_CSV.exists() else None

@st.cache_data
def load_trades():
    if not TRADE_CSV.exists():
        return None
    df = pd.read_csv(TRADE_CSV)
    for c in ("entry_date", "exit_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df

@st.cache_data
def load_equity():
    if not EQUITY_CSV.exists():
        return None
    df = pd.read_csv(EQUITY_CSV)
    for c in ("entry_date", "exit_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df

def load_state(refresh_key: int = 0) -> dict | None:
    """Load state.json, refresh_key busts cache."""
    _ = refresh_key  # used only as cache-buster via caller
    if not STATE_JSON.exists():
        return None
    try:
        with STATE_JSON.open() as f:
            return json.load(f)
    except Exception:
        return None


def load_log_lines(refresh_key: int = 0) -> list[str]:
    """Load all lines from trading_log.txt."""
    _ = refresh_key
    if not LOG_FILE.exists():
        return []
    try:
        return LOG_FILE.read_text(errors="replace").splitlines()
    except Exception:
        return []


def parse_paper_trades(lines: list[str]) -> pd.DataFrame:
    """
    Parse [PAPER] events from trading_log.txt into a trade DataFrame.
    Matches open lines then close lines by order of occurrence.
    """
    RE_TS   = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC")
    RE_OPEN = re.compile(
        r"\[PAPER\] Position opened: (LONG|SHORT) @ ([\d.]+)"
        r"\s+SL=([\d.]+)\s+TP=([\d.]+)"
    )
    RE_CLOSE = re.compile(
        r"\[PAPER\] Trade closed \(([^)]+)\): exit=([\d.]+)"
        r"\s+gross=([\d.]+)pip\s+net=([-\d.]+)pip"
        r"\s+P&L=\$?([-\d.]+)\s+balance=\$?([\d.]+)"
    )
    RE_LIMIT = re.compile(
        r"\[PAPER\] Limit order placed: (LONG|SHORT) limit=([\d.]+)"
        r"\s+sl=([\d.]+)\s+tp=([\d.]+)"
    )

    opens: list[dict]  = []
    closes: list[dict] = []
    pending: dict | None = None

    for line in lines:
        ts_m = RE_TS.match(line)
        ts   = ts_m.group(1) if ts_m else None

        if mo := RE_OPEN.search(line):
            pending = dict(
                open_ts=ts,
                direction=mo.group(1),
                entry_price=float(mo.group(2)),
                stop_price=float(mo.group(3)),
                tp_price=float(mo.group(4)),
            )
        elif RE_LIMIT.search(line) and pending is None:
            # record limit placement but don't count as open yet
            pass
        elif (mc := RE_CLOSE.search(line)) and pending is not None:
            row = {**pending,
                   "close_ts":   ts,
                   "exit_reason": mc.group(1),
                   "exit_price":  float(mc.group(2)),
                   "gross_pips":  float(mc.group(3)),
                   "net_pips":    float(mc.group(4)),
                   "pnl_usd":     float(mc.group(5)),
                   "balance":     float(mc.group(6)),
                   }
            closes.append(row)
            pending = None

    records = closes
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    for col in ("open_ts", "close_ts"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    df.insert(0, "trade_#", range(1, len(df) + 1))
    return df


@st.cache_data
def fetch_live(refresh_key: int):
    """MT5 fetch — refresh_key busts the cache on demand."""
    try:
        return get_data(), None
    except Exception as e:
        return None, str(e)


# ── Chart helpers ─────────────────────────────────────────────────────────────

def equity_fig(eq: pd.DataFrame, split_n: int = None) -> go.Figure:
    x  = np.arange(1, len(eq) + 1)
    b  = eq["balance"].values
    dd = eq["drawdown_pct"].values
    mi = int(dd.argmin())

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.04,
        subplot_titles=["Account Balance (USD)", "Drawdown (%)"],
    )
    fig.add_trace(go.Scatter(x=x, y=b, name="Balance",
                              line=dict(color="#4da6ff", width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[x[mi]], y=[b[mi]], mode="markers", name=f"Max DD {dd[mi]:.1f}%",
        marker=dict(color="red", size=14, symbol="x-thin-open",
                    line=dict(width=3, color="red")),
    ), row=1, col=1)
    if split_n:
        for r in (1, 2):
            fig.add_vline(x=split_n + 0.5, line=dict(color="orange", dash="dash", width=1.2),
                           row=r, col=1)
        fig.add_annotation(x=split_n, y=float(b.max()), text="IS | OOS",
                            showarrow=False, font=dict(color="orange", size=11))
    fig.add_trace(go.Scatter(x=x, y=dd, name="Drawdown", fill="tozeroy",
                              line=dict(color="rgba(255,80,80,0.9)", width=0.5),
                              fillcolor="rgba(255,80,80,0.25)"), row=2, col=1)
    fig.update_layout(height=480, template="plotly_dark", showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_xaxes(title_text="Trade #", row=2)
    fig.update_yaxes(title_text="USD", row=1)
    fig.update_yaxes(title_text="%",   row=2)
    return fig


def _metric_row(m: dict, prefix: str = "") -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric(f"{prefix}Trades",
              int(m["total_trades"]) if "total_trades" in m else "—")
    c2.metric(f"{prefix}Win Rate",
              f"{m['win_rate_pct']:.1f}%" if "win_rate_pct" in m else "—")
    c3.metric(f"{prefix}Expectancy",
              f"{m['expectancy_net_r']:.2f} R" if "expectancy_net_r" in m else "—")
    c4.metric(f"{prefix}Max DD",
              f"{m['max_drawdown_pct']:.1f}%" if "max_drawdown_pct" in m else "—")
    dur = m.get("avg_duration_hours")
    c5.metric(f"{prefix}Avg Duration",
              f"{dur:.0f} h" if dur is not None else "—")
    tpm = m.get("trades_per_month")
    c6.metric(f"{prefix}Trades/Month",
              f"{tpm:.1f}" if tpm is not None else "—")


# ── Page 1: Backtest Results ──────────────────────────────────────────────────

def page_backtest() -> None:
    st.title("📊 Backtest Results")

    opt_df   = load_opt()
    trade_df = load_trades()
    eq_df    = load_equity()

    missing = (["`optimisation_results.csv`"] if opt_df   is None else [] +
               ["`trade_log.csv`"]            if trade_df is None else [] +
               ["`equity_curve.csv`"]         if eq_df    is None else [])
    if missing:
        st.error(f"Missing: {', '.join(missing)}. Run `python backtest_engine.py` first.")
        return

    # ── Combo selector ────────────────────────────────────────────────────────
    def _label(row):
        return (
            f"#{int(row.name)+1}  "
            f"N={row['SWING_N']} SZ={row['MIN_SWING_SIZE']} INC={row['MIN_SWING_INCR']} "
            f"TR={int(row['MIN_TREND_SIZE'])} RR={row['TREND_RANGE_RATIO']} "
            f"LB={int(row['PULLBACK_LOOKBACK'])} SB={int(row['STOP_BUFFER'])} TP={row['TP_MODE']}"
            f"  |  OOS exp={row['OOS_expectancy_net_r']:.3f}R  IS exp={row['IS_expectancy_net_r']:.3f}R"
        )

    labels  = [_label(row) for _, row in opt_df.iterrows()]
    sel_idx = st.selectbox("Parameter combination", range(len(labels)),
                            format_func=lambda i: labels[i])
    sel     = opt_df.iloc[sel_idx]

    is_m  = {k[3:]: sel[k] for k in opt_df.columns if k.startswith("IS_")}
    oos_m = {k[4:]: sel[k] for k in opt_df.columns if k.startswith("OOS_")}

    # ── Metric rows ───────────────────────────────────────────────────────────
    st.subheader("In-Sample  (first 70% by date)")
    _metric_row(is_m)
    st.subheader("Out-of-Sample  (last 30% by date)")
    _metric_row(oos_m)

    # ── IS vs OOS side by side with delta ─────────────────────────────────────
    st.subheader("Comparison")
    compare = [
        ("total_trades",     "Total Trades",    ""),
        ("win_rate_pct",     "Win Rate",        "%"),
        ("expectancy_net_r", "Expectancy",      " R"),
        ("net_r",            "Total Net R",     " R"),
        ("max_drawdown_pct", "Max Drawdown",    "%"),
        ("trades_per_month", "Trades / Month",  ""),
    ]
    col_is, col_oos = st.columns(2)
    with col_is:
        st.markdown("**In-Sample**")
        for key, label, unit in compare:
            v = is_m.get(key)
            st.metric(label, f"{v}{unit}" if v is not None else "—")
    with col_oos:
        st.markdown("**Out-of-Sample**")
        for key, label, unit in compare:
            iv  = is_m.get(key)
            ov  = oos_m.get(key)
            delta = None
            if iv is not None and ov is not None:
                try:
                    delta = round(float(ov) - float(iv), 3)
                except Exception:
                    pass
            st.metric(label, f"{ov}{unit}" if ov is not None else "—", delta=delta)

    # ── Equity curve ──────────────────────────────────────────────────────────
    st.subheader("Equity Curve  (default-parameter run)")
    split_n = int(is_m.get("total_trades", 0)) or None
    st.plotly_chart(equity_fig(eq_df, split_n=split_n), use_container_width=True)

    # ── Trade log ─────────────────────────────────────────────────────────────
    st.subheader("Trade Log  (default-parameter run)")
    show_cols = [c for c in [
        "trade_id", "direction", "entry_date", "entry_price",
        "exit_date", "exit_price", "exit_reason",
        "net_pips", "net_r", "duration_hours",
    ] if c in trade_df.columns]
    show = trade_df[show_cols].copy()
    if "net_pips" in show.columns:
        show["net_pips"] = show["net_pips"].round(0).astype("Int64")
    st.dataframe(show, use_container_width=True, height=360)

    st.download_button(
        "⬇  Download trade log",
        data=trade_df.to_csv(index=False).encode(),
        file_name="trade_log.csv",
        mime="text/csv",
    )


# ── Page 2: Live Regime Monitor ───────────────────────────────────────────────

def page_live() -> None:
    st.title("📡 Live Regime Monitor")

    if not PIPELINE_OK:
        st.warning(
            "⚠️ **MT5 not connected — live data unavailable.**\n\n"
            "MetaTrader 5 must be installed and running on this machine to use this page. "
            f"_(Import error: {_import_err})_"
        )
        return

    # Load best params from optimisation results
    opt_df = load_opt()
    if opt_df is not None:
        best   = opt_df.iloc[0]
        params = {c: best[c] for c in PARAM_COLS}
        st.caption(f"Using best out-of-sample parameters (rank #1 by OOS expectancy): {dict(params)}")
    else:
        # Fall back to module defaults
        from swing_engine  import SWING_N, MIN_SWING_SIZE, MIN_SWING_INCREMENT
        from trend_engine  import MIN_TREND_SIZE, TREND_RANGE_RATIO
        from trade_engine  import PULLBACK_LOOKBACK, STOP_BUFFER, TP_MODE
        params = dict(
            SWING_N=SWING_N, MIN_SWING_SIZE=MIN_SWING_SIZE,
            MIN_SWING_INCR=MIN_SWING_INCREMENT, MIN_TREND_SIZE=MIN_TREND_SIZE,
            TREND_RANGE_RATIO=TREND_RANGE_RATIO, PULLBACK_LOOKBACK=PULLBACK_LOOKBACK,
            STOP_BUFFER=STOP_BUFFER, TP_MODE=TP_MODE,
        )
        st.info("No optimisation results found — using default parameters.")

    # Refresh button (bumps cache key)
    if "refresh_key" not in st.session_state:
        st.session_state.refresh_key = 0
    _, btn_col = st.columns([5, 1])
    with btn_col:
        if st.button("🔄 Refresh"):
            st.session_state.refresh_key += 1

    with st.spinner("Fetching data from MT5…"):
        df_raw, err = fetch_live(st.session_state.refresh_key)

    if err or df_raw is None:
        st.warning(
            "⚠️ **MT5 not connected — live data unavailable.**\n\n"
            "Make sure MetaTrader 5 is open and logged in to your broker account."
        )
        return

    st.caption(f"Last update: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')} "
               f"| {len(df_raw)} bars")

    # Run pipeline with best params
    swings = build_swings(
        df_raw,
        swing_n=int(params["SWING_N"]),
        min_swing_size=float(params["MIN_SWING_SIZE"]),
        min_swing_increment=float(params["MIN_SWING_INCR"]),
        pip=PIP,
    )
    df_cls = classify_trend(
        df_raw, swings,
        swing_n=int(params["SWING_N"]),
        min_swing_increment=float(params["MIN_SWING_INCR"]),
        min_trend_size=float(params["MIN_TREND_SIZE"]),
        trend_range_ratio=float(params["TREND_RANGE_RATIO"]),
        pip=PIP,
    )

    last      = df_cls.iloc[-1]
    regime    = str(last["regime"])
    close_px  = float(last["Close"])
    strength  = last.get("trend_strength_score")
    last_time = df_cls.index[-1]

    # ── Status banner ─────────────────────────────────────────────────────────
    banner = {"UPTREND": "🟢 UPTREND", "DOWNTREND": "🔴 DOWNTREND", "AMBIGUOUS": "⚫ AMBIGUOUS"}
    st.markdown(f"## {banner.get(regime, regime)}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Close Price",    f"{close_px:.5f}")
    c2.metric("Last Candle",    str(last_time)[:16])
    str_val = (f"{float(strength):.1f} pips"
               if strength is not None and not (isinstance(strength, float) and np.isnan(strength))
               else "—")
    c3.metric("Trend Strength", str_val)
    st.divider()

    # ── Swings & pullbacks ────────────────────────────────────────────────────
    bar_i     = len(df_raw) - 1
    confirmed = _confirmed_at(swings, bar_i, int(params["SWING_N"]))
    direction = "long" if regime == "UPTREND" else "short"

    col_sw, col_pb = st.columns(2)

    with col_sw:
        st.subheader("Last 10 Confirmed Swing Points")
        if confirmed.empty:
            st.info("No confirmed swings yet.")
        else:
            disp = confirmed.tail(10)[["date", "price", "type"]].copy()
            disp["price"] = disp["price"].round(5)
            st.dataframe(disp.reset_index(drop=True), use_container_width=True)

    with col_pb:
        st.subheader("Pullback Depths")
        depths = _pullback_depths(confirmed, direction, PIP) if not confirmed.empty else []
        last5  = depths[-5:]
        if last5:
            pb_df = pd.DataFrame(
                {"pullback_pips": [int(round(d)) for d in last5]},
                index=range(1, len(last5) + 1),
            )
            st.dataframe(pb_df, use_container_width=True)
            med = float(np.median(last5))
            st.metric("Median Pullback (entry offset)", f"{int(round(med))} pips")
        else:
            st.info("Fewer than 2 pullbacks — fallback to 20-bar range median.")

    st.divider()

    # ── Projected signal ──────────────────────────────────────────────────────
    st.subheader("Projected Signal  (if triggered now)")
    if regime not in ("UPTREND", "DOWNTREND"):
        st.info("Market is AMBIGUOUS — no signal.")
    else:
        df_window = df_raw.iloc[max(0, bar_i - 19): bar_i + 1]
        setup = _compute_setup(
            confirmed, direction, df_window,
            int(params["PULLBACK_LOOKBACK"]),
            float(params["STOP_BUFFER"]), PIP,
        )
        if setup:
            entry, stop, tp1 = setup
            rr = abs(tp1 - entry) / abs(entry - stop) if abs(entry - stop) > 0 else 0
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Entry",  f"{entry:.5f}")
            p2.metric("Stop",   f"{stop:.5f}",
                      delta=f"{int(round(abs(entry - stop) / PIP))} pip risk")
            p3.metric("TP1",    f"{tp1:.5f}",
                      delta=f"{int(round(abs(tp1 - entry) / PIP))} pip target")
            p4.metric("R:R",    f"1 : {rr:.2f}")
        else:
            st.info("Structure incomplete for a valid setup at current bar.")


# ── Page 3: Parameter Explorer ────────────────────────────────────────────────

def page_explorer() -> None:
    st.title("🔍 Parameter Explorer")

    opt_df = load_opt()
    if opt_df is None:
        st.error("No optimisation results. Run `python backtest_engine.py` first.")
        return

    best_oos_idx = int(opt_df["OOS_expectancy_net_r"].idxmax())

    # ── Sidebar filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Filters")
        swing_n_f = st.multiselect("SWING_N",            [2, 3],            default=[2, 3])
        size_r    = st.slider("MIN_SWING_SIZE",  15, 25, (15, 25), step=5)
        incr_r    = st.slider("MIN_SWING_INCR",   8, 15,  (8, 15))
        trend_r   = st.slider("MIN_TREND_SIZE",  40, 60, (40, 60), step=10)
        ratio_r   = st.slider("TREND_RANGE_RATIO", 0.30, 0.50, (0.30, 0.50), step=0.05)
        lb_r      = st.slider("PULLBACK_LOOKBACK",  3,  5, (3, 5))
        sb_r      = st.slider("STOP_BUFFER",        3,  8, (3, 8))
        tp_f      = st.multiselect("TP_MODE", ["full", "partial"], default=["full", "partial"])

    mask = (
        opt_df["SWING_N"].isin(swing_n_f) &
        opt_df["MIN_SWING_SIZE"].between(*size_r) &
        opt_df["MIN_SWING_INCR"].between(*incr_r) &
        opt_df["MIN_TREND_SIZE"].between(*trend_r) &
        opt_df["TREND_RANGE_RATIO"].between(*ratio_r) &
        opt_df["PULLBACK_LOOKBACK"].between(*lb_r) &
        opt_df["STOP_BUFFER"].between(*sb_r) &
        opt_df["TP_MODE"].isin(tp_f)
    )
    filt = opt_df[mask].copy()
    st.write(f"Showing **{len(filt)}** of {len(opt_df)} combinations")

    if filt.empty:
        st.warning("No combinations match the current filters.")
        return

    # ── Scatter plot ──────────────────────────────────────────────────────────
    hover_cols = [c for c in PARAM_COLS + [
        "IS_win_rate_pct", "IS_expectancy_net_r", "IS_max_drawdown_pct", "IS_total_trades",
        "OOS_win_rate_pct","OOS_expectancy_net_r","OOS_max_drawdown_pct","OOS_total_trades",
    ] if c in filt.columns]

    # Cast SWING_N to string so plotly treats it as categorical
    plot_df = filt.copy()
    plot_df["SWING_N"] = plot_df["SWING_N"].astype(str)

    fig = px.scatter(
        plot_df,
        x="IS_win_rate_pct", y="IS_expectancy_net_r",
        size="IS_total_trades", size_max=20,
        color="SWING_N",
        color_discrete_map={"2": "#4da6ff", "3": "#ff884d"},
        hover_data=hover_cols,
        template="plotly_dark",
        title="IS Win Rate % vs IS Expectancy R  (bubble size = trade count, colour = SWING_N)",
        labels={
            "IS_win_rate_pct":     "Win Rate % (IS)",
            "IS_expectancy_net_r": "Expectancy R (IS)",
        },
    )

    # Highlight best OOS combo
    if best_oos_idx in filt.index:
        bo = filt.loc[best_oos_idx]
        fig.add_trace(go.Scatter(
            x=[bo["IS_win_rate_pct"]], y=[bo["IS_expectancy_net_r"]],
            mode="markers+text", text=["★ Best OOS"],
            textposition="top center",
            marker=dict(color="yellow", size=18, symbol="star",
                        line=dict(color="white", width=1)),
            name="Best OOS combo",
        ))

    fig.add_hline(y=0, line=dict(color="white", dash="dot", width=0.8))
    st.plotly_chart(fig, use_container_width=True)

    # ── Filtered table ────────────────────────────────────────────────────────
    tbl_cols = [c for c in PARAM_COLS + [
        "IS_total_trades", "IS_win_rate_pct", "IS_expectancy_net_r", "IS_max_drawdown_pct",
        "OOS_total_trades","OOS_win_rate_pct","OOS_expectancy_net_r","OOS_max_drawdown_pct",
    ] if c in filt.columns]
    st.dataframe(
        filt[tbl_cols].sort_values("IS_expectancy_net_r", ascending=False).reset_index(drop=True),
        use_container_width=True, height=400,
    )


# ── Live-trade helpers (used by pages 4 & 5) ─────────────────────────────────

def _load_config_fresh() -> dict | None:
    p = Path("config.json")
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def parse_live_trades(lines: list[str]) -> pd.DataFrame:
    """
    Parse live (non-paper) trade events from trading_log.txt.
    Matches Entry-signal → Limit-filled → Position-closed triples.
    """
    RE_TS     = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC")
    RE_SIG    = re.compile(
        r"Entry signal: (LONG|SHORT)\s+entry=([\d.]+)\s+stop=([\d.]+)\s+tp1=([\d.]+)"
    )
    RE_FILL   = re.compile(r"Limit order filled\. Position #(\d+) opened\.")
    RE_CLOSE  = re.compile(r"Position was closed by MT5")

    records: list[dict] = []
    pending_signal: dict | None = None
    open_trade: dict | None = None
    trade_num = 0

    for line in lines:
        if "[PAPER]" in line:
            continue
        ts_m = RE_TS.match(line)
        ts   = ts_m.group(1) if ts_m else None

        if ms := RE_SIG.search(line):
            pending_signal = dict(
                direction=ms.group(1),
                entry_price=float(ms.group(2)),
                stop_price=float(ms.group(3)),
                tp_price=float(ms.group(4)),
                signal_ts=ts,
            )
        elif (mf := RE_FILL.search(line)) and pending_signal:
            trade_num += 1
            open_trade = {**pending_signal, "open_ts": ts,
                          "ticket": mf.group(1), "trade_#": trade_num}
            pending_signal = None
        elif RE_CLOSE.search(line) and open_trade:
            open_trade["close_ts"]    = ts
            open_trade["exit_reason"] = "mt5_close"
            records.append(dict(open_trade))
            open_trade = None

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    for col in ("open_ts", "close_ts"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def _mt5_account_info() -> dict | None:
    """Return live MT5 account fields or None if unavailable."""
    try:
        import MetaTrader5 as _mt5
        info = _mt5.account_info()
        if info is None:
            return None
        return {
            "balance":     float(info.balance),
            "equity":      float(info.equity),
            "margin":      float(info.margin),
            "free_margin": float(info.margin_free),
            "login":       str(info.login),
            "server":      str(info.server),
            "currency":    str(info.currency),
        }
    except Exception:
        return None


def _emergency_stop_execute() -> tuple[bool, list[str]]:
    """Close all MT5 positions and switch config to paper mode. Returns (ok, messages)."""
    msgs: list[str] = []
    closed = 0
    try:
        import MetaTrader5 as _mt5
        positions = _mt5.positions_get() or []
        for p in positions:
            side = _mt5.ORDER_TYPE_SELL if p.type == 0 else _mt5.ORDER_TYPE_BUY
            req = {
                "action":       _mt5.TRADE_ACTION_DEAL,
                "symbol":       p.symbol,
                "volume":       p.volume,
                "type":         side,
                "position":     p.ticket,
                "deviation":    30,
                "comment":      "EMERGENCY_STOP_DASHBOARD",
                "type_time":    _mt5.ORDER_TIME_GTC,
                "type_filling": _mt5.ORDER_FILLING_IOC,
            }
            res = _mt5.order_send(req)
            if res and res.retcode == _mt5.TRADE_RETCODE_DONE:
                closed += 1
                msgs.append(f"✓ Closed position #{p.ticket} {p.symbol} {p.volume} lots")
            else:
                msgs.append(
                    f"✗ Failed #{p.ticket}: {getattr(res, 'comment', '?')}"
                )
        if not positions:
            msgs.append("No open positions found in MT5.")
    except ImportError:
        msgs.append("MetaTrader5 not importable — positions not closed via API.")
    except Exception as e:
        msgs.append(f"MT5 error: {e}")

    cfg_path = Path("config.json")
    try:
        cfg = json.loads(cfg_path.read_text())
        cfg["paper_mode"] = True
        cfg_path.write_text(json.dumps(cfg, indent=2))
        msgs.append("✓ paper_mode set to true in config.json")
    except Exception as e:
        msgs.append(f"✗ Failed to update config.json: {e}")

    try:
        ts = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S")
        with LOG_FILE.open("a", encoding="utf-8") as lf:
            lf.write(
                f"{ts} UTC [CRITICAL] EMERGENCY STOP executed from dashboard. "
                f"Positions closed: {closed}. paper_mode set to true.\n"
            )
        msgs.append("✓ Action logged to trading_log.txt")
    except Exception as e:
        msgs.append(f"Log write failed: {e}")

    return True, msgs


# ── Page 4: Paper Trading Monitor ────────────────────────────────────────────

def page_paper() -> None:
    st.title("📄 Paper Trading Monitor")

    # ── Refresh controls ──────────────────────────────────────────────────────
    if "paper_refresh_key" not in st.session_state:
        st.session_state.paper_refresh_key = 0

    hdr_l, hdr_r = st.columns([5, 1])
    with hdr_r:
        if st.button("🔄 Refresh Now"):
            st.session_state.paper_refresh_key += 1

    rk    = st.session_state.paper_refresh_key
    state = load_state(rk)
    lines = load_log_lines(rk)

    # ── Status banner ─────────────────────────────────────────────────────────
    st.warning(
        "⚠  **PAPER MODE** — All orders are simulated. "
        "No real positions are opened in MetaTrader 5."
    )

    now_utc = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
    b1, b2, b3 = st.columns(3)
    if state:
        pm = state.get("paper_mode", True)
        b1.metric("Mode",       "📄 PAPER" if pm else "🔴 LIVE")
        b2.metric("Last Update", now_utc)
        cb = state.get("circuit_breaker_active", False)
        b3.metric("Circuit Breaker", "🔴 TRIGGERED" if cb else "🟢 OK")
    else:
        b1.info("No `state.json` found — start `live_trader.py` first.")
        b2.metric("Last Update", now_utc)
        b3.metric("Circuit Breaker", "—")

    st.divider()

    # ── Section 1: Current State ───────────────────────────────────────────────
    st.subheader("1 · Current State  (`state.json`)")
    if not state:
        st.info("state.json not found. Live trader has not been started yet.")
    else:
        s1, s2, s3, s4 = st.columns(4)

        bal = state.get("session_starting_balance") or state.get("current_balance", 0)
        s1.metric("Session Start Balance", f"${bal:,.2f}")

        # derive peak balance from trade history
        hist = state.get("trade_history", [])
        peak = bal
        if hist:
            balances = [t.get("balance", bal) for t in hist if "balance" in t]
            if balances:
                peak = max(balances)
        s2.metric("Peak Balance", f"${peak:,.2f}")

        dd_pct = state.get("session_drawdown_pct", 0.0)
        s3.metric("Session Drawdown", f"{dd_pct:.2f}%",
                  delta_color="inverse" if dd_pct > 0 else "normal")

        open_trade = state.get("open_trade")
        pending_ord = state.get("pending_order")
        if open_trade:
            s4.metric("Position", f"OPEN {open_trade.get('direction','').upper()}")
        elif pending_ord:
            s4.metric("Position", f"PENDING {pending_ord.get('direction','').upper()}")
        else:
            s4.metric("Position", "FLAT")

        # Open trade detail
        if open_trade:
            st.markdown("**Open Position**")
            oc1, oc2, oc3, oc4 = st.columns(4)
            oc1.metric("Direction",  open_trade.get("direction", "—").upper())
            oc2.metric("Entry",      f"{open_trade.get('entry_price', 0):.5f}")
            oc3.metric("Stop",       f"{open_trade.get('stop_price', 0):.5f}")
            oc4.metric("TP",         f"{open_trade.get('tp_price', 0):.5f}")

        if pending_ord:
            st.markdown("**Pending Limit Order**")
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("Direction",    pending_ord.get("direction", "—").upper())
            pc2.metric("Limit Price",  f"{pending_ord.get('limit_price', 0):.5f}")
            pc3.metric("Stop",         f"{pending_ord.get('stop_price', 0):.5f}")
            pc4.metric("TP",           f"{pending_ord.get('tp_price', 0):.5f}")

    st.divider()

    # ── Section 2: Paper Trade Log ─────────────────────────────────────────────
    st.subheader("2 · Paper Trade Log  (parsed from `trading_log.txt`)")
    paper_trades = parse_paper_trades(lines)

    if paper_trades.empty:
        st.info("No completed paper trades found in the log yet.")
    else:
        show_cols = [c for c in [
            "trade_#", "direction", "open_ts", "close_ts",
            "entry_price", "exit_price", "exit_reason",
            "gross_pips", "net_pips", "pnl_usd", "balance",
        ] if c in paper_trades.columns]
        disp = paper_trades[show_cols].copy()
        if "gross_pips" in disp.columns:
            disp["gross_pips"] = disp["gross_pips"].round(1)
        if "net_pips" in disp.columns:
            disp["net_pips"] = disp["net_pips"].round(1)
        st.dataframe(disp, use_container_width=True, height=320)

        wins  = (paper_trades["net_pips"] > 0).sum()
        total = len(paper_trades)
        wr    = wins / total * 100 if total else 0
        tot_pnl = paper_trades["pnl_usd"].sum()
        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("Paper Trades",  total)
        sm2.metric("Win Rate",      f"{wr:.1f}%")
        sm3.metric("Total P&L",     f"${tot_pnl:+,.2f}")
        sm4.metric("Avg Net Pips",  f"{paper_trades['net_pips'].mean():.1f}")

    st.divider()

    # ── Section 3: Paper Equity Curve ─────────────────────────────────────────
    st.subheader("3 · Paper Equity Curve")
    if paper_trades.empty:
        st.info("No trades to plot yet.")
    else:
        eq_fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.70, 0.30], vertical_spacing=0.05,
            subplot_titles=["Paper Balance (USD)", "Trade P&L (USD)"],
        )
        t_idx = paper_trades["trade_#"].values
        bal_v = paper_trades["balance"].values
        pnl_v = paper_trades["pnl_usd"].values
        colors = ["#26c26e" if p >= 0 else "#ef5350" for p in pnl_v]

        eq_fig.add_trace(
            go.Scatter(x=t_idx, y=bal_v, name="Balance",
                       line=dict(color="#4da6ff", width=2)),
            row=1, col=1,
        )
        eq_fig.add_trace(
            go.Bar(x=t_idx, y=pnl_v, name="P&L", marker_color=colors),
            row=2, col=1,
        )
        eq_fig.update_layout(height=460, template="plotly_dark", showlegend=True,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02))
        eq_fig.update_xaxes(title_text="Trade #", row=2)
        eq_fig.update_yaxes(title_text="USD", row=1)
        eq_fig.update_yaxes(title_text="USD", row=2)
        st.plotly_chart(eq_fig, use_container_width=True)

    st.divider()

    # ── Section 4: Paper vs Backtest Comparison ────────────────────────────────
    st.subheader("4 · Paper vs Backtest Comparison")
    opt_df = load_opt()
    if opt_df is None:
        st.info("No optimisation_results.csv found — run the backtest first.")
    elif paper_trades.empty:
        st.info("No completed paper trades to compare yet.")
    else:
        best_row = opt_df.iloc[0]
        metrics = [
            ("Win Rate %",   "win_rate_pct",     "%"),
            ("Expectancy R", "expectancy_net_r",  " R"),
            ("Max DD %",     "max_drawdown_pct",  "%"),
        ]

        # Paper stats — compute R using actual stop distance per trade
        p_wr  = wr  # computed above
        _sp   = (paper_trades["entry_price"] - paper_trades["stop_price"]).abs() / 0.0001
        _nr   = paper_trades["net_pips"] / _sp.replace(0, float("nan"))
        p_exp = float(_nr.mean()) if not _nr.empty else 0.0
        running_peak = paper_trades["balance"].cummax()
        paper_dd = ((running_peak - paper_trades["balance"]) / running_peak * 100).max()

        paper_stats  = {"win_rate_pct": p_wr, "expectancy_net_r": p_exp, "max_drawdown_pct": paper_dd}
        bt_oos_stats = {
            "win_rate_pct":     best_row.get("OOS_win_rate_pct", None),
            "expectancy_net_r": best_row.get("OOS_expectancy_net_r", None),
            "max_drawdown_pct": best_row.get("OOS_max_drawdown_pct", None),
        }

        cmp_l, cmp_r = st.columns(2)
        with cmp_l:
            st.markdown("**Backtest OOS  (best combo)**")
            for label, key, unit in metrics:
                v = bt_oos_stats.get(key)
                st.metric(label, f"{v:.2f}{unit}" if v is not None else "—")
        with cmp_r:
            st.markdown("**Paper Trading (live)**")
            for label, key, unit in metrics:
                pv = paper_stats.get(key)
                bv = bt_oos_stats.get(key)
                delta = None
                if pv is not None and bv is not None:
                    try:
                        delta = round(float(pv) - float(bv), 2)
                    except Exception:
                        pass
                st.metric(label, f"{pv:.2f}{unit}" if pv is not None else "—", delta=delta)

    st.divider()

    # ── Section 5: Last 30 Log Lines ──────────────────────────────────────────
    st.subheader("5 · Last 30 Log Lines  (`trading_log.txt`)")
    if not lines:
        st.info("trading_log.txt not found or empty.")
    else:
        last30 = "\n".join(lines[-30:])
        st.code(last30, language=None)

    st.divider()

    # ── Go Live Checklist ─────────────────────────────────────────────────────
    st.subheader("Go-Live Checklist")
    st.markdown(
        "Review every item before switching `paper_mode` to `false` in `config.json`."
    )
    checks = [
        "Paper mode has run for at least 2 weeks with ≥ 10 completed trades",
        "Paper win rate is within ±10% of OOS backtest win rate",
        "Paper expectancy is positive (> 0 R)",
        "Paper max drawdown is below `max_drawdown_pct` threshold in config.json",
        "MT5 account has been verified as connected (check Page 2)",
        "Risk per trade (`risk_per_trade = 0.005`) confirmed correct for live account size",
        "Circuit breaker threshold (`max_drawdown_pct`) reviewed and accepted",
    ]
    all_checked = all(
        st.checkbox(item, key=f"chk_{i}")
        for i, item in enumerate(checks)
    )
    if all_checked:
        st.success(
            "✅ All items checked. You may set `paper_mode: false` in config.json "
            "and restart with `--live-confirmed` flag."
        )
    else:
        st.info("Complete all checklist items above before going live.")

    # Auto-refresh: render content fully, then sleep 60 s and re-render
    import time as _time
    st.caption(f"Auto-refreshing every 60 s · {now_utc}")
    _time.sleep(60)
    st.session_state.paper_refresh_key += 1
    st.rerun()


# ── Page 5: Live Trading ──────────────────────────────────────────────────────

def page_live_trading() -> None:
    st.title("🔴 Live Trading")

    # ── Refresh controls ──────────────────────────────────────────────────────
    if "live_refresh_key" not in st.session_state:
        st.session_state.live_refresh_key = 0
    if "estop_armed" not in st.session_state:
        st.session_state.estop_armed = False
    if "estop_done" not in st.session_state:
        st.session_state.estop_done = False
    if "live_auto_ts" not in st.session_state:
        import time as _t
        st.session_state.live_auto_ts = _t.time()

    hdr_l, hdr_r = st.columns([5, 1])
    with hdr_r:
        if st.button("🔄 Refresh", key="live_refresh_btn"):
            st.session_state.live_refresh_key += 1

    cfg   = _load_config_fresh()
    rk    = st.session_state.live_refresh_key
    state = load_state(rk)
    lines = load_log_lines(rk)
    now_utc = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")

    is_live = cfg is not None and not cfg.get("paper_mode", True)

    # ── Emergency stop ────────────────────────────────────────────────────────
    if st.session_state.estop_done:
        st.success(
            "🛑 Emergency stop executed. All positions closed. "
            "paper_mode is now active. Restart live_trader.py to resume paper trading."
        )
    elif st.session_state.estop_armed:
        st.error(
            "⚠️  CONFIRM EMERGENCY STOP\n\n"
            "This will immediately close ALL open MT5 positions and switch config.json "
            "to paper_mode=true. This action cannot be undone."
        )
        ea_col, ec_col = st.columns(2)
        with ea_col:
            if st.button("✅ YES — EXECUTE EMERGENCY STOP", type="primary",
                         key="estop_confirm_btn"):
                _, msgs = _emergency_stop_execute()
                st.session_state.estop_armed = False
                st.session_state.estop_done  = True
                for m in msgs:
                    st.write(m)
                st.rerun()
        with ec_col:
            if st.button("❌ Cancel", key="estop_cancel_btn"):
                st.session_state.estop_armed = False
                st.rerun()
    else:
        if st.button("🔴 EMERGENCY STOP", type="primary", key="estop_arm_btn"):
            st.session_state.estop_armed = True
            st.rerun()

    st.divider()

    # ── Status banner ─────────────────────────────────────────────────────────
    if is_live:
        st.success("## 🟢  LIVE TRADING ACTIVE")
        st.caption(f"paper_mode = false · Last render: {now_utc}")
    else:
        st.warning("## 📄  PAPER MODE")
        st.caption(
            "paper_mode = true · Set paper_mode=false in config.json and restart "
            "live_trader.py with --live-confirmed to go live."
        )

    st.divider()

    # ── Section 1: MT5 Account ────────────────────────────────────────────────
    st.subheader("1 · MT5 Account")
    acct = _mt5_account_info()
    if acct is None:
        st.warning(
            "⚠️ **MT5 not connected — live data unavailable.**\n\n"
            "Open MetaTrader 5 and log in to your broker account to see live balance, "
            "equity, and margin information."
        )
    else:
        st.caption(
            f"Account: {acct['login']} · Server: {acct['server']} · "
            f"Currency: {acct['currency']}"
        )
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Balance",      f"${acct['balance']:,.2f}")
        a2.metric("Equity",       f"${acct['equity']:,.2f}")
        a3.metric("Margin Used",  f"${acct['margin']:,.2f}")
        a4.metric("Free Margin",  f"${acct['free_margin']:,.2f}")

        # Balance changes
        sb = state.get("session_starting_balance") if state else None
        if sb and sb > 0:
            delta_usd = acct["balance"] - sb
            delta_pct = delta_usd / sb * 100
            ch1, ch2 = st.columns(2)
            ch1.metric(
                "Change Since Session Start",
                f"${delta_usd:+,.2f}",
                delta=f"{delta_pct:+.2f}%",
                delta_color="normal" if delta_usd >= 0 else "inverse",
            )
            # Today approximation — use equity vs balance as proxy for open P&L
            open_pnl = acct["equity"] - acct["balance"]
            ch2.metric(
                "Open P&L (equity − balance)",
                f"${open_pnl:+,.2f}",
                delta_color="normal" if open_pnl >= 0 else "inverse",
            )

    st.divider()

    # ── Section 2: Current State ───────────────────────────────────────────────
    st.subheader("2 · Current State")
    if not state:
        st.info("state.json not found. Start live_trader.py first.")
    else:
        regime   = state.get("last_regime", "—")
        cb_on    = state.get("circuit_breaker_active", False)
        cb_rsn   = state.get("circuit_breaker_reason", "")
        last_upd = state.get("last_updated", "—")
        open_t   = state.get("open_trade")
        pending  = state.get("pending_limit")

        r_map = {"UPTREND": "🟢 UPTREND", "DOWNTREND": "🔴 DOWNTREND",
                 "AMBIGUOUS": "⚫ AMBIGUOUS"}
        s1, s2, s3 = st.columns(3)
        s1.metric("Regime",          r_map.get(regime, regime))
        s2.metric("Circuit Breaker", "🔴 TRIGGERED" if cb_on else "🟢 ACTIVE")
        s3.metric("Last Candle",     str(last_upd)[:19])

        if cb_on and cb_rsn:
            st.error(f"CB reason: {cb_rsn}")

        if open_t:
            st.markdown("**Open Live Position**")
            direction = open_t.get("direction", "").upper()
            entry_px  = open_t.get("entry", 0)
            stop_px   = open_t.get("stop", 0)
            tp_px     = open_t.get("tp1", 0)
            lot       = open_t.get("lot", 0)
            entry_ts  = open_t.get("entry_time", "—")

            # Current price from MT5 if available
            cur_px = None
            cur_pips = "—"
            cur_r    = "—"
            try:
                import MetaTrader5 as _mt5
                tick = _mt5.symbol_info_tick(cfg["symbol"] if cfg else "EURUSD")
                if tick:
                    cur_px    = float(tick.bid if direction == "LONG" else tick.ask)
                    pip       = 0.0001
                    sign      = 1 if direction == "LONG" else -1
                    gross     = sign * (cur_px - entry_px) / pip
                    cur_pips  = f"{gross:.1f}"
                    stop_dist = abs(entry_px - stop_px) / pip
                    cur_r     = f"{gross / stop_dist:.2f}" if stop_dist > 0 else "—"
            except Exception:
                pass

            ot1, ot2, ot3, ot4, ot5, ot6 = st.columns(6)
            ot1.metric("Direction",    direction)
            ot2.metric("Entry",        f"{entry_px:.5f}")
            cur_str = f"{cur_px:.5f}" if cur_px else "—"
            ot3.metric("Current Price", cur_str, delta=cur_pips + " pips" if cur_px else None)
            ot4.metric("Current R",    cur_r)
            ot5.metric("Stop",         f"{stop_px:.5f}")
            ot6.metric("TP1",          f"{tp_px:.5f}")
            st.caption(f"Entry time: {entry_ts}  |  Lots: {lot}")

        elif pending:
            st.markdown("**Pending Limit Order**")
            pd1, pd2, pd3, pd4 = st.columns(4)
            pd1.metric("Direction",   pending.get("direction", "—").upper())
            pd2.metric("Limit Price", f"{pending.get('price', 0):.5f}")
            pd3.metric("Stop",        f"{pending.get('stop', 0):.5f}")
            pd4.metric("TP1",         f"{pending.get('tp1', 0):.5f}")
        else:
            st.info("No open position or pending order.")

    st.divider()

    # ── Section 3: Live Trade Log ──────────────────────────────────────────────
    st.subheader("3 · Live Trade Log")
    live_trades = parse_live_trades(lines)

    if is_live and live_trades.empty:
        st.info("No live trades recorded yet in trading_log.txt.")
    elif not is_live and live_trades.empty:
        st.info(
            "No live trades found — system is in paper mode. "
            "Live trades will appear here after switching to live mode."
        )
    else:
        show_cols = [c for c in [
            "trade_#", "direction", "open_ts", "close_ts",
            "entry_price", "stop_price", "tp_price", "exit_reason", "ticket",
        ] if c in live_trades.columns]
        st.dataframe(live_trades[show_cols], use_container_width=True, height=300)

        total = len(live_trades)
        st.caption(f"Total live trades recorded: {total}")

    st.divider()

    # ── Section 4: Three Equity Curves ────────────────────────────────────────
    st.subheader("4 · Equity Curves  (Backtest · Paper · Live)")
    st.caption("Backtest curve loads from CSV — no MT5 connection required.")

    eq_bt     = load_equity()
    paper_trd = parse_paper_trades(lines)
    live_trd  = live_trades  # already parsed above

    fig4 = go.Figure()
    has_any = False

    if eq_bt is not None and "balance" in eq_bt.columns:
        x_bt = np.arange(1, len(eq_bt) + 1)
        fig4.add_trace(go.Scatter(
            x=x_bt, y=eq_bt["balance"].values,
            name="Backtest", line=dict(color="#888888", width=1.5, dash="dot"),
        ))
        has_any = True

    if not paper_trd.empty and "balance" in paper_trd.columns:
        x_p = paper_trd["trade_#"].values
        fig4.add_trace(go.Scatter(
            x=x_p, y=paper_trd["balance"].values,
            name="Paper", line=dict(color="#4da6ff", width=2),
        ))
        has_any = True

    if not live_trd.empty:
        # Live trades don't carry balance from logs — annotate count only
        st.caption(
            f"Live curve: {len(live_trd)} trade(s) logged. "
            "Full balance tracking requires live_trader to be running."
        )

    if not has_any:
        st.info("No equity data available yet. Run backtest and/or start paper trading.")
    else:
        fig4.update_layout(
            height=400, template="plotly_dark",
            xaxis_title="Trade #", yaxis_title="Balance (USD)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            title="Normalised equity curves — all starting from respective initial balance",
        )
        fig4.add_hline(y=10_000, line=dict(color="white", dash="dot", width=0.6))
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # ── Section 5: Live vs Paper vs Backtest Comparison ───────────────────────
    st.subheader("5 · Live vs Paper vs Backtest Comparison")

    opt_df = load_opt()

    def _safe_metric(df: pd.DataFrame, col: str, fn):
        try:
            return float(fn(df[col])) if not df.empty and col in df.columns else None
        except Exception:
            return None

    def _paper_stats(trd: pd.DataFrame) -> dict:
        if trd.empty:
            return {}
        wins   = (trd["net_pips"] > 0).sum()
        total  = len(trd)
        wr     = wins / total * 100 if total else 0
        _sp    = (trd["entry_price"] - trd["stop_price"]).abs() / 0.0001
        _nr    = trd["net_pips"] / _sp.replace(0, float("nan"))
        exp    = float(_nr.mean()) if not _nr.empty else 0.0
        avg_w  = float(_nr[trd["net_pips"] > 0].mean()) if wins else None
        avg_l  = float(_nr[trd["net_pips"] <= 0].mean()) if (total - wins) else None
        peak   = trd["balance"].cummax()
        max_dd = ((peak - trd["balance"]) / peak * 100).max()
        return dict(win_rate=wr, expectancy=exp, avg_win_r=avg_w,
                    avg_loss_r=avg_l, max_dd=max_dd, trades=total)

    def _bt_oos_stats(od: pd.DataFrame | None) -> dict:
        if od is None or od.empty:
            return {}
        best = od.iloc[0]
        return dict(
            win_rate=best.get("OOS_win_rate_pct"),
            expectancy=best.get("OOS_expectancy_net_r"),
            max_dd=best.get("OOS_max_drawdown_pct"),
        )

    paper_s  = _paper_stats(paper_trd)
    live_s   = _paper_stats(parse_paper_trades([])) if live_trd.empty else {}
    bt_s     = _bt_oos_stats(opt_df)

    metrics_def = [
        ("Win Rate %",       "win_rate",    "win_rate",    "%",  False),
        ("Expectancy R",     "expectancy",  "expectancy",  " R", False),
        ("Avg Win R",        "avg_win_r",   None,          " R", False),
        ("Avg Loss R",       "avg_loss_r",  None,          " R", True),
        ("Max Drawdown %",   "max_dd",      "max_dd",      "%",  True),
    ]

    def _fmt(v, unit):
        return f"{v:.2f}{unit}" if v is not None else "—"

    red_count = 0
    col_bt, col_paper, col_live = st.columns(3)
    with col_bt:
        st.markdown("**📊 Backtest OOS**")
        for label, key, bt_key, unit, higher_is_worse in metrics_def:
            v = bt_s.get(bt_key or key)
            st.metric(label, _fmt(v, unit))

    with col_paper:
        st.markdown("**📄 Paper Trading**")
        for label, key, bt_key, unit, higher_is_worse in metrics_def:
            pv = paper_s.get(key)
            bv = bt_s.get(bt_key or key)
            is_red = False
            if pv is not None and bv is not None:
                try:
                    if not higher_is_worse and float(pv) < float(bv) * 0.80:
                        is_red = True
                    elif higher_is_worse and float(pv) > float(bv) * 1.20:
                        is_red = True
                except Exception:
                    pass
            label_str = f"🔴 {label}" if is_red else label
            st.metric(label_str, _fmt(pv, unit))

    with col_live:
        st.markdown(f"**🔴 Live Trading {'(ACTIVE)' if is_live else '(no trades yet)'}**")
        live_warns = 0
        for label, key, bt_key, unit, higher_is_worse in metrics_def:
            lv = live_s.get(key)
            bv = bt_s.get(bt_key or key)
            is_red = False
            if lv is not None and bv is not None:
                try:
                    if not higher_is_worse and float(lv) < float(bv) * 0.80:
                        is_red = True
                        live_warns += 1
                    elif higher_is_worse and float(lv) > float(bv) * 1.20:
                        is_red = True
                        live_warns += 1
                except Exception:
                    pass
            label_str = f"🔴 {label}" if is_red else label
            st.metric(label_str, _fmt(lv, unit))

        if live_warns >= 2:
            st.error(
                "⚠️  System behavior has changed significantly. "
                "2 or more live metrics are worse than backtest by >20%. "
                "Review live trades before continuing."
            )

    st.divider()

    # ── Section 6: Recent Log Entries ─────────────────────────────────────────
    st.subheader("6 · Recent Log Entries")
    if not lines:
        st.info("trading_log.txt not found or empty.")
    else:
        def _tag(line: str) -> str:
            if "[PAPER]" in line:
                return f"[PAPER] {line}"
            for kw in ("Position", "Limit order", "Entry signal", "Trade", "Stop modified",
                        "Circuit", "HALT"):
                if kw in line:
                    return f"[LIVE]  {line}"
            return line

        last30 = [_tag(l) for l in lines[-30:]]
        last30.reverse()
        st.code("\n".join(last30), language=None)

    # Auto-refresh: render content fully, then sleep 60 s and re-render
    import time as _time2
    st.caption(f"Auto-refreshing every 60 s · {now_utc}")
    _time2.sleep(60)
    st.session_state.live_refresh_key += 1
    st.rerun()


# ── Navigation ────────────────────────────────────────────────────────────────

PAGES = {
    "📊 Backtest Results":      page_backtest,
    "📡 Live Regime Monitor":   page_live,
    "🔍 Parameter Explorer":    page_explorer,
    "📄 Paper Trading Monitor": page_paper,
    "🔴 Live Trading":          page_live_trading,
}

with st.sidebar:
    st.title("EURUSD Algo Trader")
    st.divider()
    page = st.radio("", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    if _mt5_connected():
        st.success("🟢 MT5 connected")
    else:
        st.error("🔴 MT5 not connected")
    st.caption("Run `python backtest_engine.py` to generate data files.")

PAGES[page]()
