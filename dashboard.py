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

st.set_page_config(page_title="Algo Trader", page_icon="📈", layout="wide")

# ── Optional pipeline imports ─────────────────────────────────────────────────
from constants import PIP

PIPELINE_OK = False
_import_err  = ""
try:
    from data_pipeline   import get_data
    from trade_engine    import run_trades, _wm_levels, _atr
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
        _mt5.initialize()
        info = _mt5.terminal_info()
        return info is not None and bool(info.connected)
    except Exception:
        return False


# ── Paths & constants ─────────────────────────────────────────────────────────
OPT_CSV     = Path("optimisation_results.csv")
TRADE_CSV   = Path("trade_log.csv")
EQUITY_CSV  = Path("equity_curve.csv")
STATE_JSON  = Path("state.json")
LOG_FILE    = Path("trading_log.txt")
CONFIG_PATH = Path("config.json")

ALL_SYMBOLS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD"]

RESULTS_DIR = Path("results")

PARAM_COLS = ["fixed_stop_pips", "tp_rr", "stop_atr_mult", "atr_max_pips"]

def _pip_for(sym: str) -> float:
    return 0.01 if "JPY" in sym else PIP


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


def _load_state(symbol: str) -> dict | None:
    """Load per-symbol state_{symbol}.json."""
    path = Path(f"state_{symbol}.json")
    if not path.exists():
        return None
    try:
        with path.open() as f:
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
def fetch_live(refresh_key: int, symbol: str = "EURUSD"):
    """MT5 fetch — refresh_key busts the cache on demand."""
    try:
        return get_data(symbol=symbol), None
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

def _list_runs() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        [d for d in RESULTS_DIR.iterdir() if d.is_dir() and (d / "summary.json").exists()],
        reverse=True,
    )


def page_backtest() -> None:
    st.title("📊 Backtest Results")

    _METRICS = [
        ("total_trades",     "Trades",      ""),
        ("win_rate_pct",     "Win Rate",    "%"),
        ("expectancy_net_r", "Expectancy",  " R"),
        ("net_ev",           "Net EV",      " R"),
        ("max_drawdown_pct", "Max DD",      "%"),
        ("trades_per_month", "T/Month",     ""),
    ]

    def _metric_pair(col_is, col_oos, is_m, oos_m, split_pct):
        col_is.markdown(f"**In-Sample ({split_pct}%)**")
        col_oos.markdown(f"**Out-of-Sample ({100 - split_pct}%)**")

        for key, lbl, unit in _METRICS:
            iv, ov = is_m.get(key, 0), oos_m.get(key, 0)
            fmt = lambda v, u=unit: f"{v:.1f}{u}" if isinstance(v, float) else f"{v}{u}"
            try:
                delta = round(float(ov) - float(iv), 3)
            except Exception:
                delta = None
            col_is.metric(lbl,  fmt(iv))
            col_oos.metric(lbl, fmt(ov), delta=delta)

        col_is.divider()
        col_oos.divider()
        col_is.markdown("**Trading Costs**")
        col_oos.markdown("**Trading Costs**")

        for m, col in [(is_m, col_is), (oos_m, col_oos)]:
            pre  = m.get('ev_pre_commission',  0) or 0
            post = m.get('ev_post_commission', 0) or 0
            col.metric("EV pre-commission",  f"{pre:.3f} R")
            col.metric("EV post-commission", f"{post:.3f} R")
            col.metric("Commission drag",    f"{round(pre - post, 3):.3f} R/trade")
            col.metric("Commission total",   f"${m.get('commission_total_usd', 0):,.0f}")

    # ── Run new backtest ──────────────────────────────────────────────────────
    if PIPELINE_OK:
        cfg = _load_config_fresh() or {}

        def _cfg_run_params():
            return dict(
                tp_rr         = float(cfg.get("tp_rr", 1.0)),
                tp_mode       = cfg.get("tp_mode", "full"),
                risk_pct      = float(cfg.get("risk_per_trade", 0.01)),
                session_start = int(cfg.get("session_start_utc", 7)),
                session_end   = int(cfg.get("session_end_utc", 20)),
                min_stop_pips = 5, max_lot=10.0,
                max_open_lots = float(cfg.get("max_open_lots", 0.0)),
                fixed_stop_pips = float(cfg.get("fixed_stop_pips", 25.0)),
                atr_period    = int(cfg.get("atr_period", 14)),
                atr_max_pips  = float(cfg.get("atr_max_pips", 0)),
                stop_atr_mult = float(cfg.get("stop_atr_mult", 0.8)),
            )

        with st.expander("▶ Run Backtest", expanded=False):
            rb1, rb2, rb3, rb4 = st.columns(4)
            sym_choice      = rb1.selectbox("Symbol", ["All Symbols"] + ALL_SYMBOLS,
                                            index=ALL_SYMBOLS.index(selected) + 1)
            bars            = rb2.number_input("Bars to fetch",      min_value=100, max_value=50000,      value=9999,  step=100)
            split_pct       = rb3.number_input("In-sample split %",  min_value=10,  max_value=90,         value=70,    step=5)
            initial_balance = rb4.number_input("Initial balance ($)", min_value=100, max_value=10_000_000, value=10000, step=500)
            run_btn = st.button("▶ Run with current Bot Settings", type="primary")

        if run_btn:
            split_ratio = split_pct / 100
            params      = _cfg_run_params()

            def _get_pip_info(sym: str) -> tuple:
                try:
                    import MetaTrader5 as _mt5
                    _mt5.initialize()
                    info = _mt5.symbol_info(sym)
                    if info:
                        pip       = round(info.point * 10, 6)
                        pip_value = round(info.trade_tick_value * (pip / info.trade_tick_size), 4)
                        return pip, pip_value
                except Exception:
                    pass
                return (0.01, 9.0) if "JPY" in sym else (0.0001, 10.0)

            def _run_sym(sym):
                pip, pip_value = _get_pip_info(sym)
                df       = get_data(symbol=sym, bars=bars)
                trades, fill_stats = run_trades(df, account_balance=initial_balance,
                                                pip=pip, pip_value=pip_value, **params)
                split_dt = df.index[0] + (df.index[-1] - df.index[0]) * split_ratio
                is_t     = [t for t in trades if t.entry_date <  split_dt]
                oos_t    = [t for t in trades if t.entry_date >= split_dt]
                return (
                    compute_metrics(is_t,  initial=initial_balance),
                    compute_metrics(oos_t, initial=initial_balance),
                    is_t, oos_t, trades,
                    build_equity(trades, initial=initial_balance),
                    fill_stats,
                )

            if sym_choice == "All Symbols":
                prog    = st.progress(0, "Running all symbols...")
                results = {}
                for i, sym in enumerate(ALL_SYMBOLS):
                    prog.progress(int(i / len(ALL_SYMBOLS) * 95), f"Running {sym}...")
                    try:
                        results[sym] = _run_sym(sym)
                    except Exception as e:
                        results[sym] = None
                        st.warning(f"{sym}: failed — {e}")
                all_fill_stats = {
                    "signals_total":     sum(r[6]["signals_total"]     for r in results.values() if r),
                    "fills_total":       sum(r[6]["fills_total"]       for r in results.values() if r),
                    "expirations_total": sum(r[6]["expirations_total"] for r in results.values() if r),
                }
                prog.progress(100, "Done.")

                # Summary comparison table
                rows = []
                for sym, r in results.items():
                    if r is None:
                        rows.append({"Symbol": sym, "Trades": "—", "Win Rate %": "—",
                                     "Expectancy R": "—", "OOS Net EV R": "—",
                                     "Max DD %": "—", "T/Month": "—"})
                    else:
                        _, oos_m = r[0], r[1]
                        rows.append({
                            "Symbol":       sym,
                            "Trades":       oos_m.get("total_trades", 0),
                            "Win Rate %":   round(oos_m.get("win_rate_pct", 0), 1),
                            "Expectancy R": round(oos_m.get("expectancy_net_r", 0), 4),
                            "OOS Net EV R": round(oos_m.get("net_ev", 0) or 0, 4),
                            "Max DD %":     round(oos_m.get("max_drawdown_pct", 0), 2),
                            "T/Month":      round(oos_m.get("trades_per_month", 0), 1),
                        })
                tbl_df = pd.DataFrame(rows).set_index("Symbol")

                def _color_ev(val):
                    try:
                        return "color: green" if float(val) >= 0 else "color: red"
                    except Exception:
                        return ""

                st.subheader(f"All Symbols — Out-of-Sample Summary (last {100 - split_pct}%)")
                st.dataframe(
                    tbl_df.style.map(_color_ev, subset=["OOS Net EV R"]),
                    use_container_width=True,
                )

                # ── Portfolio combined ────────────────────────────────────────
                all_combined = sorted(
                    [t for r in results.values() if r for t in r[4]],
                    key=lambda t: t.entry_date,
                )
                if all_combined:
                    first_dt         = all_combined[0].entry_date
                    last_dt          = all_combined[-1].entry_date
                    split_dt_comb    = first_dt + (last_dt - first_dt) * split_ratio
                    comb_is_t        = [t for t in all_combined if t.entry_date <  split_dt_comb]
                    comb_oos_t       = [t for t in all_combined if t.entry_date >= split_dt_comb]
                    comb_is_m        = compute_metrics(comb_is_t,  initial=initial_balance)
                    comb_oos_m       = compute_metrics(comb_oos_t, initial=initial_balance)
                    comb_eq          = build_equity(all_combined,  initial=initial_balance)
                    comb_oos_ev      = comb_oos_m.get("net_ev", 0) or 0

                    st.divider()
                    st.subheader("Portfolio — All Symbols Combined")
                    lbl = (f"Portfolio — {len(all_combined)} total trades | "
                           f"OOS EV: {'+' if comb_oos_ev >= 0 else ''}{comb_oos_ev:.4f} R/trade")
                    (st.success if comb_oos_ev >= 0 else st.error)(lbl)
                    if not comb_eq.empty:
                        st.plotly_chart(equity_fig(comb_eq, split_n=len(comb_is_t)), use_container_width=True)
                    _metric_pair(*st.columns(2), comb_is_m, comb_oos_m, split_pct)
                    _n = all_fill_stats.get("signals_total", 0)
                    _f = all_fill_stats.get("fills_total", 0)
                    _e = all_fill_stats.get("expirations_total", 0)
                    if _n:
                        st.caption(
                            f"Fill rate (all symbols): **{_f}/{_n} = "
                            f"{round(_f/_n*100,1)}%** filled · {_e} expired"
                        )
                    st.divider()

                # Per-symbol tabs
                tabs = st.tabs(ALL_SYMBOLS)
                for tab, sym in zip(tabs, ALL_SYMBOLS):
                    with tab:
                        r = results.get(sym)
                        if r is None:
                            st.error("Failed to run.")
                            continue
                        is_m, oos_m, is_t, oos_t, trades, eq, fs = r
                        oos_ev = oos_m.get("net_ev", 0) or 0
                        _fr = round(fs["fills_total"]/fs["signals_total"]*100, 1) if fs.get("signals_total") else 0
                        lbl = (f"{sym} — {len(oos_t)} OOS trades | EV: "
                               f"{'+' if oos_ev >= 0 else ''}{oos_ev:.4f} R/trade | "
                               f"Fill rate: {_fr}%")
                        (st.success if oos_ev >= 0 else st.error)(lbl)
                        if not eq.empty:
                            st.plotly_chart(equity_fig(eq, split_n=len(is_t)), use_container_width=True)
                        _metric_pair(*st.columns(2), is_m, oos_m, split_pct)

            else:
                prog = st.progress(0, f"Fetching {sym_choice} from MT5...")
                try:
                    is_m, oos_m, is_t, oos_t, all_trades, eq, sym_fill_stats = _run_sym(sym_choice)
                except Exception as e:
                    st.error(f"Failed: {e}")
                    return
                prog.progress(100, "Done.")

                oos_ev = oos_m.get("net_ev", 0) or 0
                _fr = round(sym_fill_stats["fills_total"]/sym_fill_stats["signals_total"]*100, 1) if sym_fill_stats.get("signals_total") else 0
                lbl = (f"{sym_choice} — {len(all_trades)} trades | OOS EV: "
                       f"{'+' if oos_ev >= 0 else ''}{oos_ev:.4f} R/trade | "
                       f"Fill rate: {_fr}%")
                (st.success if oos_ev >= 0 else st.error)(lbl)

                if not eq.empty:
                    st.plotly_chart(equity_fig(eq, split_n=len(is_t)), use_container_width=True)

                _metric_pair(*st.columns(2), is_m, oos_m, split_pct)
                _sn = sym_fill_stats.get("signals_total", 0)
                if _sn:
                    st.caption(
                        f"Signal fill rate: **{sym_fill_stats['fills_total']}/{_sn} = {_fr}%** filled · "
                        f"{sym_fill_stats['expirations_total']} expired"
                    )

                with st.expander("Trade Log", expanded=False):
                    if all_trades:
                        tbl = pd.DataFrame([{
                            "id": t.trade_id, "dir": t.direction,
                            "entry": str(t.entry_date)[:16], "exit": str(t.exit_date)[:16],
                            "reason": t.exit_reason,
                            "net_pips": round(t.net_pips, 1), "net_r": round(t.net_r, 4),
                            "lot": t.lot_size,
                        } for t in all_trades])
                        st.dataframe(tbl, use_container_width=True, height=350)
                        st.download_button(
                            "⬇ Download", data=tbl.to_csv(index=False).encode(),
                            file_name=f"trade_log_{sym_choice}.csv", mime="text/csv",
                        )
                    else:
                        st.warning("No trades generated.")

            return

        st.divider()

    # ── Saved runs ────────────────────────────────────────────────────────────
    runs = _list_runs()
    if not runs:
        st.info("No saved runs found in `results/`. Use Run Backtest above or run `python backtest_engine.py`.")
        return

    selected_run = st.selectbox("Saved run", options=runs, format_func=lambda p: p.name)

    with open(selected_run / "summary.json") as f:
        run_summary = json.load(f)

    sym_label = run_summary.get("symbol", "not recorded (pre-multi-symbol run)")
    st.caption(f"Symbol: {sym_label}")

    is_m_run  = run_summary.get("is",  {})
    oos_m_run = run_summary.get("oos", {})

    ev = oos_m_run.get("net_ev")
    if ev is not None:
        lbl = f"{selected_run.name} — OOS Net EV: {'+' if float(ev) >= 0 else ''}{float(ev):.4f} R/trade"
        (st.success if float(ev) >= 0 else st.error)(lbl)

    eq_path = selected_run / "equity_curve.csv"
    if eq_path.exists():
        st.plotly_chart(
            equity_fig(pd.read_csv(eq_path), split_n=int(is_m_run.get("total_trades", 0)) or None),
            use_container_width=True,
        )

    compare = [
        ("total_trades",     "Trades",        ""),
        ("win_rate_pct",     "Win Rate",       "%"),
        ("expectancy_net_r", "Expectancy",     " R"),
        ("net_ev",           "Net EV",         " R"),
        ("net_r",            "Total Net R",    " R"),
        ("max_drawdown_pct", "Max Drawdown",   "%"),
        ("trades_per_month", "Trades / Month", ""),
    ]
    col_is, col_oos = st.columns(2)
    with col_is:
        st.markdown("**In-Sample**")
        for key, label, unit in compare:
            v = is_m_run.get(key)
            st.metric(label, f"{v}{unit}" if v is not None else "—")
    with col_oos:
        st.markdown("**Out-of-Sample**")
        for key, label, unit in compare:
            iv, ov = is_m_run.get(key), oos_m_run.get(key)
            delta = None
            if iv is not None and ov is not None:
                try:
                    delta = round(float(ov) - float(iv), 3)
                except Exception:
                    pass
            st.metric(label, f"{ov}{unit}" if ov is not None else "—", delta=delta)

    chart_path = selected_run / "chart_trades.png"
    if chart_path.exists():
        st.subheader("Recent Trades Chart")
        st.image(str(chart_path), use_container_width=True)

    trade_path = selected_run / "trade_log.csv"
    if trade_path.exists():
        with st.expander("Trade Log", expanded=False):
            trade_df = pd.read_csv(trade_path)
            for c in ("entry_date", "exit_date"):
                if c in trade_df.columns:
                    trade_df[c] = pd.to_datetime(trade_df[c])
            show_cols = [c for c in [
                "trade_id", "direction", "entry_date", "entry_price",
                "exit_date", "exit_price", "exit_reason", "net_pips", "net_r", "duration_hours",
            ] if c in trade_df.columns]
            show = trade_df[show_cols].copy()
            if "net_pips" in show.columns:
                show["net_pips"] = show["net_pips"].round(0).astype("Int64")
            st.dataframe(show, use_container_width=True, height=360)
            st.download_button(
                "⬇ Download trade log",
                data=trade_df.to_csv(index=False).encode(),
                file_name=f"trade_log_{selected_run.name}.csv", mime="text/csv",
            )



# ── Parameter Studio ─────────────────────────────────────────────────────────

def page_bot_settings() -> None:
    st.title("⚙️ Bot Settings")
    st.caption("Saved to config.json — restart bot to apply.")

    cfg = _load_config_fresh() or {}

    # ── Trading mode toggle ───────────────────────────────────────────────────
    paper_mode_current = bool(cfg.get("paper_mode", True))
    working_for_real   = not paper_mode_current

    if working_for_real:
        st.error("**WORKING FOR REAL** — bot is placing live MT5 orders on your demo account.")
    else:
        st.info("**Simulation mode** — bot runs paper trades, no real MT5 orders.")

    toggle_label = "Switch to Simulation" if working_for_real else "Switch to Working for Real"
    if st.button(toggle_label, type="secondary"):
        cfg["paper_mode"] = working_for_real   # flip
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
        st.rerun()

    st.divider()

    def _i(key, default): return int(cfg.get(key, default))
    def _f(key, default): return float(cfg.get(key, default))

    def _val_changed(key, new_v):
        old_v = cfg.get(key)
        if old_v is None:
            return True
        if isinstance(new_v, str):
            return str(old_v) != new_v
        try:
            return abs(float(old_v) - float(new_v)) > 1e-9
        except Exception:
            return old_v != new_v

    # ── 1. Trading Session ────────────────────────────────────────────────────
    with st.expander("🕐 Trading Session", expanded=True):
        s1, s2 = st.columns(2)
        with s1:
            session_start = st.number_input(
                "Session start (UTC hour)", min_value=0, max_value=23,
                value=_i("session_start_utc", 7), step=1,
                help="Only take new trades at or after this UTC hour."
            )
            st.caption(f"currently: {_i('session_start_utc', 7)}")
        with s2:
            session_end = st.number_input(
                "Session end (UTC hour)", min_value=1, max_value=24,
                value=_i("session_end_utc", 20), step=1,
                help="Stop taking new trades after this UTC hour."
            )
            st.caption(f"currently: {_i('session_end_utc', 20)}")

    # ── 2. Entry & Stop ───────────────────────────────────────────────────────
    with st.expander("🎯 Entry & Stop", expanded=True):
        st.caption("Entries at previous week / month high-low levels. Stop scales with ATR.")
        e1, e2, e3 = st.columns(3)
        with e1:
            stop_atr_mult = st.number_input(
                "Stop ATR multiplier", min_value=0.0, max_value=3.0,
                value=float(cfg.get("stop_atr_mult", 0.8)), step=0.1, format="%.1f",
                help="Stop = multiplier × 14-bar ATR. Adapts per symbol. 0 = use fixed pips fallback."
            )
            st.caption(f"currently: {float(cfg.get('stop_atr_mult', 0.8)):.1f}")
        with e2:
            fixed_stop_pips = st.number_input(
                "Fixed stop fallback (pips)", min_value=5.0, max_value=100.0,
                value=float(cfg.get("fixed_stop_pips", 25.0)), step=5.0, format="%.0f",
                help="Used when stop_atr_mult = 0."
            )
            st.caption(f"currently: {float(cfg.get('fixed_stop_pips', 25.0)):.0f}")
        with e3:
            tp_rr = st.number_input(
                "TP R:R target", min_value=0.5, max_value=5.0,
                value=float(cfg.get("tp_rr", 1.0)), step=0.25, format="%.2f",
                help="Take profit = entry ± tp_rr × stop distance."
            )
            st.caption(f"currently: {float(cfg.get('tp_rr', 1.0)):.2f}")

        atr_max_pips = float(cfg.get("atr_max_pips", 0))
        tp_mode      = cfg.get("tp_mode", "full")

    # ── 3. Risk ───────────────────────────────────────────────────────────────
    with st.expander("🛡️ Risk", expanded=True):
        r1, r2 = st.columns(2)
        with r1:
            risk_pct_pct = st.number_input(
                "Risk per trade (%)", min_value=0.01, max_value=10.0,
                value=_f("risk_per_trade", 0.005) * 100, step=0.1, format="%.2f",
                help="Percentage of account balance risked per trade."
            )
            st.caption(f"currently: {_f('risk_per_trade', 0.005) * 100:.2f}%")
        with r2:
            max_drawdown_pct = st.number_input(
                "Max drawdown — circuit breaker (%)", min_value=1.0, max_value=50.0,
                value=_f("max_drawdown_pct", 5.0), step=0.5,
                help="Bot halts all trading for the session if session drawdown hits this level."
            )
            st.caption(f"currently: {_f('max_drawdown_pct', 5.0):.1f}%")
        risk_pct = risk_pct_pct / 100

    # ── Unsaved changes detection + save ─────────────────────────────────────
    new_vals = {
        "session_start_utc": int(session_start),
        "session_end_utc":   int(session_end),
        "risk_per_trade":    round(risk_pct, 4),
        "tp_rr":             round(float(tp_rr), 2),
        "stop_atr_mult":     round(float(stop_atr_mult), 1),
        "fixed_stop_pips":   float(fixed_stop_pips),
        "max_drawdown_pct":  float(max_drawdown_pct),
    }

    changed_keys = [k for k, v in new_vals.items() if _val_changed(k, v)]
    has_changes  = bool(changed_keys)

    st.divider()
    if has_changes:
        st.warning(f"Unsaved changes: {', '.join(changed_keys)}")

    if st.button("💾 Save — restart bot to apply", type="primary", disabled=not has_changes):
        new_cfg = {**cfg, **new_vals, "symbols": active_symbols}
        CONFIG_PATH.write_text(json.dumps(new_cfg, indent=2))
        st.success("Saved. Restart live_trader.py to apply.")
        st.rerun()


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

    # Refresh button (bumps cache key)
    if "refresh_key" not in st.session_state:
        st.session_state.refresh_key = 0
    _, btn_col = st.columns([5, 1])
    with btn_col:
        if st.button("🔄 Refresh"):
            st.session_state.refresh_key += 1

    st.caption(f"Symbol: **{selected}**")
    with st.spinner("Fetching data from MT5…"):
        df_raw, err = fetch_live(st.session_state.refresh_key, selected)

    if err or df_raw is None:
        st.warning(
            "⚠️ **MT5 not connected — live data unavailable.**\n\n"
            "Make sure MetaTrader 5 is open and logged in to your broker account."
        )
        return

    st.caption(f"Last update: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')} "
               f"| {len(df_raw)} bars")

    last      = df_raw.iloc[-1]
    close_px  = float(last["Close"])
    last_time = df_raw.index[-1]

    c1, c2 = st.columns(2)
    c1.metric("Close Price", f"{close_px:.5f}")
    c2.metric("Last Candle", str(last_time)[:16])
    st.divider()

    # ── W/M levels + current bar ──────────────────────────────────────────────
    live_cfg      = _load_config_fresh() or {}
    tp_rr_val     = float(live_cfg.get("tp_rr", 1.0))
    stop_atr_m    = float(live_cfg.get("stop_atr_mult", 0.8))
    fixed_sp      = float(live_cfg.get("fixed_stop_pips", 25.0))
    atr_period_v  = int(live_cfg.get("atr_period", 14))
    pip_size      = 0.01 if "JPY" in selected else PIP

    curr        = df_raw.iloc[-1]
    current_atr = _atr(df_raw, len(df_raw), atr_period_v)
    atr_pips    = current_atr / pip_size
    stop_d      = (stop_atr_m * current_atr) if stop_atr_m > 0 else (fixed_sp * pip_size)
    stop_d_pips = stop_d / pip_size

    _agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    weekly_df_live  = df_raw.resample("W").agg(_agg).dropna()
    monthly_df_live = df_raw.resample("ME").agg(_agg).dropna()
    all_levels = _wm_levels(weekly_df_live, monthly_df_live, df_raw.index[-1])

    price  = curr["Open"]
    below  = [lv for lv in all_levels if lv < price - stop_d]
    above  = [lv for lv in all_levels if lv > price + stop_d]
    lvl_lo = max(below) if below else None
    lvl_hi = min(above) if above else None

    col_bars, col_sig = st.columns(2)
    with col_bars:
        st.subheader("Current Bar")
        bar_data = pd.DataFrame([{
            "Open": curr["Open"], "High": curr["High"],
            "Low": curr["Low"], "Close": curr["Close"],
        }])
        st.dataframe(bar_data.round(5), use_container_width=True)
        st.metric("ATR (14-bar H4)", f"{atr_pips:.1f} pip")
        st.metric("Stop distance",   f"{stop_d_pips:.1f} pip  ({stop_atr_m:.1f}× ATR)")

        st.markdown("**All W/M Levels**")
        for lv in sorted(all_levels, reverse=True):
            dist = (lv - price) / pip_size
            tag  = "↑" if lv > price else "↓"
            st.caption(f"{lv:.5f}  {tag}  ({dist:+.1f} pip)")

    with col_sig:
        st.subheader("Active Setup")
        if lvl_lo is None and lvl_hi is None:
            st.info("No W/M level within range of current price.")
        else:
            if lvl_lo and curr["Low"] <= lvl_lo:
                entry = lvl_lo; sl = round(entry - stop_d, 5); tp = round(entry + tp_rr_val * stop_d, 5)
                st.success(f"**LONG** — level {entry:.5f} touched")
                p1, p2, p3 = st.columns(3)
                p1.metric("Entry", f"{entry:.5f}")
                p2.metric("Stop",  f"{sl:.5f}", delta=f"-{stop_d_pips:.0f}pip")
                p3.metric("TP",    f"{tp:.5f}",  delta=f"+{stop_d_pips*tp_rr_val:.0f}pip")
            elif lvl_hi and curr["High"] >= lvl_hi:
                entry = lvl_hi; sl = round(entry + stop_d, 5); tp = round(entry - tp_rr_val * stop_d, 5)
                st.success(f"**SHORT** — level {entry:.5f} touched")
                p1, p2, p3 = st.columns(3)
                p1.metric("Entry", f"{entry:.5f}")
                p2.metric("Stop",  f"{sl:.5f}", delta=f"+{stop_d_pips:.0f}pip")
                p3.metric("TP",    f"{tp:.5f}",  delta=f"-{stop_d_pips*tp_rr_val:.0f}pip")
            else:
                st.info("No level touched this bar — watching for fills.")
                if lvl_lo:
                    st.metric("Buy limit", f"{lvl_lo:.5f}", delta=f"{(lvl_lo-price)/pip_size:+.1f} pip")
                if lvl_hi:
                    st.metric("Sell limit", f"{lvl_hi:.5f}", delta=f"{(lvl_hi-price)/pip_size:+.1f} pip")

    st.divider()


# ── Live-trade helpers ────────────────────────────────────────────────────────

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


def _rolling_ev_df(trd_df: pd.DataFrame, pip: float = PIP) -> pd.DataFrame:
    """Compute per-trade R and rolling EV series from a paper-trade DataFrame."""
    if trd_df.empty:
        return pd.DataFrame()
    _sp = (trd_df["entry_price"] - trd_df["stop_price"]).abs() / pip
    nr  = (trd_df["net_pips"] / _sp.replace(0, float("nan"))).values
    idx = trd_df["trade_#"].values if "trade_#" in trd_df.columns else np.arange(1, len(trd_df) + 1)
    s   = pd.Series(nr)
    return pd.DataFrame({
        "trade_#":    idx,
        "net_r":      nr,
        "ev_all":     s.expanding().mean().values,
        "ev_last20":  s.rolling(20, min_periods=1).mean().values,
        "ev_last10":  s.rolling(10, min_periods=1).mean().values,
    })


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


def _sidebar_kpis(sym: str = "EURUSD") -> dict:
    """Return net_ev, max_dd, win_rate from best available source.
    Priority: paper trades > backtest OOS CSV.
    (Live trades lack net_pips in current log format.)
    """
    rk    = st.session_state.get("live_refresh_key", 0)
    lines = load_log_lines(rk)
    paper = parse_paper_trades(lines)
    _pip  = _pip_for(sym)

    if not paper.empty:
        wins  = (paper["net_pips"] > 0).sum()
        total = len(paper)
        _sp   = (paper["entry_price"] - paper["stop_price"]).abs() / _pip
        _nr   = paper["net_pips"] / _sp.replace(0, float("nan"))
        wr_f  = wins / total
        lr_f  = (total - wins) / total
        avg_nw = float(_nr[paper["net_pips"] > 0].mean())   if wins            else 0.0
        avg_nl = float(_nr[paper["net_pips"] <= 0].mean())  if (total - wins)  else 0.0
        ev     = wr_f * avg_nw - lr_f * abs(avg_nl)
        pk     = paper["balance"].cummax()
        dd     = float(((pk - paper["balance"]) / pk * 100).max())
        return {"net_ev": ev, "max_dd": dd, "win_rate": wr_f * 100, "source": "paper"}

    opt_df = load_opt()
    if opt_df is not None and not opt_df.empty:
        best = opt_df.iloc[0]
        return {
            "net_ev":   best.get("OOS_net_ev"),
            "max_dd":   best.get("OOS_max_drawdown_pct"),
            "win_rate": best.get("OOS_win_rate_pct"),
            "source":   "backtest",
        }
    return {}


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
    state = _load_state(selected)
    lines = load_log_lines(rk)
    sym_lines = [l for l in lines if f"[{selected}]" in l]
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
        st.info(f"state_{selected}.json not found. Start live_trader.py first.")
    else:
        cb_on    = state.get("circuit_breaker_active", False)
        cb_rsn   = state.get("circuit_breaker_reason", "")
        last_upd = state.get("last_updated", "—")
        open_t   = state.get("open_trade")
        pending  = state.get("pending_limit")

        s1, s2 = st.columns(2)
        s1.metric("Circuit Breaker", "🔴 TRIGGERED" if cb_on else "🟢 ACTIVE")
        s2.metric("Last Candle",     str(last_upd)[:19])

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
                tick = _mt5.symbol_info_tick(selected)
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
    st.subheader(f"3 · {selected} Live Trade Log")
    live_trades = parse_live_trades(sym_lines)

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

    # ── Section 4: Rolling EV (Paper trades — proxy for live performance) ─────
    st.subheader("4 · Rolling Expected Value")
    _live_opt_df = load_opt()
    _live_bt_ev  = None
    if _live_opt_df is not None and not _live_opt_df.empty:
        _live_bt_ev = _live_opt_df.iloc[0].get("OOS_net_ev")

    _paper_for_ev = parse_paper_trades(sym_lines)

    if live_trades.empty and _paper_for_ev.empty:
        st.info(
            "No paper or live trades with sufficient data to compute rolling EV. "
            "Live trade R is derived from paper trading logs (net_pips required)."
        )
    else:
        _ev_src = _paper_for_ev if not _paper_for_ev.empty else pd.DataFrame()
        _rev2   = _rolling_ev_df(_ev_src, pip=_pip_for(selected)) if not _ev_src.empty else pd.DataFrame()

        if _rev2.empty:
            st.info("Insufficient data for rolling EV chart.")
        else:
            rev_l1, rev_l2, rev_l3 = st.columns(3)
            rev_l1.metric("EV all trades", f"{_rev2['ev_all'].iloc[-1]:.4f} R")
            rev_l2.metric("EV last 20",    f"{_rev2['ev_last20'].iloc[-1]:.4f} R")
            rev_l3.metric("EV last 10",    f"{_rev2['ev_last10'].iloc[-1]:.4f} R")

            _ev_fig2 = go.Figure()
            _x2 = _rev2["trade_#"].values
            _ev_fig2.add_trace(go.Scatter(
                x=_x2, y=_rev2["ev_all"].values,
                name="EV all trades", line=dict(color="#4da6ff", width=2),
            ))
            _ev_fig2.add_trace(go.Scatter(
                x=_x2, y=_rev2["ev_last20"].values,
                name="EV last 20", line=dict(color="#26c26e", width=1.8, dash="dash"),
            ))
            _ev_fig2.add_trace(go.Scatter(
                x=_x2, y=_rev2["ev_last10"].values,
                name="EV last 10", line=dict(color="#ffd700", width=1.5, dash="dot"),
            ))
            _ev_fig2.add_hline(y=0, line=dict(color="white", dash="dot", width=0.7))

            if _live_bt_ev is not None:
                _ev_fig2.add_hline(
                    y=float(_live_bt_ev),
                    line=dict(color="orange", dash="longdash", width=1.2),
                    annotation_text=f"Backtest OOS EV ({_live_bt_ev:.4f} R)",
                    annotation_position="bottom right",
                )
                _thr2 = float(_live_bt_ev) * 0.80
                if _live_bt_ev > 0 and _rev2[["ev_all","ev_last20","ev_last10"]].min().min() < _thr2:
                    st.error(
                        f"⚠️  Rolling EV has dropped more than 20% below backtest EV "
                        f"({_live_bt_ev:.4f} R). Review recent trades."
                    )

            _ev_fig2.update_layout(
                height=360, template="plotly_dark",
                xaxis_title="Trade #", yaxis_title="EV (R/trade)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                title="Rolling EV (from paper trading log) vs Backtest Reference",
            )
            st.plotly_chart(_ev_fig2, use_container_width=True)

    st.divider()

    # ── Section 5: Three Equity Curves ────────────────────────────────────────
    st.subheader("5 · Equity Curves  (Backtest · Paper · Live)")
    st.caption("Backtest curve loads from CSV — no MT5 connection required.")

    eq_bt     = load_equity()
    paper_trd = parse_paper_trades(sym_lines)
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

    # ── Section 6: Live vs Paper vs Backtest Comparison ───────────────────────
    st.subheader("6 · Live vs Paper vs Backtest Comparison")

    opt_df = load_opt()

    def _safe_metric(df: pd.DataFrame, col: str, fn):
        try:
            return float(fn(df[col])) if not df.empty and col in df.columns else None
        except Exception:
            return None

    _sym_pip = _pip_for(selected)
    def _paper_stats(trd: pd.DataFrame) -> dict:
        if trd.empty:
            return {}
        wins   = (trd["net_pips"] > 0).sum()
        total  = len(trd)
        wr     = wins / total * 100 if total else 0
        _sp    = (trd["entry_price"] - trd["stop_price"]).abs() / _sym_pip
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
            net_ev=best.get("OOS_net_ev"),
            max_dd=best.get("OOS_max_drawdown_pct"),
        )

    paper_s  = _paper_stats(paper_trd)
    live_s   = _paper_stats(parse_paper_trades([])) if live_trd.empty else {}
    bt_s     = _bt_oos_stats(opt_df)

    # Paper net EV
    if not paper_trd.empty:
        _rev = _rolling_ev_df(paper_trd, pip=_sym_pip)
        if not _rev.empty:
            paper_s["net_ev"] = float(_rev["ev_all"].iloc[-1])

    metrics_def = [
        ("Win Rate %",       "win_rate",    "win_rate",    "%",  False),
        ("Expectancy R",     "expectancy",  "expectancy",  " R", False),
        ("Net EV",           "net_ev",      "net_ev",      " R", False),
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

    # ── Section 7: Recent Log Entries ─────────────────────────────────────────
    st.subheader(f"7 · Recent Log Entries  ({selected})")
    if not sym_lines:
        st.info(f"No log entries for {selected} yet.")
    else:
        def _tag(line: str) -> str:
            if "[PAPER]" in line:
                return f"[PAPER] {line}"
            for kw in ("Position", "Limit order", "Entry signal", "Trade", "Stop modified",
                        "Circuit", "HALT"):
                if kw in line:
                    return f"[LIVE]  {line}"
            return line

        last30 = [_tag(l) for l in sym_lines[-30:]]
        last30.reverse()
        st.code("\n".join(last30), language=None)

    # ── Section 8: Scale-Up Checklist ────────────────────────────────────────────
    st.subheader("8 · Scale-Up Checklist  (0.5% → 1% risk)")
    with st.expander("Show checklist", expanded=False):
        st.markdown("""
Use this checklist before increasing `risk_per_trade` from **0.5% to 1.0%** in Bot Settings.

**Strategy reference (W/M level mean-reversion, OOS backtest):**
- OOS expectancy: **+0.452 R/trade**
- OOS win rate: **68.8%**

**Minimum requirements:**
- [ ] 50+ live trades completed across all symbols
- [ ] Live expectancy ≥ 0.36R  *(within 20% of OOS +0.452R)*
- [ ] Live win rate ≥ 55%  *(within 20% of OOS 68.8%)*
- [ ] At least one drawdown period observed and recovered from
- [ ] Max live drawdown has not exceeded 10%  *(at 0.5% risk)*
- [ ] Bot has run uninterrupted for at least 3 months

**If all boxes are ticked:** change `risk_per_trade` to `0.01` in Bot Settings.

**Do not scale up if:**
- Live expectancy is below 0.25R after 50+ trades
- You are in an active drawdown
- You have modified the strategy mid-run
""")

    st.divider()

    # Auto-refresh: render content fully, then sleep 60 s and re-render
    import time as _time2
    st.caption(f"Auto-refreshing every 60 s · {now_utc}")
    _time2.sleep(60)
    st.session_state.live_refresh_key += 1
    st.rerun()


# ── Page 0: Market Overview ───────────────────────────────────────────────────

def page_overview() -> None:
    hdr_l, hdr_r = st.columns([8, 1])
    with hdr_l:
        st.title("Market Overview")
    with hdr_r:
        if st.button("Refresh"):
            st.rerun()

    st.caption(f"Last render: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}")
    st.divider()

    cols = st.columns(len(active_symbols))
    for col, sym in zip(cols, active_symbols):
        state = _load_state(sym)
        with col:
            st.markdown(f"**{sym}**")
            if state is None:
                st.markdown("NO DATA")
                st.caption("Start bot")
            else:
                cb_on   = state.get("circuit_breaker_active", False)
                open_t  = state.get("open_trade")
                pending = state.get("pending_limit")
                cur_bal = state.get("current_balance")
                ses_bal = state.get("session_starting_balance")
                dd_pct  = state.get("drawdown_pct", 0.0)

                if cb_on:
                    status = "CIRCUIT BREAK"
                    color  = "🔴"
                elif open_t:
                    dir_str = open_t.get("direction", "").upper()
                    status  = f"IN TRADE ({dir_str})"
                    color   = "🟢"
                elif pending:
                    dir_str = pending.get("direction", "").upper()
                    status  = f"PENDING ({dir_str})"
                    color   = "🟡"
                else:
                    status = "FLAT"
                    color  = "⚪"

                st.markdown(f"{color} {status}")

                if cur_bal is not None and ses_bal is not None and ses_bal > 0:
                    delta = cur_bal - ses_bal
                    sign  = "+" if delta >= 0 else ""
                    st.metric("Session P&L", f"{sign}${delta:,.2f}")
                else:
                    st.metric("Session P&L", "—")

                st.caption(f"DD: {dd_pct:.2f}%  |  CB: {'TRIGGERED' if cb_on else 'OK'}")


# ── Navigation ────────────────────────────────────────────────────────────────

PAGES = {
    "Market Overview":          page_overview,
    "⚙️ Bot Settings":          page_bot_settings,
    "📊 Backtest Results":      page_backtest,
    "📡 Live Regime Monitor":   page_live,
    "🔴 Live Trading":          page_live_trading,
}

with st.sidebar:
    st.title("Algo Trader")
    st.divider()
    page = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()

    # ── Active Pairs ──────────────────────────────────────────────────────────
    _cfg_sb      = _load_config_fresh() or {}
    _active_in_cfg = _cfg_sb.get("symbols", ALL_SYMBOLS)

    st.markdown("**Active Pairs**")
    active_symbols = [s for s in ALL_SYMBOLS
                      if st.checkbox(s, value=(s in _active_in_cfg), key=f"cb_{s}")]
    if not active_symbols:
        active_symbols = list(_active_in_cfg[:1])

    if set(active_symbols) != set(_active_in_cfg):
        _cfg_sb["symbols"] = active_symbols
        CONFIG_PATH.write_text(json.dumps(_cfg_sb, indent=2))
        st.warning("Restart bot to apply")

    # ── View Symbol ───────────────────────────────────────────────────────────
    st.markdown("**View Symbol**")
    selected = st.radio("Symbol", active_symbols, label_visibility="collapsed", key="sym_radio")

    st.divider()
    if _mt5_connected():
        st.success("🟢 MT5 connected")
    else:
        st.error("🔴 MT5 not connected")
    st.caption("Run `python backtest_engine.py` to generate data files.")

    # ── KPI indicators ────────────────────────────────────────────────────────
    st.divider()
    _kpis = _sidebar_kpis(selected)
    if _kpis:
        _src_label = "paper" if _kpis.get("source") == "paper" else "backtest OOS"
        st.caption(f"KPIs from {_src_label}")

        _ev = _kpis.get("net_ev")
        if _ev is not None:
            if _ev >= 0:
                st.success(f"EV: **+{_ev:.4f} R/trade**")
            else:
                st.error(f"EV: **{_ev:.4f} R/trade**")

        _dd = _kpis.get("max_dd")
        if _dd is not None:
            _dd = float(_dd)
            if _dd < 5:
                st.success(f"Max DD: **{_dd:.1f}%**")
            elif _dd < 10:
                st.warning(f"Max DD: **{_dd:.1f}%**")
            else:
                st.error(f"Max DD: **{_dd:.1f}%**")

        _wr = _kpis.get("win_rate")
        if _wr is not None:
            _wr = float(_wr)
            if _wr >= 40:
                st.success(f"Win Rate: **{_wr:.1f}%**")
            elif _wr >= 30:
                st.warning(f"Win Rate: **{_wr:.1f}%**")
            else:
                st.error(f"Win Rate: **{_wr:.1f}%**")

PAGES[page]()
