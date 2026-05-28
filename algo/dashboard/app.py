"""
OI-Z Signal Bot — Dashboard

Sections:
  Sidebar  : strategy explainer + editable risk settings
  Status   : account health, alerts
  Positions: open positions with live P&L and stop health
  Signals  : scanner output with per-condition breakdown
  Universe : heatmap grid
  History  : equity curve + trade log
"""

import os
import sys
import json
import time
import math
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timezone

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pybit.unified_trading import HTTP
from algo.engine.state   import load_state, load_trade_log
from algo.engine.signals import FZ_THRESH, ATR_THRESH, OI_Z_THRESH, EXIT_Z

CONFIG_PATH = os.path.join(_ROOT, "algo", "config", "config.json")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OI-Z Signal Bot",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# ── Thresholds for staleness warning ─────────────────────────────────────────
STALE_WARN_HOURS = 4    # amber warning
STALE_ERR_HOURS  = 12   # red error

# ── CSS tweaks ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1rem; }
  .stMetric label  { font-size: 0.78rem; color: #95a5a6; }
  div[data-testid="stMetricValue"] { font-size: 1.3rem; }
</style>
""", unsafe_allow_html=True)


# ── Config helpers ────────────────────────────────────────────────────────────

def load_config() -> dict:
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {"strategy": {}, "engine": {}}


def save_config(cfg: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


# ── API / data loaders ────────────────────────────────────────────────────────

@st.cache_resource
def _session():
    return HTTP(testnet=False)


@st.cache_data(ttl=25)
def _live_prices(symbols: tuple) -> dict:
    try:
        tickers = _session().get_tickers(category="spot")["result"]["list"]
        return {t["symbol"]: float(t["lastPrice"]) for t in tickers}
    except Exception:
        return {}


@st.cache_data(ttl=20)
def _live_equity() -> float | None:
    try:
        w = _session().get_wallet_balance(accountType="UNIFIED", coin="USDT")
        for c in w["result"]["list"][0]["coin"]:
            if c["coin"] == "USDT":
                v = c.get("availableToWithdraw") or c.get("walletBalance")
                return float(v) if v else None
    except Exception:
        return None


@st.cache_data(ttl=30)
def _state_cached():
    return load_state()


@st.cache_data(ttl=120)
def _trades_cached():
    return load_trade_log()


# ── Formatting ────────────────────────────────────────────────────────────────

def _na(v) -> bool:
    return v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))


def _fmt(v, fmt=".2f", prefix="", suffix="", na="—"):
    return na if _na(v) else f"{prefix}{v:{fmt}}{suffix}"


def _age(iso: str) -> tuple[float, str]:
    """Returns (hours_old, human_str)."""
    try:
        dt    = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        secs  = (datetime.now(timezone.utc) - dt).total_seconds()
        h, m  = int(secs // 3600), int((secs % 3600) // 60)
        label = f"{h//24}d {h%24}h ago" if h >= 24 else (f"{h}h {m}m ago" if h else f"{m}m ago")
        return secs / 3600, label
    except Exception:
        return 999, "unknown"


def _cond_icon(met: bool) -> str:
    return "✅" if met else "❌"


def _pnl_color(v) -> str:
    if isinstance(v, (int, float)) and not _na(v):
        return "color: #2ecc71" if v >= 0 else "color: #e74c3c"
    return ""


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 OI-Z Signal Bot")
    st.caption("Automated Bybit spot trading · Dutch ESMA-compliant")
    st.markdown("---")

    # Refresh controls
    c_ref, c_auto = st.columns(2)
    with c_ref:
        refresh_clicked = st.button("🔄 Refresh", use_container_width=True)
    with c_auto:
        auto_refresh = st.checkbox("Auto 30s", value=False)

    if refresh_clicked:
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    # ── Strategy explanation ──────────────────────────────────────────────────
    with st.expander("📖 How this bot works", expanded=False):
        st.markdown(f"""
**Strategy in plain English:**

This bot looks for coins where the *crowd has left* (funding rate is low
compared to its recent history) and the market has *gone quiet* (low
recent volatility). It buys in anticipation of the next move, then exits
when the crowd comes back.

**Why it works:**

Small-cap crypto tends to mean-revert. When a coin is under-crowded AND
compressed, the conditions for a move are ideal. When funding rises again,
the crowd is back and the edge is gone.

**Risk management:**

Every trade risks exactly **{load_config().get('strategy',{}).get('risk_frac',0.01)*100:.1f}%**
of your account. A stop-loss order is placed automatically at entry.
The engine stops trading if session losses exceed the circuit-breaker level.
""")

    # ── Signal conditions (always visible) ───────────────────────────────────
    st.markdown("#### Signal Conditions")
    st.markdown(f"""
**Entry** — ALL 3 must be true:
| # | Condition | Threshold |
|---|-----------|-----------|
| 1 | Funding Z-Score | < **{FZ_THRESH}** |
| 2 | ATR Ratio (5/10) | < **{ATR_THRESH}** |
| 3 | OI Z-Score | > **{OI_Z_THRESH}** |

**Exit** — any position is closed when:
Funding Z-Score rises above **{EXIT_Z}**
""")

    st.markdown("---")

    # ── Risk settings ─────────────────────────────────────────────────────────
    st.markdown("#### ⚙️ Risk Settings")
    st.caption("Takes effect on the next engine run.")

    cfg   = load_config()
    strat = cfg.get("strategy", {})
    eng   = cfg.get("engine",   {})

    new_risk   = st.slider("Risk per trade (%)",   0.5, 3.0,
                           float(strat.get("risk_frac", 0.01)) * 100,    step=0.1,
                           help="Percentage of your account risked on each trade.")
    new_maxpos = st.slider("Max open positions",    1,  10,
                           int(strat.get("max_positions", 5)),
                           help="The bot will never open more than this many positions at once.")
    new_stop_d = st.slider("Stop lookback (days)",  5,  20,
                           int(strat.get("stop_days", 10)),
                           help="Stop-loss is placed at the lowest low over this many days.")
    new_cb     = st.slider("Circuit breaker (%)",   5.0, 25.0,
                           float(eng.get("max_session_drawdown", 0.10)) * 100, step=1.0,
                           help="Bot halts all trading if session losses reach this level.")

    if st.button("💾 Save Settings", use_container_width=True):
        cfg["strategy"]["risk_frac"]          = round(new_risk / 100, 4)
        cfg["strategy"]["max_positions"]      = new_maxpos
        cfg["strategy"]["stop_days"]          = new_stop_d
        cfg["engine"]["max_session_drawdown"] = round(new_cb / 100, 4)
        save_config(cfg)
        st.success("✅ Saved. Active on next engine run.")

    st.markdown("---")
    st.caption("**Schedule:** 00:05 · 08:05 · 16:05 UTC")
    st.code("python -m algo.live.main", language="bash")


# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(30)
    st.rerun()

# ── Load data ─────────────────────────────────────────────────────────────────
state       = _state_cached()
positions   = state.get("positions", [])
scan_res    = state.get("scan_results", [])
last_scan   = state.get("last_scan")
equity_snap = state.get("equity")
trade_log   = _trades_cached()
live_eq     = _live_equity()
display_eq  = live_eq or equity_snap
cfg_now     = load_config()
max_pos     = cfg_now.get("strategy", {}).get("max_positions", 5)
cb_pct      = cfg_now.get("engine",   {}).get("max_session_drawdown", 0.10) * 100

# Scan age
scan_hours, scan_age_label = (999, "never") if not last_scan else _age(last_scan)

# ── Alerts ────────────────────────────────────────────────────────────────────
unprotected = [p for p in positions if not p.get("stop_order_id")]

if not last_scan:
    st.info("ℹ️ **Bot not started yet.** Run `python -m algo.live.main` to begin.")
elif scan_hours > STALE_ERR_HOURS:
    st.error(f"🚨 **Data is stale ({scan_age_label}).** The engine hasn't run in over "
             f"{STALE_ERR_HOURS}h. Run `python -m algo.live.main` immediately.")
elif scan_hours > STALE_WARN_HOURS:
    st.warning(f"⚠️ Last scan was {scan_age_label}. Engine should run at 00:05 / 08:05 / 16:05 UTC.")

if unprotected:
    syms = ", ".join(p["symbol"] for p in unprotected)
    st.error(f"🚨 **Unprotected positions: {syms}** — no stop-loss order on record. "
             "Log into Bybit immediately and place stop orders manually.")

# ── Page header ───────────────────────────────────────────────────────────────
st.title("📈 OI-Z Signal Bot")

# ── Section 1: Account Health ─────────────────────────────────────────────────
st.subheader("Account Health")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric(
        "USDT Balance",
        _fmt(display_eq, ".2f", "$") if display_eq else "—",
        help="Available USDT balance (fetched live from Bybit).",
    )

with c2:
    n_entry = sum(1 for r in scan_res if r.get("entry_signal"))
    n_watch = sum(1 for r in scan_res if r.get("conditions_met", 0) >= 2 and not r.get("entry_signal"))
    st.metric(
        "Entry Signals",
        n_entry,
        delta=f"{n_watch} watching" if n_watch else None,
        help="Coins meeting ALL 3 entry conditions right now. 'Watching' = 2/3 conditions met.",
    )

with c3:
    st.metric(
        "Open Positions",
        f"{len(positions)} / {max_pos}",
        help=f"Positions held vs maximum allowed ({max_pos}). Set in risk settings.",
    )

with c4:
    if last_scan:
        delta_color = "normal" if scan_hours < STALE_WARN_HOURS else "inverse"
        st.metric(
            "Last Scan",
            scan_age_label,
            help="How long ago the signal engine last ran. Should be under 4 hours.",
        )
    else:
        st.metric("Last Scan", "Never", help="Engine hasn't run yet.")

with c5:
    if display_eq and equity_snap and equity_snap > 0:
        session_dd = max(0, (equity_snap - display_eq) / equity_snap * 100)
        st.metric(
            "Session Drawdown",
            f"{session_dd:.1f}%",
            delta=f"CB at {cb_pct:.0f}%",
            delta_color="inverse" if session_dd > cb_pct * 0.7 else "off",
            help=f"Loss from session start. Circuit breaker halts trading at {cb_pct:.0f}%.",
        )
    else:
        st.metric("Circuit Breaker", f"{cb_pct:.0f}%",
                  help=f"Engine halts if session drawdown reaches {cb_pct:.0f}%.")

st.markdown("---")

# ── Section 2: Open Positions ─────────────────────────────────────────────────
st.subheader("Open Positions")

if not positions:
    st.info("No open positions. The bot is waiting for entry signals.")
else:
    prices = _live_prices(tuple(p["symbol"] for p in positions))
    rows   = []

    for pos in positions:
        sym   = pos["symbol"]
        entry = float(pos.get("entry_price") or 0)
        stop  = float(pos.get("stop") or 0)
        qty   = float(pos.get("qty") or 0)
        curr  = prices.get(sym, entry)

        pnl      = (curr - entry) * qty
        ret      = (curr - entry) / entry * 100 if entry else 0
        risk_usd = (entry - stop) * qty
        rr       = pnl / risk_usd if risk_usd else 0
        stop_dist = (curr - stop) / curr * 100 if curr else 0

        try:
            dt_entry  = datetime.fromisoformat(pos.get("entry_date", "").replace("Z", "+00:00"))
            days_held = (datetime.now(timezone.utc) - dt_entry).days
        except Exception:
            days_held = 0

        stop_status = "✅ Protected" if pos.get("stop_order_id") else "🚨 NO STOP"

        rows.append({
            "Symbol":      sym,
            "Entry Date":  pos.get("entry_date", "")[:10],
            "Days":        days_held,
            "Entry $":     entry,
            "Current $":   curr,
            "Stop $":      stop,
            "Stop Dist %": stop_dist,
            "P&L $":       pnl,
            "Return %":    ret,
            "R":           rr,
            "Stop":        stop_status,
            "fz@entry":    pos.get("entry_fz"),
            "oiz@entry":   pos.get("entry_oiz"),
        })

    pos_df = pd.DataFrame(rows)
    for col in ["fz@entry", "oiz@entry"]:
        pos_df[col] = pd.to_numeric(pos_df[col], errors="coerce")

    def _stop_color(v):
        return "color: #e74c3c; font-weight: bold" if isinstance(v, str) and "NO" in v else "color: #2ecc71"

    fmt_pos = {
        "Entry $":     "${:.6f}",
        "Current $":   "${:.6f}",
        "Stop $":      "${:.6f}",
        "Stop Dist %": "{:.1f}%",
        "P&L $":       "${:+.2f}",
        "Return %":    "{:+.2f}%",
        "R":           "{:+.2f}R",
        "fz@entry":    "{:.2f}",
        "oiz@entry":   "{:.2f}",
    }

    st.dataframe(
        pos_df.style
            .map(_pnl_color,   subset=["P&L $", "Return %", "R"])
            .map(_stop_color,  subset=["Stop"])
            .format(fmt_pos, na_rep="—"),
        use_container_width=True,
        height=min(400, 55 + len(rows) * 38),
    )

    with st.expander("ℹ️ Column guide — Positions"):
        st.markdown("""
| Column | Meaning |
|--------|---------|
| Stop Dist % | How far current price is from the stop. The closer to 0%, the closer to being stopped out. |
| R | Risk/Reward multiple. +1R = you've gained what you risked. +2R = gained twice what you risked. |
| Stop | Whether a stop-loss order is live on the exchange. 🚨 NO STOP is a critical warning. |
| fz@entry | Funding Z-Score at the time of entry (what triggered the signal). |
""")

st.markdown("---")

# ── Section 3: Signal Scanner ─────────────────────────────────────────────────
st.subheader("Signal Scanner")
st.caption(
    "Each row is one coin in the universe. "
    "**Entry** = ALL 3 conditions met. **Watch** = 2/3. "
    "The ✅/❌ columns show exactly which conditions pass."
)

if not scan_res:
    st.info("No scan results yet. Run `python -m algo.live.main` to populate.")
else:
    f1, f2 = st.columns([2, 2])
    with f1:
        show_mode = st.selectbox(
            "Show",
            options=["Signals & Watching (≥1 condition)", "Entry signals only",
                     "Watching only (2 conditions)", "Full universe (all 80 coins)"],
            index=0,
            help="Filter which coins are shown in the table below.",
        )
    with f2:
        sort_mode = st.selectbox(
            "Sort by",
            options=["Readiness (best entry first)", "Funding Z (most negative)",
                     "OI Z-Score (highest)", "Symbol (A–Z)"],
            index=0,
        )

    # Filter
    if show_mode == "Entry signals only":
        display = [r for r in scan_res if r.get("entry_signal")]
    elif show_mode == "Watching only (2 conditions)":
        display = [r for r in scan_res if r.get("conditions_met", 0) == 2]
    elif show_mode == "Full universe (all 80 coins)":
        display = scan_res
    else:
        display = [r for r in scan_res if r.get("conditions_met", 0) >= 1]

    # Sort
    if sort_mode == "Funding Z (most negative)":
        display = sorted(display, key=lambda r: r.get("funding_z", 99) or 99)
    elif sort_mode == "OI Z-Score (highest)":
        display = sorted(display, key=lambda r: -(r.get("oi_z", 0) or 0))
    elif sort_mode == "Symbol (A–Z)":
        display = sorted(display, key=lambda r: r.get("symbol", ""))
    # else: default sort from scanner (entry first, then conditions_met desc)

    rows = []
    for r in display:
        fz   = r.get("funding_z")
        oiz  = r.get("oi_z")
        atr  = r.get("atr_ratio")

        fz_ok  = not _na(fz)  and fz  < FZ_THRESH
        oiz_ok = not _na(oiz) and oiz > OI_Z_THRESH
        atr_ok = not _na(atr) and atr < ATR_THRESH

        if r.get("entry_signal"):
            badge = "🟢 ENTRY"
        elif r.get("exit_signal"):
            badge = "🔴 EXIT"
        elif r.get("conditions_met", 0) >= 2:
            badge = "🟡 WATCH"
        else:
            badge = f"⚪ {r.get('conditions_met', 0)}/3"

        rows.append({
            "Signal":         badge,
            "Symbol":         r.get("symbol", ""),
            "💰 Funding":     _cond_icon(fz_ok),
            "📊 OI":          _cond_icon(oiz_ok),
            "🔇 ATR":         _cond_icon(atr_ok),
            "Funding Z":      fz  if not _na(fz)  else np.nan,
            "OI Z-Score":     oiz if not _na(oiz) else np.nan,
            "ATR Ratio":      atr if not _na(atr) else np.nan,
            "Price":          r.get("close"),
            "Stop $":         r.get("stop"),
            "Stop %":         r.get("stop_pct"),
        })

    scan_df = pd.DataFrame(rows)

    def _fz_style(v):
        if not isinstance(v, float) or _na(v): return ""
        if v < FZ_THRESH:  return "color: #2ecc71; font-weight: bold"
        if v > EXIT_Z:     return "color: #e74c3c"
        return "color: #95a5a6"

    def _oiz_style(v):
        if not isinstance(v, float) or _na(v): return ""
        return "color: #2ecc71; font-weight: bold" if v > OI_Z_THRESH else "color: #95a5a6"

    def _atr_style(v):
        if not isinstance(v, float) or _na(v): return ""
        return "color: #2ecc71; font-weight: bold" if v < ATR_THRESH else "color: #95a5a6"

    def _row_bg(row):
        sig = row.get("Signal", "")
        if sig.startswith("🟢"): return ["background-color: #0d2818"] * len(row)
        if sig.startswith("🔴"): return ["background-color: #2d0a0a"] * len(row)
        if sig.startswith("🟡"): return ["background-color: #2d200a"] * len(row)
        return [""] * len(row)

    fmt_scan = {
        "Funding Z":  "{:+.2f}",
        "OI Z-Score": "{:+.2f}",
        "ATR Ratio":  "{:.3f}",
        "Price":      "${:.4f}",
        "Stop $":     "${:.4f}",
        "Stop %":     "{:.1f}%",
    }

    st.dataframe(
        scan_df.style
            .apply(_row_bg, axis=1)
            .map(_fz_style,  subset=["Funding Z"])
            .map(_oiz_style, subset=["OI Z-Score"])
            .map(_atr_style, subset=["ATR Ratio"])
            .format(fmt_scan, na_rep="—"),
        use_container_width=True,
        height=min(650, 55 + len(rows) * 35),
    )

    with st.expander("📘 Column guide — Signals"):
        st.markdown(f"""
| Column | What it measures | Entry condition |
|--------|-----------------|-----------------|
| 💰 Funding | Is the crowd sentiment unusually low? | ✅ when Funding Z < **{FZ_THRESH}** |
| 📊 OI | Is open interest building? | ✅ when OI Z-Score > **{OI_Z_THRESH}** |
| 🔇 ATR | Is the market unusually quiet? | ✅ when ATR Ratio < **{ATR_THRESH}** |
| Funding Z | 90-day z-score of daily funding rate payments. Negative = shorts paying longs = bearish crowd. | |
| OI Z-Score | 90-day z-score of perp open interest. Rising OI = new money entering the market. | |
| ATR Ratio | ATR(5 days) ÷ ATR(10 days). Below 1 = volatility contracting = compression. | |
| Stop % | Max loss if stopped out (stop placed at {cfg_now.get('strategy',{}).get('stop_days',10)}-day low). | |

**Green** = value passes the threshold.
**Exit** signals (red rows) mean funding has risen above {EXIT_Z} — crowd has returned, edge is gone.
""")

st.markdown("---")

# ── Section 4: Universe Heatmap ────────────────────────────────────────────────
with st.expander("🌐 Universe Overview — all coins at a glance", expanded=False):
    st.caption(
        "Green = entry zone · Red = exit zone · Amber = watching · Dark = neutral. "
        "Sorted by readiness (entry signals first)."
    )
    if scan_res:
        coins = sorted(scan_res, key=lambda r: (
            -int(r.get("entry_signal", False)),
            -int(r.get("exit_signal", False)),
            -r.get("conditions_met", 0),
        ))
        cols_per_row = 8
        for i in range(0, len(coins), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, r in enumerate(coins[i: i + cols_per_row]):
                sym  = r.get("symbol", "").replace("USDT", "")
                fz   = r.get("funding_z")
                oiz  = r.get("oi_z")
                atr  = r.get("atr_ratio")
                cond = r.get("conditions_met", 0)

                if r.get("entry_signal"):
                    bg, border = "#0d2818", "#00b894"
                elif r.get("exit_signal"):
                    bg, border = "#2d0a0a", "#d63031"
                elif cond >= 2:
                    bg, border = "#2d200a", "#f39c12"
                else:
                    bg, border = "#12131a", "#2d3436"

                fz_s  = _fmt(fz,  "+.1f") if not _na(fz)  else "—"
                oiz_s = _fmt(oiz, "+.1f") if not _na(oiz) else "—"
                atr_s = _fmt(atr, ".2f")  if not _na(atr) else "—"

                with cols[j]:
                    st.markdown(
                        f"""<div style="background:{bg};border:1px solid {border};
                                        border-radius:6px;padding:6px 4px;
                                        text-align:center;font-size:10px;line-height:1.75">
                            <b style="font-size:11px">{sym}</b><br>
                            fz {fz_s}<br>oiz {oiz_s}<br>atr {atr_s}
                        </div>""",
                        unsafe_allow_html=True,
                    )
    else:
        st.info("No scan data available.")

st.markdown("---")

# ── Section 5: Trade History ──────────────────────────────────────────────────
with st.expander("📜 Trade History", expanded=True):
    if not trade_log:
        st.info(
            "No completed trades yet. Trades appear here after positions are opened and closed. "
            "The first entry signal is **MNTUSDT** — check the Signal Scanner above."
        )
    else:
        df = pd.DataFrame(trade_log)
        num_cols = ["entry_price","exit_price","qty","ret_pct","pnl_usdt","entry_fz","entry_oiz","entry_atr"]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # ── Summary stats ────────────────────────────────────────────────────
        if "ret_pct" in df.columns and len(df) > 0:
            wins   = (df["ret_pct"] > 0).sum()
            wr     = wins / len(df)
            avg    = df["ret_pct"].mean()
            tot    = df["pnl_usdt"].sum() if "pnl_usdt" in df.columns else 0
            best   = df["ret_pct"].max()
            worst  = df["ret_pct"].min()

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Total Trades", len(df))
            m2.metric("Win Rate",     f"{wr*100:.0f}%",
                      help=f"{int(wins)} winners out of {len(df)} trades.")
            m3.metric("Avg Return",   f"{avg:+.2f}%",
                      help="Average return per closed trade.")
            m4.metric("Total P&L",    f"${tot:+.2f}",
                      help="Sum of all closed trade profits/losses in USDT.")
            m5.metric("Best Trade",   f"{best:+.2f}%")
            m6.metric("Worst Trade",  f"{worst:+.2f}%")

        # ── Equity curve ─────────────────────────────────────────────────────
        if "pnl_usdt" in df.columns and "closed_at" in df.columns:
            eq_df = df[["closed_at","pnl_usdt"]].dropna().copy()
            eq_df["closed_at"] = pd.to_datetime(eq_df["closed_at"], errors="coerce")
            eq_df = eq_df.sort_values("closed_at").dropna(subset=["closed_at"])
            if not eq_df.empty:
                total_pnl  = df["pnl_usdt"].sum()
                start_eq   = (display_eq - total_pnl) if display_eq else 5000
                eq_df["Equity ($)"] = start_eq + eq_df["pnl_usdt"].cumsum()
                st.line_chart(eq_df.set_index("closed_at")["Equity ($)"], height=220)

        # ── Exit reason breakdown ─────────────────────────────────────────────
        if "exit_reason" in df.columns:
            reason_counts = df["exit_reason"].value_counts()
            r1, r2 = st.columns(2)
            with r1:
                for reason, count in reason_counts.items():
                    pct = count / len(df) * 100
                    st.markdown(f"**{reason.capitalize()}** exits: {count} ({pct:.0f}%)")

        # ── Trade table ──────────────────────────────────────────────────────
        disp_cols = [c for c in [
            "closed_at","symbol","entry_price","exit_price",
            "ret_pct","pnl_usdt","exit_reason","entry_fz","entry_oiz",
        ] if c in df.columns]

        fmt_trd = {
            "entry_price": "${:.4f}", "exit_price": "${:.4f}",
            "ret_pct":     "{:+.2f}%","pnl_usdt":  "${:+.2f}",
            "entry_fz":    "{:.2f}",  "entry_oiz":  "{:.2f}",
        }

        st.dataframe(
            df[disp_cols].sort_values("closed_at", ascending=False).style
                .map(_pnl_color, subset=[c for c in ["ret_pct","pnl_usdt"] if c in disp_cols])
                .format({k: v for k, v in fmt_trd.items() if k in disp_cols}, na_rep="—"),
            use_container_width=True,
            height=400,
        )
