"""
OI-Z Signal Bot — live dashboard.

Sections:
  1. Engine status + equity
  2. Open positions (with live P&L)
  3. Signal scanner (today's setups, ranked)
  4. Universe overview (all coins, signal state heat map)
  5. Trade history (equity curve + log)

Run: streamlit run algo/dashboard/app.py
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
from pybit.unified_trading import HTTP

# Add project root (tradingbot/) so all algo.* imports resolve correctly
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from algo.engine.state import load_state, load_trade_log
from algo.engine.signals import FZ_THRESH, ATR_THRESH, OI_Z_THRESH, EXIT_Z

st.set_page_config(page_title="OI-Z Signal Bot", layout="wide", page_icon="📈")

_SESSION = HTTP(testnet=False)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(v, decimals=2, suffix=""):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:+.{decimals}f}{suffix}" if suffix else f"{v:.{decimals}f}"


def _color_fz(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if v < FZ_THRESH:
        return "background-color: #1a4a1a"   # dark green — entry zone
    if v > EXIT_Z:
        return "background-color: #4a1a1a"   # dark red — exit zone
    return ""


def _signal_badge(row):
    if row.get("entry_signal"):
        return "🟢 ENTRY"
    if row.get("exit_signal"):
        return "🔴 EXIT"
    c = row.get("conditions_met", 0)
    return f"{'🟡' if c >= 2 else '⚪'} {c}/3"


@st.cache_data(ttl=30)
def _live_prices(symbols: list[str]) -> dict:
    prices = {}
    try:
        tickers = _SESSION.get_tickers(category="spot")["result"]["list"]
        prices  = {t["symbol"]: float(t["lastPrice"]) for t in tickers}
    except Exception:
        pass
    return prices


@st.cache_data(ttl=60)
def _load_state_cached():
    return load_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("OI-Z Bot")
    st.caption("Frozen spec: funding_z × ATR × oi_z")
    st.markdown("---")
    st.markdown(f"**Entry:** funding_z < {FZ_THRESH}  +  ATR < {ATR_THRESH}  +  oi_z > {OI_Z_THRESH}")
    st.markdown(f"**Exit:**  funding_z > {EXIT_Z}")
    st.markdown(f"**Risk:**  1% per trade, max 5 positions")
    st.markdown("---")
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    auto = st.checkbox("Auto-refresh (30s)", value=False)

if auto:
    time.sleep(30)
    st.rerun()

# ── Load state ────────────────────────────────────────────────────────────────

state       = _load_state_cached()
positions   = state.get("positions", [])
scan_res    = state.get("scan_results", [])
last_scan   = state.get("last_scan")
equity_snap = state.get("equity")

# ── Section 1: Engine status ──────────────────────────────────────────────────

st.title("OI-Z Signal Bot")

c1, c2, c3, c4 = st.columns(4)

# Try live equity
live_equity = None
try:
    w = _SESSION.get_wallet_balance(accountType="UNIFIED", coin="USDT")
    for coin in w["result"]["list"][0]["coin"]:
        if coin["coin"] == "USDT":
            live_equity = float(coin["availableToWithdraw"] or coin["walletBalance"])
            break
except Exception:
    live_equity = equity_snap

with c1:
    eq_display = f"${live_equity:,.2f}" if live_equity else "—"
    st.metric("USDT Balance", eq_display)

with c2:
    st.metric("Open Positions", len(positions), delta=None)

with c3:
    n_entry = sum(1 for r in scan_res if r.get("entry_signal"))
    n_watch = sum(1 for r in scan_res if r.get("conditions_met", 0) >= 2)
    st.metric("Active Signals", n_entry, delta=f"{n_watch} watching")

with c4:
    if last_scan:
        try:
            dt  = datetime.fromisoformat(last_scan.replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - dt)
            h   = int(age.total_seconds() // 3600)
            m   = int((age.total_seconds() % 3600) // 60)
            lbl = f"{h}h {m}m ago" if h else f"{m}m ago"
        except Exception:
            lbl = last_scan[:16]
        st.metric("Last Scan", lbl)
    else:
        st.metric("Last Scan", "Never")
        st.caption("Run: python -m algo.live.main")

st.markdown("---")

# ── Section 2: Open positions ─────────────────────────────────────────────────

st.subheader("Open Positions")

if not positions:
    st.info("No open positions.")
else:
    symbols   = [p["symbol"] for p in positions]
    prices    = _live_prices(symbols)

    rows = []
    for pos in positions:
        sym   = pos["symbol"]
        entry = float(pos.get("entry_price", 0) or 0)
        stop  = float(pos.get("stop", 0) or 0)
        qty   = float(pos.get("qty", 0) or 0)
        curr  = prices.get(sym, entry)
        pnl   = (curr - entry) * qty
        ret   = (curr - entry) / entry * 100 if entry else 0
        risk  = (entry - stop) * qty
        rr    = pnl / risk if risk else 0
        dist_stop = (curr - stop) / curr * 100 if curr else 0

        entry_date = pos.get("entry_date", "")[:10]
        rows.append({
            "Symbol":      sym,
            "Entry Date":  entry_date,
            "Entry":       entry,
            "Current":     curr,
            "Stop":        stop,
            "Qty":         qty,
            "P&L $":       pnl,
            "Ret %":       ret,
            "R":           rr,
            "Stop Dist %": dist_stop,
            "fz entry":    pos.get("entry_fz"),
            "oiz entry":   pos.get("entry_oiz"),
        })

    pos_df = pd.DataFrame(rows)

    def _style_pnl(v):
        if isinstance(v, float):
            return f"color: {'#2ecc71' if v > 0 else '#e74c3c'}"
        return ""

    st.dataframe(
        pos_df.style
            .applymap(_style_pnl, subset=["P&L $", "Ret %", "R"])
            .format({
                "Entry":       "${:.4f}",
                "Current":     "${:.4f}",
                "Stop":        "${:.4f}",
                "Qty":         "{:.4f}",
                "P&L $":       "${:+.2f}",
                "Ret %":       "{:+.2f}%",
                "R":           "{:+.2f}R",
                "Stop Dist %": "{:.1f}%",
                "fz entry":    "{:.2f}",
                "oiz entry":   "{:.2f}",
            }, na_rep="—"),
        use_container_width=True,
        height=200,
    )

st.markdown("---")

# ── Section 3: Signal scanner ─────────────────────────────────────────────────

st.subheader("Signal Scanner")

if not scan_res:
    st.info("No scan results yet. Run the engine or click Refresh.")
else:
    col_a, col_b = st.columns([1, 4])
    with col_a:
        show_all = st.checkbox("Show all coins", value=False)

    display = scan_res if show_all else [r for r in scan_res if r.get("conditions_met", 0) >= 1]

    rows = []
    for r in display:
        fz  = r.get("funding_z")
        oiz = r.get("oi_z")
        atr = r.get("atr_ratio")
        rows.append({
            "Signal":     _signal_badge(r),
            "Symbol":     r.get("symbol", ""),
            "funding_z":  fz  if fz  is not None else np.nan,
            "oi_z":       oiz if oiz is not None else np.nan,
            "atr_ratio":  atr if atr is not None else np.nan,
            "Close":      r.get("close"),
            "Stop":       r.get("stop"),
            "Stop %":     r.get("stop_pct"),
            "Cond":       r.get("conditions_met", 0),
        })

    scan_df = pd.DataFrame(rows)

    def _style_signal_row(row):
        if row.get("Signal", "").startswith("🟢"):
            return ["background-color: #1a3a1a"] * len(row)
        if row.get("Signal", "").startswith("🔴"):
            return ["background-color: #3a1a1a"] * len(row)
        return [""] * len(row)

    def _color_fz_cell(v):
        if not isinstance(v, float) or math.isnan(v):
            return ""
        if v < FZ_THRESH:
            return "color: #2ecc71; font-weight: bold"
        if v > EXIT_Z:
            return "color: #e74c3c"
        if v < 0:
            return "color: #f39c12"
        return "color: #95a5a6"

    def _color_oiz(v):
        if not isinstance(v, float) or math.isnan(v):
            return ""
        if v > OI_Z_THRESH:
            return "color: #2ecc71"
        if v < 0:
            return "color: #95a5a6"
        return ""

    def _color_atr(v):
        if not isinstance(v, float) or math.isnan(v):
            return ""
        return "color: #2ecc71" if v < ATR_THRESH else ""

    st.dataframe(
        scan_df.style
            .apply(_style_signal_row, axis=1)
            .applymap(_color_fz_cell,  subset=["funding_z"])
            .applymap(_color_oiz,      subset=["oi_z"])
            .applymap(_color_atr,      subset=["atr_ratio"])
            .format({
                "funding_z":  "{:+.2f}",
                "oi_z":       "{:+.2f}",
                "atr_ratio":  "{:.3f}",
                "Close":      "${:.4f}",
                "Stop":       "${:.4f}",
                "Stop %":     "{:.1f}%",
            }, na_rep="—"),
        use_container_width=True,
        height=min(600, 35 + len(rows) * 35),
    )

st.markdown("---")

# ── Section 4: Universe heat map ─────────────────────────────────────────────

st.subheader("Universe Overview")

if scan_res:
    # Compact grid: 6 columns, each coin shown as a color tile
    cols_per_row = 6
    coins_sorted = sorted(scan_res, key=lambda r: r.get("symbol", ""))

    for i in range(0, len(coins_sorted), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, r in enumerate(coins_sorted[i:i + cols_per_row]):
            sym = r.get("symbol", "")
            fz  = r.get("funding_z")
            oiz = r.get("oi_z")
            atr = r.get("atr_ratio")
            cond = r.get("conditions_met", 0)

            if r.get("entry_signal"):
                bg = "#1a4a1a"
            elif r.get("exit_signal"):
                bg = "#4a1a1a"
            elif cond >= 2:
                bg = "#3a3a1a"
            else:
                bg = "#1a1a2a"

            fz_str  = f"{fz:+.2f}"  if fz  is not None and not math.isnan(fz)  else "—"
            oiz_str = f"{oiz:+.2f}" if oiz is not None and not math.isnan(oiz) else "—"
            atr_str = f"{atr:.2f}"  if atr is not None and not math.isnan(atr) else "—"

            with cols[j]:
                st.markdown(
                    f"""<div style="background:{bg};border-radius:6px;padding:8px 6px;
                                   text-align:center;font-size:11px;line-height:1.6">
                        <b>{sym.replace('USDT','')}</b><br>
                        fz {fz_str}<br>
                        oiz {oiz_str}<br>
                        atr {atr_str}
                    </div>""",
                    unsafe_allow_html=True,
                )

st.markdown("---")

# ── Section 5: Trade history ──────────────────────────────────────────────────

st.subheader("Trade History")

log_data = load_trade_log()

if not log_data:
    st.info("No completed trades yet.")
else:
    trade_df = pd.DataFrame(log_data)
    for col in ["entry_price","exit_price","qty","ret_pct","pnl_usdt","entry_fz","entry_oiz","entry_atr"]:
        if col in trade_df.columns:
            trade_df[col] = pd.to_numeric(trade_df[col], errors="coerce")

    # Equity curve
    if "pnl_usdt" in trade_df.columns and "closed_at" in trade_df.columns:
        eq_df = trade_df[["closed_at","pnl_usdt"]].dropna().copy()
        eq_df["closed_at"] = pd.to_datetime(eq_df["closed_at"], errors="coerce")
        eq_df = eq_df.sort_values("closed_at")
        eq_df["equity"] = 5000 + eq_df["pnl_usdt"].cumsum()

        st.line_chart(eq_df.set_index("closed_at")["equity"], height=200)

    # Summary metrics
    if "ret_pct" in trade_df.columns:
        m1, m2, m3, m4, m5 = st.columns(5)
        wr  = (trade_df["ret_pct"] > 0).mean()
        avg = trade_df["ret_pct"].mean()
        tot = trade_df["pnl_usdt"].sum() if "pnl_usdt" in trade_df.columns else 0
        m1.metric("Trades", len(trade_df))
        m2.metric("Win Rate", f"{wr*100:.0f}%")
        m3.metric("Avg Return", f"{avg:+.2f}%")
        m4.metric("Total P&L", f"${tot:+.2f}")
        best = trade_df["ret_pct"].max()
        m5.metric("Best Trade", f"{best:+.2f}%")

    # Trade log table
    display_cols = [c for c in ["closed_at","symbol","entry_price","exit_price",
                                 "ret_pct","pnl_usdt","exit_reason","entry_fz","entry_oiz"]
                    if c in trade_df.columns]
    st.dataframe(
        trade_df[display_cols].sort_values("closed_at", ascending=False).style.format({
            "entry_price":  "${:.4f}",
            "exit_price":   "${:.4f}",
            "ret_pct":      "{:+.2f}%",
            "pnl_usdt":     "${:+.2f}",
            "entry_fz":     "{:.2f}",
            "entry_oiz":    "{:.2f}",
        }, na_rep="—"),
        use_container_width=True,
        height=400,
    )
