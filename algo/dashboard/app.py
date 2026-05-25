import json
import time
import os
import sys
import pandas as pd
import streamlit as st
from pybit.unified_trading import HTTP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from signals.compute import compute_oi_signal, compute_regime, compute_funding

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.json")

st.set_page_config(page_title="Trading Bot", layout="wide")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_config(cfg: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


@st.cache_data(ttl=60)
def fetch_spreads() -> pd.DataFrame:
    s = HTTP(testnet=False)
    tickers = s.get_tickers(category="spot")["result"]["list"]

    EXCLUDE = {
        "USDCUSDT", "USDEUSDT", "USD1USDT", "USDTUSDT", "BUSDUSDT",
        "TUSDUSDT", "DAIUSDT", "XAUTUSDT", "PAXGUSDT", "RLUSDUSDT",
        "USDSUSDT", "PYUSDUSDT", "USDYUSDT",
    }

    symbols = [
        t["symbol"] for t in tickers
        if t["symbol"].endswith("USDT") and t["symbol"] not in EXCLUDE
    ]
    vol_map = {t["symbol"]: float(t["turnover24h"] or 0) for t in tickers}

    rows = []
    for sym in symbols:
        try:
            ob = s.get_orderbook(category="spot", symbol=sym, limit=1)["result"]
            bid = float(ob["b"][0][0]) if ob["b"] else None
            ask = float(ob["a"][0][0]) if ob["a"] else None
            if not bid or not ask:
                continue
            rows.append({
                "symbol":      sym,
                "bid":         bid,
                "ask":         ask,
                "spread_pct":  round((ask - bid) / bid * 100, 4),
                "vol_24h_usd": vol_map.get(sym, 0),
            })
            time.sleep(0.03)
        except Exception:
            pass

    return pd.DataFrame(rows).sort_values("spread_pct").reset_index(drop=True)


@st.cache_data(ttl=3600)
def fetch_regime(symbol: str) -> str:
    try:
        return compute_regime(symbol)
    except Exception:
        return "ERROR"


@st.cache_data(ttl=300)
def fetch_oi_signal(symbol: str, lookback: int) -> dict:
    try:
        return compute_oi_signal(symbol, lookback)
    except Exception:
        return {"oi_chg_24h": 0.0, "oi_quintile_num": 0, "oi_quintile": "ERR"}


@st.cache_data(ttl=300)
def fetch_funding(symbol: str) -> float:
    try:
        return compute_funding(symbol)
    except Exception:
        return 0.0


# ── Sidebar ───────────────────────────────────────────────────────────────────

cfg = load_config()

st.sidebar.title("Control Panel")

st.sidebar.subheader("Watchlist filters")
max_spread = st.sidebar.number_input(
    "Max spread %",
    min_value=0.01,
    max_value=5.0,
    value=float(cfg["watchlist"]["max_spread_pct"]),
    step=0.05,
    format="%.2f",
)
min_vol = st.sidebar.number_input(
    "Min 24h volume (USD)",
    min_value=0,
    max_value=50_000_000,
    value=int(cfg["watchlist"]["min_volume_usd"]),
    step=100_000,
    format="%d",
)
max_vol = st.sidebar.number_input(
    "Max 24h volume (USD)",
    min_value=0,
    max_value=500_000_000,
    value=int(cfg["watchlist"]["max_volume_usd"]),
    step=1_000_000,
    format="%d",
)
changed = (
    max_spread != cfg["watchlist"]["max_spread_pct"]
    or min_vol != cfg["watchlist"]["min_volume_usd"]
    or max_vol != cfg["watchlist"]["max_volume_usd"]
)
if changed:
    cfg["watchlist"]["max_spread_pct"] = max_spread
    cfg["watchlist"]["min_volume_usd"] = min_vol
    cfg["watchlist"]["max_volume_usd"] = max_vol
    save_config(cfg)
    st.sidebar.success("Saved")

st.sidebar.markdown("---")
st.sidebar.subheader("Signal filter")
filter_enabled = st.sidebar.checkbox(
    "Enable OI filter",
    value=cfg["signal_filter"]["enabled"],
)
oi_max_q = st.sidebar.selectbox(
    "Max OI quintile (pre-committed: Q3)",
    options=[1, 2, 3, 4, 5],
    index=cfg["signal_filter"]["oi_max_quintile"] - 1,
)
sf = cfg["signal_filter"]
if filter_enabled != sf["enabled"] or oi_max_q != sf["oi_max_quintile"]:
    cfg["signal_filter"]["enabled"] = filter_enabled
    cfg["signal_filter"]["oi_max_quintile"] = oi_max_q
    save_config(cfg)
    st.sidebar.success("Saved")

st.sidebar.markdown("---")
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption("Regime: 1h cache  |  OI/funding: 5min  |  Spreads: 1min")

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("Trading Bot")

# Market Conditions
st.subheader("Market Conditions — BTCUSDT")

with st.spinner("Loading regime (4800 bars, cached 1h)..."):
    regime = fetch_regime("BTCUSDT")

oi = fetch_oi_signal("BTCUSDT", cfg["signal_filter"]["oi_lookback_bars"])
funding = fetch_funding("BTCUSDT")

gate_allowed = (
    not cfg["signal_filter"]["enabled"]
    or oi["oi_quintile_num"] <= cfg["signal_filter"]["oi_max_quintile"]
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Regime", regime)
c2.metric("OI 24h Change", f'{oi["oi_chg_24h"]:+.2f}%', oi["oi_quintile"])
c3.metric("Funding Rate", f'{funding:+.4f}%')
c4.metric("Signal Gate", "OPEN" if gate_allowed else "BLOCKED")

if not gate_allowed:
    st.warning(
        f"Gate blocked — OI 24h change is {oi['oi_quintile']} "
        f"(threshold: max Q{cfg['signal_filter']['oi_max_quintile']})"
    )

st.markdown("---")

# Watchlist
with st.spinner("Loading spreads from Bybit..."):
    df = fetch_spreads()

filtered = df[
    (df["spread_pct"] <= max_spread) &
    (df["vol_24h_usd"] >= min_vol) &
    (df["vol_24h_usd"] <= max_vol)
].copy()

col1, col2, col3 = st.columns(3)
col1.metric("Pairs in watchlist", len(filtered))
col2.metric("Max spread filter", f"{max_spread:.2f}%")
col3.metric("Total pairs scanned", len(df))

st.subheader("Watchlist")
st.dataframe(
    filtered[["symbol", "spread_pct", "vol_24h_usd"]].rename(columns={
        "symbol":      "Symbol",
        "spread_pct":  "Spread %",
        "vol_24h_usd": "24h Volume (USD)",
    }).style.format({
        "Spread %":         "{:.4f}%",
        "24h Volume (USD)": "${:,.0f}",
    }),
    use_container_width=True,
    height=600,
)
