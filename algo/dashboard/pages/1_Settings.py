import json
import os
import streamlit as st

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.json")

st.set_page_config(page_title="Settings", layout="wide")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_config(cfg: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


cfg = load_config()

st.title("Settings")

# ── Watchlist ─────────────────────────────────────────────────────────────────

st.subheader("Watchlist")
st.caption("Filters applied to the universe of USDT spot pairs.")

wl = cfg["watchlist"]
col1, col2, col3 = st.columns(3)

max_spread = col1.number_input(
    "Max spread %",
    min_value=0.01, max_value=5.0,
    value=float(wl["max_spread_pct"]),
    step=0.05, format="%.2f",
)
min_vol = col2.number_input(
    "Min 24h volume (USD)",
    min_value=0, max_value=50_000_000,
    value=int(wl["min_volume_usd"]),
    step=100_000, format="%d",
)
max_vol = col3.number_input(
    "Max 24h volume (USD)",
    min_value=0, max_value=500_000_000,
    value=int(wl["max_volume_usd"]),
    step=1_000_000, format="%d",
)

st.markdown("---")

# ── Filters ───────────────────────────────────────────────────────────────────

st.subheader("Filters")

filt = cfg.get("filters", {})

c1, c2 = st.columns(2)

with c1:
    st.markdown("**Regime filter (200d MA)**")
    st.caption("Hard gate. Only allows longs when price is above the 200-day MA. BEAR = blocked.")
    regime_enabled = st.toggle("Enable regime filter", value=filt.get("regime_enabled", True))

with c2:
    st.markdown("**OI 1-week veto**")
    st.caption(
        "Soft veto. Blocks a trade when OI 168h change is in Q5 (most crowded). "
        "Studied on small caps — inconsistency between time periods, use with caution."
    )
    oi_168h_veto_enabled = st.toggle("Enable OI 168h veto", value=filt.get("oi_168h_veto_enabled", False))

st.markdown("---")

# ── Funding context reference ──────────────────────────────────────────────────

st.subheader("Funding Phase reference")
st.caption("Not a filter — context signal used by the trade engine to adjust hold time and entry conditions.")

ref = {
    "Phase":      ["🔵 SLEEPING", "🟢 CALM", "⚪ NEUTRAL", "🟡 WARMING", "🔴 EXCITED"],
    "Quintile":   ["Q1", "Q2", "Q3", "Q4", "Q5"],
    "Meaning":    [
        "Crowd absent, funding very low/negative",
        "Quiet, below-average funding",
        "Normal conditions",
        "Crowd building, above-average funding",
        "Crowd fully in, very high funding",
    ],
    "Action hint": [
        "Watch for wakeup — potential entry setup",
        "Good entry conditions",
        "Neutral",
        "Ok to enter or hold",
        "Hold existing trades longer — don't enter new ones",
    ],
}
st.dataframe(ref, use_container_width=True, hide_index=True)

st.markdown("---")

# ── Save ──────────────────────────────────────────────────────────────────────

if st.button("Save settings", type="primary"):
    cfg["watchlist"]["max_spread_pct"] = max_spread
    cfg["watchlist"]["min_volume_usd"] = min_vol
    cfg["watchlist"]["max_volume_usd"] = max_vol
    cfg["filters"] = {
        "regime_enabled": regime_enabled,
        "oi_168h_veto_enabled": oi_168h_veto_enabled,
    }
    save_config(cfg)
    st.success("Saved")
