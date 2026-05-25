import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from signals.compute import compute_oi_signal, compute_regime, compute_funding


def check_gate(symbol: str, cfg: dict) -> dict:
    """
    Returns gate state for symbol.

    Result keys: oi_chg_24h, oi_quintile, oi_quintile_num,
                 regime, funding_rate, allowed (bool), reason (str)
    """
    sf = cfg.get("signal_filter", {})
    lookback = sf.get("oi_lookback_bars", 500)
    max_q = sf.get("oi_max_quintile", 3)
    enabled = sf.get("enabled", True)

    oi = compute_oi_signal(symbol, lookback)
    regime = compute_regime(symbol)
    funding = compute_funding(symbol)

    if not enabled:
        allowed, reason = True, "filter disabled"
    elif oi["oi_quintile_num"] > max_q:
        allowed = False
        reason = f"OI {oi['oi_quintile']} > Q{max_q} threshold"
    else:
        allowed = True
        reason = f"OI {oi['oi_quintile']} within Q{max_q} threshold"

    return {
        "symbol": symbol,
        "regime": regime,
        "funding_rate": funding,
        **oi,
        "allowed": allowed,
        "reason": reason,
    }
