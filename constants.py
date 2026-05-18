"""constants.py — shared constants imported by all modules."""
PIP            = 0.0001   # EURUSD price increment (1 pip = 0.0001)
PIP_VALUE      = 10.0     # USD per pip per standard lot (EURUSD)
RISK_PCT       = 0.005    # default risk fraction per trade (0.5%)
MAGIC          = 20240101 # MT5 magic number identifying our orders
MAX_LIMIT_BARS = 20       # pending limit cancelled if unfilled after this many 4H bars
