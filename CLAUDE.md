# tradingbot — Project Context for Claude

## Philosophy
- No assumptions. Every component is hypothesis → test → prove → add.
- 100% accurate simulation: factor in everything that happens live before accepting results.
- Goal: positive EV, good Sharpe, high trade frequency. Realistic expectations.
- No general trading advice. Proof and facts only.

## What this is
Bybit linear perpetuals (USDT-margined) trading bot, built from scratch.
Working directory: `C:\Users\tigod\OneDrive\Documents\tradingbot`

## Current state
- Scaffold in place, API keys configured in `algo/.env`
- Historical data fetch verified clean: `algo/data/fetch_and_check.py`
- Ready to build data pipeline

## Regulatory constraint
- User is Dutch — cannot trade derivatives or perpetuals (ESMA rules)
- Spot only, with up to 10x margin leverage on Bybit
- category="spot" for all API calls, not "linear"
- Can still go long AND short via spot margin (borrow USDT to buy, borrow crypto to sell)

## Data findings
- Bybit spot BTCUSDT: clean data, no gaps, no bad candles
- Daily bars go back to July 2021 (~5 years)
- Intraday depth TBD (still checking)
- Public kline endpoint needs NO auth — always use `HTTP(testnet=False)` for data
- Testnet has fake/garbage prices — never use it for historical data
- BYBIT_TESTNET=false in .env

## Universe
- 447 USDT spot pairs on Bybit
- Target: $500K–$10M daily 24h volume (USD)
  - Floor $500K: below this spreads are too wide and exits are unreliable
  - Ceiling $10M: above this market becomes too efficient, signal edge shrinks
- This range is where signal inefficiency is real and liquidity is still workable
- Configured in config.json watchlist.min_volume_usd / max_volume_usd

## Build order (evolves as we test)
1. Data pipeline — fetch, cache, serve OHLCV  ← next
2. (TBD based on what data shows)

## Signal findings
Tested on 5 volume tiers (ETH $200M, SOL $50M, ADA $10M, UNI $1M, ALGO $300K), 1H bars, 4 years.
Signals tested: OI 24h change, OI 72h change, OI Z-Score (90d), Funding raw, Funding Z-Score, Rel. Volume.
Horizons: 4h, 24h, 168h forward return. Edge threshold: Q5-Q1 spread ≥ 0.4%.

**OI 24h change — retire as a gate**
- Fast and noisy: BTC median quintile run = 2h, mean 3.4h, max ~2 days
- Direction inconsistent across coins (negative SOL/ADA, positive UNI) — no universal edge
- Current dashboard gate (block Q4/Q5) is not statistically justified — drop it

**OI 72h change — better, use as soft indicator only**
- Stronger edges than 24h: SOL bear -1.22%, UNI bull -1.66% at 24h horizon
- Direction mostly negative (high OI growth = bearish) but consistency checks fail post-2024
- Do not use as a hard gate; track as a soft regime indicator

**Funding Z-Score (90d) — context signal, not a gate**
- Most consistent across time and regimes
- ALGO: first half +0.297%, second half +0.488% (positive and strengthening)
- UNI: first half +0.254%, second half +0.717% (consistent and strengthening)
- NOT used as a simple pass/fail filter — used to understand where we are in the cycle:
  - Q1/Q2 (calm) → crowd absent, price cheap → good conditions to look for entries
  - Q3/Q4 (warming) → move already building → ok to hold, late to enter
  - Q4/Q5 (excited) → crowd fully in → think about exits, not entries
- Rising z-score confirms a move is starting; Q5 is the exit warning zone

**Funding raw in BULL regime — large edge for smaller coins**
- UNI BULL 168h forward: +6.57% Q5-Q1 spread
- ALGO BULL 168h forward: +7.84% Q5-Q1 spread
- Effect is significant and consistent across both time halves
- High funding in bull market = strong positive predictor at weekly horizon

**Relative Volume — drop entirely**
- Never clears 0.4% threshold with statistical significance
- No consistent edge at any horizon or regime

**Volume tier insight**
- Signal strength increases as daily volume decreases
- ETH ($200M): nearly no actionable edge from these inputs
- ADA/UNI/ALGO tier: real, consistent edges — better hunting ground

**How signals are used**
- Funding Z-Score: context signal — calm (Q1/Q2) = look for entries, excited (Q4/Q5) = look for exits
- OI 168h: soft veto only — Q5 means trade is crowded, skip it
- 200d MA regime: direction bias, NOT hard on/off filter
  - Bull (above 200MA): favor longs, short only on strong signal
  - Bear (below 200MA): favor shorts, long only on strong signal
  - Reason: "only trade in bull" = dead money in bear markets; price oscillates in both regimes

**Relative Volume — permanently dropped**
- Tested on real data: never cleared 0.4% edge threshold
- Practitioner literature says it works — our data says it doesn't. Trust the data.

**Consolidation as setup condition**
- Tight consolidation = small, well-defined stop = better R:R geometrically
- Measure with ATR compression: ATR(short) / ATR(longer) — ratio below threshold = quiet
- Key: use SHORT lookback windows matched to our swing duration (5–15 bars), not 20–125 bar windows designed for position traders
- Exact windows TBD — to be calibrated empirically on our data

**Trade target framework**
- Think in R (risk units), not percentages
- Stop goes at logical structure below/above entry
- Target 2–3R per trade
- 5–10% moves are typical but irrelevant — what matters is getting more than we risk
- High trade frequency preferred over waiting for large moves

**Strategy direction (current)**
- Trade the oscillation, not trend breakouts — small caps oscillate, they don't trend cleanly
- Swing trading: buy swing lows (in bull bias), short swing highs (in bear bias)
- Enter when: coin is quiet (ATR compressed) + funding z-score is calm + price gives swing signal
- Exit when: target R hit, or crowd arrives (z-score rising into Q4/Q5)
- Always looking for trades in both directions — no dead capital

**Academic backing**
- Small caps (98% of crypto by count) are mean-reversion dominant at daily horizon
- Momentum emerges at 1–4 week horizon post-compression — the window we're targeting
- Retail follows momentum (buys after moves, not before) — z-score measures this crowding
- Low-vol regime precedes large moves (MSGARCH literature) — compression is real and measurable

**Carry cost**
- USDT borrow rate ~0.02%/day baseline → ~0.09% weekly at 3x leverage
- Negligible against 2–3R target for 1–4 week holds
- Factor into backtest. Monitor live rate via API before entry (can spike in bull runs)
- API: GET /v5/spot-margin-trade/data?currency=USDT

**Next step**
- Define the entry trigger: what does "swing low is in" look like as a measurable price signal?
- Calibrate ATR compression window to match swing duration seen in charts
- Build backtest once entry trigger is defined and hypothesis is clear

## Key decisions
- Single Bybit API key (public endpoints don't need auth for market data)
- Pybit is the Bybit Python SDK
- No dead money — strategy must be active in bull and bear regimes
- Everything applied to live trading must be proven on data first
