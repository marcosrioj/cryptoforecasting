README
======

CryptoForecast (Bybit + ML/TA)
------------------------------
Multi-timeframe crypto forecaster with console UX. The script fetches Bybit kline data, computes
technical indicators, and runs an ensemble of models (LightGBM + ElasticNet) to produce short-term
forecasts per timeframe and an overall decision.

Key features
 - Bybit Kline API (default category = linear)
 - Ensemble ML: LightGBM (price), LightGBM (return→price), ElasticNet (return→price)
 - Technical Indicators: RSI, MACD, EMA, Bollinger Bands, ATR, StochRSI, MFI, OBV
 - Colorized console output for numeric values (green=BUY, yellow=FLAT, red=SELL)
 - Uses closed candles only (no repaint)

Output order: [OVERALL], then 1w, 1d, 4h, 1h, 5m.

Installation
------------
Create a virtualenv and install required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -U scikit-learn lightgbm ta aiohttp numpy pandas colorama
```

If LightGBM fails to compile on Linux, install build tools first:

```bash
sudo apt-get update && sudo apt-get install -y build-essential
```

Files
-----
- `cryptoforecast.py`  — main script
- `logs/<scriptname>/<PAIR>/<YYYY-MM-DD_HH-MM-SS>.log` — per-run summary logs

Quick run examples
------------------
Single run (default BTCUSDT, linear category):

```bash
python3 cryptoforecast.py
```

Specify another symbol:

```bash
python3 cryptoforecast.py --symbol ETHUSDT
```

Run continuously (aligned to 5m by default):

```bash
python3 cryptoforecast.py --loop
```

Run with custom interval (seconds):

```bash
python3 cryptoforecast.py --loop --every 60
```

 - `--category` : Bybit category (linear, inverse, spot)
 - `--compact`  : compact summary output
 - `--no-color` : disable ANSI colors
 - `--no-icons` : disable emojis/icons
 - `--strategy` / `--strategy-category` : removed — the script runs all registered strategies by default

Interactive configuration
-------------------------
 When you run the script without any CLI arguments (and from a TTY), the script opens an
 interactive prompt and asks for the main runtime parameters. This is convenient for quick
 exploration or when you prefer not to type flags. Behavior:

 Prompts for: symbol, loop (yes/no), `--every` (seconds), Bybit category, compact mode, colors/icons.
 Strategy selection is no longer asked interactively because the script now runs all strategies by default.
- Each prompt shows valid options (when relevant) and the default value; press Enter to accept the default.
- The interactive prompt only triggers when stdin is a TTY and no CLI args were provided. Running with
  any flags (or from a non-interactive environment) keeps the previous, non-interactive behavior.

Example interactive run
```text
$ python3 cryptoforecast.py
Interactive mode — configure runtime parameters. Press Enter to accept the default shown.
Symbol (e.g. BTCUSDT) (default: BTCUSDT): ETHUSDT
Run continuously (loop)? [y/N] (default: False): y
Seconds between runs when --loop (integer) (default: 300): 60
Bybit category. Options: linear, inverse, spot. (default: linear): spot
Compact summary mode? [y/N] (default: False):
Enable ANSI colors? [Y/n] (default: True): y
Enable icons/emojis? [Y/n] (default: True): n
 -- now the script runs according to the choices you entered --
-- now the script runs according to the choices you entered --
```

Shell helper (optional)
-----------------------
`cryptoforecast.sh` can be a small wrapper:

```bash
#!/bin/bash
source .venv/bin/activate
python3 cryptoforecast.py "$@"

chmod +x cryptoforecast.sh
```

Strategies (categories & quick descriptions)
The repository registers a set of strategies grouped by category. The script now runs all
registered strategies by default in a single run: it computes the shared multi-timeframe data
(1w, 1d, 4h, 1h, 5m) once, prints the main header and timeframe blocks, and then appends each
strategy's summary block below the timeframes. Some strategies implement heuristics with the

Note: each strategy's printed "Reason" now includes an "Applies to" hint indicating the trading
category and the main timeframe(s) that strategy targets (for quick human context).
available kline+indicator data while a few remain placeholders requiring external data (noted below).

Day Trading
 - Scalping — many quick trades on 1–5m TFs; uses tight stops and fast execution (stub).
 - Breakout — intraday opening-range / breakout setups on short TFs (stub).
 - Range Trading — mean-reversion inside short-term ranges; buy at support / sell at resistance (stub).
 - AI/ML Algorithmic — high-frequency ML-driven signals (requires model training/data; stub).
 - Multi-Indicator “High Win Rate” — confluence of multiple indicators across timeframes (stub).
 - Price Action / ICT — liquidity & market-structure based entries (stub).
 - Arbitrage — cross-exchange or spot/futures price differences (placeholder — needs market data).

Swing Trading
 - Trend-Following — trade multi-day trends (4H/1D), enter on pullbacks (stub).
 - Breakout & Momentum Swing — multi-day breakouts from consolidations (stub).
 - Support/Resistance Range Swings — trade bounces on daily/4H ranges (stub).
 - Sentiment & News-Driven Swings — uses social/news data (requires APIs; stub).
 - Ichimoku Cloud Swing — Ichimoku-based entries on 4H/1D (stub).

Long-Term Investing
 - HODL (Buy-and-Hold) — long-term holding; no per-run action (note).
 - DCA — dollar-cost averaging strategy (simulation stub).
 - Staking & Yield — reward/compounding simulation (requires staking data; stub).
 - Diversified Portfolio — manage basket/rebalance simulation (stub).
 - Value Investing — fundamental, long-term picks (stub).

Implemented strategy logic & usage examples
-----------------------------------------
The script now computes the shared multi-timeframe data (1w, 1d, 4h, 1h, 5m) once per run and
passes it to each strategy. Strategies then apply their own decision logic using that shared data.

Implemented (heuristic) strategies
 - Scalping (Day Trading): uses the 5m timeframe (EMA short vs mid, StochRSI and AI ensemble as filter).
 - Breakout (Day Trading): breakout detection on 5m with volume confirmation proxy (OBV slope).
 - RangeTrading (Day Trading): Bollinger Bands position on 1h/5m to enter mean-reversion trades.
 - AI_ML (Day Trading): ensemble ENS delta weighted across 1h/5m.
 - MultiIndicator (Day Trading): confluence on 1h/4h of EMA trend + RSI + Stoch.
 - PriceAction (Day Trading): wick/liquidity heuristics on 1h/4h (BB position + MACD).

Implemented Swing strategies (heuristic)
 - TrendFollowing: 1d/4h EMA trend + RSI pullback entries.
 - BreakoutMomentumSwing: daily breakout strength + OBV slope.
 - SupportResistanceSwing: daily BB position bounces.
 - IchimokuSwing: simplified Ichimoku proxy using EMA crossovers.
 - SentimentSwing: volume/OBV proxy for sentiment (note: uses volume as a proxy; real sentiment needs APIs).

Long-term helpers
 - HODL: uses weekly EMA trend to label HOLD and give a simple note.
 - DCA: recommends DCA buys when price is significantly below the long-term EMA.
 - Staking/Diversified/ValueInvesting/Arbitrage: placeholders — require external data (staking APYs, portfolio holdings, fundamentals, or cross-exchange orderbooks).

Usage examples
--------------
Run everything (default behavior — computes core once and runs all strategies):

```bash
python3 cryptoforecast.py
```

Run continuously (aligned to 5m by default):

```bash
python3 cryptoforecast.py --loop
```

Run in compact mode:

```bash
python3 cryptoforecast.py --compact
```

If you want to run only a subset of strategies or reintroduce CLI selection, it's possible to
add a convenience flag (e.g. `--list-strategies` or `--strategy`) — tell me if you'd like that
added back in and I'll implement it.

Notes & next steps
------------------
- Most added strategies are currently stubs. They either call `FirstOne` as a base or print a
  short note that external data/APIs (orderbooks, social feeds, cross-exchange prices) are required.
- If you want concrete implementations for any strategy (e.g., true scalping rules on 1–5m TFs,
  ORB breakout rules, a DCA scheduler, or sentiment integration), tell me which one(s) to
  prioritize and I will implement them in the code (or wire the necessary external APIs).
- This project isn't financial advice. Use proper risk management and test in paper mode.

Troubleshooting
---------------
- If you see import errors for sklearn, ensure you installed `scikit-learn` (not `sklearn`).
- If HTTP calls fail, check network, Bybit API limits, and increase `--every` when using `--loop`.
- If LightGBM fails to compile, install system build tools as noted above.

License & disclaimer
--------------------
Provided as-is for research and educational purposes. Not financial advice.
