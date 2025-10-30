CryptoForecast (Bybit + ML/TA)

Multi-timeframe crypto forecaster with console UX. The project fetches Bybit kline data, computes

CryptoForecast (Bybit + ML/TA)
------------------------------
technical indicators, and runs an ensemble of models (LightGBM + ElasticNet) plus many
heuristic strategies to produce per-timeframe forecasts and standardized strategy summaries.

This README describes the refactor, how to run the project, the new five-level signal system,
and the watchlist notifier (SMTP-only) used to alert on STRONGBUY events.

## What changed / high-level summary

- The original forecasting logic was extracted and exposed as `FirstOne()` and a shared
  core implementation `_core_forecast()` that computes multi-timeframe features once per run
  and is reused by all strategies.
- A `STRATEGIES` registry groups named strategies into categories (Day Trading, Swing Trading,
  Long-Term). The script now runs all registered strategies by default in a single run.
- Standardized rendering: the script prints the shared timeframe blocks (1w, 1d, 4h, 1h, 5m)
  once, then appends each strategy's summary block in a consistent format (Strategy, Decision,
  Reason, Applies to).
- Interactive mode: when run without CLI args from a TTY, the script prompts for runtime
  options (symbol, loop, every, category, compact, colors, icons) for faster experimentation.
- Signals: replaced the old BUY/FLAT/SELL mapping with a five-level signal system:
  STRONGBUY, BUY, FLAT, SELL, STRONGSELL. These are used across indicators, timeframe
  summaries, strategies, and aggregated voting.
- Watchlist notifier: `watchlist_notify.py` checks a list of symbols hourly and sends an email
  when any symbol's overall decision becomes `STRONGBUY`. This script now uses simple SMTP
  authentication (EMAIL_USER + EMAIL_PASS) by default.

## Five-level signal scale

The project uses the following signal labels consistently:

- STRONGBUY — strongest bullish signal
- BUY       — bullish
- FLAT      — neutral / no-action
- SELL      — bearish
- STRONGSELL— strongest bearish signal

Color mapping (console):
- STRONGBUY : darker green
- BUY       : bright green
- FLAT      : yellow
- SELL      : bright red
- STRONGSELL: darker red

How signals are derived:
- Model deltas (predicted vs last) use base thresholds (`BUY_BPS` / `SELL_BPS`). Values exceeding
  the base threshold produce BUY/SELL; values exceeding a STRONG multiplier (default x3)
  produce STRONG{BUY/SELL}.
- Indicators (RSI, MACD, EMA distance, BB position, ATR, StochRSI, MFI, OBV) are mapped to
  the same five-level labels using conservative cutoffs to make interpretation human-friendly.

Aggregation rules:
- Each timeframe's signal maps to a numeric multiplier: STRONGBUY=+2, BUY=+1, FLAT=0,
  SELL=-1, STRONGSELL=-2.
- `vote = sum(weight[tf] * multiplier(sig_tf) for tf in timeframes)` and
  `total_weight = sum(weights for available timeframes)`.
- `overall` becomes `STRONGBUY` if vote >= 0.6 * total_weight, `STRONGSELL` if vote <= -0.6 * total_weight,
  otherwise BUY/SELL/FLAT depending on vote sign and magnitude.

Why this helps:
- Strong confluence across timeframes and strong per-timeframe signals push the aggregate into
  STRONGBUY / STRONGSELL, while lone / weak signals result in BUY/SELL/FLAT.

## Files

- `cryptoforecast.py`  — main script, strategy registry, and core forecasting logic
- `watchlist_notify.py` — watchlist helper that runs `_core_forecast` for several symbols and
  sends email alerts via SMTP when an overall `STRONGBUY` occurs
- `cryptoforecast.sh`  — optional shell helper/wrapper
- `logs/` — per-run summary logs: `logs/cryptoforecast/<PAIR>/...`

## Installation

Create a virtualenv and install requirements:

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

## Quick run examples

Single run (default BTCUSDT):

```bash
python3 cryptoforecast.py
```

Specify a symbol:

```bash
python3 cryptoforecast.py --symbol ETHUSDT
```

Run continuously (aligned to 5m by default):

```bash
python3 cryptoforecast.py --loop
```

Run in compact mode:

```bash
python3 cryptoforecast.py --compact
```

Notes on CLI changes:
- `--strategy` and `--strategy-category` were removed. The script computes the shared core once
  and runs all registered strategies by default.

## Interactive configuration

If you run `cryptoforecast.py` without CLI arguments from a TTY, the script will prompt for
runtime parameters (symbol, loop, every, category, compact, colors, icons). Press Enter to
accept defaults. This is handy for quick experimentation.

## Strategies (categories & brief descriptions)

Strategies are registered and grouped for clarity. The script prints the timeframe blocks first
and appends each strategy's summary block below.

Each strategy includes a short "Applies to" hint describing its intended timeframe and trading
style (Day Trading, Swing Trading, Long-Term).

Day Trading (examples)
- Scalping — 5m/1h heuristics (EMA/fast indicators)
- Breakout — intraday breakout setups on short TFs
- Range Trading — mean-reversion inside short ranges
- AI/ML Algorithmic — ensemble-driven short-term signals
- Multi-Indicator — confluence across indicators and short TFs
- Price Action / ICT — wick/liquidity heuristics
- Arbitrage — placeholder (requires cross-exchange data)

Swing Trading (examples)
- Trend-Following — 1d/4h trend + pullback entries
- Breakout Momentum Swing — multi-day breakout + volume
- Support/Resistance Swing — daily/4h range bounces
- Ichimoku Swing — Ichimoku-style entries (simplified)
- Sentiment Swing — volume/OBV proxy for sentiment (real sentiment needs external APIs)

Long-Term (examples)
- HODL — weekly EMA trend labeling
- DCA — recommends DCA buys when below long-term EMA
- Staking/Yield/Diversified/ValueInvesting — helpers/placeholders (require external data)

Implemented vs placeholders
- Many strategies include heuristic logic and run using the shared multi-timeframe data
  (1w, 1d, 4h, 1h, 5m). Several strategies are intentionally left as placeholders where
  real external data (orderbooks, sentiment APIs, staking APYs) is required.

## Watchlist notifier (SMTP-only)

A helper `watchlist_notify.py` runs the core forecast for a watchlist of symbols (default 10)
and emails you when any symbol's overall decision becomes `STRONGBUY`.

Configuration (required):
- Set `EMAIL_USER` to your SMTP username (Gmail address for Gmail SMTP).
- Set `EMAIL_PASS` to your SMTP password. For Gmail, use a Gmail App Password (recommended).
- Optional: `EMAIL_TO` recipient address (defaults to `EMAIL_USER`).
- Optional: `WATCHLIST` comma-separated symbols to override the default watchlist.

Basic usage examples

Send a test email and exit (recommended before running as a daemon):

```bash
export EMAIL_USER="marcosrioj@gmail.com"
export EMAIL_PASS="your_app_password"
python3 watchlist_notify.py --test-email --email-to marcosrioj@gmail.com
```

Run once (check all symbols immediately):

```bash
EMAIL_USER=you@gmail.com EMAIL_PASS=your_app_password python3 watchlist_notify.py --once
```

Run as a daemon (hourly checks):

```bash
EMAIL_USER=you@gmail.com EMAIL_PASS=your_app_password python3 watchlist_notify.py
```

Override the watchlist with `--symbols` or use the `WATCHLIST` env var:

```bash
python3 watchlist_notify.py --symbols BTCUSDT,ETHUSDT
```

Notes and troubleshooting
- If you see authentication errors (SMTPAuthenticationError 535) when using Gmail, ensure you:
  - Are using an App Password (recommended) if the account has 2FA enabled.
  - `EMAIL_USER` is the full email address and `EMAIL_PASS` is the app password.
- If your SMTP provider prefers STARTTLS (port 587) instead of SSL (port 465), we can
  adapt `watchlist_notify.py` to use STARTTLS.
- Keep credentials out of version control. Use environment variables or a secrets manager.

## Logging

Each run writes a brief summary to `logs/cryptoforecast/<PAIR>/` with a timestamped filename
for later inspection.

## Tests & quick validations

- Use `python3 watchlist_notify.py --test-email` to validate email configuration.
- Run `python3 cryptoforecast.py --symbol BTCUSDT --compact` for a compact single-run output.

## Next steps and optional improvements

- Add a `--list-strategies` CLI flag to enumerate registered strategies.
- Add STARTTLS option/flags for SMTP host/port selection (if needed for your provider).
- Implement more concrete strategies (scalping rules, ORB, DCA scheduler, sentiment API wiring).
- Optionally add a small helper to obtain an OAuth2 refresh token if you prefer OAuth for Gmail.

If you want any of the above implemented, tell me which and I will add it.

## Security & disclaimer

Do not commit credentials. This project is for research and educational purposes only. Not
financial advice.


Key features
 - Bybit Kline API (default category = linear)
 - Ensemble ML: LightGBM (price), LightGBM (return→price), ElasticNet (return→price)
 - Technical Indicators: RSI, MACD, EMA, Bollinger Bands, ATR, StochRSI, MFI, OBV
 - Colorized console output for numeric values (STRONGBUY / BUY = green shades, FLAT = yellow, SELL / STRONGSELL = red shades)
 - Uses closed candles only (no repaint)

New: five-level signal scale
---------------------------------
This version uses a five-level signal system everywhere through the script and logs:

- STRONGBUY — strongest bullish signal
- BUY       — bullish
- FLAT      — neutral / no-action
- SELL      — bearish
- STRONGSELL— strongest bearish signal

Color mapping (console):
- STRONGBUY : darker green (slightly darker than BUY)
- BUY       : bright green
- FLAT      : yellow
- SELL      : bright red
- STRONGSELL: darker red (slightly darker than SELL)

How signals are derived
- Per-timeframe model deltas (predicted vs last) are converted to these five signals using
  base thresholds (configured by `BUY_BPS` / `SELL_BPS`) and a multiplier for strong signals
  (default x3). That means a prediction that exceeds the base BUY threshold becomes `BUY`,
  while one that exceeds 3x that threshold becomes `STRONGBUY`.
- Indicator helper displays (RSI, MACD, EMA distance, BB position, ATR, StochRSI, MFI, OBV)
  are mapped to the same five-level labels based on conservative, human-friendly cutoffs.
- The AI/ensemble predictions produce ENS/A/B/C deltas which are converted to five-level signals
  the same way and are used by strategies as filters.

Overall aggregation rules
- Each timeframe's signal is mapped to a numeric multiplier: STRONGBUY=+2, BUY=+1, FLAT=0,
  SELL=-1, STRONGSELL=-2. The script multiplies the timeframe weight (configured by `WEIGHTS`)
  by this multiplier and sums across timeframes to compute an aggregate `vote`.
- The script computes `total_weight = sum(weights for available timeframes)` and treats the
  overall as `STRONGBUY` if vote >= 0.6 * total_weight, `STRONGSELL` if vote <= -0.6 * total_weight,
  otherwise `BUY`/`SELL`/`FLAT` depending on sign.

Why this helps
- The five-level mapping makes strong confluence (many timeframes + strong per-tf signals)
  surface as a stronger overall recommendation (STRONGBUY / STRONGSELL). Individual indicator
  fields in the timeframe blocks also expose strength (for example, a very low RSI shows
  up as STRONGBUY in that indicator column).

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

Watchlist notifier (new)
------------------------
There's a helper script `watchlist_notify.py` that checks a list of symbols hourly and sends
an email when any symbol's overall decision becomes `STRONGBUY`.

Setup:
 - Set `EMAIL_USER` and `EMAIL_PASS` in the environment. For Gmail use an App Password (recommended).
 - Optionally set `EMAIL_TO` for recipient (defaults to `EMAIL_USER`).
 - Optionally set `WATCHLIST` as a comma-separated list of symbols to monitor; otherwise the
   default 10-symbol list will be used (BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, ADAUSDT, MATICUSDT,
   DOGEUSDT, LTCUSDT, LINKUSDT, XRPUSDT).

OAuth2 (recommended for Gmail)
------------------------------
Instead of using `EMAIL_PASS`, you can authenticate using OAuth2 (preferred). To use OAuth2 set
the following environment variables (obtain these from your Google Cloud project / OAuth consent):

- `OAUTH2_CLIENT_ID`
- `OAUTH2_CLIENT_SECRET`
- `OAUTH2_REFRESH_TOKEN`  (you must obtain a refresh token via the OAuth flow once)
- `OAUTH2_USER`           (email address to use/send from)

Then run the notifier with the `--use-oauth` flag (or the script will auto-detect a refresh token
and prefer OAuth):

```bash
OAUTH2_CLIENT_ID=... OAUTH2_CLIENT_SECRET=... OAUTH2_REFRESH_TOKEN=... OAUTH2_USER=you@gmail.com \
    python3 watchlist_notify.py --test-email --use-oauth
```

Notes:
- Getting a refresh token requires an OAuth2 consent flow (interactive) and `offline` access scope.
- The script exchanges the refresh token for an access token at runtime and uses XOAUTH2 with SMTP.
- OAuth2 avoids storing app passwords and is generally more secure. If you want, I can add a small
  helper to guide obtaining a refresh token (requires creating a Google Cloud OAuth client ID).

Run once (for testing):

```bash
EMAIL_USER=you@gmail.com EMAIL_PASS=your_app_password python3 watchlist_notify.py --once
```

Run as a daemon (hourly checks):

```bash
EMAIL_USER=you@gmail.com EMAIL_PASS=your_app_password python3 watchlist_notify.py
```

Advanced: use `--symbols` to pass a custom comma-separated list, or `WATCHLIST` env var to persist.

Security note: Do not commit `EMAIL_PASS` to version control. Prefer process managers or secret
stores to inject credentials. Gmail app passwords are recommended for SMTP.

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
