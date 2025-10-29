README.TXT
===========

CryptoForecast (Bybit + ML/TA)
------------------------------
A multi-timeframe crypto forecaster that combines:
- Bybit Kline API (default category = linear)
- Ensemble ML: LightGBM (price), LightGBM (return→price), ElasticNet (return→price)
- Technical Indicators: RSI, MACD, EMA(3/24), Bollinger Bands(20), ATR(14), StochRSI, MFI, OBV
- Colorized console output for NUMBERS ONLY (green = BUY, yellow = FLAT, red = SELL)

Output order: [OVERALL] FIRST, then blocks for 1w, 1d, 4h, 1h, 5m.
All timeframes use CLOSED CANDLES ONLY (no repaint).


1) INSTALLATION
---------------
1. Create and activate a Python virtual environment:
   python3 -m venv .venv
   source .venv/bin/activate

2. Install dependencies:
   pip install -U pip
   pip install aiohttp numpy pandas scikit-learn lightgbm ta colorama


2) FILES
--------
- cryptoforecast.py  (main script)
- Logs: logs/<scriptname>/<PAIR>/<YYYY-MM-DD_HH-MM-SS>.log
- Rolling CSV: <scriptname>/summary.csv

Note: Colors apply ONLY to the console. Logs and CSV are plain (uncolored) text.


3) HOW TO RUN
-------------
A) Run with Python directly

- Single run with default pair (BTCUSDT) and default category (linear):
  python3 cryptoforecast.py

- Change symbol:
  python3 cryptoforecast.py --symbol ETHUSDT

- Continuous loop aligned to 5-minute boundaries:
  python3 cryptoforecast.py --loop

- Continuous loop with custom cadence (e.g., every 60 seconds; not 5m aligned):
  python3 cryptoforecast.py --loop --every 60

- Change Bybit category (e.g., spot, inverse):
  python3 cryptoforecast.py --category spot


B) Optional shell helper (run.sh)

Create a run.sh with:
  #!/bin/bash
  source .venv/bin/activate
  python3 cryptoforecast.py "$@"

Make it executable:
  chmod +x run.sh

Examples:
  ./run.sh                          # single run, BTCUSDT, linear
  ./run.sh --symbol ETHUSDT         # single run, ETHUSDT
  ./run.sh --loop                   # run forever, aligned to 5-minute ticks
  ./run.sh --loop --every 60        # run forever, every 60s (not aligned)
  ./run.sh --category spot          # use Bybit spot category


4) CONSOLE OUTPUT FORMAT
------------------------
Order: [OVERALL] first, then [1w], [1d], [4h], [1h], [5m].

Example:
  SUMMARY
  symbol: BTCUSDT  category: linear  generated_at: 2025-10-29 01:00:07 UTC
  thresholds_bps: buy>=10, sell<=-10
  weights: {'1w': 5, '1d': 4, '4h': 3, '1h': 2, '5m': 1}
  execution_time_seconds: 1.23

  [OVERALL]
  signals={1w:SELL, 1d:SELL, 4h:FLAT, 1h:BUY, 5m:FLAT}
  weights={'1w': 5, '1d': 4, '4h': 3, '1h': 2, '5m': 1}  vote=-3
  Signal=SELL

  [1w]
  last=114498.30  pred=112170.25  Δ%=-2.03  MAPE=37.45%  dir_acc=46.55% conf=57.86%
  ind=[rsi14:49.96, macd:-1151.58, ema_diff%:2.60%, bb_pos%:57.11%, bb_width%:19.41%, atr%:8.27%, stoch_k:0.26, mfi:47.56, obv_slope%:-23.83%]
  AI=[A:-6.56%, B:4.91%, C:-1.13%, ENS:-2.03%, agree=2/3, overlay=No]
  Signal=SELL

Notes:
- [OVERALL] lists per-timeframe Signals WITH COLORS in the console (BUY=green, FLAT=yellow, SELL=red).
- In timeframe blocks, ONLY the numbers are colorized. The AI and ind lines show numbers only (no BUY/FLAT/SELL alongside numbers).
- Signal lines remain categorical (colored words).


5) LOGS AND CSV
----------------
- Log file (plain text):
  logs/<scriptname>/<PAIR>/<YYYY-MM-DD_HH-MM-SS>.log
  Contains only SUMMARY lines per timeframe and the overall decision.
  No colors.

- Rolling CSV (plain text):
  <scriptname>/summary.csv
  Appends numeric metrics and signals per timeframe for easier analysis.


6) LEGEND (ALL METRICS & COLORS)
--------------------------------
Colors in the console apply ONLY to numbers. Logs/CSV are uncolored.

- Δ%  (Predicted change vs last close, percent)
  Thresholds (converted to basis points):
    ≥ +0.10% (≥ +10 bps)  => green (BUY)
    ≤ −0.10% (≤ −10 bps)  => red (SELL)
    otherwise             => yellow (FLAT)

- MAPE  (Mean Absolute Percentage Error from time-series cross-validation; lower is better)
    < 5%    => green
    5–15%   => yellow
    > 15%   => red

- dir_acc  (Directional accuracy % across CV folds; higher is better)
    ≥ 60%   => green
    50–60%  => yellow
    < 50%   => red

- conf  (Heuristic confidence combining CV error and ensemble dispersion)
    ≥ 85%   => green
    60–85%  => yellow
    < 60%   => red

- rsi14  (RSI with window 14)
    < 30   => green (oversold bias)
    > 70   => red (overbought bias)
    45–55  => yellow (sideways/neutral zone)
    otherwise => yellow

- macd  (MACD histogram = MACD − signal)
    > 0        => green (bullish momentum)
    < 0        => red (bearish momentum)
    approx 0   => yellow (low momentum; we treat |hist/price| < 0.0005 as near-zero)

- ema_diff%  ((EMA3 − EMA24) / EMA24 × 100)
    > 0                => green (short-term uptrend)
    < 0                => red (short-term downtrend)
    |x| < 0.05%        => yellow (near flat)

- bb_pos%  (Position within Bollinger bands, 0% = lower, 100% = upper)
    0% (lower)  => green
    100% (upper)=> red
    middle      => yellow

- bb_width%  ((Upper − Lower) / Price × 100; volatility proxy)
    informational => yellow

- atr%  (ATR(14) / Price × 100; volatility)
    informational => yellow

- stoch_k  (StochRSI %K)
    ≤ 20   => green (oversold)
    ≥ 80   => red (overbought)
    else   => yellow

- mfi  (Money Flow Index)
    ≤ 20   => green (oversold)
    ≥ 80   => red (overbought)
    else   => yellow

- obv_slope%  (5-bar OBV percentage slope)
    > 0           => green (accumulation)
    < 0           => red (distribution)
    |x| < 0.10%   => yellow (flat)

- vote  (Weighted sum of timeframe Signals using weights)
    > 0   => green (BUY)
    = 0   => yellow (FLAT)
    < 0   => red (SELL)


7) AI LEGEND (MODELS & WHAT YOU SEE)
------------------------------------
Models used (Ensemble):
- A — LightGBM (price): Regresses next close directly.
- B — LightGBM (return→price): Regresses next return (%) and converts to price.
- C — ElasticNet (return→price): Linear with L1/L2 mixing on next return (%).
- ENS — Weighted Ensemble of A/B/C with default weights 0.5 / 0.3 / 0.2.
- Time-Series CV: 3 folds for MAPE/dir_acc; no leakage; closed-candle alignment.

AI line in console:
  AI=[A:-0.85%, B:+0.40%, C:-1.10%, ENS:-0.95%, agree=2/3, overlay=No]
- A/B/C/ENS show NUMBERS ONLY: predicted percentage change vs last close.
  Each number is colorized by the same Δ% thresholds (≥ +0.10% green, ≤ −0.10% red, otherwise yellow).
- agree shows how many base models (A/B/C) match the direction of ENS (e.g., 2/3).
- overlay indicates whether the RSI/MACD overlay neutralized a borderline trend to FLAT.

Signal per timeframe:
- Printed as: Signal=BUY | FLAT | SELL (colored word).
- [OVERALL] shows each timeframe Signal (colored), weights, weighted vote (colored), and global Signal.


8) NOTES & GOOD PRACTICES
-------------------------
- Uses CLOSED CANDLES ONLY for each timeframe to prevent look-ahead and repaint issues.
- Confidence (conf) is heuristic: it penalizes high CV error and high ensemble disagreement.
- Change Bybit category with --category if your symbol belongs to spot or inverse.
- This tool is NOT financial advice. Use your own risk management.
- If Bybit throttles requests, increase --every when running in loop mode.
- If LightGBM import fails, ensure compatible Python and OS-specific wheels.


9) QUICK TROUBLESHOOTING
------------------------
- No colors? Your terminal may not support ANSI colors. Try a different terminal emulator.
- HTTP errors / timeouts? Could be network or rate limiting. Retry with a longer interval.
- Pandas dtype errors? Ensure numeric features only (the script already enforces numeric selection).

