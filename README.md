README.TXT
===========

CryptoForecast (Bybit + ML/TA) — UX Edition
-------------------------------------------
A multi-timeframe crypto forecaster with improved console UX:
- Bybit Kline API (default category = linear)
- Ensemble ML: LightGBM (price), LightGBM (return→price), ElasticNet (return→price)
- Technical Indicators: RSI, MACD, EMA(3/24), Bollinger Bands(20), ATR(14), StochRSI, MFI, OBV
- Colorized console output for NUMBERS ONLY (green = BUY, yellow = FLAT, red = SELL)
- Enhanced readability: banners, dividers, icons, aligned columns, grouped sections
- **Timeframe title now includes the Signal** → e.g., `[1w]  - Signal: SELL`
- Optional compact summary mode, togglable icons and colors

Output order: [OVERALL] FIRST, then blocks for 1w, 1d, 4h, 1h, 5m.
All timeframes use CLOSED CANDLES ONLY (no repaint).

IMPORTANT: Weekly (1w) support is fully enabled. To ensure 1w appears for newer pairs,
the script uses per-timeframe minimum history: 1w=30 candles, others=200.


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

- Compact summary mode (minimal screen usage):
  python3 cryptoforecast.py --compact

- Disable colors (ANSI):
  python3 cryptoforecast.py --no-color

- Disable icons/emojis:
  python3 cryptoforecast.py --no-icons


B) Optional shell helper (run.sh)

Create a run.sh with:
  #!/bin/bash
  source .venv/bin/activate
  python3 cryptoforecast.py "$@"

Make it executable:
  chmod +x run.sh

Examples:
  ./run.sh                          # single run, BTCUSDT, linear
  ./run.sh --symbol MAVIAUSDT       # single run, custom pair
  ./run.sh --loop                   # run forever, aligned to 5-minute ticks
  ./run.sh --loop --every 60        # run forever, every 60s (not aligned)
  ./run.sh --compact                # compact mode
  ./run.sh --no-color               # disable ANSI colors
  ./run.sh --no-icons               # disable icons/emojis


4) CONSOLE OUTPUT FORMAT (ENHANCED)
-----------------------------------
- Header banner with aligned metadata.
- [OVERALL] block shows per-timeframe Signals (colored words), weights, vote, and final Decision.
- **Each timeframe title now includes Signal**:  
  Example: `🕒 [1w]  - Signal: SELL`
- Each timeframe block groups information into:
  * Metrics row (aligned): last, pred, Δ%, MAPE, dir_acc, conf
  * Indicators row: RSI14, MACD, EMAΔ%, BBpos%, BBwidth%, ATR%, StochK, MFI, OBV%
  * AI ensemble row: A/B/C/ENS deltas (%), agree, overlay
- Horizontal dividers between sections for scannability.
- Optional compact summary mode (one-liners) for dashboards/bots.

Example (full mode with 1w INCLUDED):
  ──────────────────────────────────────────────
                🚀 CRYPTOFORECAST SUMMARY
  ──────────────────────────────────────────────
  Symbol     : BTCUSDT
  Category   : linear
  Generated  : 2025-10-29 07:13:02 UTC
  Thresholds : BUY ≥ +10 bps | SELL ≤ -10 bps
  Weights    : 1w=5, 1d=4, 4h=3, 1h=2, 5m=1
  Exec time  : 9.71 s
  ──────────────────────────────────────────────
  🧭 [OVERALL]
  Signals    : 1w:FLAT, 1d:BUY, 4h:SELL, 1h:SELL, 5m:SELL
  Weights    : {'1w': 5, '1d': 4, '4h': 3, '1h': 2, '5m': 1}  Vote: SELL (-2)
  Decision   : SELL
  ──────────────────────────────────────────────
  🕒 [1w]  - Signal: SELL
  last=114498.30  pred=112170.25  Δ%=-2.03  MAPE= 37.45%  dir_acc= 46.55%  conf= 57.86%
  Indicators : RSI14=49.96, MACD=-1151.58, EMAΔ%=2.60%, BBpos%=57.11%, BBwidth%=19.41%, ATR%=8.27%, StochK=0.26, MFI=47.56, OBV%=-23.83%
  AI         : A=-6.56%, B=4.91%, C=-1.13%, ENS=-2.03%, agree=2/3, overlay=No
  ──────────────────────────────────────────────
  📅 [1d]  - Signal: BUY
  ...

Example (compact mode):
  ──────────────────────────────────────────────
  🧭 BTCUSDT: SELL (vote=-2)
  TF Signals : 1w=FLAT, 1d=BUY, 4h=SELL, 1h=SELL, 5m=SELL
  Δ%         : 1w:-2.03% | 1d:27.97% | 4h:-1.24% | 1h:-0.62% | 5m:-0.14%
  ──────────────────────────────────────────────


5) LOGS AND CSV
----------------
- Log file (plain text, always written on each run):
  logs/<scriptname>/<PAIR>/<YYYY-MM-DD_HH-MM-SS>.log
  Contains only SUMMARY lines per timeframe (including 1w when available) and the overall decision.

- Rolling CSV (plain text):
  <scriptname>/summary.csv
  Appends numeric metrics and signals per timeframe for easier analysis.


6) NOTES & PRACTICES
--------------------
- CLOSED CANDLES ONLY per timeframe to avoid look-ahead/repaint.
- Confidence (conf) is heuristic, blending CV error and ensemble disagreement.
- Change Bybit category with --category if your symbol belongs to spot or inverse.
- Not financial advice; manage your risk.
- If throttled by Bybit, increase --every with --loop.
- If LightGBM import fails, ensure a compatible Python version and OS-specific wheel.

7) QUICK TROUBLESHOOTING
------------------------
- No colors? Use --no-color or another terminal that supports ANSI.
- HTTP errors? Likely network or rate limiting; retry or increase --every.
- Pandas dtype errors? Ensure numeric features only; the script already selects numerics.

