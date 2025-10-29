README.TXT
===========

CryptoForecast (Bybit + ML/TA)
------------------------------
Multi-timeframe crypto forecaster with console UX:
- Bybit Kline API (default category = linear)
- Ensemble ML: LightGBM (price), LightGBM (return→price), ElasticNet (return→price)
- Technical Indicators: RSI, MACD, EMA(3/24), Bollinger Bands(20), ATR(14), StochRSI, MFI, OBV
- Colorized console output for NUMBERS ONLY (green = BUY, yellow = FLAT, red = SELL)
- Timeframe title includes the Signal → e.g., `[1w]  - Signal: SELL`
- **`last` e `pred`**:  
  - `last` é exibido **exatamente** como vem da API (string).  
  - `pred` é formatado com **o mesmo número de casas decimais de `last`**.
- **CSV removido** — somente logs de SUMMARY por execução.

Output order: [OVERALL], depois 1w, 1d, 4h, 1h, 5m.  
Usa **apenas candles FECHADOS** por timeframe (sem repaint).

INSTALL
-------
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -U scikit-learn lightgbm ta aiohttp numpy pandas colorama

(Se o LightGBM falhar no Linux: `sudo apt-get update && sudo apt-get install -y build-essential`)

FILES
-----
- cryptoforecast.py  (script principal)
- Logs: logs/<scriptname>/<PAIR>/<YYYY-MM-DD_HH-MM-SS>.log

RUN
---
Single run (BTCUSDT, linear):
  python3 cryptoforecast.py

Outro símbolo:
  python3 cryptoforecast.py --symbol ETHUSDT

Loop alinhado a 5m:
  python3 cryptoforecast.py --loop

Loop com intervalo custom (ex.: 60s):
  python3 cryptoforecast.py --loop --every 60

Categoria Bybit (spot/inverse):
  python3 cryptoforecast.py --category spot

Modo compacto:
  python3 cryptoforecast.py --compact

Sem cores:
  python3 cryptoforecast.py --no-color

Sem ícones:
  python3 cryptoforecast.py --no-icons

SHELL HELPER (opcional)
-----------------------
cryptoforecast.sh:
  #!/bin/bash
  source .venv/bin/activate
  python3 cryptoforecast.py "$@"

chmod +x cryptoforecast.sh

Exemplos:
  ./cryptoforecast.sh --symbol ETHUSDT
  ./cryptoforecast.sh --loop
  ./cryptoforecast.sh --loop --every 60
  ./cryptoforecast.sh --compact
  ./cryptoforecast.sh --no-color --no-icons

NOTES
-----
- Confiança (conf) é heurística (erro de CV + divergência do ensemble).
- Ajuste --category conforme o mercado do par.
- Não é aconselhamento financeiro.
- Se sofrer rate-limit, aumente --every ao usar --loop.

TROUBLESHOOT
------------
- ImportError `sklearn.model_model_selection`: troque para `from sklearn.model_selection import TimeSeriesSplit`.
- Instalou `sklearn`? Desinstale e instale **scikit-learn**.
- Sem cores? Use --no-color ou outro terminal.
- Erros HTTP? Rede / limite; tente novamente ou aumente --every.
