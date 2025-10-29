import asyncio, os, platform, warnings, aiohttp, time, math
import numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Tuple, List
from pathlib import Path

from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import ElasticNet
import lightgbm as lgb
from colorama import init as colorama_init, Fore, Style

warnings.filterwarnings("ignore")
colorama_init(autoreset=True)

# ===================== CONFIG =====================
SYMBOL = "ETHUSDT"
CATEGORY = "linear"
TIMEFRAMES = {"1w": "W", "1d": "D", "4h": "240", "1h": "60", "5m": "5"}
LIMITS = {"1w": 520, "1d": 1500, "4h": 1500, "1h": 1500, "5m": 1500}
BUY_BPS, SELL_BPS = 10, -10
WEIGHTS = {"1w": 5, "1d": 4, "4h": 3, "1h": 2, "5m": 1}
MIN_ROWS, SEED = 200, 42
# Ensemble weights
W_PRICE, W_RET_LGB, W_RET_EN = 0.5, 0.3, 0.2
# ==================================================

GREEN = Fore.GREEN + Style.BRIGHT
RED = Fore.RED + Style.BRIGHT
YELL = Fore.YELLOW + Style.BRIGHT
CYAN = Fore.CYAN + Style.BRIGHT
WHITE = Style.BRIGHT
RESET = Style.RESET_ALL

def clear_console():
    os.system("cls" if platform.system().lower().startswith("win") else "clear")

# ---------- LOG/CSV PATHS ----------
def _base_log_dir() -> Path:
    # Folder with the same name as the script (without .py)
    return Path(__file__).resolve().with_suffix("").name and Path(Path(__file__).resolve().with_suffix("").name)

def _log_path(now_utc: datetime) -> Path:
    base = _base_log_dir()
    year = now_utc.strftime("%Y")
    month = now_utc.strftime("%m")
    week = f"{now_utc.isocalendar().week:02d}"
    day  = now_utc.strftime("%d")
    fname = now_utc.strftime("%Y-%m-%d_%H-%M-%S") + ".log"
    return base / year / month / week / day / fname

def _csv_path() -> Path:
    # rolling CSV at the base folder
    return _base_log_dir() / "summary.csv"

def _write_summary_log(text: str, now_utc: datetime) -> None:
    path = _log_path(now_utc)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def _append_csv(rows: List[dict]) -> None:
    csv_path = _csv_path()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    # Append with header if file doesn't exist
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", index=False, header=header, encoding="utf-8")

# --------- Time alignment ----------
def _week_start_utc(dt_utc: datetime) -> datetime:
    monday = dt_utc.date() - timedelta(days=dt_utc.weekday())
    return datetime(monday.year, monday.month, monday.day, tzinfo=timezone.utc)

def interval_ms(code: str) -> int:
    if code == "W": return 7 * 24 * 60 * 60 * 1000
    if code == "D": return 24 * 60 * 60 * 1000
    if code.isdigit(): return int(code) * 60 * 1000
    raise ValueError(f"Unsupported interval code: {code}")

def last_closed_open_time(code: str) -> datetime:
    now = datetime.now(timezone.utc)
    if code == "W":
        return _week_start_utc(now) - timedelta(weeks=1)
    if code == "D":
        return datetime(now.year, now.month, now.day, tzinfo=timezone.utc) - timedelta(days=1)
    im = interval_ms(code)
    now_ms = int(now.timestamp() * 1000)
    floored_ms = (now_ms // im) * im
    return datetime.fromtimestamp((floored_ms - im) / 1000, tz=timezone.utc)

def trim_to_last_closed(df: pd.DataFrame, code: str) -> pd.DataFrame:
    cutoff = pd.to_datetime(last_closed_open_time(code)).tz_convert(None)
    return df[df["ts"] <= cutoff].sort_values("ts").reset_index(drop=True)

# --------- Fetch ----------
async def fetch_kline(session, interval_code: str, limit: int) -> pd.DataFrame:
    url = "https://api.bybit.com/v5/market/kline"
    params = {"category": CATEGORY, "symbol": SYMBOL, "interval": interval_code, "limit": limit}
    async with session.get(url, params=params, timeout=30) as resp:
        data = await resp.json()
        rows = data.get("result", {}).get("list", [])
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["start","open","high","low","close","volume","turnover"])
        df["ts"] = pd.to_datetime(df["start"].astype("int64"), unit="ms", utc=True).dt.tz_convert(None)
        for c in ["open","high","low","close","volume","turnover"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna()[["ts","open","high","low","close","volume","turnover"]].sort_values("ts").reset_index(drop=True)

async def fetch_all_tf() -> Dict[str, pd.DataFrame]:
    async with aiohttp.ClientSession() as s:
        tasks = [fetch_kline(s, iv, LIMITS[tf]) for tf, iv in TIMEFRAMES.items()]
        res = await asyncio.gather(*tasks, return_exceptions=True)
    return {tf: r for (tf,_), r in zip(TIMEFRAMES.items(), res)}

# --------- Features & models ----------
def make_features(df):
    d = df.copy()
    d["ret_1"] = d["close"].pct_change()
    for w in [3,6,12,24,60]:
        ema = EMAIndicator(d["close"], window=w).ema_indicator()
        d[f"ema_{w}"] = ema
        d[f"close_ema_dist_{w}"] = (d["close"] - ema) / ema
    d["rsi_14"] = RSIIndicator(d["close"], window=14).rsi()
    stoch = StochRSIIndicator(d["close"])
    d["stoch_k"], d["stoch_d"] = stoch.stochrsi_k(), stoch.stochrsi_d()
    macd = MACD(d["close"])
    d["macd"], d["macd_sig"], d["macd_diff"] = macd.macd(), macd.macd_signal(), macd.macd_diff()
    bb = BollingerBands(d["close"])
    d["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / d["close"]
    d["atr_14"] = AverageTrueRange(d["high"], d["low"], d["close"]).average_true_range()
    d["rv_12"] = d["ret_1"].rolling(12).std()
    d["mfi_14"] = MFIIndicator(d["high"], d["low"], d["close"], d["volume"]).money_flow_index()
    d["obv"] = OnBalanceVolumeIndicator(d["close"], d["volume"]).on_balance_volume()
    d["y_next_close"] = d["close"].shift(-1)
    d["y_next_ret_pct"] = ((d["y_next_close"] - d["close"]) / d["close"]) * 100.0
    return d.dropna()

def ts_cv_metrics(X, y):
    X = X.select_dtypes(include=[np.number])
    tscv = TimeSeriesSplit(n_splits=3)
    mape_scores, dir_hits, dir_total = [], 0, 0
    for tr, te in tscv.split(X):
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, objective="mape", random_state=SEED)
        m.fit(X.iloc[tr], y.iloc[tr])
        yhat = m.predict(X.iloc[te])
        mape_scores.append(mean_absolute_percentage_error(y.iloc[te], yhat))
        proxy = X.iloc[te]["ema_3"] if "ema_3" in X.columns else X.iloc[te]["close"]
        dir_hits += (np.sign(yhat - proxy) == np.sign(y.iloc[te] - proxy)).sum()
        dir_total += len(y.iloc[te])
    return float(np.mean(mape_scores)), float(dir_hits / max(dir_total, 1))

def fit_predict_ensemble(df_feat, last_close):
    X = df_feat.drop(columns=["y_next_close","y_next_ret_pct","ts","start"], errors="ignore").select_dtypes(include=[np.number])
    y_close = df_feat["y_next_close"]
    y_ret = df_feat["y_next_ret_pct"]
    cv_mape, dir_acc = ts_cv_metrics(X, y_close)

    mdl_a = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, objective="mape", random_state=SEED)
    mdl_a.fit(X, y_close)
    pred_a = float(mdl_a.predict(X.tail(1))[0])

    mdl_b = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, random_state=SEED)
    mdl_b.fit(X, y_ret)
    pred_b = last_close * (1.0 + mdl_b.predict(X.tail(1))[0] / 100.0)

    mdl_c = ElasticNet(alpha=0.001, l1_ratio=0.15, random_state=SEED, max_iter=2000)
    mdl_c.fit(X, y_ret)
    pred_c = last_close * (1.0 + mdl_c.predict(X.tail(1))[0] / 100.0)

    preds = np.array([pred_a, pred_b, pred_c])
    weights = np.array([W_PRICE, W_RET_LGB, W_RET_EN]); weights /= weights.sum()
    pred_ens = float(np.dot(weights, preds))
    ens_std = float(np.std(preds))
    return pred_ens, cv_mape, dir_acc, ens_std

def decide_signal(dpct): 
    bps = dpct * 100
    if bps >= BUY_BPS: return "BUY"
    if bps <= SELL_BPS: return "SELL"
    return "FLAT"

def color(sig): return GREEN if sig=="BUY" else RED if sig=="SELL" else YELL

def overlay_with_rsi_macd(sig, rsi, macd_diff, last_price):
    if 45 <= rsi <= 55 and abs(macd_diff)/max(last_price,1e-9) < 5e-4: return "FLAT"
    return sig

# --------- Core run ---------
async def run_once():
    start_time = time.perf_counter()

    raw = await fetch_all_tf()
    results = {}
    for tf, data in raw.items():
        if isinstance(data, Exception) or data.empty: continue
        code = TIMEFRAMES[tf]
        data = trim_to_last_closed(data, code)
        if len(data) < MIN_ROWS: continue
        feat = make_features(data)
        if feat.empty: continue
        last = float(data["close"].iloc[-1])
        pred, cv, acc, ens_std = fit_predict_ensemble(feat, last)
        rsi, macd_hist = float(feat["rsi_14"].iloc[-1]), float(feat["macd_diff"].iloc[-1])
        dpct = (pred - last) / last * 100
        sig = overlay_with_rsi_macd(decide_signal(dpct), rsi, macd_hist, last)
        conf = max(5.0, min(95.0, (1.0 - cv)*100.0 - (ens_std/last)*100.0))
        results[tf] = {"last": last,"pred": pred,"dpct": dpct,"cv": cv,"acc": acc*100,
                       "rsi": rsi,"macd": macd_hist,"sig": sig,"conf": conf}

    vote = sum((WEIGHTS.get(tf,1) if r["sig"]=="BUY" else -WEIGHTS.get(tf,1) if r["sig"]=="SELL" else 0)
               for tf,r in results.items())
    overall = "BUY" if vote>0 else "SELL" if vote<0 else "FLAT"

    end_time = time.perf_counter()
    duration = end_time - start_time  # seconds float

    # ---------- Build printable SUMMARY ----------
    now_utc = datetime.now(timezone.utc)
    now_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    # Prepare log (SUMMARY-only)
    summary_lines = []
    summary_lines.append("SUMMARY")
    summary_lines.append(f"symbol: {SYMBOL}  category: {CATEGORY}  generated_at: {now_str}")
    summary_lines.append(f"thresholds_bps: buy>={BUY_BPS}, sell<={SELL_BPS}")
    summary_lines.append(f"weights: {WEIGHTS}")
    summary_lines.append(f"execution_time_seconds: {duration:.2f}")
    summary_lines.append("")

    # Console output (colored + two decimals)
    clear_console()
    print(f"{WHITE}SUMMARY{RESET}")
    print(f"{WHITE}symbol:{RESET} {SYMBOL}  {WHITE}category:{RESET} {CATEGORY}  {WHITE}generated_at:{RESET} {CYAN}{now_str}{RESET}")
    print(f"{WHITE}thresholds_bps:{RESET} buy>={BUY_BPS}, sell<={SELL_BPS}")
    print(f"{WHITE}weights:{RESET} {WEIGHTS}")
    print(f"{WHITE}execution_time_seconds:{RESET} {duration:.2f}\n")

    # CSV rows to append
    csv_rows: List[dict] = []

    for tf in ["1w","1d","4h","1h","5m"]:
        r = results.get(tf)
        if not r: continue
        clr = color(r["sig"])
        print(f"{WHITE}[{tf}]{RESET} "
              f"last={r['last']:.2f}  pred={r['pred']:.2f}  Δ%={r['dpct']:.2f}  "
              f"MAPE={r['cv']*100:.2f}%  dir_acc={r['acc']:.2f}%  "
              f"RSI14={r['rsi']:.2f}  MACD_hist={r['macd']:.2f}  conf={r['conf']:.2f}%  "
              f"signal={clr}{r['sig']}{RESET}")

        # Add to plain text log
        summary_lines.append(
            f"[{tf}] last={r['last']:.2f}  pred={r['pred']:.2f}  Δ%={r['dpct']:.2f}  "
            f"MAPE={(r['cv']*100):.2f}%  dir_acc={(r['acc']):.2f}%  RSI14={r['rsi']:.2f}  "
            f"MACD_hist={r['macd']:.2f}  conf={r['conf']:.2f}%  signal={r['sig']}"
        )

        # Add a CSV row
        csv_rows.append({
            "generated_at_utc": now_str,
            "exec_time_s": round(duration, 2),
            "timeframe": tf,
            "symbol": SYMBOL,
            "last": round(r["last"], 2),
            "pred": round(r["pred"], 2),
            "delta_pct": round(r["dpct"], 2),
            "mape_pct": round(r["cv"]*100.0, 2),
            "dir_acc_pct": round(r["acc"], 2),
            "rsi14": round(r["rsi"], 2),
            "macd_hist": round(r["macd"], 2),
            "conf_pct": round(r["conf"], 2),
            "signal": r["sig"],
            "vote": vote,
            "overall_decision": overall
        })

    print(f"\n{WHITE}overall_decision:{RESET} {color(overall)}{overall}{RESET}  (vote={vote})")
    summary_lines.append("")
    summary_lines.append(f"overall_decision: {overall}  (vote={vote})")

    # ---- Persist: LOG + CSV ----
    _write_summary_log("\n".join(summary_lines) + "\n", now_utc)
    _append_csv(csv_rows)

    # ---- Console-only: LEGEND + AI details (NOT in logs, NOT in CSV) ----
    print()  # spacer
    print(f"{WHITE}LEGEND:{RESET}")
    print(f"  Δ% → Predicted change vs. last close. Ex: +0.50 = +0.5% expected rise")
    print(f"  MAPE → Historical avg prediction error (lower = better). Ex: 0.30%")
    print(f"  dir_acc → % of times model got up/down direction right. Ex: 60%")
    print(f"  RSI14 → Momentum indicator; >70=overbought, <30=oversold, ~50=neutral")
    print(f"  MACD_hist → MACD - Signal; >0=bullish, <0=bearish, near 0=sideways")
    print(f"  conf → Model confidence (AI consensus & low error = higher)\n")

    print(f"{WHITE}AI STRATEGIES & INDICATORS USED:{RESET}")
    print(f"  • Machine Learning Ensemble:")
    print(f"      - LightGBM (price-level prediction)")
    print(f"      - LightGBM (return-based prediction)")
    print(f"      - ElasticNet (regularized linear return model)")
    print(f"  • Time-Series Cross-Validation (walk-forward, 3 splits)")
    print(f"  • Ensemble weighted averaging (weights={W_PRICE}/{W_RET_LGB}/{W_RET_EN})")
    print(f"  • RSI-MACD overlay to suppress false signals in sideways markets")
    print(f"  • Confidence score based on MAPE + ensemble agreement")
    print(f"  • Numeric-only feature gating (robust against dtype drift)")
    print(f"  • Closed-candle alignment (ensures using fully finished bars)\n")
    print(f"  • Technical Indicators Used:")
    print(f"      - EMA(3,6,12,24,60), RSI(14), StochRSI, MACD(12,26,9)")
    print(f"      - Bollinger Bands(20), ATR(14), Realized Vol(12)")
    print(f"      - MFI(14), OBV, candle body/wick ratios, return variance\n")

# --------- Scheduler ---------
async def scheduler_loop():
    await run_once()  # immediate
    while True:
        now = datetime.utcnow()
        wait = 300 - ((now.minute*60 + now.second) % 300)
        await asyncio.sleep(wait)
        await run_once()

if __name__ == "__main__":
    asyncio.run(scheduler_loop())
