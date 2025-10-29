import asyncio, os, platform, warnings, aiohttp, time
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
    return Path(__file__).resolve().with_suffix("").name and Path(Path(__file__).resolve().with_suffix("").name)

def _log_path(now_utc: datetime) -> Path:
    base = _base_log_dir()
    year = now_utc.strftime("%Y"); month = now_utc.strftime("%m")
    week = f"{now_utc.isocalendar().week:02d}"; day  = now_utc.strftime("%d")
    fname = now_utc.strftime("%Y-%m-%d_%H-%M-%S") + ".log"
    return base / year / month / week / day / fname

def _csv_path() -> Path:
    return _base_log_dir() / "summary.csv"

def _write_summary_log(text: str, now_utc: datetime) -> None:
    path = _log_path(now_utc); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def _append_csv(rows: List[dict]) -> None:
    csv_path = _csv_path(); csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows); header = not csv_path.exists()
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
    if code == "W": return _week_start_utc(now) - timedelta(weeks=1)
    if code == "D": return datetime(now.year, now.month, now.day, tzinfo=timezone.utc) - timedelta(days=1)
    im = interval_ms(code); now_ms = int(now.timestamp() * 1000)
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
    stoch = StochRSIIndicator(d["close"], window=14, smooth1=3, smooth2=3)
    d["stoch_k"], d["stoch_d"] = stoch.stochrsi_k(), stoch.stochrsi_d()
    macd = MACD(d["close"], window_slow=26, window_fast=12, window_sign=9)
    d["macd"], d["macd_sig"], d["macd_diff"] = macd.macd(), macd.macd_signal(), macd.macd_diff()
    bb = BollingerBands(d["close"], window=20, window_dev=2)
    d["bb_h"] = bb.bollinger_hband(); d["bb_l"] = bb.bollinger_lband(); d["bb_m"] = bb.bollinger_mavg()
    d["bb_width"] = (d["bb_h"] - d["bb_l"]) / d["close"]
    d["atr_14"] = AverageTrueRange(d["high"], d["low"], d["close"], window=14).average_true_range()
    d["rv_12"] = d["ret_1"].rolling(12).std()
    d["mfi_14"] = MFIIndicator(d["high"], d["low"], d["close"], d["volume"], window=14).money_flow_index()
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
    mdl_a.fit(X, y_close); pred_a = float(mdl_a.predict(X.tail(1))[0])

    mdl_b = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, random_state=SEED)
    mdl_b.fit(X, y_ret);   pred_b_ret = float(mdl_b.predict(X.tail(1))[0])
    pred_b = last_close * (1.0 + pred_b_ret / 100.0)

    mdl_c = ElasticNet(alpha=0.001, l1_ratio=0.15, random_state=SEED, max_iter=2000)
    mdl_c.fit(X, y_ret);   pred_c_ret = float(mdl_c.predict(X.tail(1))[0])
    pred_c = last_close * (1.0 + pred_c_ret / 100.0)

    preds = np.array([pred_a, pred_b, pred_c])
    weights = np.array([W_PRICE, W_RET_LGB, W_RET_EN]); weights /= weights.sum()
    pred_ens = float(np.dot(weights, preds))
    ens_std = float(np.std(preds))

    # provide individual predictions for AI signals
    preds_dict = {"A": pred_a, "B": pred_b, "C": pred_c, "ENS": pred_ens}
    return pred_ens, cv_mape, dir_acc, ens_std, preds_dict

def decide_signal_from_prices(pred_price, last_price) -> str:
    dpct = (pred_price - last_price) / last_price * 100.0
    bps = dpct * 100.0
    if bps >= BUY_BPS: return "BUY"
    if bps <= SELL_BPS: return "SELL"
    return "FLAT"

def decide_signal(dpct): 
    bps = dpct * 100
    if bps >= BUY_BPS: return "BUY"
    if bps <= SELL_BPS: return "SELL"
    return "FLAT"

def color(sig): return GREEN if sig=="BUY" else RED if sig=="SELL" else YELL

def overlay_with_rsi_macd(sig, rsi, macd_diff, last_price):
    if 45 <= rsi <= 55 and abs(macd_diff)/max(last_price,1e-9) < 5e-4: return "FLAT"
    return sig

# ---- Indicator signal helpers ----
def rsi_signal(v):   return "OB" if v >= 70 else ("OS" if v <= 30 else "NEU")
def macd_signal(v):  return "BULL" if v > 0 else ("BEAR" if v < 0 else "NEU")
def ema_trend(short, long): 
    return "UP" if short > long else ("DOWN" if short < long else "NEU")
def bb_pos(close, bb_l, bb_m, bb_h):
    if close >= bb_h: return "UPPER"
    if close <= bb_l: return "LOWER"
    return "MIDDLE"
def atr_level(atr, close):
    # normalize ATR% of price; >1.5% considered high
    return "HIGH" if (atr / max(close,1e-9)) * 100.0 >= 1.5 else "NORMAL"
def stoch_signal(k): return "OB" if k >= 80 else ("OS" if k <= 20 else "NEU")
def mfi_signal(v):   return "OB" if v >= 80 else ("OS" if v <= 20 else "NEU")
def obv_trend(obv_series):
    if len(obv_series) < 6: return "NEU"
    return "UP" if obv_series.iloc[-1] > obv_series.iloc[-6] else ("DOWN" if obv_series.iloc[-1] < obv_series.iloc[-6] else "NEU")

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
        pred, cv, acc, ens_std, preds = fit_predict_ensemble(feat, last)
        rsi_v, macd_hist = float(feat["rsi_14"].iloc[-1]), float(feat["macd_diff"].iloc[-1])

        dpct = (pred - last) / last * 100
        base_sig = decide_signal(dpct)
        sig = overlay_with_rsi_macd(base_sig, rsi_v, macd_hist, last)
        conf = max(5.0, min(95.0, (1.0 - cv)*100.0 - (ens_std/last)*100.0))

        # indicator signals
        ema_s, ema_l = float(feat["ema_3"].iloc[-1]), float(feat["ema_24"].iloc[-1])
        bb_l, bb_m, bb_h = float(feat["bb_l"].iloc[-1]), float(feat["bb_m"].iloc[-1]), float(feat["bb_h"].iloc[-1])
        atr_v = float(feat["atr_14"].iloc[-1]); stoch_k = float(feat["stoch_k"].iloc[-1])
        mfi_v = float(feat["mfi_14"].iloc[-1])
        obv_sig = obv_trend(feat["obv"])

        ind_sig = {
            "rsi":  rsi_signal(rsi_v),
            "macd": macd_signal(macd_hist),
            "ema":  ema_trend(ema_s, ema_l),
            "bb":   bb_pos(last, bb_l, bb_m, bb_h),
            "atr":  atr_level(atr_v, last),
            "stoch": stoch_signal(stoch_k),
            "mfi":  mfi_signal(mfi_v),
            "obv":  obv_sig
        }

        # AI per-model signals
        ai_A = decide_signal_from_prices(preds["A"], last)
        ai_B = decide_signal_from_prices(preds["B"], last)
        ai_C = decide_signal_from_prices(preds["C"], last)
        ai_ens = decide_signal_from_prices(preds["ENS"], last)
        ai_agree = sum(s in ("BUY","SELL") and s == ai_ens for s in [ai_A, ai_B, ai_C])
        overlay_applied = "Yes" if sig != base_sig else "No"

        results[tf] = {
            "last": last, "pred": pred, "dpct": dpct,
            "cv": cv, "acc": acc*100.0,
            "rsi": rsi_v, "macd": macd_hist,
            "sig": sig, "conf": conf,
            "ind_sig": ind_sig,
            "ai": {"A": ai_A, "B": ai_B, "C": ai_C, "ENS": ai_ens, "agree": ai_agree, "overlay": overlay_applied}
        }

    vote = sum((WEIGHTS.get(tf,1) if r["sig"]=="BUY" else -WEIGHTS.get(tf,1) if r["sig"]=="SELL" else 0)
               for tf,r in results.items())
    overall = "BUY" if vote>0 else "SELL" if vote<0 else "FLAT"

    end_time = time.perf_counter(); duration = end_time - start_time

    # ---------- Build printable SUMMARY ----------
    now_utc = datetime.now(timezone.utc); now_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    # Plain-text log (SUMMARY-only)
    summary_lines = [
        "SUMMARY",
        f"symbol: {SYMBOL}  category: {CATEGORY}  generated_at: {now_str}",
        f"thresholds_bps: buy>={BUY_BPS}, sell<={SELL_BPS}",
        f"weights: {WEIGHTS}",
        f"execution_time_seconds: {duration:.2f}",
        ""
    ]

    # Console SUMMARY
    clear_console()
    print(f"{WHITE}SUMMARY{RESET}")
    print(f"{WHITE}symbol:{RESET} {SYMBOL}  {WHITE}category:{RESET} {CATEGORY}  {WHITE}generated_at:{RESET} {CYAN}{now_str}{RESET}")
    print(f"{WHITE}thresholds_bps:{RESET} buy>={BUY_BPS}, sell<={SELL_BPS}")
    print(f"{WHITE}weights:{RESET} {WEIGHTS}")
    print(f"{WHITE}execution_time_seconds:{RESET} {duration:.2f}\n")

    # CSV rows
    csv_rows: List[dict] = []

    # Per-timeframe printing (numbers), then signals line
    for tf in ["1w","1d","4h","1h","5m"]:
        r = results.get(tf)
        if not r: continue
        clr = color(r["sig"])
        print(f"{WHITE}[{tf}]{RESET} "
              f"last={r['last']:.2f}  pred={r['pred']:.2f}  Δ%={r['dpct']:.2f}  "
              f"MAPE={r['cv']*100:.2f}%  dir_acc={r['acc']:.2f}%  "
              f"RSI14={r['rsi']:.2f}  MACD_hist={r['macd']:.2f}  conf={r['conf']:.2f}%  "
              f"signal={clr}{r['sig']}{RESET}")

        # Signals/Results line (console only)
        isg = r["ind_sig"]; ai = r["ai"]
        print(f"    ind=[rsi:{isg['rsi']}, macd:{isg['macd']}, ema:{isg['ema']}, "
              f"bb:{isg['bb']}, atr:{isg['atr']}, stoch:{isg['stoch']}, mfi:{isg['mfi']}, obv:{isg['obv']}]  "
              f"AI=[A:{ai['A']}, B:{ai['B']}, C:{ai['C']}, ens:{ai['ENS']}, agree={ai['agree']}/3, overlay={ai['overlay']}]")

        # Add to plain text log (numbers only)
        summary_lines.append(
            f"[{tf}] last={r['last']:.2f}  pred={r['pred']:.2f}  Δ%={r['dpct']:.2f}  "
            f"MAPE={(r['cv']*100):.2f}%  dir_acc={(r['acc']):.2f}%  RSI14={r['rsi']:.2f}  "
            f"MACD_hist={r['macd']:.2f}  conf={r['conf']:.2f}%  signal={r['sig']}"
        )

        # CSV row (include signals)
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
            "overall_decision": overall,
            # indicator & AI signals
            "rsi_sig": isg["rsi"],
            "macd_sig": isg["macd"],
            "ema_trend": isg["ema"],
            "bb_pos": isg["bb"],
            "atr_level": isg["atr"],
            "stoch_sig": isg["stoch"],
            "mfi_sig": isg["mfi"],
            "obv_trend": isg["obv"],
            "ai_A": ai["A"], "ai_B": ai["B"], "ai_C": ai["C"], "ai_ens": ai["ENS"],
            "ai_agree": ai["agree"], "overlay_applied": ai["overlay"]
        })

    print(f"\n{WHITE}overall_decision:{RESET} {color(overall)}{overall}{RESET}  (vote={vote})")
    summary_lines.append("")
    summary_lines.append(f"overall_decision: {overall}  (vote={vote})")

    # Persist log + CSV
    _write_summary_log("\n".join(summary_lines) + "\n", now_utc)
    _append_csv(csv_rows)

    # Console-only LEGEND and AI sections
    print()
    print(f"{WHITE}LEGEND (how to read each metric and signal):{RESET}")
    print(f"  Δ% → Predicted percentage change in price vs. last close.")
    print(f"        • Positive = potential upside (BUY bias)")
    print(f"        • Negative = potential downside (SELL bias)")
    print(f"  MAPE → Mean Absolute Percentage Error — historical prediction error of the model.")
    print(f"        • Lower values = higher model accuracy (confidence improves when <5%).")
    print(f"  dir_acc → Directional accuracy (% of times model correctly predicted up/down moves).")
    print(f"        • >60% = generally good; 50% = random; <50% = poor predictive alignment.")
    print(f"  RSI14 → Relative Strength Index (momentum).")
    print(f"        • >70 = overbought (SELL pressure likely)")
    print(f"        • <30 = oversold (BUY pressure likely)")
    print(f"        • 45–55 = neutral/sideways momentum.")
    print(f"  MACD_hist → Difference between MACD and its signal line.")
    print(f"        • Positive = bullish momentum")
    print(f"        • Negative = bearish momentum")
    print(f"        • Near zero = low momentum / possible consolidation.")
    print(f"  EMA trend → Short EMA(3) vs Long EMA(24).")
    print(f"        • UP = short-term strength; price above long trend (BUY tendency)")
    print(f"        • DOWN = short-term weakness (SELL tendency)")
    print(f"        • NEU = both moving averages converging (indecision).")
    print(f"  BB pos → Position of last close inside Bollinger Bands(20).")
    print(f"        • UPPER = near resistance / overbought zone (SELL bias)")
    print(f"        • LOWER = near support / oversold zone (BUY bias)")
    print(f"        • MIDDLE = neutral zone / trend continuation likely.")
    print(f"  ATR level → Volatility level measured by ATR(14).")
    print(f"        • HIGH (>1.5%) = strong volatility / possible breakouts")
    print(f"        • NORMAL = steady or ranging conditions.")
    print(f"  StochRSI / MFI → Oscillators for short-term reversals.")
    print(f"        • OB (Overbought) = SELL risk high")
    print(f"        • OS (Oversold) = BUY opportunity possible")
    print(f"        • NEU = balanced or mid-range momentum.")
    print(f"  OBV → On-Balance Volume trend (volume-based strength).")
    print(f"        • UP = accumulation (BUY interest)")
    print(f"        • DOWN = distribution (SELL pressure)")
    print(f"        • NEU = flat volume trend / uncertainty.")
    print(f"  conf → Model confidence (%).")
    print(f"        • Combines low error (MAPE) + high ensemble agreement.")
    print(f"        • >85% = strong conviction; <60% = low reliability.")
    print(f"  AI A/B/C → Signals from individual AI models:")
    print(f"        • A = LightGBM (price prediction), B = LightGBM (return), C = ElasticNet (return).")
    print(f"        • ens = ensemble consensus (weighted combination).")
    print(f"        • agree = how many models match the ensemble (3/3 = strong confirmation).")
    print(f"        • overlay = whether RSI/MACD filters adjusted the final signal to FLAT.")
    print()


    print(f"{WHITE}AI STRATEGIES & INDICATORS USED:{RESET}")
    print(f"  • Machine Learning Ensemble: LGBM(price), LGBM(return→price), ElasticNet(return→price)")
    print(f"  • Time-Series CV (walk-forward, 3 splits); numeric-only feature gating")
    print(f"  • Closed-candle alignment; RSI/MACD overlay to avoid chop")
    print(f"  • Confidence from low CV error + ensemble agreement")
    print(f"  • Indicators: EMA(3,6,12,24,60), RSI(14), StochRSI, MACD(12,26,9),")
    print(f"               Bollinger Bands(20), ATR(14), Realized Vol(12), MFI(14), OBV,")
    print(f"               candle body/wick ratios & distances to EMAs\n")

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
