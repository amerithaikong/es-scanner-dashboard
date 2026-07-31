#!/usr/bin/env python3
"""
ES Regime Scanner v2 - live dashboard + confidence-scored alert engine
------------------------------------------------------------------------
Alert-only. Paper trading decision support. Never places orders.

1H BIAS (weighted score, direction from regression slope):
  - Linear regression slope + quality (dual lookback 40/20, best-R2 fit wins,
    so fresh trends are detected instead of blocked by a stale long window)
  - 50 EMA vs 200 EMA
  - Market structure: HH/HL vs LH/LL from swing fractals
  - Session VWAP position (RTH only; excluded from score outside RTH)

SETUP (stateful): once price pulls back to/through the regression midline in a
qualified trend, the setup stays ARMED for up to ARM_HOURS, waiting for a
trigger - no more "everything must line up in one snapshot".
Invalidation: pullback deeper than 2.75 sigma (trend likely broken).

5m TRIGGER (scored; momentum-resume bar is mandatory):
  - Bar closes beyond prior bar's high/low (momentum resumes)   [mandatory]
  - RSI crosses back through 50 in trend direction
  - MACD histogram improving (3 rising/falling bars)
  - Volume expansion on the signal bar
  - Price on the right side of session VWAP (RTH only)

AVOID FILTERS (any one blocks the alert):
  - Ranging market: |slope| below threshold or bias score < 60%
  - Extremely low volume: signal-bar volume < 35% of its 20-bar average
  - Extended move: 1H travel over last 6 bars > 3x ATR(14)  (climax - don't chase)
  - Chasing: price already back beyond midline +0.35 sigma in trend direction

CONFIDENCE = weighted bias + trigger points / available points. Alert fires
only at >= CONF_MIN (default 75%), max 2/day, 120 min cooldown.

v2.1 reliability changes:
  - REMOVED the yfinance fallback. Direct Yahoo hosts are permanently 429'd
    from this server's IP, so the fallback could never succeed - but
    yf.download() has no request timeout, so a silently-dropped connection
    hung the scanner thread forever (the 80,000s stall). All data now goes
    through yahoo_chart(), where every request has a hard 10s timeout.
  - prev_close no longer requires a separate daily fetch every loop. It is
    refreshed on the slow cadence and, if that fails, derived from the 1h
    dataframe already in memory (last close of the prior NY trading date).
  - Watchdog thread: if the scanner loop's heartbeat is older than
    WATCHDOG_SEC (default 300s), the process exits with code 1 so the
    hosting platform (Render etc.) restarts it automatically.
"""

import os
import smtplib
import threading
import time
import traceback
from datetime import datetime, timezone
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request

# ----------------------------------------------------------------------
# Config - env vars override where noted
# ----------------------------------------------------------------------
SYMBOL = "ES=F"
LOOKBACKS = (40, 20)                                      # dual regression windows
MIN_R2 = float(os.environ.get("MIN_R2", 0.45))
MIN_SLOPE = float(os.environ.get("MIN_SLOPE", 0.30))      # pts per 1h bar
BIAS_MIN_PCT = float(os.environ.get("BIAS_MIN_PCT", 60))  # % of bias weight to arm
CONF_MIN = float(os.environ.get("CONF_MIN", 75))          # % to fire an alert
ARM_HOURS = float(os.environ.get("ARM_HOURS", 8))         # armed setup lifetime
PULLBACK_Z = 0.25          # long: armed when z <= +0.25 (at/through midline)
INVALID_Z = 2.75           # pullback deeper than this against trend = broken
CHASE_Z = 0.35             # no entry if price already beyond mid +0.35s in trend dir
RSI_LEN, EMA_FAST, EMA_SLOW = 14, 50, 200
STOP_PTS, TARGET1_PTS = 10.0, 10.0
MAX_ALERTS_PER_DAY, COOLDOWN_MIN = 2, 120
POLL_FAST = int(os.environ.get("POLL_FAST", 30))
POLL_SLOW = int(os.environ.get("POLL_SLOW", 180))
# Alerts only fire inside this ET window (scanning never stops).
ALERT_WINDOW_START = os.environ.get("ALERT_WINDOW_START", "05:00")  # ET, premarket open
ALERT_WINDOW_END = os.environ.get("ALERT_WINDOW_END", "20:00")      # ET, RTH close
WATCHDOG_SEC = int(os.environ.get("WATCHDOG_SEC", 300))   # restart if loop stalls
CHART_BARS = 200

# Weights (points). VWAP weights are excluded from the denominator outside RTH.
W_BIAS = {"slope_dir": 10, "slope_strength": 10, "r2": 15, "ema": 15,
          "structure": 15, "vwap_bias": 10}
W_TRIG = {"momentum_bar": 12, "rsi_cross": 10, "macd_hist": 6,
          "volume": 7, "vwap_side": 5}

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_APP_PASSWORD = os.environ.get("SMTP_APP_PASSWORD", "")
SMS_TO = os.environ.get("SMS_TO", "")

app = Flask(__name__)
LOCK = threading.Lock()
STATE = {
    "last_price": None, "prev_close": None, "change_pts": None, "change_pct": None,
    "price_updated": None,
    "charts": {"5m": {"t": [], "c": []}, "15m": {"t": [], "c": []}, "1h": {"t": [], "c": []}},
    "bias": None,            # direction, pct, factors[], slope, r2, z, lookback, ...
    "setup": None,           # {direction, armed_at, expires_at}  when armed
    "trigger": None,         # factors[], mandatory_ok
    "avoid": [],             # active avoid-filter reasons
    "confidence": None,
    "alerts": [], "alerts_today": 0, "alerts_date": "", "last_alert_ts": 0.0,
    "feed": {"status": "starting", "detail": "", "errors": 0},
    "loop": {"n": 0, "phase": "boot", "ts": None, "epoch": None},
}


# ----------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------
def linreg(closes: np.ndarray):
    x = np.arange(len(closes), dtype=float)
    b, a = np.polyfit(x, closes, 1)
    fitted = a + b * x
    resid = closes - fitted
    ss_tot = float(np.sum((closes - closes.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 0.0
    std = float(np.std(resid, ddof=1)) if len(closes) > 2 else 0.0
    return float(b), r2, std, float(fitted[-1])


def best_regression(closes: pd.Series):
    """Fit each lookback; return the fit with the higher R^2 (fresh-trend friendly)."""
    best = None
    for lb in LOOKBACKS:
        arr = closes.tail(lb).to_numpy(dtype=float)
        slope, r2, std, fitted_last = linreg(arr)
        cand = {"lookback": lb, "slope": slope, "r2": r2, "std": std,
                "fitted_last": fitted_last}
        if best is None or r2 > best["r2"]:
            best = cand
    return best


def rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd_hist(series: pd.Series) -> pd.Series:
    macd = series.ewm(span=12, adjust=False).mean() - series.ewm(span=26, adjust=False).mean()
    return macd - macd.ewm(span=9, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> float:
    h, l, c = df["High"], df["Low"], df["Close"].shift()
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])


def swing_structure(htf: pd.DataFrame, k: int = 2, scan: int = 60):
    """Fractal swings on 1H. Returns 'HH/HL', 'LH/LL', or 'MIXED'."""
    df = htf.tail(scan)
    highs, lows = df["High"].to_numpy(), df["Low"].to_numpy()
    sh, sl = [], []
    for i in range(k, len(df) - k):
        if highs[i] == max(highs[i - k:i + k + 1]):
            sh.append(highs[i])
        if lows[i] == min(lows[i - k:i + k + 1]):
            sl.append(lows[i])
    if len(sh) < 2 or len(sl) < 2:
        return "MIXED"
    hh, hl = sh[-1] > sh[-2], sl[-1] > sl[-2]
    lh, ll = sh[-1] < sh[-2], sl[-1] < sl[-2]
    if hh and hl:
        return "HH/HL"
    if lh and ll:
        return "LH/LL"
    return "MIXED"


def session_vwap(ltf: pd.DataFrame):
    """RTH VWAP anchored at 09:30 ET. Returns (vwap, applicable, in_rth)."""
    idx = ltf.index.tz_convert("America/New_York")
    now = datetime.now(timezone.utc).astimezone(idx.tz)
    in_rth = (now.weekday() < 5 and
              (now.hour, now.minute) >= (9, 30) and now.hour < 16)
    if not in_rth:
        return None, False, False
    day = now.date()
    mask = (idx.date == day) & (idx.time >= pd.Timestamp("09:30").time())
    sess = ltf.loc[mask]
    if len(sess) < 3 or float(sess["Volume"].sum()) <= 0:
        return None, False, True
    tp = (sess["High"] + sess["Low"] + sess["Close"]) / 3
    vwap = float((tp * sess["Volume"]).sum() / sess["Volume"].sum())
    return vwap, True, True


# ----------------------------------------------------------------------
# Bias / trigger / avoid evaluation
# ----------------------------------------------------------------------
def factor(name, ok, na=False, detail=""):
    return {"name": name, "ok": bool(ok), "na": bool(na), "detail": detail}


def eval_bias(htf: pd.DataFrame, ltf: pd.DataFrame):
    closed = htf.iloc[:-1]
    close = closed["Close"]
    reg = best_regression(close)
    slope, r2, std = reg["slope"], reg["r2"], reg["std"]
    last = float(close.iloc[-1])
    z = (last - reg["fitted_last"]) / std if std > 0 else 0.0

    direction = "LONG" if slope >= MIN_SLOPE else "SHORT" if slope <= -MIN_SLOPE else None
    bull = slope > 0

    ema_f = float(close.ewm(span=EMA_FAST, adjust=False).mean().iloc[-1])
    ema_s = float(close.ewm(span=EMA_SLOW, adjust=False).mean().iloc[-1])
    ema_ok = (ema_f > ema_s) if bull else (ema_f < ema_s)

    structure = swing_structure(closed)
    struct_ok = (structure == "HH/HL") if bull else (structure == "LH/LL")

    vwap, vwap_ok_applicable, in_rth = session_vwap(ltf.iloc[:-1])
    if vwap_ok_applicable:
        px = float(ltf["Close"].iloc[-1])
        vwap_ok = (px > vwap) if bull else (px < vwap)
        vwap_f = factor("Above VWAP" if bull else "Below VWAP", vwap_ok,
                        detail=f"vwap {vwap:.2f}")
    else:
        vwap_ok = None
        vwap_f = factor("VWAP position", False, na=True,
                        detail="RTH only" if not in_rth else "no volume data")

    factors = [
        factor(f"Regression slope {'positive' if bull else 'negative'}",
               direction is not None, detail=f"{slope:+.2f} pts/bar, lb {reg['lookback']}"),
        factor("Slope strength", abs(slope) >= 2 * MIN_SLOPE or
               (direction is not None and abs(slope) >= MIN_SLOPE),
               detail=f"|{slope:.2f}| vs min {MIN_SLOPE}"),
        factor("Trend quality R\u00b2", r2 >= MIN_R2, detail=f"{r2:.2f}"),
        factor(f"50 EMA {'>' if bull else '<'} 200 EMA", ema_ok,
               detail=f"{ema_f:.0f} / {ema_s:.0f}"),
        factor("Structure " + ("HH/HL" if bull else "LH/LL"), struct_ok,
               detail=structure),
        vwap_f,
    ]
    keys = ["slope_dir", "slope_strength", "r2", "ema", "structure", "vwap_bias"]
    pts = avail = 0.0
    for f, kname in zip(factors, keys):
        if f["na"]:
            continue
        avail += W_BIAS[kname]
        if f["ok"]:
            pts += W_BIAS[kname]
    # partial credit for slope strength
    if direction is not None and not factors[1]["ok"]:
        pts += W_BIAS["slope_strength"] * min(1.0, abs(slope) / (2 * MIN_SLOPE))

    pct = round(100 * pts / avail, 1) if avail else 0.0
    return {
        "direction": direction, "pct": pct, "pts": pts, "avail": avail,
        "factors": factors, "slope": round(slope, 3), "r2": round(r2, 3),
        "z": round(z, 2), "resid_std": round(std, 2),
        "fitted_last": round(reg["fitted_last"], 2), "lookback": reg["lookback"],
        "structure": structure, "ema_fast": round(ema_f, 2), "ema_slow": round(ema_s, 2),
        "vwap": round(vwap, 2) if vwap else None, "in_rth": in_rth,
        "fitted_last_ts": close.index[-1].tz_convert("UTC").isoformat()
        if close.index.tz else close.index[-1].tz_localize("UTC").isoformat(),
        "trend": "UP" if bull and direction else "DOWN" if direction else "FLAT",
        "quality_ok": direction is not None and pct >= BIAS_MIN_PCT,
        "updated": datetime.now(timezone.utc).isoformat(),
    }


def eval_trigger(ltf: pd.DataFrame, direction: str, bias: dict):
    closed = ltf.iloc[:-1]
    close = closed["Close"]
    long_ = direction == "LONG"

    c = float(close.iloc[-1])
    prev_h, prev_l = float(closed["High"].iloc[-2]), float(closed["Low"].iloc[-2])
    momentum = c > prev_h if long_ else c < prev_l

    r = rsi(close, RSI_LEN)
    r_now, r_win = float(r.iloc[-1]), r.iloc[-5:-1]
    rsi_ok = (r_win.min() < 50 and r_now > 50) if long_ else (r_win.max() > 50 and r_now < 50)

    h = macd_hist(close)
    h3 = h.iloc[-3:].to_numpy()
    macd_ok = bool(np.all(np.diff(h3) > 0)) if long_ else bool(np.all(np.diff(h3) < 0))

    vol = closed["Volume"]
    vavg = float(vol.rolling(20).mean().iloc[-1])
    vol_na = not np.isfinite(vavg) or vavg <= 0
    vol_ok = (not vol_na) and float(vol.iloc[-1]) >= 1.1 * vavg

    vwap = bias.get("vwap")
    if vwap:
        vwap_ok = c > vwap if long_ else c < vwap
        vwap_f = factor("Above VWAP" if long_ else "Below VWAP", vwap_ok)
    else:
        vwap_f = factor("VWAP side", False, na=True, detail="RTH only")

    factors = [
        factor("Momentum resumes (closes beyond prior bar)", momentum,
               detail=f"close {c:.2f}"),
        factor(f"RSI crossed {'above' if long_ else 'below'} 50", rsi_ok,
               detail=f"RSI {r_now:.0f}"),
        factor("MACD histogram improving", macd_ok),
        factor("Volume increasing on signal bar", vol_ok, na=vol_na,
               detail="" if vol_na else f"{float(vol.iloc[-1])/vavg:.1f}x avg"),
        vwap_f,
    ]
    keys = ["momentum_bar", "rsi_cross", "macd_hist", "volume", "vwap_side"]
    pts = avail = 0.0
    for f, kname in zip(factors, keys):
        if f["na"]:
            continue
        avail += W_TRIG[kname]
        if f["ok"]:
            pts += W_TRIG[kname]
    return {"factors": factors, "pts": pts, "avail": avail,
            "mandatory_ok": momentum, "entry": c}


def eval_avoid(htf: pd.DataFrame, ltf: pd.DataFrame, bias: dict):
    reasons = []
    if bias["direction"] is None:
        reasons.append("Ranging market - regression slope too flat")
    elif bias["pct"] < BIAS_MIN_PCT:
        reasons.append(f"Bias score {bias['pct']:.0f}% < {BIAS_MIN_PCT:.0f}% - mixed signals")

    closed = htf.iloc[:-1]
    a = atr(closed, 14)
    if np.isfinite(a) and a > 0:
        travel = abs(float(closed["Close"].iloc[-1]) - float(closed["Close"].iloc[-7]))
        if travel > 3 * a:
            reasons.append(f"Extended move: {travel:.0f} pts in 6h > 3x ATR ({a:.0f})")

    if bias["direction"]:
        z = bias["z"]
        chasing = z > CHASE_Z if bias["direction"] == "LONG" else z < -CHASE_Z
        if chasing:
            reasons.append(f"Chasing: price {z:+.1f}\u03c3 past midline in trend direction")

    vol = ltf.iloc[:-1]["Volume"]
    vavg = float(vol.rolling(20).mean().iloc[-1])
    if np.isfinite(vavg) and vavg > 0 and float(vol.iloc[-1]) < 0.35 * vavg:
        reasons.append("Extremely low volume period")
    return reasons


# ----------------------------------------------------------------------
# Alerts
# ----------------------------------------------------------------------
def send_sms(body: str):
    if not (SMTP_USER and SMTP_APP_PASSWORD and SMS_TO):
        return
    try:
        msg = MIMEText(body)
        msg["From"], msg["To"], msg["Subject"] = SMTP_USER, SMS_TO, "ES Alert"
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_APP_PASSWORD)
            s.send_message(msg)
    except Exception as e:
        print(f"[warn] SMS failed: {e}")


def format_alert(direction, conf, bias, trig, entry, stop, t1):
    lines = [f"{direction} ES", f"Confidence: {conf:.0f}%", "Reason:"]
    for f in bias["factors"] + trig["factors"]:
        mark = "\u2013" if f["na"] else ("\u2713" if f["ok"] else "\u2717")
        lines.append(f"{mark} {f['name']}")
    lines += ["", f"Entry:\n{entry:.2f}", f"Stop:\n{stop:.2f}",
              f"Scale 1/2 at {t1:.2f}, stop->BE, trail runner on 5m swings"]
    return "\n".join(lines)
  
def in_alert_window():
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:          # no alerts Sat/Sun
        return False
    hhmm = now_et.strftime("%H:%M")
    return ALERT_WINDOW_START <= hhmm < ALERT_WINDOW_END

def maybe_alert(direction, conf, bias, trig):
    if not in_alert_window():
        return False
    now = time.time()
    today = datetime.now(timezone.utc).date().isoformat()
    with LOCK:
        if STATE["alerts_date"] != today:
            STATE["alerts_date"], STATE["alerts_today"] = today, 0
        if STATE["alerts_today"] >= MAX_ALERTS_PER_DAY:
            return False
        if now - STATE["last_alert_ts"] < COOLDOWN_MIN * 60:
            return False
        entry = trig["entry"]
        stop = entry - STOP_PTS if direction == "LONG" else entry + STOP_PTS
        t1 = entry + TARGET1_PTS if direction == "LONG" else entry - TARGET1_PTS
        alert = {"ts": datetime.now(timezone.utc).isoformat(), "direction": direction,
                 "confidence": round(conf), "entry": round(entry, 2),
                 "stop": round(stop, 2), "t1": round(t1, 2),
                 "factors": [{"name": f["name"], "ok": f["ok"], "na": f["na"]}
                             for f in bias["factors"] + trig["factors"]]}
        STATE["alerts"].insert(0, alert)
        STATE["alerts"] = STATE["alerts"][:20]
        STATE["alerts_today"] += 1
        STATE["last_alert_ts"] = now
    body = format_alert(direction, conf, bias, trig, entry, stop, t1)
    print("ALERT\n" + body)
    send_sms(body)
    return True


# ----------------------------------------------------------------------
# Data + scanner loop
# ----------------------------------------------------------------------
YH_HOSTS = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")
# Optional relay (e.g. Cloudflare Worker) that forwards /v8/finance/chart/* to
# Yahoo from a non-blocked IP. Set YH_PROXY in Render's Environment tab.
YH_PROXY = os.environ.get("YH_PROXY", "").rstrip("/")
YH_BASES = ([YH_PROXY] if YH_PROXY else []) + [f"https://{h}" for h in YH_HOSTS]
YH_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}
_yh_session = requests.Session()
_yh_session.headers.update(YH_HEADERS)

# Scanner gets its OWN session: requests.Session is not thread-safe
_scanner_session = requests.Session()
_scanner_session.headers.update(YH_HEADERS)


def get_with_deadline(url, deadline=25):
    """GET that cannot hang. Runs in a disposable thread with its OWN
    fresh session, so an abandoned (timed-out) thread can never poison
    state shared with future requests."""
    result = {}

    def _run():
        s = requests.Session()
        s.headers.update(YH_HEADERS)
        try:
            result["r"] = s.get(url, timeout=10)
        except Exception as e:
            result["e"] = e
        finally:
            s.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(deadline)
    if t.is_alive():
        raise RuntimeError(f"request exceeded {deadline}s hard deadline "
                           f"(DNS or socket stall)")
    if "e" in result:
        raise result["e"]
    return result["r"]


def yahoo_chart(interval, range_, session=None):
    """Direct Yahoo v8 chart API via proxy/hosts, hard deadline per request."""
    session = session or _yh_session
    last_err = None
    for attempt, base in enumerate(YH_BASES + YH_BASES[:1]):
        try:
            url = (f"{base}/v8/finance/chart/{requests.utils.quote(SYMBOL)}"
                   f"?interval={interval}&range={range_}")
            r = get_with_deadline( url)
            if r.status_code == 429:
                raise RuntimeError("HTTP 429 rate limited")
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            ts = res.get("timestamp")
            q = res["indicators"]["quote"][0]
            if not ts:
                raise RuntimeError("no timestamps in response")
            df = pd.DataFrame(
                {"Open": q["open"], "High": q["high"], "Low": q["low"],
                 "Close": q["close"], "Volume": q["volume"]},
                index=pd.to_datetime(ts, unit="s", utc=True),
            ).dropna(subset=["Close"])
            if len(df) < 5:
                raise RuntimeError(f"empty {interval} response")
            df["Volume"] = df["Volume"].fillna(0)
            return df
        except Exception as e:
            last_err = e
            print(f"[warn] fetch attempt {attempt+1} {base}: "
                  f"{type(e).__name__}: {str(e)[:100]}", flush=True)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"yahoo chart api failed: {last_err}")



def fetch(interval: str, period: str) -> pd.DataFrame:
    # yfinance fallback removed: from this server's IP the direct Yahoo hosts
    # are hard 429-blocked, so yf.download() could never succeed - and it has
    # no request timeout, which is what silently froze the scanner thread.
    return yahoo_chart(interval, period, session =_scanner_session)


def prev_close_from_htf(htf: pd.DataFrame):
    """Fallback: derive previous session close from 1h bars already in memory
    (last close of the most recent NY-date before today)."""
    try:
        idx = (htf.index.tz_convert("America/New_York") if htf.index.tz
               else htf.index.tz_localize("UTC").tz_convert("America/New_York"))
        today = datetime.now(timezone.utc).astimezone(idx.tz).date()
        dates = np.array(idx.date)
        mask = dates < today
        if not mask.any():
            return None
        last_prior_date = dates[mask][-1]
        closes = htf["Close"].to_numpy(dtype=float)
        return float(closes[dates == last_prior_date][-1])
    except Exception:
        return None


def mark(phase, bump=False):
    with LOCK:
        if bump:
            STATE["loop"]["n"] += 1
        STATE["loop"]["phase"] = phase
        STATE["loop"]["ts"] = datetime.now(timezone.utc).isoformat()
        STATE["loop"]["epoch"] = time.time()


def scanner_loop():
    print(f"[boot] pid {os.getpid()} scanner starting", flush=True)
    last_slow, htf, prev_close = 0.0, None, None
    while True:
        try:
            mark("fetch 5m", bump=True)
            ltf = fetch("5m", "2d")
            closes = ltf["Close"]
            last_price = float(closes.iloc[-1])
            idx = ltf.index.tz_convert("UTC") if ltf.index.tz else ltf.index.tz_localize("UTC")

            slow_due = htf is None or time.time() - last_slow >= POLL_SLOW
            if slow_due:
                mark("fetch 1h")
                htf = fetch("1h", "60d")
                last_slow = time.time()

                # prev_close: refresh on the slow cadence only. Daily fetch
                # first; if it fails, derive from the 1h data we already have.
                mark("prev close")
                try:
                    daily = fetch("1d", "5d")
                    prev_close = float(daily["Close"].iloc[-2])
                except Exception as e:
                    print(f"[warn] daily fetch failed ({e}); deriving prev close from 1h")
                    derived = prev_close_from_htf(htf)
                    prev_close = derived if derived else STATE["prev_close"]
            else:
                prev_close = prev_close or STATE["prev_close"]

            mark("evaluate")
            bias = eval_bias(htf, ltf)
            avoid = eval_avoid(htf, ltf, bias)

            # ---- stateful pullback arming ----
            now = time.time()
            setup = STATE["setup"]
            if setup and (now > setup["expires_at"] or
                          setup["direction"] != bias["direction"]):
                setup = None
            if bias["quality_ok"] and bias["direction"]:
                z = bias["z"]
                pulled = z <= PULLBACK_Z if bias["direction"] == "LONG" else z >= -PULLBACK_Z
                broken = z < -INVALID_Z if bias["direction"] == "LONG" else z > INVALID_Z
                if broken:
                    setup = None
                elif pulled and setup is None:
                    setup = {"direction": bias["direction"], "armed_at": now,
                             "expires_at": now + ARM_HOURS * 3600}
            else:
                setup = None

            trig, conf = None, None
            if setup:
                trig = eval_trigger(ltf, setup["direction"], bias)
                denom = bias["avail"] + trig["avail"]
                conf = 100 * (bias["pts"] + trig["pts"]) / denom if denom else 0.0
                if trig["mandatory_ok"] and not avoid and conf >= CONF_MIN:
                    if maybe_alert(setup["direction"], conf, bias, trig):
                        setup = None  # consumed

            with LOCK:
                STATE["last_price"] = round(last_price, 2)
                STATE["price_updated"] = datetime.now(timezone.utc).isoformat()
                if prev_close:
                    STATE["prev_close"] = round(prev_close, 2)
                    STATE["change_pts"] = round(last_price - prev_close, 2)
                    STATE["change_pct"] = round((last_price / prev_close - 1) * 100, 2)
                def series(frame):
                    fidx = (frame.index.tz_convert("UTC") if frame.index.tz
                            else frame.index.tz_localize("UTC"))[-CHART_BARS:]
                    return {"t": [t.isoformat() for t in fidx],
                            "c": [round(float(v), 2)
                                  for v in frame["Close"].iloc[-CHART_BARS:]]}
                m15 = ltf.resample("15min").agg(
                    {"Close": "last"}).dropna(subset=["Close"])
                STATE["charts"] = {"5m": series(ltf), "15m": series(m15),
                                   "1h": series(htf)}
                STATE["bias"], STATE["avoid"] = bias, avoid
                STATE["setup"] = setup
                STATE["trigger"] = ({"factors": trig["factors"],
                                     "mandatory_ok": trig["mandatory_ok"]} if trig else None)
                STATE["confidence"] = round(conf, 1) if conf is not None else None
                STATE["feed"] = {"status": "live", "detail": "", "errors": 0}

        except Exception as e:
            with LOCK:
                STATE["feed"]["errors"] += 1
                STATE["feed"]["status"] = "stale" if STATE["last_price"] else "error"
                STATE["feed"]["detail"] = str(e)[:200]
            print(f"[error] scanner: {e}")
            traceback.print_exc()

        mark("sleep")
        time.sleep(POLL_FAST)


def watchdog_loop():
    """Self-heal: if the scanner heartbeat is older than WATCHDOG_SEC, exit
    the process so the hosting platform restarts it. This catches any future
    hang (library bug, DNS stall, etc.) that a try/except cannot."""
    while True:
        time.sleep(30)
        with LOCK:
            last = STATE["loop"].get("epoch")
            phase = STATE["loop"].get("phase")
    
        if not last:
            continue
        age = time.time() - last
        if age > 90:
            print(f"[watchdog] loop quiet for {int(age)}s in phase '{phase}'", flush=True)
        if age > WATCHDOG_SEC:
            print(f"[watchdog] stalled in '{phase}' {int(age)}s - exiting for restart", flush=True)
            os._exit(1)

_scanner_pid, _start_lock = None, threading.Lock()


def start_scanner():
    global _scanner_pid
    with _start_lock:
        if _scanner_pid != os.getpid():
            threading.Thread(target=scanner_loop, daemon=True).start()
            threading.Thread(target=watchdog_loop, daemon=True).start()
            _scanner_pid = os.getpid()


start_scanner()


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.before_request
def _ensure_scanner():
    start_scanner()
  
@app.route("/")
def index():
    return render_template("index.html", symbol=SYMBOL)


@app.route("/api/status")
def api_status():
    with LOCK:
        setup = STATE["setup"]
        return jsonify({
            "symbol": SYMBOL,
            "last_price": STATE["last_price"], "prev_close": STATE["prev_close"],
            "change_pts": STATE["change_pts"], "change_pct": STATE["change_pct"],
            "price_updated": STATE["price_updated"],
            "bias": STATE["bias"], "avoid": STATE["avoid"],
            "setup": ({"direction": setup["direction"],
                       "expires_in": max(0, int(setup["expires_at"] - time.time()))}
                      if setup else None),
            "trigger": STATE["trigger"], "confidence": STATE["confidence"],
            "alerts": STATE["alerts"], "alerts_today": STATE["alerts_today"],
            "max_alerts": MAX_ALERTS_PER_DAY,
            "cooldown_remaining": max(0, int(COOLDOWN_MIN * 60 -
                                             (time.time() - STATE["last_alert_ts"])))
            if STATE["last_alert_ts"] else 0,
            "feed": STATE["feed"], "loop": STATE["loop"],
            "params": {"min_r2": MIN_R2, "min_slope": MIN_SLOPE,
                       "bias_min_pct": BIAS_MIN_PCT, "conf_min": CONF_MIN,
                       "stop_pts": STOP_PTS, "target1_pts": TARGET1_PTS},
        })


@app.route("/api/chart")
def api_chart():
    tf = request.args.get("tf", "5m")
    if tf not in ("5m", "15m", "1h"):
        tf = "5m"
    with LOCK:
        return jsonify({"tf": tf, "chart": STATE["charts"].get(tf, {"t": [], "c": []}),
                        "regime": STATE["bias"]})


@app.route("/api/debug")
def api_debug():
    """One-shot Yahoo connectivity probe from this server, per host."""
    out = {"hosts": {}, "proxy_configured": bool(YH_PROXY),
           "loop": STATE["loop"], "feed": STATE["feed"]}
    for base in YH_BASES:
        try:
            url = (f"{base}/v8/finance/chart/"
                   f"{requests.utils.quote(SYMBOL)}?interval=5m&range=2d")
            r = _yh_session.get(url, timeout=10)
            body = r.text[:120].replace("\n", " ")
            n = 0
            try:
                n = len(r.json()["chart"]["result"][0].get("timestamp") or [])
            except Exception:
                pass
            out["hosts"][base] = {"http": r.status_code, "bars": n, "head": body}
        except Exception as e:
            out["hosts"][base] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
    return jsonify(out)


@app.route("/healthz")
def healthz():
    with LOCK:
        last = STATE["loop"].get("epoch")
    if last and time.time() - last > WATCHDOG_SEC:
        return "stalled", 503
    return "ok"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
