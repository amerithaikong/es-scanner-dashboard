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
import re
import smtplib
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo


import numpy as np
import pandas as pd
import requests
# Render containers sometimes have broken IPv6 routing: DNS returns an
# IPv6 address, connects to it hang. Force urllib3 to use IPv4 only.
import socket
import urllib3.util.connection as _urllib3_cn
_urllib3_cn.allowed_gai_family = lambda: socket.AF_INET
from flask import Flask, jsonify, render_template, request

# ----------------------------------------------------------------------
# Config - env vars override where noted
# ----------------------------------------------------------------------
SYMBOL = "ES=F"
LOOKBACKS = (40, 20)                                      # dual regression windows
MIN_R2 = float(os.environ.get("MIN_R2", 0.60))
MIN_SLOPE = float(os.environ.get("MIN_SLOPE", 0.30))      # pts per 1h bar
MIN_ATR_TRADE = float(os.environ.get("MIN_ATR_TRADE", 2.5))
BIAS_MIN_PCT = float(os.environ.get("BIAS_MIN_PCT", 60))  # % of bias weight to arm
CONF_MIN = float(os.environ.get("CONF_MIN", 60))          # % to fire an alert
ARM_HOURS = float(os.environ.get("ARM_HOURS", 8))         # armed setup lifetime
PULLBACK_Z = 0.25          # long: armed when z <= +0.25 (at/through midline)
RETRACE_Z = float(os.environ.get("RETRACE_Z", 0.6))
INVALID_Z = 2.75           # pullback deeper than this against trend = broken
CHASE_Z = 0.35             # no entry if price already beyond mid +0.35s in trend dir
TRIG_Z_MAX = float(os.environ.get("TRIG_Z_MAX", 1.0))   # trigger only within this many sigma of midline
CLOSE_POS_MIN = float(os.environ.get("CLOSE_POS_MIN", 0.7))  # trigger bar must close in top/bottom 30%
# Retest entry: after the trigger bar, wait for price to come back to the
# breakout level (prior bar high/low) before alerting. 0 = alert at market.
RETEST_BARS = int(os.environ.get("RETEST_BARS", 6))     # how many 5m bars to wait
RETEST_TOL = float(os.environ.get("RETEST_TOL", 1.5))   # pts of slack around the level
 # If the retest never comes but price is still within this many pts of the
# level when the wait expires, take it at market instead of dropping it. 0 = off.
RETEST_CHASE_PTS = float(os.environ.get("RETEST_CHASE_PTS", 3.0))
RSI_LEN, EMA_FAST, EMA_SLOW = 14, 50, 200
STOP_PTS      = float(os.environ.get("STOP_PTS", 5.25))       # fixed stop
TARGET1_PTS   = float(os.environ.get("TARGET1_PTS", 9.0))   # fixed target
STOP_ATR_MULT = float(os.environ.get("STOP_ATR_MULT", 1.5))
MIN_STOP_PTS  = float(os.environ.get("MIN_STOP_PTS", STOP_PTS))   # min = max = fixed
MAX_STOP_PTS  = float(os.environ.get("MAX_STOP_PTS", STOP_PTS))
TARGET_R      = float(os.environ.get("TARGET_R", 1.0))       # unused while target is fixed
# Exit plan shown on alerts / dashboard - indicative only, managed manually live
PARTIAL_PCT   = int(os.environ.get("PARTIAL_PCT", 50))       # % closed at target
TRAIL_PTS     = float(os.environ.get("TRAIL_PTS", 9.75))     # trail width after target

SWING_LOOKBACK = int(os.environ.get("SWING_LOOKBACK", 12))   # 5m bars for swing stop

MAX_BAR_ATR = float(os.environ.get("MAX_BAR_ATR", 1.8))   # trigger bar width cap
MAX_EXT_EMA = float(os.environ.get("MAX_EXT_EMA", 1.2))   # ATRs above 5m 20EMA

MAX_ALERTS_PER_DAY = int(os.environ.get("MAX_ALERTS_PER_DAY", 0))   # 0 = unlimited
COOLDOWN_MIN = int(os.environ.get("COOLDOWN_MIN", 45))
DUPE_PTS = float(os.environ.get("DUPE_PTS", 12.0))
CONF_MIN_OVERNIGHT = float(os.environ.get("CONF_MIN_OVERNIGHT", 75))
LOSS_COOLDOWN_MIN = int(os.environ.get("LOSS_COOLDOWN_MIN", 150))  # same-direction block after a stop-out

POLL_FAST = int(os.environ.get("POLL_FAST", 30))
POLL_SLOW = int(os.environ.get("POLL_SLOW", 180))
# Alerts only fire inside this ET window (scanning never stops).
ALERT_WINDOW_START = os.environ.get("ALERT_WINDOW_START", "06:30")  # ET, premarket open
ALERT_WINDOW_END = os.environ.get("ALERT_WINDOW_END", "16:00")      # ET, RTH close
WATCHDOG_SEC = int(os.environ.get("WATCHDOG_SEC", 300))   # restart if loop stalls
# Feed resilience: exponential backoff after consecutive fetch failures
# (POLL_FAST, 2x, 4x ... capped at BACKOFF_MAX seconds) and a per-host
# cooldown after a 429 so we stop hammering an edge that already said no.
BACKOFF_MAX = int(os.environ.get("BACKOFF_MAX", 600))
HOST_COOLDOWN_429 = int(os.environ.get("HOST_COOLDOWN_429", 300))
FEED_LOG_MAX = 50
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
    "ext_z": None, 
    "ext_dir": None,
    "trigger": None,         # factors[], mandatory_ok
    "avoid": [],             # active avoid-filter reasons
    "confidence": None,
    "alerts": [], "alerts_prev": [], "alerts_today": 0, "alerts_date": "", "last_alert_ts": 0.0,
        "last_resolved": True,
    "last_loss": None,       # {"ts", "direction"} of the most recent stop-out
    "pending": None,         # trigger fired, waiting for retest fill (see RETEST_BARS)
    "why_not": [],           # plain-English list of what is blocking an alert right now
    "feed": {"status": "starting", "detail": "", "errors": 0,
             "since": None, "last_ok": None, "backoff_s": 0, "last_http": None},
    "feed_log": deque(maxlen=FEED_LOG_MAX),   # newest first; see feed_log_add()
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


def session_vwap(ltf: pd.DataFrame, now=None):
    """RTH VWAP anchored 09:30 ET; Globex VWAP anchored 18:00 ET otherwise."""
    idx = ltf.index.tz_convert("America/New_York")
    now = (now or datetime.now(timezone.utc)).astimezone(idx.tz)
    in_rth = (now.weekday() < 5 and
              (now.hour, now.minute) >= (9, 30) and now.hour < 16)
    if in_rth:
        anchor = now.replace(hour=9, minute=30, second=0, microsecond=0)
    else:
        anchor = now.replace(hour=18, minute=0, second=0, microsecond=0)
        if now.hour < 18:
            anchor -= pd.Timedelta(days=1)
    sess = ltf.loc[idx >= anchor]
    if len(sess) < 3 or float(sess["Volume"].sum()) <= 0:
        return None, False, in_rth
    tp = (sess["High"] + sess["Low"] + sess["Close"]) / 3
    vwap = float((tp * sess["Volume"]).sum() / sess["Volume"].sum())
    return vwap, True, in_rth

# ----------------------------------------------------------------------
# Bias / trigger / avoid evaluation
# ----------------------------------------------------------------------
def factor(name, ok, na=False, detail=""):
    return {"name": name, "ok": bool(ok), "na": bool(na), "detail": detail}


def eval_bias(htf: pd.DataFrame, ltf: pd.DataFrame, asof=None):
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

    vwap, vwap_ok_applicable, in_rth = session_vwap(ltf.iloc[:-1], now=asof)
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
    a5 = atr(closed, 14)
    a5 = float(a5) if np.isfinite(a5) and a5 > 0 else 0.0
    swing = (float(closed["Low"].tail(SWING_LOOKBACK).min()) if long_
             else float(closed["High"].tail(SWING_LOOKBACK).max()))
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    bar_mid = float((closed["High"].iloc[-1] + closed["Low"].iloc[-1]) / 2)
    limit = max(bar_mid, ema20) if long_ else min(bar_mid, ema20)
    limit = min(limit, c) if long_ else max(limit, c)
   
  
    prev_h, prev_l = float(closed["High"].iloc[-2]), float(closed["Low"].iloc[-2])
    bar_h, bar_l = float(closed["High"].iloc[-1]), float(closed["Low"].iloc[-1])
    close_pos = (c - bar_l) / (bar_h - bar_l) if bar_h > bar_l else 1.0
    strong_close = close_pos >= CLOSE_POS_MIN if long_ else close_pos <= 1 - CLOSE_POS_MIN
    momentum = (c > prev_h if long_ else c < prev_l) and strong_close

    r = rsi(close, RSI_LEN)
    r_now, r_win = float(r.iloc[-1]), r.iloc[-5:-1]
    rsi_ok = (r_win.min() < 50 and r_now > 50) if long_ else (r_win.max() > 50 and r_now < 50)
    rsi_side = r_now > 50 if long_ else r_now < 50   # half credit: right side, no fresh cross

    h = macd_hist(close)
    h3 = h.iloc[-3:].to_numpy()
    macd_ok = bool(np.all(np.diff(h3) > 0)) if long_ else bool(np.all(np.diff(h3) < 0))

    vol = closed["Volume"]
    vavg = float(vol.rolling(20).mean().iloc[-1])
    vlast = float(vol.iloc[-1])
    vol_na = (not np.isfinite(vavg)) or vavg <= 0 or vlast <= 0
    vol_ok = (not vol_na) and vlast >= 1.1 * vavg

    vwap = bias.get("vwap")
    if vwap:
        vwap_ok = c > vwap if long_ else c < vwap
        vwap_f = factor("Above VWAP" if long_ else "Below VWAP", vwap_ok)
    else:
        vwap_f = factor("VWAP side", False, na=True, detail="RTH only")

    factors = [
        factor("Momentum resumes (beyond prior bar, strong close)", momentum,
               detail=f"close {c:.2f}, {close_pos:.0%} of bar"),
        factor(f"RSI crossed {'above' if long_ else 'below'} 50", rsi_ok,
               detail=f"RSI {r_now:.0f}" + ("" if rsi_ok else " (no fresh cross: half credit)" if rsi_side else "")),

        factor("MACD histogram improving", macd_ok),
        factor("Volume increasing on signal bar", vol_ok, na=vol_na,
               detail="" if vol_na else f"{float(vol.iloc[-1])/vavg:.1f}x avg"),
        vwap_f,
    ]
    keys = ["momentum_bar", "rsi_cross", "macd_hist", "volume", "vwap_side"]
    pts = avail = 0.0
    for f, kname in zip(factors, keys):
        avail += W_TRIG[kname]
        if f["ok"]:
            pts += W_TRIG[kname]
    if rsi_side and not rsi_ok:
        pts -= W_TRIG["rsi_cross"] / 2   # counted full above; take half back      
    return {"factors": factors, "pts": pts, "avail": avail,
            "mandatory_ok": momentum, "entry": c,
            "level": prev_h if long_ else prev_l, "close_pos": round(close_pos, 2),
            "atr5": round(a5, 2), "swing": round(swing, 2),
            "ema20": round(ema20, 2), "limit": round(limit * 4) / 4}

def eval_avoid(htf: pd.DataFrame, ltf: pd.DataFrame, bias: dict, armed: bool = False):
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

    if bias["direction"] and not armed:
        z = bias["z"]
        chasing = z > CHASE_Z if bias["direction"] == "LONG" else z < -CHASE_Z
        if chasing:
            reasons.append(f"Chasing: price {z:+.1f}\u03c3 past midline in trend direction")

    vol = ltf.iloc[:-1]["Volume"]
    vavg = float(vol.rolling(20).mean().iloc[-1])
    vlast = float(vol.iloc[-1])
    if np.isfinite(vavg) and vavg > 0 and 0 < vlast < 0.35 * vavg:
        reasons.append("Extremely low volume period")
    l5 = ltf.iloc[:-1]
    a5 = atr(l5, 14)
    if np.isfinite(a5) and a5 > 0 and bias["direction"]:
        if a5 < MIN_ATR_TRADE:
            reasons.append(f"Tape too quiet: 5m ATR {a5:.1f} "
                           f"< {MIN_ATR_TRADE} - target unreachable")
        bar_rng = float(l5["High"].iloc[-1] - l5["Low"].iloc[-1])
        if bar_rng > MAX_BAR_ATR * a5:
            reasons.append(f"Trigger bar too wide: {bar_rng:.1f} pts "
                           f"> {MAX_BAR_ATR}x 5m ATR ({a5:.1f})")
        e20 = float(l5["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
        ext = (float(l5["Close"].iloc[-1]) - e20) / a5
        if bias["direction"] == "LONG" and ext > MAX_EXT_EMA:
            reasons.append(f"Extended {ext:.1f} ATR above 5m 20EMA - wait for pullback")
        if bias["direction"] == "SHORT" and ext < -MAX_EXT_EMA:
            reasons.append(f"Extended {abs(ext):.1f} ATR below 5m 20EMA - wait for pullback")

    return reasons

# ----------------------------------------------------------------------
# Session rollover
# ----------------------------------------------------------------------
SCANNER_TZ = ZoneInfo(os.environ.get("SCANNER_TZ", "America/Los_Angeles"))

def trading_day() -> str:
    """Local calendar date. Rolls at local midnight."""
    return datetime.now(timezone.utc).astimezone(SCANNER_TZ).date().isoformat()

def roll_session() -> bool:
    """Wipe per-day state when the local date changes. Call with LOCK held."""
    today = trading_day()
    if STATE["alerts_date"] == today:
        return False
    STATE["alerts_date"] = today
    STATE["alerts_today"] = 0
    STATE["last_alert_ts"] = 0.0
    STATE["setup"] = None
    STATE["pending"] = None
    STATE["alerts_prev"] = STATE["alerts"][:20]
    STATE["alerts"] = []
    return True


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
    lines += ["", f"Limit: {trig.get('limit', entry):.2f}  (mkt {entry:.2f})",
              f"Stop: {stop:.2f}  ({abs(entry - stop):.2f} pts)",
              f"Target: {t1:.2f}  (+{abs(t1 - entry):.0f} pts)",
              f"Plan: close {PARTIAL_PCT}% at target, stop->BE, "
              f"trail rest {TRAIL_PTS:.0f} pts (manual)"]

    return "\n".join(lines)
  
def in_alert_window(now=None):
    now_et = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:          # no alerts Sat/Sun
        return False
    hhmm = now_et.strftime("%H:%M")
    return ALERT_WINDOW_START <= hhmm < ALERT_WINDOW_END
  
ET = ZoneInfo("America/New_York")

def session_date():
    """CME trading day - starts 18:00 ET, so evening bars belong to the next date."""
    et = datetime.now(timezone.utc).astimezone(ET)
    d = et.date()
    if et.hour >= 18:
        d += timedelta(days=1)
    return d.isoformat()
  
def maybe_alert(direction, conf, bias, trig):
    if not in_alert_window():
        return False
    now = time.time()
    with LOCK:
        roll_session()
        if MAX_ALERTS_PER_DAY and STATE["alerts_today"] >= MAX_ALERTS_PER_DAY:
            return False
        if now - STATE["last_alert_ts"] < COOLDOWN_MIN * 60:
            return False
        ll = STATE.get("last_loss")
        if (ll and ll["direction"] == direction
                and now - ll["ts"] < LOSS_COOLDOWN_MIN * 60):
            return False
        prev = STATE["alerts"][0] if STATE["alerts"] else None
        if (prev and prev["direction"] == direction
                and not STATE["last_resolved"]
                and abs(trig["entry"] - prev["entry"]) < DUPE_PTS):
            return False
        entry = trig["entry"]
        a5 = trig.get("atr5") or 0.0
        vol_risk = STOP_ATR_MULT * a5 if a5 > 0 else STOP_PTS
        struct_risk = (entry - trig["swing"] + 0.75) if direction == "LONG" \
                      else (trig["swing"] - entry + 0.75)
        risk = max(vol_risk, struct_risk)
        risk = round(min(max(risk, MIN_STOP_PTS), MAX_STOP_PTS) * 4) / 4
        t1_pts = TARGET1_PTS
        stop = entry - risk if direction == "LONG" else entry + risk
        t1 = entry + t1_pts if direction == "LONG" else entry - t1_pts

        alert = {"ts": datetime.now(timezone.utc).isoformat(), "limit": trig.get("limit"), 
                 "risk": risk, "direction": direction,
                 "confidence": round(conf), "entry": round(entry, 2),
                 "stop": round(stop, 2), "t1": round(t1, 2),
                 "session": STATE["alerts_date"],
                 "factors": [{"name": f["name"], "ok": f["ok"], "na": f["na"]}
                             for f in bias["factors"] + trig["factors"]]}
        STATE["alerts"].insert(0, alert)
        STATE["alerts"] = STATE["alerts"][:20]
        STATE["alerts_today"] += 1
        STATE["last_alert_ts"] = now
        STATE["last_resolved"] = False
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
            result["stage"] = "dns"
            host = url.split("/")[2]
            t0 = time.time()
            socket.getaddrinfo(host, 443, socket.AF_INET)
            result["dns_s"] = round(time.time() - t0, 1)
            result["stage"] = "http"
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
                           f"(stalled at stage: {result.get('stage')}, "
                           f"dns took {result.get('dns_s', '?')}s)")
    if "e" in result:
        raise result["e"]
    return result["r"]


class YahooHTTPError(RuntimeError):
    """Non-2xx from a Yahoo host. Carries status + host so the feed log can
    say *which* hop failed and *how* (429 block vs 500 backend vs 401)."""
    def __init__(self, status, base, head=""):
        self.status, self.base, self.head = status, base, head
        super().__init__(f"HTTP {status} from {base} {head!r}")


# base -> epoch until which we skip it (set after a 429)
_host_cooldown = {}
_host_last = {}   # base -> {"http": .., "ts": .., "msg": ..}  (for /api/debug)


def _note_host(base, http=None, msg=""):
    _host_last[base] = {"http": http, "msg": msg[:120],
                        "ts": datetime.now(timezone.utc).isoformat()}


def feed_log_add(status, msg, http=None, host=None):
    """Ring buffer of feed events (newest first). Survives until restart, so
    a week-long outage leaves a trail instead of one truncated string."""
    with LOCK:
        STATE["feed_log"].appendleft({
            "ts": datetime.now(timezone.utc).isoformat(),
            "status": status, "http": http, "host": host, "msg": str(msg)[:200]})


def yahoo_chart(interval, range_, session=None):
    """Direct Yahoo v8 chart API via proxy/hosts, hard deadline per request.
    Hosts that returned 429 recently are skipped for HOST_COOLDOWN_429s."""
    session = session or _yh_session
    last_err = None
    now = time.time()
    bases = [b for b in YH_BASES if _host_cooldown.get(b, 0) <= now]
    if not bases:   # everything is cooling down - try the first one anyway
        bases = YH_BASES[:1]
    for attempt, base in enumerate(bases):
        try:
            url = (f"{base}/v8/finance/chart/{requests.utils.quote(SYMBOL)}"
                   f"?interval={interval}&range={range_}")
            r = get_with_deadline(url)
            head = r.text[:80].replace("\n", " ")
            if r.status_code == 429:
                _host_cooldown[base] = time.time() + HOST_COOLDOWN_429
                raise YahooHTTPError(429, base, head)
            if r.status_code >= 400:
                raise YahooHTTPError(r.status_code, base, head)
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
            _note_host(base, r.status_code, f"ok {len(df)} bars")
            return df
        except Exception as e:
            last_err = e
            http = getattr(e, "status", None)
            _note_host(base, http, f"{type(e).__name__}: {e}")
            print(f"[warn] fetch attempt {attempt+1} {base}: "
                  f"{type(e).__name__}: {str(e)[:100]}", flush=True)
            if attempt < len(bases) - 1:
                time.sleep(1.5)
    raise RuntimeError(f"yahoo chart api failed: {last_err}") from last_err



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
            with LOCK:
              roll_session()
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
            avoid = eval_avoid(htf, ltf, bias, armed=bool(STATE["setup"]))

            # ---- stateful pullback arming ----
            now = time.time()
            setup = STATE["setup"]
            if setup and (now > setup["expires_at"] or
                          setup["direction"] != bias["direction"]):
                setup = None
            if bias["quality_ok"] and bias["direction"]:
                z = bias["z"]
                if STATE["ext_dir"] != bias["direction"] or STATE["ext_z"] is None:
                    STATE["ext_z"], STATE["ext_dir"] = z, bias["direction"]
                if bias["direction"] == "LONG":
                    STATE["ext_z"] = max(STATE["ext_z"], z)
                    pulled = z <= PULLBACK_Z or (STATE["ext_z"] - z) >= RETRACE_Z
                    broken = z < -INVALID_Z
                else:
                    STATE["ext_z"] = min(STATE["ext_z"], z)
                    pulled = z >= -PULLBACK_Z or (z - STATE["ext_z"]) >= RETRACE_Z
                    broken = z > INVALID_Z
                if broken:
                    setup = None
                  
                elif pulled and setup is None:
                    setup = {"direction": bias["direction"], 
                             "armed_at": now,
                             "expires_at": now + ARM_HOURS * 3600}
            else:
                setup = None

            trig, conf = None, None
            near_mid, conf_needed = True, CONF_MIN
            # --- stage 2: a trigger already fired, wait for the retest fill ---
            pend = STATE["pending"]
            if pend:
                bar = ltf.iloc[-1]          # forming 5m bar
                lvl, d = pend["level"], pend["direction"]
                touched = (float(bar["Low"]) <= lvl + RETEST_TOL if d == "LONG"
                           else float(bar["High"]) >= lvl - RETEST_TOL)
                px_now = float(bar["Close"])
                close_enough = (RETEST_CHASE_PTS > 0 and
                                abs(px_now - lvl) <= RETEST_CHASE_PTS and
                                (px_now >= lvl if d == "LONG" else px_now <= lvl))
                if bias["direction"] != d:
                    STATE["pending"] = None
                    setup = None
                    STATE["ext_z"] = bias["z"]
                elif now >= pend["expires_at"] and not touched and close_enough:
                    print(f"[retest] {d} no retest, taking market {px_now:.2f} "
                          f"({abs(px_now - lvl):.1f} pts from {lvl})", flush=True)
                    t2 = dict(pend["trig"]); t2["entry"] = px_now; t2["limit"] = px_now
                    if maybe_alert(d, pend["conf"], pend["bias"], t2):
                        setup = None
                        STATE["ext_z"] = bias["z"]
                    STATE["pending"] = None
                elif now >= pend["expires_at"]:
                    print(f"[retest] {d} expired without fill at {lvl}", flush=True)
                    STATE["pending"] = None
                    setup = None
                    STATE["ext_z"] = bias["z"]
                elif touched:
                    t2 = dict(pend["trig"]); t2["entry"] = lvl; t2["limit"] = lvl
                    if maybe_alert(d, pend["conf"], pend["bias"], t2):
                        setup = None
                        STATE["ext_z"] = bias["z"]
                    STATE["pending"] = None
            # --- stage 1: look for a trigger bar ---
            elif setup:
                trig = eval_trigger(ltf, setup["direction"], bias)
                denom = bias["avail"] + trig["avail"]
                conf = 100 * (bias["pts"] + trig["pts"]) / denom if denom else 0.0
                conf_needed = CONF_MIN if bias.get("in_rth") else CONF_MIN_OVERNIGHT
                near_mid = abs(bias["z"]) <= TRIG_Z_MAX
                if trig["mandatory_ok"] and near_mid and not avoid and conf >= conf_needed:
                    if RETEST_BARS > 0:
                        STATE["pending"] = {
                            "direction": setup["direction"], "level": trig["level"],
                            "sig_close": trig["entry"], "conf": conf,
                            "created_at": now, "expires_at": now + RETEST_BARS * 300,
                            "bias": bias, "trig": trig}
                        print(f"[retest] {setup['direction']} trigger at {trig['entry']}, "
                              f"waiting for {trig['level']} ({RETEST_BARS} bars)", flush=True)
                    elif maybe_alert(setup["direction"], conf, bias, trig):
                        setup = None  # consumed
                        STATE["ext_z"] = bias["z"]

            # ---- "why not firing" readout for the dashboard ----
            why = []
            pend = STATE["pending"]
            if pend:
                why.append(f"Retest pending: waiting for price to touch "
                           f"{pend['level']:.2f} ({max(0, int((pend['expires_at'] - now) / 60))} min left)")
            elif not bias["direction"]:
                why.append(f"No bias: 1H slope {bias['slope']:+.2f} is inside "
                           f"±{MIN_SLOPE} pts/bar (ranging)")
            elif not bias["quality_ok"]:
                why.append(f"Bias {bias['pct']:.0f}% < {BIAS_MIN_PCT:.0f}% needed to arm")
            elif not setup:
                need = f"z <= +{PULLBACK_Z}" if bias["direction"] == "LONG" else f"z >= -{PULLBACK_Z}"
                why.append(f"Not armed: waiting for pullback to midline "
                           f"(z {bias['z']:+.2f}, need {need} or {RETRACE_Z}σ retrace)")
            else:
                if trig and not trig["mandatory_ok"]:
                    why.append(f"Momentum bar ✗: last 5m close is not beyond the prior "
                               f"bar with a strong close ({trig.get('close_pos', 0):.0%} of bar, "
                               f"need {'>=' if setup['direction'] == 'LONG' else '<='} "
                               f"{CLOSE_POS_MIN if setup['direction'] == 'LONG' else 1 - CLOSE_POS_MIN:.0%})")
                if not near_mid:
                    why.append(f"Too far from midline: z {bias['z']:+.2f}, need |z| <= {TRIG_Z_MAX}")
                if conf is not None and conf < conf_needed:
                    why.append(f"Confidence {conf:.0f}% < {conf_needed:.0f}% "
                               f"({'RTH' if bias.get('in_rth') else 'overnight'} floor)")
                for a in avoid:
                    why.append("Avoid filter: " + a)
                if not in_alert_window():
                    why.append(f"Outside alert window {ALERT_WINDOW_START}-{ALERT_WINDOW_END} ET")
                cd = COOLDOWN_MIN * 60 - (now - STATE["last_alert_ts"])
                if STATE["last_alert_ts"] and cd > 0:
                    why.append(f"Cooldown: {int(cd / 60)} min since last alert")
                ll = STATE.get("last_loss")
                if ll and ll["direction"] == setup["direction"]:
                    lc = LOSS_COOLDOWN_MIN * 60 - (now - ll["ts"])
                    if lc > 0:
                        why.append(f"Loss cooldown: no {ll['direction']} for {int(lc / 60)} more min")
                if MAX_ALERTS_PER_DAY and STATE["alerts_today"] >= MAX_ALERTS_PER_DAY:
                    why.append(f"Daily cap reached ({MAX_ALERTS_PER_DAY})")
                if not why:
                    why.append("All gates clear - armed and waiting for the next qualifying 5m bar")

            with LOCK:
                STATE["why_not"] = why
                STATE["last_price"] = round(last_price, 2)
                STATE["price_updated"] = datetime.now(timezone.utc).isoformat()
                p = STATE["alerts"][0] if STATE["alerts"] else None
                if p and not STATE["last_resolved"]:
                    if p["direction"] == "LONG":
                        done = last_price >= p["t1"] or last_price <= p["stop"]
                    else:
                        done = last_price <= p["t1"] or last_price >= p["stop"]
                    if done:
                        STATE["last_resolved"] = True
                        stopped = (last_price <= p["stop"] if p["direction"] == "LONG"
                                   else last_price >= p["stop"])
                        if stopped:
                            STATE["last_loss"] = {"ts": time.time(),
                                                  "direction": p["direction"]}
                            STATE["ext_z"] = None   # force a fresh pullback before re-arming
                            setup = None
                  
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
                was_down = STATE["feed"]["errors"] > 0
                STATE["feed"] = {"status": "live", "detail": "", "errors": 0,
                                 "since": None, "backoff_s": 0, "last_http": 200,
                                 "last_ok": datetime.now(timezone.utc).isoformat()}
            if was_down:
                feed_log_add("recovered", "feed back to live")
            sleep_s = POLL_FAST

        except Exception as e:
            cause = e.__cause__ if isinstance(e.__cause__, Exception) else e
            http = getattr(cause, "status", None)
            host = getattr(cause, "base", None)
            with LOCK:
                f = STATE["feed"]
                f["errors"] += 1
                f["status"] = "stale" if STATE["last_price"] else "error"
                f["detail"] = str(e)[:200]
                f["last_http"] = http
                f["since"] = f["since"] or datetime.now(timezone.utc).isoformat()
                n = f["errors"]
                # 30s, 60s, 120s, 240s, 480s, then BACKOFF_MAX (default 600s)
                sleep_s = min(BACKOFF_MAX, POLL_FAST * (2 ** min(n - 1, 8)))
                f["backoff_s"] = sleep_s
            feed_log_add("error", f"{type(cause).__name__}: {cause}", http, host)
            print(f"[error] scanner ({n} in a row, next try in {sleep_s}s): {e}",
                  flush=True)
            if n == 1:
                traceback.print_exc()

        # Sleep in short slices, heartbeating each one, so a long backoff
        # never looks like a stall to the watchdog.
        end = time.time() + sleep_s
        while True:
            mark("sleep" if sleep_s == POLL_FAST else f"backoff {sleep_s}s")
            left = end - time.time()
            if left <= 0:
                break
            time.sleep(min(15, left))


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
        roll_session()
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
            "why_not": STATE["why_not"],
            "pending": ({"direction": STATE["pending"]["direction"],
                         "level": STATE["pending"]["level"],
                         "expires_in": max(0, int(STATE["pending"]["expires_at"] - time.time()))}
                        if STATE["pending"] else None),
            "alerts": STATE["alerts"], "alerts_prev": STATE["alerts_prev"],
            "alerts_today": STATE["alerts_today"], "session": STATE["alerts_date"],
            "max_alerts": MAX_ALERTS_PER_DAY or None, 
            "cooldown_remaining": max(0, int(COOLDOWN_MIN * 60 -
                                             (time.time() - STATE["last_alert_ts"])))
            if STATE["last_alert_ts"] else 0,
            "feed": STATE["feed"], "loop": STATE["loop"],
            "feed_log": list(STATE["feed_log"])[:10],
            "params": {"min_r2": MIN_R2, "min_slope": MIN_SLOPE,
                       "bias_min_pct": BIAS_MIN_PCT, "conf_min": CONF_MIN,
                       "stop_pts": STOP_PTS, "target1_pts": TARGET1_PTS,
                       "partial_pct": PARTIAL_PCT, "trail_pts": TRAIL_PTS,
                       "conf_min_overnight": CONF_MIN_OVERNIGHT},
           # display-only: lets the page describe the rules it is actually running
                       "arm_hours": ARM_HOURS, "cooldown_min": COOLDOWN_MIN,
                       "loss_cooldown_min": LOSS_COOLDOWN_MIN,
                       "retest_bars": RETEST_BARS, "retest_chase_pts": RETEST_CHASE_PTS,
                       "window_start": ALERT_WINDOW_START, "window_end": ALERT_WINDOW_END,
                       "max_alerts": MAX_ALERTS_PER_DAY or None},

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
    now = time.time()
    with LOCK:
        out = {"hosts": {}, "proxy_configured": bool(YH_PROXY),
               "loop": dict(STATE["loop"]), "feed": dict(STATE["feed"]),
               "feed_log": list(STATE["feed_log"]),
               "host_last_result": dict(_host_last),
               "host_cooldown_remaining_s": {
                   b: max(0, int(t - now)) for b, t in _host_cooldown.items()}}
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

# ----------------------------------------------------------------------
# Replay: run the exact live gates over recent history, grade each alert
# ----------------------------------------------------------------------
REPLAY = {"status": "idle", "result": None, "error": None, "started": None}
_replay_lock = threading.Lock()


def _replay_run(days, params):
    ltf = yahoo_chart("5m", "60d")
    htf = yahoo_chart("1h", "60d")
    start = ltf.index[-1] - pd.Timedelta(days=days)
    i0 = int(np.searchsorted(ltf.index, start))
    i0 = max(i0, 250)
    hor = int(params.get("horizon_bars", 36))
    conf_min = float(params.get("conf_min", CONF_MIN))
    conf_min_on = float(params.get("conf_min_overnight", CONF_MIN_OVERNIGHT))
    trig_z = float(params.get("trig_z_max", TRIG_Z_MAX))
    stop_pts = float(params.get("stop_pts", STOP_PTS))
    tgt_pts = float(params.get("target1_pts", TARGET1_PTS))
    retest_bars = int(params.get("retest_bars", RETEST_BARS))
    retest_tol = float(params.get("retest_tol", RETEST_TOL))
    chase = float(params.get("retest_chase_pts", RETEST_CHASE_PTS))
    use_window = bool(int(params.get("window", 1)))

    setup = pend = last_loss = None
    ext_z = ext_dir = None
    last_alert_ts = 0.0
    alerts, blocks, armed_bars, bars = [], {}, 0, 0
    hi = htf.index

    def block(k):
        blocks[k] = blocks.get(k, 0) + 1

    for i in range(i0, len(ltf)):
        t = ltf.index[i]
        ltf_w = ltf.iloc[max(0, i - 400):i + 1]
        htf_w = htf.iloc[:int(np.searchsorted(hi, t, side="right"))]
        if len(htf_w) < 60:
            continue
        bars += 1
        now = t.timestamp()
        bias = eval_bias(htf_w, ltf_w, asof=t.to_pydatetime())
        avoid = eval_avoid(htf_w, ltf_w, bias, armed=bool(setup))

        if setup and (now > setup["expires_at"] or setup["direction"] != bias["direction"]):
            setup = None
        if bias["quality_ok"] and bias["direction"]:
            z = bias["z"]
            if ext_dir != bias["direction"] or ext_z is None:
                ext_z, ext_dir = z, bias["direction"]
            if bias["direction"] == "LONG":
                ext_z = max(ext_z, z)
                pulled = z <= PULLBACK_Z or (ext_z - z) >= RETRACE_Z
                broken = z < -INVALID_Z
            else:
                ext_z = min(ext_z, z)
                pulled = z >= -PULLBACK_Z or (z - ext_z) >= RETRACE_Z
                broken = z > INVALID_Z
            if broken:
                setup = None
            elif pulled and setup is None:
                setup = {"direction": bias["direction"], "armed_at": now,
                         "expires_at": now + ARM_HOURS * 3600}
        else:
            setup = None
            block("no bias / bias < min" if not bias["direction"] or not bias["quality_ok"] else "x")

        def fire(d, conf, trig, entry, kind):
            nonlocal last_alert_ts, last_loss, setup, ext_z
            if use_window and not in_alert_window(t.to_pydatetime()):
                block("outside alert window"); return False
            if now - last_alert_ts < COOLDOWN_MIN * 60:
                block("cooldown"); return False
            if last_loss and last_loss["direction"] == d and now - last_loss["ts"] < LOSS_COOLDOWN_MIN * 60:
                block("loss cooldown"); return False
            stop = entry - stop_pts if d == "LONG" else entry + stop_pts
            t1 = entry + tgt_pts if d == "LONG" else entry - tgt_pts
            fwd = ltf.iloc[i + 1:i + 1 + hor]
            sgn = 1 if d == "LONG" else -1
            mae = mfe = 0.0
            outcome, bars_to = "timeout", None
            for k, (_, b) in enumerate(fwd.iterrows()):
                mae = max(mae, (entry - float(b["Low"])) * sgn)
                mfe = max(mfe, (float(b["High"]) - entry) * sgn)
                hit_stop = float(b["Low"]) <= stop if d == "LONG" else float(b["High"]) >= stop
                hit_t1 = float(b["High"]) >= t1 if d == "LONG" else float(b["Low"]) <= t1
                if hit_stop:
                    outcome, bars_to = "stop", k + 1; break
                if hit_t1:
                    outcome, bars_to = "target", k + 1; break
            et = t.tz_convert("America/New_York")
            alerts.append({
                "ts_et": et.strftime("%Y-%m-%d %H:%M"), "direction": d, "entry_kind": kind,
                "confidence": round(conf, 1), "entry": round(entry, 2), "stop": round(stop, 2),
                "t1": round(t1, 2), "outcome": outcome, "bars_to_resolve": bars_to,
                "mae": round(mae, 2), "mfe": round(mfe, 2),
                "in_rth": bool(bias.get("in_rth")), "z": bias["z"], "r2": bias["r2"],
                "slope": bias["slope"], "bias_pct": bias["pct"],
                "factors_ok": [f["name"] for f in bias["factors"] + trig["factors"] if f["ok"]],
                "factors_fail": [f["name"] for f in bias["factors"] + trig["factors"] if not f["ok"]],
            })
            last_alert_ts = now
            if outcome == "stop":
                last_loss = {"ts": now + (bars_to or 0) * 300, "direction": d}
            setup = None
            ext_z = bias["z"]
            return True

        if pend:
            bar = ltf_w.iloc[-1]
            lvl, d = pend["level"], pend["direction"]
            touched = (float(bar["Low"]) <= lvl + retest_tol if d == "LONG"
                       else float(bar["High"]) >= lvl - retest_tol)
            px = float(bar["Close"])
            close_enough = chase > 0 and abs(px - lvl) <= chase and (px >= lvl if d == "LONG" else px <= lvl)
            if bias["direction"] != d:
                pend = None; setup = None; ext_z = bias["z"]; block("retest: bias flipped")
            elif touched:
                fire(d, pend["conf"], pend["trig"], lvl, "retest"); pend = None
            elif now >= pend["expires_at"] and close_enough:
                fire(d, pend["conf"], pend["trig"], px, "market-after-no-retest"); pend = None
            elif now >= pend["expires_at"]:
                pend = None; setup = None; ext_z = bias["z"]; block("retest: never came back")
        elif setup:
            armed_bars += 1
            trig = eval_trigger(ltf_w, setup["direction"], bias)
            denom = bias["avail"] + trig["avail"]
            conf = 100 * (bias["pts"] + trig["pts"]) / denom if denom else 0.0
            need = conf_min if bias.get("in_rth") else conf_min_on
            near = abs(bias["z"]) <= trig_z
            if not trig["mandatory_ok"]:
                block("momentum bar")
            elif not near:
                block("too far from midline")
            elif avoid:
                for a in avoid:
                    block("avoid: " + re.sub(r"[\d.+-]+", "#", a.split(" - ")[0]))
            elif conf < need:
                block("confidence < min")
            else:
                if retest_bars > 0:
                    pend = {"direction": setup["direction"], "level": trig["level"],
                            "conf": conf, "expires_at": now + retest_bars * 300, "trig": trig}
                else:
                    fire(setup["direction"], conf, trig, trig["entry"], "market")

    n = len(alerts)
    wins = sum(a["outcome"] == "target" for a in alerts)
    stops = sum(a["outcome"] == "stop" for a in alerts)
    rth = [a for a in alerts if a["in_rth"]]
    days_covered = max(1, len({a["ts_et"][:10] for a in alerts}) or days)
    return {
        "days": days, "bars_evaluated": bars, "bars_armed": armed_bars,
        "params": {"conf_min": conf_min, "conf_min_overnight": conf_min_on, "trig_z_max": trig_z,
                   "stop_pts": stop_pts, "target1_pts": tgt_pts, "retest_bars": retest_bars,
                   "retest_tol": retest_tol, "retest_chase_pts": chase, "window": use_window,
                   "horizon_bars": hor},
        "summary": {"alerts": n, "per_day": round(n / days, 2), "targets": wins, "stops": stops,
                    "timeouts": n - wins - stops,
                    "win_rate_decided": round(100 * wins / (wins + stops), 1) if wins + stops else None,
                    "expectancy_pts": round((wins * tgt_pts - stops * stop_pts) / n, 2) if n else None,
                    "avg_mae": round(float(np.mean([a["mae"] for a in alerts])), 2) if n else None,
                    "avg_mfe": round(float(np.mean([a["mfe"] for a in alerts])), 2) if n else None,
                    "rth_alerts": len(rth),
                    "rth_targets": sum(a["outcome"] == "target" for a in rth),
                    "rth_stops": sum(a["outcome"] == "stop" for a in rth)},
        "blocked_by": dict(sorted(blocks.items(), key=lambda kv: -kv[1])),
        "alerts": alerts,
        "note": ("Replay evaluates once per completed 5m bar and sees the whole retest bar at once, "
                 "so fills are slightly optimistic vs live. Stops/targets checked on bar highs/lows."),
    }


@app.route("/api/replay")
def api_replay():
    """GET /api/replay?days=10[&conf_min=70&trig_z_max=1.0&stop_pts=6&target1_pts=10
    &retest_bars=6&retest_tol=1.5&retest_chase_pts=3&window=1&horizon_bars=36]
    Starts a background replay (first call) and returns the result once done."""
    days = max(1, min(int(request.args.get("days", 10)), 30))
    params = {k: v for k, v in request.args.items() if k != "days"}
    with _replay_lock:
        if REPLAY["status"] == "running":
            return jsonify({"status": "running", "started": REPLAY["started"]})
        if (REPLAY["status"] == "done" and REPLAY["result"]
                and REPLAY["result"]["days"] == days
                and request.args.get("fresh") is None
                and all(str(REPLAY["result"]["params"].get(k)) == v for k, v in params.items()
                        if k in REPLAY["result"]["params"])):
            return jsonify({"status": "done", **REPLAY["result"]})
        REPLAY.update({"status": "running", "result": None, "error": None,
                       "started": datetime.now(timezone.utc).isoformat()})

    def _bg():
        try:
            res = _replay_run(days, params)
            REPLAY.update({"status": "done", "result": res})
        except Exception as e:
            REPLAY.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
            traceback.print_exc()
    threading.Thread(target=_bg, daemon=True).start()
    return jsonify({"status": "running", "started": REPLAY["started"],
                    "hint": "reload this URL in ~20-60s"})
@app.route("/healthz")
def healthz():
    with LOCK:
        last = STATE["loop"].get("epoch")
    if last and time.time() - last > WATCHDOG_SEC:
        return "stalled", 503
    return "ok"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
