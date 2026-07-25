from flask import Flask, jsonify, render_template_string
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

app = Flask(__name__)

CONFIG = {
    "symbol": "ES=F",
    "htf_interval": "1h", "htf_period": "30d", "htf_lookback": 40,
    "min_r2": 0.55, "min_slope_pts": 0.35,
    "pullback_z_min": 1.0, "pullback_z_max": 2.5,
    "ltf_interval": "5m", "ltf_period": "5d",
    "rsi_len": 14, "rsi_reset_level": 40, "rsi_trigger_level": 45,
    "rsi_reset_window": 8, "ema_len": 20, "vol_mult": 1.1,
}

def linreg_channel(closes):
    x = np.arange(len(closes), dtype=float)
    b, a = np.polyfit(x, closes, 1)
    fitted = a + b * x
    resid = closes - fitted
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((closes - closes.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    resid_std = float(np.std(resid, ddof=1)) if len(closes) > 2 else 0.0
    return float(b), r2, resid_std, float(fitted[-1])

def rsi(series, length):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def fetch_bars(interval, period):
    df = yf.download(CONFIG["symbol"], interval=interval, period=period,
                      progress=False, prepost=True, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    return df.iloc[:-1]

def get_status():
    htf = fetch_bars(CONFIG["htf_interval"], CONFIG["htf_period"])
    window = htf["Close"].tail(CONFIG["htf_lookback"]).to_numpy(dtype=float)
    slope, r2, resid_std, fitted_last = linreg_channel(window)
    last_close = float(window[-1])
    prev_close = float(window[-2]) if len(window) > 1 else last_close
    change = last_close - prev_close
    change_pct = (change / prev_close * 100) if prev_close else 0.0
    z = (last_close - fitted_last) / resid_std if resid_std > 0 else 0.0

    direction = None
    if r2 >= CONFIG["min_r2"] and abs(slope) >= CONFIG["min_slope_pts"]:
        lo, hi = CONFIG["pullback_z_min"], CONFIG["pullback_z_max"]
        if slope > 0 and -hi <= z <= -lo:
            direction = "LONG"
        elif slope < 0 and lo <= z <= hi:
            direction = "SHORT"

    confirmations = {}
    confirmed = False
    if direction:
        ltf = fetch_bars(CONFIG["ltf_interval"], CONFIG["ltf_period"])
        close = ltf["Close"]
        ema = close.ewm(span=CONFIG["ema_len"], adjust=False).mean()
        r = rsi(close, CONFIG["rsi_len"])
        vol_avg = ltf["Volume"].rolling(20).mean()
        c = float(close.iloc[-1])
        prev_high = float(ltf["High"].iloc[-2])
        prev_low = float(ltf["Low"].iloc[-2])
        r_now = float(r.iloc[-1])
        r_window = r.iloc[-CONFIG["rsi_reset_window"]:-1]
        v_ok = float(ltf["Volume"].iloc[-1]) >= CONFIG["vol_mult"] * float(vol_avg.iloc[-1])
        if direction == "LONG":
            confirmations["RSI reset & turn up"] = bool(r_window.min() < CONFIG["rsi_reset_level"] and r_now > CONFIG["rsi_trigger_level"])
            confirmations["Close above 20 EMA"] = bool(c > float(ema.iloc[-1]))
            confirmations["Momentum bar (broke prior high)"] = bool(c > prev_high)
        else:
            confirmations["RSI reset & turn down"] = bool(r_window.max() > (100 - CONFIG["rsi_reset_level"]) and r_now < (100 - CONFIG["rsi_trigger_level"]))
            confirmations["Close below 20 EMA"] = bool(c < float(ema.iloc[-1]))
            confirmations["Momentum bar (broke prior low)"] = bool(c < prev_low)
        confirmations["Volume ≥ 1.1x avg"] = bool(v_ok)
        confirmed = all(confirmations.values())

    return {
        "price": round(last_close, 2),
        "change": round(change, 2),
        "change_pct": round(change_pct, 2),
        "slope": round(slope, 3),
        "r2": round(r2, 3),
        "z": round(z, 2),
        "direction": direction,
        "confirmations": confirmations,
        "confirmed": confirmed,
        "server_time": datetime.now().strftime("%H:%M:%S"),
    }

def get_history():
    ltf = fetch_bars(CONFIG["ltf_interval"], CONFIG["ltf_period"])
    tail = ltf.tail(150)
    return {
        "labels": [ts.strftime("%H:%M") for ts in tail.index],
        "closes": [round(float(c), 2) for c in tail["Close"]],
    }

@app.route("/api/status")
def api_status():
    try:
        return jsonify(get_status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/history")
def api_history():
    try:
        return jsonify(get_history())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return render_template_string(HTML)

HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>ES Scanner</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0a0a0a;
    --card: #111111;
    --border: #232323;
    --text: #ededed;
    --muted: #888888;
    --green: #2ecc71;
    --red: #ef4444;
    --amber: #f5a623;
    --blue: #3b82f6;
  }
  * { box-sizing: border-box; }
  body {
    background: var(--bg); color: var(--text); margin: 0; padding: 32px 20px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  }
  .wrap { max-width: 920px; margin: 0 auto; }
  .top { display:flex; align-items:center; justify-content:space-between; margin-bottom:24px; flex-wrap:wrap; gap:12px; }
  .brand { display:flex; align-items:center; gap:10px; }
  .dot { width:8px; height:8px; border-radius:50%; background:var(--green); box-shadow:0 0 8px var(--green); animation: pulse 1.6s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }
  .brand h1 { font-size:15px; font-weight:600; margin:0; letter-spacing:-0.2px; }
  .brand span { color: var(--muted); font-size:12px; }
  .badge-symbol { background:#161616; border:1px solid var(--border); padding:5px 12px; border-radius:100px; font-size:12px; color:var(--muted); }
  .badge-symbol b { color: var(--text); }

  .price-row { display:flex; align-items:baseline; gap:12px; margin: 4px 0 20px; }
  .price { font-size:42px; font-weight:700; letter-spacing:-1px; }
  .chg { font-size:15px; font-weight:600; padding:3px 9px; border-radius:6px; }
  .chg.up { color: var(--green); background: rgba(46,204,113,0.1); }
  .chg.down { color: var(--red); background: rgba(239,68,68,0.1); }

  .card { background: var(--card); border:1px solid var(--border); border-radius:16px; padding:20px; margin-bottom:16px; }
  .card h2 { font-size:12px; text-transform:uppercase; letter-spacing:0.6px; color:var(--muted); margin:0 0 14px; font-weight:600; }
  canvas { max-height:220px; }

  .grid { display:grid; grid-template-columns: repeat(3,1fr); gap:12px; }
  .stat { background:#0d0d0d; border:1px solid var(--border); border-radius:12px; padding:14px; }
  .stat .label { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px; }
  .stat .val { font-size:20px; font-weight:700; }

  .regime-badge { display:inline-flex; align-items:center; gap:6px; padding:6px 14px; border-radius:100px; font-weight:700; font-size:13px; }
  .regime-badge.LONG { background: rgba(46,204,113,0.12); color: var(--green); border:1px solid rgba(46,204,113,0.3); }
  .regime-badge.SHORT { background: rgba(239,68,68,0.12); color: var(--red); border:1px solid rgba(239,68,68,0.3); }
  .regime-badge.NONE { background: rgba(136,136,136,0.1); color: var(--muted); border:1px solid var(--border); }

  .check-row { display:flex; justify-content:space-between; align-items:center; padding:10px 0; border-bottom:1px solid var(--border); font-size:13px; }
  .check-row:last-child { border-bottom:none; }
  .check-row .ok { color: var(--green); }
  .check-row .no { color: var(--red); }

  .confirmed-banner { margin-top:12px; text-align:center; padding:12px; border-radius:10px; background: rgba(245,166,35,0.1); border:1px solid rgba(245,166,35,0.3); color:var(--amber); font-weight:700; font-size:13px; letter-spacing:0.3px; }

  .scan-wrap { display:flex; align-items:center; gap:10px; margin-top:6px; }
  .scan-track { flex:1; height:4px; background:#1a1a1a; border-radius:100px; overflow:hidden; }
  .scan-fill { height:100%; background: linear-gradient(90deg, var(--blue), #60a5fa); width:0%; transition: width 1s linear; }
  .scan-label { font-size:11px; color:var(--muted); min-width:120px; text-align:right; }
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div class="brand">
      <div class="dot"></div>
      <div>
        <h1>ES Regime Scanner</h1>
        <span>Linear regression channel · alert-only · paper trading</span>
      </div>
    </div>
    <div class="badge-symbol"><b>ES=F</b> · CME E-mini S&amp;P 500</div>
  </div>

  <div class="price-row">
    <div class="price" id="price">--</div>
    <div class="chg" id="chg">--</div>
  </div>

  <div class="card">
    <h2>Price · Last 150 Bars (5m)</h2>
    <canvas id="chart"></canvas>
  </div>

  <div class="card">
    <h2>1H Regime</h2>
    <div style="margin-bottom:16px;"><span class="regime-badge NONE" id="regimeBadge">LOADING</span></div>
    <div class="grid">
      <div class="stat"><div class="label">Slope (pts/bar)</div><div class="val" id="slope">--</div></div>
      <div class="stat"><div class="label">R²</div><div class="val" id="r2">--</div></div>
      <div class="stat"><div class="label">Pullback Z</div><div class="val" id="z">--</div></div>
    </div>
  </div>

  <div class="card">
    <h2>5m Confirmation Checklist</h2>
    <div id="checks"><div class="check-row"><span>Loading...</span></div></div>
    <div id="confirmedBanner"></div>
  </div>

  <div class="card">
    <div class="scan-wrap">
      <div class="scan-track"><div class="scan-fill" id="scanFill"></div></div>
      <div class="scan-label" id="scanLabel">scanning…</div>
    </div>
  </div>
</div>

<script>
let chart;
const POLL_SECONDS = 60;
let secondsLeft = POLL_SECONDS;

function initChart(labels, data) {
  const ctx = document.getElementById('chart').getContext('2d');
  const gradient = ctx.createLinearGradient(0, 0, 0, 220);
  gradient.addColorStop(0, 'rgba(59,130,246,0.35)');
  gradient.addColorStop(1, 'rgba(59,130,246,0)');
  chart = new Chart(ctx, {
    type: 'line',
    data: { labels: labels, datasets: [{ data: data, borderColor:'#3b82f6', backgroundColor:gradient, fill:true, tension:0.3, pointRadius:0, borderWidth:2 }] },
    options: {
      responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{display:false} },
      scales:{
        x:{ ticks:{ color:'#666', maxTicksLimit:6, font:{size:10} }, grid:{ display:false } },
        y:{ ticks:{ color:'#666', font:{size:10} }, grid:{ color:'#1a1a1a' } }
      }
    }
  });
}

async function loadHistory() {
  const res = await fetch('/api/history');
  const d = await res.json();
  if (d.error) return;
  if (!chart) initChart(d.labels, d.closes);
  else { chart.data.labels = d.labels; chart.data.datasets[0].data = d.closes; chart.update(); }
}

async function loadStatus() {
  const res = await fetch('/api/status');
  const d = await res.json();
  if (d.error) return;

  document.getElementById('price').innerText = d.price.toLocaleString();
  const chgEl = document.getElementById('chg');
  const up = d.change >= 0;
  chgEl.className = 'chg ' + (up ? 'up' : 'down');
  chgEl.innerText = (up ? '+' : '') + d.change + ' (' + (up ? '+' : '') + d.change_pct + '%)';

  document.getElementById('slope').innerText = d.slope;
  document.getElementById('r2').innerText = d.r2;
  document.getElementById('z').innerText = d.z;

  const badge = document.getElementById('regimeBadge');
  const dir = d.direction || 'NONE';
  badge.className = 'regime-badge ' + dir;
  badge.innerText = d.direction ? (d.direction + ' ZONE ARMED') : 'NO SETUP';

  const checksDiv = document.getElementById('checks');
  const entries = Object.entries(d.confirmations || {});
  checksDiv.innerHTML = entries.length ? entries.map(([k,v]) =>
    `<div class="check-row"><span>${k}</span><span class="${v?'ok':'no'}">${v?'✓ PASS':'✗ WAIT'}</span></div>`
  ).join('') : '<div class="check-row"><span style="color:var(--muted)">No setup zone active — standing down</span></div>';

  document.getElementById('confirmedBanner').innerHTML = d.confirmed
    ? '<div class="confirmed-banner">⚡ ALL CONFIRMED — ALERT WOULD FIRE NOW</div>' : '';
}

async function refreshAll() {
  await Promise.all([loadStatus(), loadHistory()]);
  secondsLeft = POLL_SECONDS;
}

function tickScanBar() {
  secondsLeft -= 1;
  if (secondsLeft <= 0) { refreshAll(); secondsLeft = POLL_SECONDS; }
  const pct = ((POLL_SECONDS - secondsLeft) / POLL_SECONDS) * 100;
  document.getElementById('scanFill').style.width = pct + '%';
  document.getElementById('scanLabel').innerText = 'next scan in ' + secondsLeft + 's';
}

refreshAll();
setInterval(tickScanBar, 1000);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, port=5050)
