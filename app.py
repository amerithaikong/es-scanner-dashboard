from flask import Flask, jsonify, render_template_string
import numpy as np
import pandas as pd
import yfinance as yf

app = Flask(__name__)

CONFIG = {
    "symbol": "ES=F",
    "htf_interval": "1h",
    "htf_period": "30d",
    "htf_lookback": 40,
    "min_r2": 0.55,
    "min_slope_pts": 0.35,
    "pullback_z_min": 1.0,
    "pullback_z_max": 2.5,
    "ltf_interval": "5m",
    "ltf_period": "5d",
    "rsi_len": 14,
    "rsi_reset_level": 40,
    "rsi_trigger_level": 45,
    "rsi_reset_window": 8,
    "ema_len": 20,
    "vol_mult": 1.1,
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
        confirmations["Volume >= 1.1x avg"] = bool(v_ok)
        confirmed = all(confirmations.values())

    return {
        "price": round(last_close, 2),
        "slope": round(slope, 3),
        "r2": round(r2, 3),
        "z": round(z, 2),
        "direction": direction,
        "confirmations": confirmations,
        "confirmed": confirmed,
    }

@app.route("/api/status")
def api_status():
    try:
        return jsonify(get_status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return render_template_string(HTML)

HTML = """
<!doctype html>
<html>
<head>
<title>ES Regime Scanner</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body { background:#0b0e14; color:#e6e6e6; font-family: -apple-system, sans-serif; padding:24px; }
h1 { font-size:20px; color:#8ab4f8; }
.card { background:#151a23; border-radius:12px; padding:20px; margin-top:16px; }
.row { display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #222; }
.label { color:#9aa4b2; }
.badge { padding:4px 10px; border-radius:6px; font-weight:600; }
.long { background:#1e3a2f; color:#5fd98a; }
.short { background:#3a1e1e; color:#d95f5f; }
.none { background:#22262e; color:#9aa4b2; }
.confirmed { background:#3a331e; color:#f2c94c; }
.check { color:#5fd98a; }
.x { color:#d95f5f; }
#updated { color:#5a6472; font-size:12px; margin-top:12px; }
</style>
</head>
<body>
<h1>ES Linear Regression Scanner — Live</h1>
<div class="card" id="card">Loading...</div>
<div id="updated"></div>
<script>
async function refresh() {
  const res = await fetch('/api/status');
  const d = await res.json();
  if (d.error) {
    document.getElementById('card').innerHTML = '<div class="row">Error: ' + d.error + '</div>';
    return;
  }
  const dirClass = d.direction === 'LONG' ? 'long' : d.direction === 'SHORT' ? 'short' : 'none';
  let confHtml = '';
  for (const [k, v] of Object.entries(d.confirmations)) {
    confHtml += `<div class="row"><span class="label">${k}</span><span class="${v ? 'check' : 'x'}">${v ? '✓' : '✗'}</span></div>`;
  }
  document.getElementById('card').innerHTML = `
    <div class="row"><span class="label">Price</span><span>${d.price}</span></div>
    <div class="row"><span class="label">Regime</span><span class="badge ${dirClass}">${d.direction || 'NO SETUP'}</span></div>
    <div class="row"><span class="label">Slope (pts/bar)</span><span>${d.slope}</span></div>
    <div class="row"><span class="label">R²</span><span>${d.r2}</span></div>
    <div class="row"><span class="label">Pullback z-score</span><span>${d.z}</span></div>
    ${confHtml}
    ${d.confirmed ? '<div class="row"><span class="badge confirmed">ALL CONFIRMED — ALERT WOULD FIRE</span></div>' : ''}
  `;
  document.getElementById('updated').innerText = 'Updated ' + new Date().toLocaleTimeString();
}
refresh();
setInterval(refresh, 60000);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, port=5000)
