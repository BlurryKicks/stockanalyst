from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import yaml
import os
from datetime import datetime, timezone
import json  # for parsing stringified JSON bodies

app = Flask(__name__)

# ---------- helpers to accept stringified JSON from n8n ----------
def _normalize_ohlcv(payload: dict):
    """Accept ohlcv as list OR as a JSON string; normalize to list."""
    ohlcv = payload.get("ohlcv", [])
    if isinstance(ohlcv, str):
        try:
            ohlcv = json.loads(ohlcv)
        except Exception:
            # leave as-is if parsing fails
            pass
    payload["ohlcv"] = ohlcv
    return payload

def _maybe_parse_dict(value):
    """If value is a JSON string, parse to dict; otherwise return as-is."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value

# -------- Safe config loads (works even if YAMLs are missing) --------
DEFAULT_CFG = {
    "weights": {"posterior": 0.35, "regime_quality": 0.25, "confluence": 0.25, "sentiment_align": 0.15},
    "penalties": {"news_proximity": 15, "spread_widening": 10, "liquidity_thin": 10, "mdl_buffer_low": 10, "signal_conflict": 10},
    "regime": {"adx_like": 20, "ema_slope_min": 0.0, "atr_low": 0.006, "atr_high": 0.014},
}
try:
    with open("model_config.yaml", "r") as f:
        CFG = yaml.safe_load(f) or DEFAULT_CFG
except FileNotFoundError:
    CFG = DEFAULT_CFG

DEFAULT_PAIRS = {
    "pairs": ["EURUSD", "GBPUSD", "USDJPY", "USDCAD"],
    "timeframes": {"bias": ["1d", "4h", "1h"], "entry": ["15m", "5m"]},
    "session_et": {"asia": ["20:00", "03:00"], "london": ["03:00", "07:00"], "ny": ["08:00", "12:00"]},
}
try:
    with open("pairs.yaml", "r") as f:
        PAIRS = yaml.safe_load(f) or DEFAULT_PAIRS
except FileNotFoundError:
    PAIRS = DEFAULT_PAIRS

# ---------------------- TA helpers ----------------------
def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def vwap(df):
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_vp = (tp * vol).cumsum()
    cum_vol = vol.cumsum()
    return cum_vp / cum_vol

def atr_percent(df, n=14):
    a = atr(df, n)
    return (a / df["close"]).fillna(0.0)

def ema_slope(close, n=50):
    e = ema(close, n)
    # smoother slope: average of last 5 bar change
    return (e.iloc[-1] - e.iloc[-5]) / 5.0

def to_df(ohlcv):
    df = pd.DataFrame(ohlcv)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").set_index("ts")
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            df[col] = np.nan
    if "volume" not in df.columns:
        df["volume"] = 1.0
    df = df.ffill().bfill()
    return df

# ------------------ Regime + scoring -------------------
def regime_tag(df):
    slope = ema_slope(df["close"], 50)
    v = vwap(df)
    price_side = "above" if df["close"].iloc[-1] >= v.iloc[-1] else "below"
    atrp = atr_percent(df, 14).iloc[-1]

    if atrp < CFG["regime"]["atr_low"]:
        vol_bucket = "Low"
    elif atrp > CFG["regime"]["atr_high"]:
        vol_bucket = "High"
    else:
        vol_bucket = "Normal"

    liq_state = "OK"  # placeholder; can be overridden by spread/depth

    if slope > CFG["regime"]["ema_slope_min"] and price_side == "above":
        reg = "Trend-Up"
    elif slope < -CFG["regime"]["ema_slope_min"] and price_side == "below":
        reg = "Trend-Down"
    else:
        reg = "Mean-Revert" if abs(slope) < 1e-5 else "Chop"

    return {
        "regime": reg,
        "vol_bucket": vol_bucket,
        "liq_state": liq_state,
        "ema50_slope": float(slope),
        "atrp": float(atrp),
        "vwap_side": price_side,
    }

def confluence_bits(high_tf_trend_agrees, vwap_side_agrees, context_agrees):
    parts = [high_tf_trend_agrees, vwap_side_agrees, context_agrees]
    score = sum(1 for p in parts if p)
    return score, len(parts)

def page_score(posterior, regime_quality, conf_ratio, sent_align,
               news=False, spread_wide=False, liq_thin=False,
               mdl_low=False, conflict=False):
    w = CFG["weights"]
    base = 100.0 * (
        w["posterior"] * posterior
        + w["regime_quality"] * regime_quality
        + w["confluence"] * conf_ratio
        + w["sentiment_align"] * sent_align
    )
    P = CFG["penalties"]
    pen = 0.0
    pen += P["news_proximity"] if news else 0.0
    pen += P["spread_widening"] if spread_wide else 0.0
    pen += P["liquidity_thin"] if liq_thin else 0.0
    pen += P["mdl_buffer_low"] if mdl_low else 0.0
    pen += P["signal_conflict"] if conflict else 0.0
    return max(0.0, min(100.0, base - pen))

# ----------------------- Routes ------------------------

@app.route("/")
def home():
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trading Analysis API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            h2 { color: #666; border-bottom: 2px solid #eee; padding-bottom: 10px; }
            .endpoint { background: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007cba; }
            .method { background: #007cba; color: white; padding: 3px 8px; border-radius: 3px; font-size: 12px; }
            .status { text-align: center; padding: 20px; background: #e8f5e8; border-radius: 5px; margin: 20px 0; }
            code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
            .footer { text-align: center; margin-top: 30px; color: #888; font-size: 14px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Trading Analysis API</h1>
            <div class="status">
                <strong>✅ Server Status: Online</strong><br>
                <small>Server Time: {{ current_time }}</small>
            </div>
            <h2>📚 Available Endpoints</h2>
            <div class="endpoint">
                <h3><span class="method">GET</span> /health</h3>
                <p>Health check endpoint that returns server status and current time.</p>
                <p><strong>Response:</strong> <code>{"ok": true, "time": "..."}</code></p>
            </div>
            <div class="endpoint">
                <h3><span class="method">POST</span> /features</h3>
                <p>Processes OHLCV trading data and calculates technical indicators (EMA, ATR, VWAP).</p>
                <p><strong>Input:</strong> JSON with pair, timeframe, OHLCV data, session, spread_state</p>
                <p><strong>Output:</strong> Technical analysis features</p>
            </div>
            <div class="endpoint">
                <h3><span class="method">POST</span> /regime</h3>
                <p>Performs regime analysis on market data to determine trend conditions.</p>
                <p><strong>Input:</strong> JSON with pair, OHLCV data, high_tf_bias</p>
                <p><strong>Output:</strong> Regime classification and quality metrics</p>
            </div>
            <div class="endpoint">
                <h3><span class="method">POST</span> /ensemble</h3>
                <p>Provides ensemble scoring for trading decisions based on multiple factors.</p>
                <p><strong>Input:</strong> JSON with features, regime, sentiment, market conditions</p>
                <p><strong>Output:</strong> Comprehensive trading score and reasoning</p>
            </div>
            <div class="footer">
                <p>Trading Analysis API • Running on Flask</p>
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(
        html_template,
        current_time=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )

@app.route("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "time": datetime.now(timezone.utc).isoformat(),
            "status": "healthy",
            "version": "1.0.0",
        }
    )

@app.route("/features", methods=["POST"])
def features():
    """
    Input JSON:
    {
      "pair": "EURUSD",
      "timeframe": "5m",
      "ohlcv": [{"ts":"...","open":...,"high":...,"low":...,"close":...,"volume":...}, ...],
      "session": "london",
      "spread_state": "OK"
    }
    """
    data = request.get_json(force=True)
    data = _normalize_ohlcv(data)  # normalize ohlcv (handles string or list)

    df = to_df(data["ohlcv"])
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["atr14"] = atr(df, 14)
    df["atrp14"] = df["atr14"] / df["close"]
    df["vwap"] = vwap(df)

    out = {
        "pair": data.get("pair", ""),
        "timeframe": data.get("timeframe", ""),
        "last_close": float(df["close"].iloc[-1]),
        "ema20": float(df["ema20"].iloc[-1]),
        "ema50": float(df["ema50"].iloc[-1]),
        "ema200": float(df["ema200"].iloc[-1]),
        "atrp14": float(df["atrp14"].iloc[-1]),
        "vwap": float(df["vwap"].iloc[-1]),
        "spread_state": data.get("spread_state", "Unknown"),
        "session": data.get("session", "unknown"),
    }
    return jsonify(out)

@app.route("/regime", methods=["POST"])
def regime():
    """
    Input JSON:
    {
      "pair": "...",
      "ohlcv": [...],  # may be stringified
      "high_tf_bias": "up"|"down"|"neutral"
    }
    """
    data = request.get_json(force=True)
    data = _normalize_ohlcv(data)

    df = to_df(data["ohlcv"])
    r = regime_tag(df)

    rq_map = {"Trend-Up": 1.0, "Trend-Down": 1.0, "Mean-Revert": 0.6, "Chop": 0.4}
    r["regime_quality"] = rq_map.get(r["regime"], 0.5)
    r["pair"] = data.get("pair", "")
    r["high_tf_bias"] = data.get("high_tf_bias", "neutral")
    return jsonify(r)

@app.route("/ensemble", methods=["POST"])
def ensemble():
    """
    Input JSON:
    {
      "pair": "USDJPY",
      "features": {...},  # may be stringified
      "regime": {...},    # may be stringified
      "sentiment_skew": -1..+1,
      "news_lockout": false,
      "spread_wide": false,
      "liq_thin": false,
      "mdl_buffer_low": false,
      "context_direction": "up"|"down"|"flat",
      "intended_direction": "long"|"short"
    }
    """
    j = request.get_json(force=True)
    # allow features/regime to arrive as strings
    j["features"] = _maybe_parse_dict(j.get("features", {}))
    j["regime"] = _maybe_parse_dict(j.get("regime", {}))

    feats = j.get("features", {})
    reg = j.get("regime", {})

    price = feats.get("last_close", 0.0)
    ema50 = feats.get("ema50", price)
    vwap_last = feats.get("vwap", price)
    intended = j.get("intended_direction", "long")

    posterior = 0.55
    if intended == "long":
        if price >= ema50:
            posterior += 0.15
        if price >= vwap_last:
            posterior += 0.15
    else:
        if price <= ema50:
            posterior += 0.15
        if price <= vwap_last:
            posterior += 0.15
    posterior = min(0.95, max(0.05, posterior))

    high_tf_bias = reg.get("high_tf_bias", "neutral")
    regime_name = reg.get("regime", "Chop")
    vwap_side = reg.get("vwap_side", "above")
    context_dir = j.get("context_direction", "flat")

    if intended == "long":
        high_tf_ok = (high_tf_bias == "up") or ("Trend-Up" in regime_name)
        vwap_ok = (vwap_side == "above")
        context_ok = (context_dir in ["up", "flat"])
    else:
        high_tf_ok = (high_tf_bias == "down") or ("Trend-Down" in regime_name)
        vwap_ok = (vwap_side == "below")
        context_ok = (context_dir in ["down", "flat"])

    conf_count = (1 if high_tf_ok else 0) + (1 if vwap_ok else 0) + (1 if context_ok else 0)
    conf_total = 3
    conf_ratio = conf_count / conf_total

    skew = float(j.get("sentiment_skew", 0.0))
    sent_align = abs(skew) if ((skew >= 0 and intended == "long") or (skew <= 0 and intended == "short")) else 0.0
    sent_align = min(1.0, max(0.0, sent_align))

    regime_quality = float(reg.get("regime_quality", 0.5))
    conflict = (("Trend-Up" in regime_name and intended == "short") or
                ("Trend-Down" in regime_name and intended == "long"))

    score = page_score(
        posterior=posterior,
        regime_quality=regime_quality,
        conf_ratio=conf_ratio,
        sent_align=sent_align,
        news=j.get("news_lockout", False),
        spread_wide=j.get("spread_wide", False),
        liq_thin=j.get("liq_thin", False),
        mdl_low=j.get("mdl_buffer_low", False),
        conflict=conflict,
    )

    reasons = []
    if high_tf_ok:
        reasons.append("High-TF bias aligned")
    if vwap_ok:
        reasons.append("VWAP side aligned")
    if context_ok:
        reasons.append("Context aligned")
    if j.get("news_lockout", False):
        reasons.append("News lockout penalty")
    if j.get("spread_wide", False):
        reasons.append("Spread penalty")
    if conflict:
        reasons.append("Regime conflict penalty")

    out = {
        "pair": j.get("pair", ""),
        "direction": intended,
        "posterior": round(posterior, 3),
        "regime_quality": round(regime_quality, 3),
        "confluence": f"{conf_count}/{conf_total}",
        "sentiment_align": round(sent_align, 3),
        "page_score": round(score, 1),
        "reasons": reasons,
    }
    return jsonify(out)

# ----------------------- Errors ------------------------
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found", "message": "Check the API documentation at /"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error", "message": "Something went wrong"}), 500

# ----------------------- Entrypoint ---------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting Flask server on port {port}")
    print(f"📡 Server will be accessible at http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
