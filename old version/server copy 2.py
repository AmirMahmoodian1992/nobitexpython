# server.py
# pip install fastapi uvicorn requests pandas
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import datetime as dt
from typing import List, Dict, Any, Tuple
import math

app = FastAPI()
UDF_URL = "https://apiv2.nobitex.ir/market/udf/history"

# -----------------------
# Low-level: data fetch
# -----------------------
def fetch_udf(symbol: str, resolution: str, start_ts: int, end_ts: int, timeout: int = 15) -> List[Dict[str, Any]]:
    """Fetch UDF-style history and return list of OHLC dicts with unix-second time."""
    params = {"symbol": symbol, "resolution": resolution, "from": start_ts, "to": end_ts}
    r = requests.get(UDF_URL, params=params, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    if j.get("s") != "ok":
        raise RuntimeError(f"UDF error: {j}")
    out = []
    t = j.get("t", [])
    o = j.get("o", [])
    h = j.get("h", [])
    l = j.get("l", [])
    c = j.get("c", [])
    v = j.get("v", [])
    n = len(t)
    for i in range(n):
        out.append({
            "time": int(t[i]),
            "open": float(o[i]) if i < len(o) else float(c[i]),
            "high": float(h[i]) if i < len(h) else float(c[i]),
            "low":  float(l[i]) if i < len(l) else float(c[i]),
            "close":float(c[i]),
            "volume": float(v[i]) if i < len(v) else 0.0
        })
    return out

# -----------------------
# Helpers: math & stats
# -----------------------
def compute_max_drawdown(equity_series: List[float]) -> float:
    """Return max drawdown as positive fraction (e.g. 0.2 -> 20% drawdown)."""
    peak = -float('inf')
    max_dd = 0.0
    for v in equity_series:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd

# -----------------------
# Indicator: accurate seeding
# -----------------------
def compute_ema_from_series(closes: List[float], period: int) -> float:
    """
    Compute EMA by iterating the full series (recommended).
    Uses SMA of first `period` values as initial seed, then recursive EMA.
    Returns last EMA value (float).
    """
    if not closes:
        return 0.0
    if period <= 0:
        raise ValueError("period must be > 0")
    n = len(closes)
    k = 2.0 / (period + 1.0)
    if n < period:
        # not enough history: seed with first price and run EMA across available values
        ema = closes[0]
        for i in range(1, n):
            ema = closes[i] * k + ema * (1 - k)
        return ema
    # seed with SMA of first `period`
    seed = sum(closes[:period]) / period
    ema = seed
    for i in range(period, n):
        ema = closes[i] * k + ema * (1 - k)
    return ema

def compute_wilder_avgs_from_series(closes: List[float], period: int) -> Tuple[float, float]:
    """
    Compute Wilder RSI avgUp and avgDown by iterating over the full historical closes.
    Returns (avgUp, avgDown) as final values.
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    n = len(closes)
    if n < 2:
        return (1e-8, 1e-8)
    # if insufficient length for full initialization, fallback to small nonzero
    if n <= period:
        up = down = 0.0
        for i in range(1, n):
            d = closes[i] - closes[i-1]
            if d > 0:
                up += d
            else:
                down += -d
        # average over available intervals (avoid zero)
        denom = max(1, n-1)
        return (up/denom if up>0 else 1e-8, down/denom if down>0 else 1e-8)
    # proper Wilder initialization: first avg is simple avg of gains/losses of first `period` intervals
    up = down = 0.0
    for i in range(1, period+1):
        d = closes[i] - closes[i-1]
        if d > 0: up += d
        else: down += -d
    avg_up = up / period
    avg_down = down / period
    # iterate remaining values with Wilder smoothing
    for i in range(period+1, n):
        d = closes[i] - closes[i-1]
        gain = d if d > 0 else 0.0
        loss = -d if d < 0 else 0.0
        avg_up = (avg_up * (period - 1) + gain) / period
        avg_down = (avg_down * (period - 1) + loss) / period
    return (avg_up if avg_up>0 else 1e-8, avg_down if avg_down>0 else 1e-8)

# -----------------------
# PnL helper
# -----------------------
def pnl_for_close(position: Dict[str, Any], exit_price: float, fee_factor: float) -> Tuple[float, float]:
    """
    Compute realized PnL for closing a position and exit fee.
    position holds 'side','units','entry_price','entry_fee'.
    Returns (pnl, exit_fee).
    """
    units = position["units"]
    entry_price = position["entry_price"]
    entry_fee = position.get("entry_fee", 0.0)
    exit_fee = abs(units * exit_price) * fee_factor
    if position["side"] == "long":
        pnl = units * (exit_price - entry_price) - (entry_fee + exit_fee)
    else:
        pnl = (entry_price * units) - (exit_price * units) - (entry_fee + exit_fee)
    return float(pnl), float(exit_fee)

# -----------------------
# Candidate detector (your requested semantics)
# -----------------------
def detect_crossover_by_minute_candidates(
    ema_fast_seed: float,
    ema_slow_seed: float,
    avgUp_seed: float,
    avgDown_seed: float,
    last_hour_close: float,
    minute_bars: List[Dict[str, Any]],
    fast_period_hours: int,
    slow_period_hours: int,
    rsi_period_hours: int,
    rsi_overbought: float,
    rsi_oversold: float
) -> Dict[str, Any]:
    """
    For each minute bar, treat minute.close as a hypothetical hourly close and compute
    single-step EMA & Wilder-RSI candidate. Return earliest candidate dict with:
      { signal, time, price, ema_fast_candidate, ema_slow_candidate, rsi_candidate, minute_bar }
    or None if no candidate inside this hour.
    """
    # standard per-bar k
    k_fast_bar = 2.0 / (fast_period_hours + 1.0)
    k_slow_bar = 2.0 / (slow_period_hours + 1.0)
    prev_diff = ema_fast_seed - ema_slow_seed
    P = rsi_period_hours
    if P <= 0:
        raise ValueError("rsi_period_hours must be > 0")

    for mb in minute_bars:
        price = float(mb["close"])
        tstamp = int(mb["time"])

        ema_fast_cand = k_fast_bar * price + (1.0 - k_fast_bar) * ema_fast_seed
        ema_slow_cand = k_slow_bar * price + (1.0 - k_slow_bar) * ema_slow_seed
        cand_diff = ema_fast_cand - ema_slow_cand

        # Wilder RSI one-step candidate (use delta vs last_hour_close)
        d = price - last_hour_close
        gain = d if d > 0 else 0.0
        loss = -d if d < 0 else 0.0
        avgUp_cand = (avgUp_seed * (P - 1) + gain) / P
        avgDown_cand = (avgDown_seed * (P - 1) + loss) / P
        rsi_cand = 100.0 - 100.0 / (1.0 + (avgUp_cand / (avgDown_cand or 1e-12)))

        crossed_up = (prev_diff <= 0.0 and cand_diff > 0.0)
        crossed_down = (prev_diff >= 0.0 and cand_diff < 0.0)

        if crossed_up and rsi_cand > rsi_oversold:
            return {
                "signal": "long",
                "time": tstamp,
                "price": price,
                "ema_fast_candidate": ema_fast_cand,
                "ema_slow_candidate": ema_slow_cand,
                "rsi_candidate": rsi_cand,
                "minute_bar": mb
            }
        if crossed_down and rsi_cand < rsi_overbought:
            return {
                "signal": "short",
                "time": tstamp,
                "price": price,
                "ema_fast_candidate": ema_fast_cand,
                "ema_slow_candidate": ema_slow_cand,
                "rsi_candidate": rsi_cand,
                "minute_bar": mb
            }

    return None

# -----------------------
# Walk-forward intraminute simulator (full multi-hour)
# -----------------------
def run_walkforward_intraminute(
    hourly_candles: List[Dict[str, Any]],
    minute_candles_all: List[Dict[str, Any]],
    lookback_hours: int = 12,
    fast_period_hours: int = 9,
    slow_period_hours: int = 21,
    rsi_period_hours: int = 14,
    rsi_overbought: float = 65.0,
    rsi_oversold: float = 35.0,
    initial_capital: float = 1000.0,
    fee_pct: float = 0.1,
    position_size_pct: float = 1.0
) -> Dict[str, Any]:
    """
    Walk forward across the last `lookback_hours` hourly candles.
    minute_candles_all must cover the same span; we group minutes by hour.
    Returns aggregated trades, markers, minute-equity curve and stats.
    """
    # group minute candles by hour start timestamp
    minute_by_hour: Dict[int, List[Dict[str,Any]]] = {}
    for mb in minute_candles_all:
        t = int(mb["time"])
        hour_start = t - (t % 3600)
        minute_by_hour.setdefault(hour_start, []).append(mb)
    for k in minute_by_hour:
        minute_by_hour[k].sort(key=lambda x: int(x["time"]))

    n_hours = len(hourly_candles)
    if n_hours == 0:
        return {"error": "no hourly candles"}

    # ensure we have enough history for seeding; choose start_idx such that we can seed EMAs reasonably
    min_seed = max(fast_period_hours, slow_period_hours, rsi_period_hours)
    # choose index window: last `lookback_hours` hours
    start_idx = max(min_seed, max(0, n_hours - lookback_hours))
    end_idx = n_hours  # exclusive

    trades: List[Dict[str,Any]] = []
    markers: List[Dict[str,Any]] = []
    equity_curve: List[Dict[str,Any]] = []

    fee_factor = fee_pct / 100.0
    capital = float(initial_capital)
    position = None  # current open position dict or None

    hourly_closes = [float(h["close"]) for h in hourly_candles]

    for i in range(start_idx, end_idx):
        hour = hourly_candles[i]
        hour_start = int(hour["time"])
        # seed from earlier closed hours (all hours index < i)
        prior_hourly_closes = hourly_closes[:i]
        # compute EMA and Wilder avg seeds by iterating whole prior series (accurate)
        ema_fast_seed = compute_ema_from_series(prior_hourly_closes, fast_period_hours)
        ema_slow_seed = compute_ema_from_series(prior_hourly_closes, slow_period_hours)
        avgUp_seed, avgDown_seed = compute_wilder_avgs_from_series(prior_hourly_closes, rsi_period_hours)
        last_hour_close = prior_hourly_closes[-1] if prior_hourly_closes else hourly_closes[max(0, i-1)]

        minute_bars = minute_by_hour.get(hour_start, [])
        minute_bars.sort(key=lambda x: int(x["time"]))

        candidate = detect_crossover_by_minute_candidates(
            ema_fast_seed=ema_fast_seed,
            ema_slow_seed=ema_slow_seed,
            avgUp_seed=avgUp_seed,
            avgDown_seed=avgDown_seed,
            last_hour_close=last_hour_close,
            minute_bars=minute_bars,
            fast_period_hours=fast_period_hours,
            slow_period_hours=slow_period_hours,
            rsi_period_hours=rsi_period_hours,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold
        )

        # iterate minutes and build equity; execute at candidate minute if present
        if minute_bars:
            for mb in minute_bars:
                p = float(mb["close"])
                t = int(mb["time"])

                if candidate is not None and t == int(candidate["time"]):
                    signal = candidate["signal"]
                    exec_price = candidate["price"]
                    exec_time = candidate["time"]
                    position_notional = capital * float(position_size_pct)
                    units = position_notional / exec_price if exec_price > 0 else 0.0
                    entry_fee = units * exec_price * fee_factor

                    if position is None:
                        position = {"side": signal, "entry_time": exec_time, "entry_price": exec_price, "units": units, "entry_fee": entry_fee}
                        markers.append({"time": exec_time, "position":"belowBar", "color":"green" if signal=="long" else "red", "shape":"circle", "text":"L" if signal=="long" else "S"})
                    elif position["side"] != signal:
                        pnl, exit_fee = pnl_for_close(position, exec_price, fee_factor)
                        trade = {
                            "side": "Long" if position["side"]=="long" else "Short",
                            "entry_time": position["entry_time"], "entry_price": position["entry_price"],
                            "exit_time": exec_time, "exit_price": exec_price,
                            "units": position["units"], "entry_fee": position["entry_fee"],
                            "exit_fee": exit_fee, "pnl": pnl
                        }
                        trades.append(trade)
                        markers.append({"time": exec_time, "position":"aboveBar", "color":"green" if position["side"]=="long" else "red", "shape":"circle", "text":"L" if position["side"]=="long" else "S"})
                        capital += pnl
                        # open opposite
                        position = {"side": signal, "entry_time": exec_time, "entry_price": exec_price, "units": units, "entry_fee": entry_fee}
                        markers.append({"time": exec_time, "position":"belowBar", "color":"green" if signal=="long" else "red", "shape":"circle", "text":"L" if signal=="long" else "S"})
                    else:
                        # same-side -> ignore
                        pass

                # mark-to-market
                if position is None:
                    equity = capital
                else:
                    if position["side"] == "long":
                        unreal = position["units"] * (p - position["entry_price"]) - position.get("entry_fee", 0.0)
                    else:
                        unreal = (position["entry_price"] - p) * position["units"] - position.get("entry_fee", 0.0)
                    equity = capital + unreal
                equity_curve.append({"time": t, "equity": equity})
        else:
            # no minute data -> use hourly close as fallback equity point
            p = float(hour["close"])
            t = int(hour["time"]) + 3599
            if position is None:
                equity = capital
            else:
                if position["side"] == "long":
                    unreal = position["units"] * (p - position["entry_price"]) - position.get("entry_fee", 0.0)
                else:
                    unreal = (position["entry_price"] - p) * position["units"] - position.get("entry_fee", 0.0)
                equity = capital + unreal
            equity_curve.append({"time": t, "equity": equity})

    # stats
    net_profit = (equity_curve[-1]["equity"] - float(initial_capital)) if equity_curve else 0.0
    stats = {
        "initial_capital": float(initial_capital),
        "final_equity": equity_curve[-1]["equity"] if equity_curve else float(initial_capital),
        "net_profit": net_profit,
        "trades": len(trades),
        "max_drawdown_pct": compute_max_drawdown([e["equity"] for e in equity_curve]) * 100.0 if equity_curve else 0.0
    }
    return {"trades": trades, "markers": markers, "equity": equity_curve, "stats": stats}

# -----------------------
# Endpoints
# -----------------------
@app.get("/candles")
def candles(symbol: str = Query("BTCUSDT"), resolution: str = Query("60"), days: int = Query(7)):
    now = int(dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).timestamp())
    start = now - int(days) * 86400
    try:
        data = fetch_udf(symbol, resolution, start, now)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return JSONResponse(content={"candles": data})

@app.get("/intraminute_backtest")
def intraminute_backtest(
    symbol: str = Query("BTCUSDT"),
    days: int = Query(7),
    fast: int = Query(9),
    slow: int = Query(21),
    rsi_len: int = Query(14),
    rsi_ob: float = Query(65.0),
    rsi_os: float = Query(35.0),
    initial_capital: float = Query(1000.0),
    fee_pct: float = Query(0.1),
    position_size_pct: float = Query(1.0)
) -> JSONResponse:
    """
    Run strategy for *current* hour: seed with hourly history then test minute closes inside current hour.
    Returns hourly history + minute_current_hour + strategy result.
    """
    try:
        now = int(dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).timestamp())
        start = now - int(days) * 86400
        hourly = fetch_udf(symbol, "60", start, now)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"hourly fetch failed: {e}")
    if not hourly:
        raise HTTPException(status_code=400, detail="no hourly candles returned")

    # current hour start
    now_dt = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0, tzinfo=dt.timezone.utc)
    hour_start_ts = int(now_dt.timestamp())
    try:
        minute_candles = fetch_udf(symbol, "1", hour_start_ts, hour_start_ts + 3600)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"minute fetch failed: {e}")

    result = run_walkforward_intraminute(
        hourly_candles=hourly,
        minute_candles_all=minute_candles,
        lookback_hours=1,
        fast_period_hours=int(fast),
        slow_period_hours=int(slow),
        rsi_period_hours=int(rsi_len),
        rsi_overbought=float(rsi_ob),
        rsi_oversold=float(rsi_os),
        initial_capital=float(initial_capital),
        fee_pct=float(fee_pct),
        position_size_pct=float(position_size_pct)
    )

    return JSONResponse(content={"hourly": hourly, "minute_current_hour": minute_candles, **result})

@app.get("/walkforward")
def walkforward(
    symbol: str = Query("BTCUSDT"),
    hours: int = Query(12),
    fast: int = Query(9),
    slow: int = Query(21),
    rsi_len: int = Query(14),
    rsi_ob: float = Query(65.0),
    rsi_os: float = Query(35.0),
    initial_capital: float = Query(1000.0),
    fee_pct: float = Query(0.1),
    position_size_pct: float = Query(1.0),
    days_back_for_history: int = Query(2)
) -> JSONResponse:
    """
    Walk-forward intraminute backtest for the last `hours` hours.
    Fetches hourly history for seeding and minute candles covering the entire hour window.
    """
    now_ts = int(dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).timestamp())
    start_ts = now_ts - int(days_back_for_history) * 86400
    try:
        hourly = fetch_udf(symbol, "60", start_ts, now_ts)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"hourly fetch failed: {e}")
    if not hourly:
        raise HTTPException(status_code=400, detail="no hourly candles returned")

    latest_hour_start = hourly[-1]["time"] - (hourly[-1]["time"] % 3600)
    earliest_hour_start = latest_hour_start - (hours - 1) * 3600
    try:
        minute_all = fetch_udf(symbol, "1", earliest_hour_start, latest_hour_start + 3600)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"minute fetch failed: {e}")

    result = run_walkforward_intraminute(
        hourly_candles=hourly,
        minute_candles_all=minute_all,
        lookback_hours=hours,
        fast_period_hours=int(fast),
        slow_period_hours=int(slow),
        rsi_period_hours=int(rsi_len),
        rsi_overbought=float(rsi_ob),
        rsi_oversold=float(rsi_os),
        initial_capital=float(initial_capital),
        fee_pct=float(fee_pct),
        position_size_pct=float(position_size_pct)
    )

    return JSONResponse(content={
        "hourly_window_start": earliest_hour_start,
        "hourly_window_end": latest_hour_start + 3600,
        **result
    })

# Simple simulation endpoint (place_order)
@app.post("/place_order")
async def place_order(req: Request):
    payload = await req.json()
    print("PLACE_ORDER received:", payload)
    return JSONResponse(content={"ok": True, "accepted": True, "payload": payload})

# Legacy /backtest preserved (next-open semantics)
#@app.get("/backtest_legacy")
#def backtest_legacy(...):  # keep your old implementation if you want; omitted here for brevity
 #    raise HTTPException(status_code=410, detail="Use /walkforward or /intraminute_backtest")


@app.get("/debug_intraminute")
def debug_intraminute(
    symbol: str = Query("BTCUSDT"),
    days: int = Query(2),
    fast: int = Query(9),
    slow: int = Query(21),
    rsi_len: int = Query(14),
    rsi_ob: float = Query(65.0),
    rsi_os: float = Query(35.0),
):
    """
    Debug endpoint: For each hour in the fetched hourly window, return:
      - seed values used (ema_fast_seed, ema_slow_seed, avgUp_seed, avgDown_seed)
      - per-minute diagnostics: price, ema_fast_cand, ema_slow_cand, cand_diff, prev_diff,
        avgUp_cand, avgDown_cand, rsi_cand, crossed_up, crossed_down
      - the earliest candidate found (if any)
    """
    now_ts = int(dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).timestamp())
    start_ts = now_ts - int(days) * 86400
    try:
        hourly = fetch_udf(symbol, "60", start_ts, now_ts)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"hourly fetch failed: {e}")
    if not hourly:
        raise HTTPException(status_code=400, detail="no hourly candles returned")

    # fetch minute candles covering the same span (so we can inspect inside each hour)
    latest_hour_start = hourly[-1]["time"] - (hourly[-1]["time"] % 3600)
    earliest_hour_start = hourly[0]["time"] - (hourly[0]["time"] % 3600)
    try:
        minute_all = fetch_udf(symbol, "1", earliest_hour_start, latest_hour_start + 3600)
    except Exception as e:
        minute_all = []

    # group minute candles by hour
    minute_by_hour = {}
    for mb in minute_all:
        t = int(mb["time"])
        hour_start = t - (t % 3600)
        minute_by_hour.setdefault(hour_start, []).append(mb)
    for k in minute_by_hour:
        minute_by_hour[k].sort(key=lambda x: int(x["time"]))

    hourly_closes = [float(h["close"]) for h in hourly]
    results = []
    P = int(rsi_len)
    k_fast_bar = 2.0 / (int(fast) + 1.0)
    k_slow_bar = 2.0 / (int(slow) + 1.0)

    # We will iterate hours and re-create the exact per-minute candidate math
    for i, hour in enumerate(hourly):
        hour_start = int(hour["time"])
        prior_hourly_closes = hourly_closes[:i]
        # seed calculations (reuse your existing helpers)
        ema_fast_seed = compute_ema_from_series(prior_hourly_closes, int(fast))
        ema_slow_seed = compute_ema_from_series(prior_hourly_closes, int(slow))
        avgUp_seed, avgDown_seed = compute_wilder_avgs_from_series(prior_hourly_closes, int(rsi_len))
        last_hour_close = prior_hourly_closes[-1] if prior_hourly_closes else hourly_closes[max(0, i-1)]

        prev_diff = ema_fast_seed - ema_slow_seed

        minutes = minute_by_hour.get(hour_start, [])
        minute_diags = []
        earliest_candidate = None

        for mb in minutes:
            price = float(mb["close"])
            tstamp = int(mb["time"])

            ema_fast_cand = k_fast_bar * price + (1.0 - k_fast_bar) * ema_fast_seed
            ema_slow_cand = k_slow_bar * price + (1.0 - k_slow_bar) * ema_slow_seed
            cand_diff = ema_fast_cand - ema_slow_cand

            d = price - float(last_hour_close)
            gain = d if d > 0 else 0.0
            loss = -d if d < 0 else 0.0
            avgUp_cand = (avgUp_seed * (P - 1) + gain) / P
            avgDown_cand = (avgDown_seed * (P - 1) + loss) / P
            rsi_cand = 100.0 - 100.0 / (1.0 + (avgUp_cand / (avgDown_cand or 1e-12)))

            crossed_up = (prev_diff <= 0.0 and cand_diff > 0.0)
            crossed_down = (prev_diff >= 0.0 and cand_diff < 0.0)

            minute_diags.append({
                "time": tstamp,
                "price": price,
                "ema_fast_cand": ema_fast_cand,
                "ema_slow_cand": ema_slow_cand,
                "cand_diff": cand_diff,
                "prev_diff": prev_diff,
                "avgUp_cand": avgUp_cand,
                "avgDown_cand": avgDown_cand,
                "rsi_cand": rsi_cand,
                "crossed_up": crossed_up,
                "crossed_down": crossed_down
            })

            if earliest_candidate is None:
                if crossed_up and rsi_cand > float(rsi_os):
                    earliest_candidate = {"signal": "long", "time": tstamp, "price": price, "rsi": rsi_cand}
                if crossed_down and rsi_cand < float(rsi_ob):
                    earliest_candidate = {"signal": "short", "time": tstamp, "price": price, "rsi": rsi_cand}
            # Note: we intentionally do NOT mutate prev_diff per minute here because the original detector uses prev_diff (seed)
            # If you want a running prev_diff for stepwise detection, change accordingly.

        results.append({
            "hour_start": hour_start,
            "ema_fast_seed": ema_fast_seed,
            "ema_slow_seed": ema_slow_seed,
            "avgUp_seed": avgUp_seed,
            "avgDown_seed": avgDown_seed,
            "last_hour_close": last_hour_close,
            "minute_count": len(minutes),
            "minutes": minute_diags,
            "candidate": earliest_candidate
        })

    return JSONResponse(content={"symbol": symbol, "hours": len(hourly), "debug": results})

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
