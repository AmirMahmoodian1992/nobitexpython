# ema_debug_compare.py
import requests, math
import datetime as dt
from typing import List, Dict, Any, Optional
import pandas as pd

UDF_URL = "https://apiv2.nobitex.ir/market/udf/history"

# --------------------------
# Helpers (same as before)
# --------------------------
def to_iso(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).isoformat()

def detect_and_normalize_times(raw_candles: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    if not raw_candles:
        return []
    # detect ms vs s
    sample = raw_candles[0].get("time", raw_candles[0].get("t", None))
    ms = False
    try:
        if int(sample) > 1_000_000_000_000:
            ms = True
    except Exception:
        ms = False
    out = []
    for c in raw_candles:
        t = int(c.get("time", c.get("t", 0)))
        if ms:
            t = t // 1000
        out.append({
            "time": int(t),
            "open": float(c.get("open", c.get("o", 0.0))),
            "high": float(c.get("high", c.get("h", 0.0))),
            "low": float(c.get("low", c.get("l", 0.0))),
            "close": float(c.get("close", c.get("c", 0.0))),
            "volume": float(c.get("volume", c.get("v", 0.0)))
        })
    return out

def compute_ema_series_local(closes: List[float], period: int) -> List[Optional[float]]:
    """SMA seed at period then recursive ema (same as in your script)."""
    if period <= 0:
        raise ValueError("period>0")
    n = len(closes)
    if n == 0:
        return []
    k = 2.0 / (period + 1.0)
    emas = [None] * n
    if n < period:
        ema = closes[0]
        emas[0] = ema
        for i in range(1, n):
            ema = closes[i]*k + ema*(1-k)
            emas[i] = ema
        return emas
    seed = sum(closes[:period]) / period
    for i in range(period-1):
        emas[i] = closes[i]
    emas[period-1] = seed
    ema = seed
    for i in range(period, n):
        ema = closes[i]*k + ema*(1-k)
        emas[i] = ema
    return emas

def detect_crosses_from_series(times: List[int], fast_ema: List[Optional[float]], slow_ema: List[Optional[float]]) -> List[Dict[str,Any]]:
    crosses = []
    prev_diff = None
    for i in range(len(times)):
        f = fast_ema[i] if i < len(fast_ema) else None
        s = slow_ema[i] if i < len(slow_ema) else None
        if f is None or s is None:
            prev_diff = None
            continue
        diff = f - s
        if prev_diff is not None:
            if prev_diff <= 0 and diff > 0:
                crosses.append({"index": i, "time": times[i], "type": "cross_up", "fast": f, "slow": s})
            elif prev_diff >= 0 and diff < 0:
                crosses.append({"index": i, "time": times[i], "type": "cross_down", "fast": f, "slow": s})
        prev_diff = diff
    return crosses

# --------------------------
# Main compare routine
# --------------------------
def fetch_candles(symbol="BTCUSDT", resolution="60", days_back=30, udf_url=UDF_URL, timeout=20):
    now = int(dt.datetime.now(dt.timezone.utc).timestamp())
    start = now - int(days_back)*86400
    params = {"symbol": symbol, "resolution": resolution, "from": start, "to": now}
    r = requests.get(udf_url, params=params, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    if j.get("s") != "ok":
        raise RuntimeError(f"UDF returned not ok: {j}")
    t = j.get("t", [])
    o = j.get("o", [])
    h = j.get("h", [])
    l = j.get("l", [])
    c = j.get("c", [])
    v = j.get("v", [])
    raw = []
    for i in range(len(t)):
        raw.append({
            "time": int(t[i]),
            "open": float(o[i]) if i < len(o) else float(c[i]),
            "high": float(h[i]) if i < len(h) else float(c[i]),
            "low": float(l[i]) if i < len(l) else float(c[i]),
            "close": float(c[i]),
            "volume": float(v[i]) if i < len(v) else 0.0
        })
    return detect_and_normalize_times(raw)

def compare_emas_and_crosses(symbol="BTCUSDT", res="60", days=30, fast=9, slow=21):
    candles = fetch_candles(symbol=symbol, resolution=res, days_back=days)
    if not candles:
        print("No candles returned")
        return
    candles.sort(key=lambda x: x["time"])
    times = [c["time"] for c in candles]
    closes = [c["close"] for c in candles]

    # local EMA (your implementation)
    ema_fast_local = compute_ema_series_local(closes, fast)
    ema_slow_local = compute_ema_series_local(closes, slow)

    # pandas EMA (adjust=False -> recursive formula)
    s = pd.Series(closes)
    ema_fast_pd = s.ewm(span=fast, adjust=False).mean().to_list()
    ema_slow_pd = s.ewm(span=slow, adjust=False).mean().to_list()

    # numeric comparison
    diffs_fast = [ (i, None if ema_fast_local[i] is None else abs(ema_fast_local[i] - ema_fast_pd[i])) for i in range(len(closes)) ]
    diffs_slow = [ (i, None if ema_slow_local[i] is None else abs(ema_slow_local[i] - ema_slow_pd[i])) for i in range(len(closes)) ]

    # compute max difference after index > max(period) (ignore first seed region)
    start_check = max(fast, slow)
    max_diff_fast = max((d for i,d in diffs_fast[start_check:] if d is not None), default=0.0)
    max_diff_slow = max((d for i,d in diffs_slow[start_check:] if d is not None), default=0.0)

    print(f"candles: {len(closes)}  start_time={to_iso(times[0])}  end_time={to_iso(times[-1])}")
    print(f"Max abs diff after index {start_check}: fast={max_diff_fast:.12f}  slow={max_diff_slow:.12f}")
    if max_diff_fast > 1e-8 or max_diff_slow > 1e-8:
        print("Differences exist beyond numerical noise. Inspect first 40 rows below:")
    else:
        print("EMA implementations match (within tiny numerical error) after seed region.")

    # Show diagnostic head rows
    print("\nIndex | Time (UTC)           | Close      | ema_fast_local | ema_fast_pd | diff ")
    for i in range(min(40, len(closes))):
        ef_local = ema_fast_local[i]
        ef_pd = ema_fast_pd[i]
        diff = None if ef_local is None else (ef_local - ef_pd)
        print(f"{i:4d} | {to_iso(times[i])} | {closes[i]:10.6f} | {ef_local!s:13} | {ef_pd:13.9f} | {diff!s}")

    # compute crosses
    crosses_local = detect_crosses_from_series(times, ema_fast_local, ema_slow_local)
    crosses_pd = detect_crosses_from_series(times, ema_fast_pd, ema_slow_pd)

    print("\nCROSSES (local implementation):", len(crosses_local))
    for c in crosses_local:
        print(c["type"], to_iso(c["time"]), "idx", c["index"], "price", closes[c["index"]], "fast", c["fast"], "slow", c["slow"])
    print("\nCROSSES (pandas ewm):", len(crosses_pd))
    for c in crosses_pd:
        print(c["type"], to_iso(c["time"]), "idx", c["index"], "price", closes[c["index"]], "fast", c["fast"], "slow", c["slow"])

    # Compare cross lists (timestamps/indices)
    idxs_local = [(c["index"], c["type"]) for c in crosses_local]
    idxs_pd = [(c["index"], c["type"]) for c in crosses_pd]
    only_local = [x for x in idxs_local if x not in idxs_pd]
    only_pd = [x for x in idxs_pd if x not in idxs_local]
    print("\nCrosses only in local impl:", only_local)
    print("Crosses only in pandas impl:", only_pd)

# --------------------------
# Run when executed
# --------------------------
if __name__ == "__main__":
    # change symbol/resolution/days as needed
    compare_emas_and_crosses(symbol="BTCUSDT", res="60", days=30, fast=9, slow=21)
