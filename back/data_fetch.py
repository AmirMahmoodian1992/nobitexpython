# back/data_fetch.py
import requests
from typing import List, Dict, Any

UDF_URL = "https://apiv2.nobitex.ir/market/udf/history"

# ... (rest of the file remains unchanged)

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