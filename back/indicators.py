# app/indicators.py
from typing import List, Optional, Tuple

# -----------------------
# Helpers: math & stats
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

# New helper: full EMA series (useful for hourly series diagnostics)
def compute_ema_series(closes: List[float], period: int) -> List[float]:
    """
    Compute full EMA series (one ema per input close) using SMA seed for first 'period' bars.
    Returns list of EMA floats same length as closes (if closes empty -> []).
    """
    if not closes:
        return []
    if period <= 0:
        raise ValueError("period must be > 0")
    n = len(closes)
    k = 2.0 / (period + 1.0)
    emas = [0.0] * n
    if n < period:
        ema = closes[0]
        emas[0] = ema
        for i in range(1, n):
            ema = closes[i] * k + ema * (1 - k)
            emas[i] = ema
        return emas
    # seed with SMA of first period
    seed = sum(closes[:period]) / period
    ema = seed
    for i in range(period):
        emas[i] = None
    emas[period-1] = seed
    for i in range(period, n):
        ema = closes[i] * k + ema * (1 - k)
        emas[i] = ema
    for i in range(0, period-1):
        emas[i] = closes[i]
    return emas

# -----------------------
# RSI series (Wilder / ta.rsi equivalent)
# -----------------------
def compute_rsi_series(closes: List[float], period: int) -> List[Optional[float]]:
    """
    Compute Wilder-style RSI series aligned to closes.
    Returns list of floats or None for bars before RSI is defined.
    """
    n = len(closes)
    if n == 0:
        return []
    if period <= 0:
        raise ValueError("period must be > 0")
    rsi = [None] * n
    if n < 2:
        return rsi
    # deltas
    deltas = [0.0] * n
    for i in range(1, n):
        deltas[i] = closes[i] - closes[i-1]
    # initial avg gain/loss for first 'period' intervals (indexes 1..period)
    if n <= period:
        # not enough data for full initialization: compute simple avg over available
        up = down = 0.0
        count = 0
        for i in range(1, n):
            d = deltas[i]
            if d > 0: up += d
            else: down += -d
            count += 1
        if count == 0:
            return rsi
        avg_up = up / count
        avg_down = down / count
        rs = avg_up / (avg_down or 1e-12)
        rsi[-1] = 100.0 - 100.0 / (1.0 + rs)
        return rsi
    # proper initialization
    up = down = 0.0
    for i in range(1, period+1):
        d = deltas[i]
        if d > 0: up += d
        else: down += -d
    avg_up = up / period
    avg_down = down / period
    # RSI value at index period
    rs = avg_up / (avg_down or 1e-12)
    rsi[period] = 100.0 - 100.0 / (1.0 + rs)
    # iterate
    for i in range(period+1, n):
        d = deltas[i]
        gain = d if d > 0 else 0.0
        loss = -d if d < 0 else 0.0
        avg_up = (avg_up * (period - 1) + gain) / period
        avg_down = (avg_down * (period - 1) + loss) / period
        rs = avg_up / (avg_down or 1e-12)
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return rsi