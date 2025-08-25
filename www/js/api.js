// api.js
const BASE = "http://127.0.0.1:8000";

export function buildUrl(path, params = {}) {
  const url = new URL(BASE + path);
  Object.keys(params).forEach(k => {
    if (params[k] === undefined || params[k] === null || params[k] === "") return;
    url.searchParams.set(k, params[k]);
  });
  return url.toString();
}

export async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

export async function postJSON(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(body)
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

export function candlesUrl(resolution = '60', days = 2, symbol = 'BTCUSDT') {
  return buildUrl('/candles', { resolution, days, symbol });
}

export function walkforwardUrl(params = {}) {
  return buildUrl('/walkforward', params);
}

export function intraminuteUrl(params = {}) {
  return buildUrl('/intraminute_backtest', params);
}

export function placeOrderUrl() {
  return BASE + '/place_order';
}
