// run.js
import * as api from './api.js';
import * as charts from './charts.js';
import * as utils from './utils.js';
import * as ui from './ui.js';

const { priceChart, candleSeries, emaShortSeries, emaLongSeries, equityChart, equitySeries, setSeriesMarkers } = charts.createCharts();

// UI wiring
ui.runBtn.addEventListener('click', runButton);
ui.stopBtn.addEventListener('click', stopRun);

let pollInterval = null;
let isRunning = false;
let lastPlacedSignalIds = new Set();

async function runButton() {
  if (isRunning) return;
  isRunning = true;
  ui.runBtn.disabled = true;
  utils.$('#tradesTable tbody').innerHTML = '<tr><td colspan="5" class="muted">Loading…</td></tr>';
  try {
    await runStrategyOnce();
  } catch (err) {
    alert('Run failed: ' + (err.message || err));
    console.error(err);
  } finally {
    ui.runBtn.disabled = false;
    isRunning = false;
  }
}

function stopRun() {
  if (pollInterval) clearInterval(pollInterval);
  pollInterval = null;
  ui.setSeedWarn('');
  ui.setStats('');
}

async function runStrategyOnce() {
  const opts = ui.getUIOptions();
  const symbol = 'BTCUSDT';

  // 1) fetch hourly for chart + seed info
  const hourlyUrl = api.candlesUrl('60', opts.days, symbol);
  const hourlyResp = await api.fetchJSON(hourlyUrl);
  const hourly = (hourlyResp.candles || []).map(c => ({ time: c.time, open:+c.open, high:+c.high, low:+c.low, close:+c.close, volume:+c.volume }));
  if (hourly.length) {
    candleSeries.setData(hourly);
    priceChart.timeScale().fitContent();
  }

  const maxPeriod = Math.max(opts.short, opts.long);
  if (hourly.length < maxPeriod * 3) {
    ui.setSeedWarn(`<div class="warn">Only ${hourly.length} hourly bars — recommended >= ${maxPeriod*3} for stable seeding</div>`);
  } else {
    ui.setSeedWarn(`<div class="muted">Seed history: ${hourly.length} hourly bars</div>`);
  }

  // Mode handling
  const mode = opts.mode;

  // BACKTEST (hourly walkforward or conventional)
  if (mode === 'new' || mode === 'conventional') {
    const hoursWindow = Math.max(12, opts.days * 24);
    // define backParams in outer scope so it exists for later use (fixes ReferenceError)
    const backParams = {
      symbol,
      hours: hoursWindow,
      fast: opts.short,
      slow: opts.long,
      rsi_len: 14,
      rsi_ob: 65.0,
      rsi_os: 35.0,
      initial_capital: opts.initial_capital,
      fee_pct: opts.fee_pct,
      position_size_pct: 1.0,
      days_back_for_history: opts.days
    };

    if (mode === 'conventional') {
      backParams.method = 'conventional';
      if (opts.startISO) backParams.start_ts = Math.floor(new Date(opts.startISO).getTime()/1000);
      if (opts.endISO) backParams.end_ts = Math.floor(new Date(opts.endISO).getTime()/1000);
    }
    debugger;
    const url = api.walkforwardUrl(backParams);
    const j = await api.fetchJSON(url);

    const trades = j.trades || [];
    const markers = j.markers || [];
    const equity = j.equity || [];
    const stats = j.stats || {};

    const mappedMarkers = (markers || []).map(m => ({
      time: m.time,
      position: m.position || 'belowBar',
      color: m.color || (m.text === 'L' ? '#7cf2a6' : '#ff9b9b'),
      shape: m.shape || 'circle',
      text: m.text || (m.position === 'belowBar' ? 'L' : 'S')
    }));
    setSeriesMarkers(candleSeries, mappedMarkers);

    equitySeries.setData((equity || []).map(e => ({ time: e.time, value: e.equity })));
    equityChart.timeScale().fitContent();

    // EMAs: prefer server-provided for conventional
    if (j.method === 'conventional' && j.fast_ema && j.slow_ema && j.hourly) {
      const sdata = j.fast_ema.map((v,i) => ({ time: j.hourly[i].time, value: v })).filter(x=>x.value!=null);
      const ldata = j.slow_ema.map((v,i) => ({ time: j.hourly[i].time, value: v })).filter(x=>x.value!=null);
      emaShortSeries.setData(sdata);
      emaLongSeries.setData(ldata);

      // optionally render signals table
      if (j.signals) {
        const execSignals = j.signals.filter(s => s.long_signal || s.short_signal).map(s => ({ time: s.time, long: s.long_signal, short: s.short_signal }));
        utils.renderTradesIntoTable(j.trades || []);
      }
    } else {
      ui.setSeedWarn(`<div class="warn">no signal foound</div>`);

      // fallback compute EMAs client-side
      // utils.computeAndDrawEMAFromHourly(hourly, opts.short, emaShortSeries);
      // utils.computeAndDrawEMAFromHourly(hourly, opts.long, emaLongSeries);
      // utils.renderTradesIntoTable(trades || []);
    }

    ui.setStats(`<div class="stat">Trades: ${stats.trades || trades.length}</div>
      <div class="stat">Net PnL: ${ (stats.net_profit||0).toFixed(2) }</div>
      <div class="stat">Final Equity: ${ (stats.final_equity||0).toFixed(2) }</div>
      <div class="stat">Max Drawdown: ${ (stats.max_drawdown_pct||0).toFixed(2) }%</div>`);

    return { type:'backtest', trades, equity, stats };
  }

  // INTRAHOUR / LIVE (uses intraminute endpoint)
  if (mode === 'intrahour' || mode === 'live') {
    const p = {
      symbol,
      days: opts.days,
      fast: opts.short,
      slow: opts.long,
      rsi_len: 14,
      rsi_ob: 65.0,
      rsi_os: 35.0,
      initial_capital: opts.initial_capital,
      fee_pct: opts.fee_pct,
      position_size_pct: 1.0
    };

    const j = await api.fetchJSON(api.intraminuteUrl(p));
    const h = j.hourly || [];
    const minute_current = j.minute_current_hour || [];
    const trades = j.trades || [];
    const markers = j.markers || [];
    const equity = j.equity || [];
    const stats = j.stats || {};

    if (h.length) {
      const hh = h.map(c => ({ time: c.time, open:+c.open, high:+c.high, low:+c.low, close:+c.close }));
      candleSeries.setData(hh);
      priceChart.timeScale().fitContent();
    }

    // EMAs: server didn't provide arrays here in our setup — compute client fallback
    utils.computeAndDrawEMAFromHourly(hourly, opts.short, emaShortSeries);
    utils.computeAndDrawEMAFromHourly(hourly, opts.long, emaLongSeries);

    const mappedMarkers2 = (markers || []).map(m => ({
      time: m.time,
      position: m.position || 'belowBar',
      color: m.color || (m.text === 'L' ? '#7cf2a6' : '#ff9b9b'),
      shape: m.shape || 'circle',
      text: m.text || (m.position === 'belowBar' ? 'L' : 'S')
    }));
    setSeriesMarkers(candleSeries, mappedMarkers2);

    equitySeries.setData((equity || []).map(e => ({ time: e.time, value: e.equity })));
    equityChart.timeScale().fitContent();

    utils.renderTradesIntoTable(trades.length ? trades : (j.signals || []), '#tradesTable tbody');

    ui.setStats(`<div class="stat">Trades/Signals: ${ (trades.length || (j.signals||[]).length) }</div>
      <div class="stat">Final Equity: ${ (stats.final_equity||0).toFixed(2) }</div>
      <div class="stat">Max Drawdown: ${ (stats.max_drawdown_pct||0).toFixed(2) }%</div>`);

    if (mode === 'live' && markers && markers.length) {
      const newly = markers.filter(m => {
        const id = `${m.time}_${m.text || ''}`;
        if (lastPlacedSignalIds.has(id)) return false;
        lastPlacedSignalIds.add(id);
        return true;
      });
      for (const s of newly) {
        const payload = {
          symbol,
          action: s.position === 'aboveBar' ? 'close' : 'open',
          side: (s.text === 'L' || s.text === 'B') ? 'long' : 'short',
          time: s.time
        };
        try { await api.postJSON(api.placeOrderUrl(), payload); } catch(e){ console.warn('place_order failed', e); }
      }
    }

    return { type:'intrahour', trades, markers, equity, stats };
  }

  throw new Error('unknown mode: ' + mode);
}
