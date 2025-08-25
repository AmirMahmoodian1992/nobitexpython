// charts.js
export function createCharts(priceElemId='priceChart', equityElemId='equityChart') {
  const priceChart = LightweightCharts.createChart(document.getElementById(priceElemId), {
    layout:{background:'#071226', textColor:'#dfeeff'},
    grid:{vertLines:{color:'rgba(255,255,255,0.03)'}, horzLines:{color:'rgba(255,255,255,0.03)'}},
    rightPriceScale:{borderColor:'#063248'},
    timeScale:{timeVisible:true, secondsVisible:false}
  });
  const candleSeries = priceChart.addSeries(LightweightCharts.CandlestickSeries);
  const emaShortSeries = priceChart.addSeries(LightweightCharts.LineSeries, { lineWidth:1.5 });
  const emaLongSeries  = priceChart.addSeries(LightweightCharts.LineSeries, { lineWidth:1.5 });

  const equityChart = LightweightCharts.createChart(document.getElementById(equityElemId), {
    layout:{background:'#071226', textColor:'#dfeeff'},
    rightPriceScale:{borderColor:'#063248'},
    timeScale:{timeVisible:true, secondsVisible:false}
  });
  const equitySeries = equityChart.addSeries(LightweightCharts.LineSeries, { priceFormat:{type:'price', precision:2}, lineWidth:2 });

  function setSeriesMarkers(series, markers) {
    if (!series) return;
    if (typeof series.setMarkers === 'function') {
      try { series.setMarkers(markers); return; } catch(e){}
    }
    if (typeof LightweightCharts !== 'undefined' && typeof LightweightCharts.createSeriesMarkers === 'function') {
      try { LightweightCharts.createSeriesMarkers(series, markers); return; } catch(e){}
    }
    console.warn('Cannot set markers');
  }

  return {
    priceChart, candleSeries, emaShortSeries, emaLongSeries,
    equityChart, equitySeries, setSeriesMarkers
  };
}
