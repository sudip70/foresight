import { elements, state } from "../state/store.js";
import { formatCurrency, formatDate } from "../utils/formatters.js";
import { cssVar } from "../utils/dom.js";

export function chartLabels(history, forecastPath) {
  const historyDates = history.map((point) => point.date);
  const forecastDates = forecastPath.map((point) => point.date);
  if (historyDates.at(-1) && historyDates.at(-1) === forecastDates[0]) {
    return historyDates.concat(forecastDates.slice(1));
  }
  return historyDates.concat(forecastDates);
}

export function visibleHistory(history) {
  const rangeToPoints = {
    "1m": 22,
    "3m": 66,
    "6m": 132,
    "1y": 252,
  };
  const points = rangeToPoints[state.chartRange];
  return points ? history.slice(-points) : history;
}

export function forecastSeries(history, forecastPath) {
  const startsAtHistoryEnd = history.at(-1)?.date === forecastPath[0]?.date;
  return Array(startsAtHistoryEnd ? history.length - 1 : history.length)
    .fill(null)
    .concat(forecastPath.map((point) => point.price));
}

export function renderForecastChart(forecast) {
  if (!window.Chart) {
    elements.chartFallback.textContent = "Chart.js is unavailable. Metrics are still shown.";
    return;
  }
  elements.chartFallback.textContent = "";
  const history = visibleHistory(forecast.historical_prices);
  const basePath = forecast.forecast_paths.base;
  const labels = chartLabels(history, basePath);
  const startsAtHistoryEnd = history.at(-1)?.date === basePath[0]?.date;
  const historyData = history
    .map((point) => point.price)
    .concat(Array(startsAtHistoryEnd ? basePath.length - 1 : basePath.length).fill(null));
  if (elements.forecastDateRange) {
    const start = basePath[0]?.date || forecast.forecast_start_date;
    const end = forecast.forecast_paths.base.at(-1)?.date;
    elements.forecastDateRange.textContent = `${formatDate(start)} - ${formatDate(end)} Projection`;
  }

  if (state.chart) {
    state.chart.destroy();
  }
  const chartAccent = cssVar("--accent", "#008755");
  const chartBlue = cssVar("--blue", "#1f66d1");
  const chartDanger = cssVar("--danger", "#e1192d");
  const chartGrid = cssVar("--chart-grid", "#edf2f7");
  const chartText = cssVar("--muted", "#52617a");
  state.chart = new window.Chart(elements.forecastChart, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Historical",
          data: historyData,
          borderColor: chartAccent,
          backgroundColor: "transparent",
          pointRadius: 0,
          borderWidth: 3,
        },
        {
          label: "Bull",
          data: forecastSeries(history, forecast.forecast_paths.bull),
          borderColor: chartBlue,
          borderDash: [6, 4],
          pointRadius: 0,
          borderWidth: 2,
        },
        {
          label: "Base",
          data: forecastSeries(history, forecast.forecast_paths.base),
          borderColor: chartAccent,
          borderDash: [6, 4],
          pointRadius: 0,
          borderWidth: 2,
        },
        {
          label: "Bear",
          data: forecastSeries(history, forecast.forecast_paths.bear),
          borderColor: chartDanger,
          borderDash: [6, 4],
          pointRadius: 0,
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { intersect: false, mode: "index" },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (context) => `${context.dataset.label}: ${formatCurrency(context.parsed.y)}`,
          },
        },
      },
      scales: {
        x: {
          grid: { color: chartGrid },
          ticks: { color: chartText, maxTicksLimit: 8 },
        },
        y: {
          grid: { color: chartGrid },
          ticks: {
            color: chartText,
            callback: (value) => formatCurrency(value),
          },
        },
      },
    },
  });
}
