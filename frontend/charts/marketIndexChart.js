import { elements, state } from "../state/store.js";
import {
  formatDate,
  formatIndexValue,
  formatSignedNumber,
  formatSignedPercent,
  isMissing,
  toneForValue,
} from "../utils/formatters.js";
import { cssVar, metricCard, setError } from "../utils/dom.js";

export function movingAverage(points, windowSize) {
  let total = 0;
  return points.map((point, index) => {
    const value = Number(point.close || 0);
    total += value;
    if (index >= windowSize) {
      total -= Number(points[index - windowSize].close || 0);
    }
    return index < windowSize - 1 ? null : total / windowSize;
  });
}

export function renderMarketIndexHistory(result) {
  const history = result?.history || [];
  if (!history.length) {
    setError(elements.marketIndexSummary, "Market index history unavailable.");
    if (elements.marketIndexChartFallback) {
      elements.marketIndexChartFallback.textContent = "No history returned for this index.";
    }
    return;
  }
  state.marketIndexHistory = result;
  state.marketIndexSymbol = result.symbol || state.marketIndexSymbol;
  if (elements.marketIndexSelect) {
    const hasSelectedOption = Array.from(elements.marketIndexSelect.options).some(
      (option) => option.value === state.marketIndexSymbol,
    );
    if (!hasSelectedOption) {
      elements.marketIndexSelect.innerHTML =
        `<option value="${escapeHtml(state.marketIndexSymbol)}">${escapeHtml(result.label || state.marketIndexSymbol)}</option>`;
    }
    elements.marketIndexSelect.value = state.marketIndexSymbol;
  }

  const summary = result.summary || {};
  const firstDate = summary.first_date || history[0]?.date;
  const latestDate = summary.latest_date || result.as_of_date || history.at(-1)?.date;
  if (elements.marketIndexDateRange) {
    elements.marketIndexDateRange.textContent = `${result.label || result.symbol} - ${formatDate(firstDate)} to ${formatDate(latestDate)}`;
  }
  elements.marketIndexSummary.innerHTML = [
    metricCard("Latest", formatIndexValue(summary.latest_close || history.at(-1)?.close)),
    metricCard(
      "Range return",
      formatSignedPercent(summary.range_return),
      "Total return over the selected chart range",
      "",
      isMissing(summary.range_return) ? "" : toneForValue(summary.range_return),
    ),
    metricCard(
      "Daily move",
      `${formatSignedNumber(summary.change)} (${formatSignedPercent(summary.change_percent)})`,
      "",
      "",
      isMissing(summary.change) ? "" : toneForValue(summary.change),
    ),
    metricCard("High / low", `${formatIndexValue(summary.high)} / ${formatIndexValue(summary.low)}`),
  ].join("");

  if (!window.Chart) {
    elements.marketIndexChartFallback.textContent = "Chart.js is unavailable. Index history is still summarized.";
    return;
  }
  elements.marketIndexChartFallback.textContent = "";
  if (state.marketIndexChart) {
    state.marketIndexChart.destroy();
  }

  const chartAccent = cssVar("--accent", "#008755");
  const chartBlue = cssVar("--blue", "#1f66d1");
  const chartGrid = cssVar("--chart-grid", "#edf2f7");
  const chartText = cssVar("--muted", "#52617a");
  state.marketIndexChart = new window.Chart(elements.marketIndexChart, {
    type: "line",
    data: {
      labels: history.map((point) => formatDate(point.date)),
      datasets: [
        {
          label: "Close",
          data: history.map((point) => point.close),
          borderColor: chartAccent,
          backgroundColor: "transparent",
          pointRadius: 0,
          borderWidth: 3,
        },
        {
          label: "20-day average",
          data: movingAverage(history, 20),
          borderColor: chartBlue,
          borderDash: [6, 4],
          backgroundColor: "transparent",
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
            label: (context) => `${context.dataset.label}: ${formatIndexValue(context.parsed.y)}`,
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
            callback: (value) => formatIndexValue(value),
          },
        },
      },
    },
  });
}
