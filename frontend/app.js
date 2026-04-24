const elements = {
  apiBase: document.querySelector("#apiBase"),
  apiStatus: document.querySelector("#apiStatus"),
  saveApiBase: document.querySelector("#saveApiBase"),
  amount: document.querySelector("#amount"),
  risk: document.querySelector("#risk"),
  riskValue: document.querySelector("#riskValue"),
  horizon: document.querySelector("#horizon"),
  windowSize: document.querySelector("#windowSize"),
  refreshDashboard: document.querySelector("#refreshDashboard"),
  marketHighlights: document.querySelector("#marketHighlights"),
  macroSnapshot: document.querySelector("#macroSnapshot"),
  marketTable: document.querySelector("#marketTable"),
  tickerSelect: document.querySelector("#tickerSelect"),
  runTickerForecast: document.querySelector("#runTickerForecast"),
  forecastChart: document.querySelector("#forecastChart"),
  chartFallback: document.querySelector("#chartFallback"),
  tickerMetrics: document.querySelector("#tickerMetrics"),
  tickerNarrative: document.querySelector("#tickerNarrative"),
  simulationTickers: document.querySelector("#simulationTickers"),
  runSimulation: document.querySelector("#runSimulation"),
  simulationSummary: document.querySelector("#simulationSummary"),
  simulationWarnings: document.querySelector("#simulationWarnings"),
  simulationClasses: document.querySelector("#simulationClasses"),
  simulationAssets: document.querySelector("#simulationAssets"),
  simulationTrades: document.querySelector("#simulationTrades"),
  runRlAllocation: document.querySelector("#runRlAllocation"),
  runBacktest: document.querySelector("#runBacktest"),
  rlSummary: document.querySelector("#rlSummary"),
  rlClasses: document.querySelector("#rlClasses"),
  backtestSummary: document.querySelector("#backtestSummary"),
  backtestWarnings: document.querySelector("#backtestWarnings"),
  healthBlock: document.querySelector("#healthBlock"),
  modelsBlock: document.querySelector("#modelsBlock"),
};

const state = {
  apiBase: localStorage.getItem("stockify-api-base") || "http://localhost:8000",
  universe: null,
  chart: null,
};

const literacy = {
  bear: "Bear scenario: a weaker outcome estimated from return and volatility.",
  base: "Base scenario: the central estimate, not a guaranteed target.",
  bull: "Bull scenario: a stronger outcome if conditions are favorable.",
  volatility: "Volatility estimates how much the price or portfolio may swing.",
  drawdown: "Drawdown measures the largest historical drop from a previous high.",
  confidence: "Confidence falls when data is noisy, volatile, or the forecast band is wide.",
  sharpe: "Sharpe compares return against volatility. Higher is generally better.",
  diversification: "Diversification spreads exposure so one asset does not dominate results.",
};

const formatCurrency = (value) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(Number(value || 0));

const formatPercent = (value) => `${(Number(value || 0) * 100).toFixed(2)}%`;

const formatNumber = (value, digits = 2) => Number(value || 0).toFixed(digits);

function metricCard(label, value, tooltip) {
  const tip = tooltip ? `<span class="term" title="${tooltip}">?</span>` : "";
  return `
    <article class="metric">
      <h3>${label} ${tip}</h3>
      <strong>${value}</strong>
    </article>
  `;
}

function setLoading(target, message = "Loading...") {
  target.innerHTML = `<p class="muted">${message}</p>`;
}

async function callApi(path, options = {}) {
  const response = await fetch(`${state.apiBase}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const payload = await response.text();
    throw new Error(payload || `Request failed: ${response.status}`);
  }
  return response.json();
}

function dashboardPayload(extra = {}) {
  return {
    horizon_days: Number(elements.horizon.value),
    risk: Number(elements.risk.value),
    window_size: Number(elements.windowSize.value),
    strict_validation: false,
    ...extra,
  };
}

function allocationPayload() {
  return {
    amount: Number(elements.amount.value),
    risk: Number(elements.risk.value),
    duration: Number(elements.horizon.value),
    window_size: Number(elements.windowSize.value),
    strict_validation: false,
  };
}

function renderUniverse(universe) {
  const buildOptions = () =>
    universe.asset_classes
      .map(
        (group) => `
          <optgroup label="${group.asset_class}">
            ${group.tickers
              .map((entry) => `<option value="${entry.ticker}">${entry.ticker}</option>`)
              .join("")}
          </optgroup>
        `,
      )
      .join("");

  elements.tickerSelect.innerHTML = buildOptions();
  elements.simulationTickers.innerHTML = buildOptions();
  if (universe.tickers.length > 0) {
    elements.tickerSelect.value = universe.tickers[0].ticker;
  }
}

function renderMacro(snapshot) {
  elements.macroSnapshot.innerHTML = "";
  const regimeLabel = ["Bull", "Normal", "Bear"][snapshot.global_regime] || "Normal";
  const regime = document.createElement("div");
  regime.className = "row two-column";
  regime.innerHTML = `<strong>Detected regime</strong><span>${regimeLabel}</span>`;
  elements.macroSnapshot.appendChild(regime);

  snapshot.macro.slice(0, 6).forEach((entry) => {
    const row = document.createElement("div");
    row.className = "row two-column";
    row.innerHTML = `
      <strong>${entry.name}</strong>
      <span>${formatNumber(entry.value, 4)}</span>
    `;
    elements.macroSnapshot.appendChild(row);
  });
}

function compactForecast(entry) {
  return {
    ticker: entry.ticker,
    assetClass: entry.asset_class,
    base: formatPercent(entry.returns.base),
    bear: formatPercent(entry.returns.bear),
    bull: formatPercent(entry.returns.bull),
    volatility: formatPercent(entry.risk_metrics.annualized_volatility),
    confidence: `${entry.confidence_label} ${formatPercent(entry.confidence)}`,
  };
}

function renderMarket(result) {
  renderMacro(result.macro_snapshot);
  const bestBase = compactForecast(result.highlights.best_base_case);
  const bestAdjusted = compactForecast(result.highlights.best_risk_adjusted);
  const downside = compactForecast(result.highlights.highest_downside_risk);
  const stable = compactForecast(result.highlights.most_stable);

  elements.marketHighlights.innerHTML = [
    metricCard("Best base case", `${bestBase.ticker} ${bestBase.base}`, literacy.base),
    metricCard("Best risk-adjusted", bestAdjusted.ticker, literacy.sharpe),
    metricCard("Highest downside risk", `${downside.ticker} ${downside.bear}`, literacy.bear),
    metricCard("Most stable", `${stable.ticker} ${stable.volatility}`, literacy.volatility),
  ].join("");

  elements.marketTable.innerHTML = `
    <div class="row table-head market-row">
      <strong>Ticker</strong>
      <span>Class</span>
      <span>Base</span>
      <span>Bear</span>
      <span>Bull</span>
      <span>Confidence</span>
      <span></span>
    </div>
  `;

  result.ranked_tickers.forEach((entry) => {
    const row = document.createElement("div");
    row.className = "row market-row";
    row.innerHTML = `
      <strong>${entry.ticker}</strong>
      <span>${entry.asset_class}</span>
      <span>${formatPercent(entry.returns.base)}</span>
      <span>${formatPercent(entry.returns.bear)}</span>
      <span>${formatPercent(entry.returns.bull)}</span>
      <span>${entry.confidence_label}</span>
      <button class="small-button" data-view-ticker="${entry.ticker}">View</button>
    `;
    elements.marketTable.appendChild(row);
  });
}

function chartLabels(history, forecastPath) {
  return [
    ...history.map((point) => point.date),
    ...forecastPath.slice(1).map((point) => point.date),
  ];
}

function forecastSeries(history, forecastPath) {
  return Array(history.length - 1)
    .fill(null)
    .concat(forecastPath.map((point) => point.price));
}

function renderForecastChart(forecast) {
  if (!window.Chart) {
    elements.chartFallback.textContent = "Chart.js is unavailable. Metrics are still shown.";
    return;
  }
  elements.chartFallback.textContent = "";
  const history = forecast.historical_prices;
  const basePath = forecast.forecast_paths.base;
  const labels = chartLabels(history, basePath);
  const historyData = history
    .map((point) => point.price)
    .concat(Array(basePath.length - 1).fill(null));

  if (state.chart) {
    state.chart.destroy();
  }
  state.chart = new window.Chart(elements.forecastChart, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Historical",
          data: historyData,
          borderColor: "#234c6f",
          backgroundColor: "transparent",
          pointRadius: 0,
          borderWidth: 2,
        },
        {
          label: "Bear",
          data: forecastSeries(history, forecast.forecast_paths.bear),
          borderColor: "#a23b2a",
          borderDash: [6, 4],
          pointRadius: 0,
          borderWidth: 2,
        },
        {
          label: "Base",
          data: forecastSeries(history, forecast.forecast_paths.base),
          borderColor: "#1f6b3a",
          pointRadius: 0,
          borderWidth: 3,
        },
        {
          label: "Bull",
          data: forecastSeries(history, forecast.forecast_paths.bull),
          borderColor: "#9b6a14",
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
        legend: { position: "bottom" },
        tooltip: {
          callbacks: {
            label: (context) => `${context.dataset.label}: ${formatCurrency(context.parsed.y)}`,
          },
        },
      },
      scales: {
        x: { ticks: { maxTicksLimit: 8 } },
        y: {
          ticks: {
            callback: (value) => formatCurrency(value),
          },
        },
      },
    },
  });
}

function renderTickerForecast(forecast) {
  renderForecastChart(forecast);
  elements.tickerMetrics.innerHTML = [
    metricCard("Current price", formatCurrency(forecast.latest_price)),
    metricCard("Bear target", formatCurrency(forecast.target_prices.bear), literacy.bear),
    metricCard("Base target", formatCurrency(forecast.target_prices.base), literacy.base),
    metricCard("Bull target", formatCurrency(forecast.target_prices.bull), literacy.bull),
    metricCard("Base return", formatPercent(forecast.returns.base)),
    metricCard(
      "Annualized volatility",
      formatPercent(forecast.risk_metrics.annualized_volatility),
      literacy.volatility,
    ),
    metricCard(
      "Max drawdown",
      formatPercent(forecast.risk_metrics.max_historical_drawdown),
      literacy.drawdown,
    ),
    metricCard("Confidence", `${forecast.confidence_label} ${formatPercent(forecast.confidence)}`, literacy.confidence),
  ].join("");
  elements.tickerNarrative.innerHTML = `
    <strong>${forecast.ticker}</strong>
    <p>${forecast.plain_language}</p>
    <p class="muted">${forecast.literacy.bear_base_bull}</p>
  `;
}

function renderMetricBlock(target, entries) {
  target.innerHTML = entries.map(([label, value, tooltip]) => metricCard(label, value, tooltip)).join("");
}

function renderAllocationTable(target, allocations) {
  target.innerHTML = "";
  allocations.forEach((allocation) => {
    const row = document.createElement("div");
    row.className = "row allocation-row";
    row.innerHTML = `
      <strong>${allocation.ticker || allocation.asset_class}</strong>
      <span>${allocation.asset_class || ""}</span>
      <span>${formatPercent(allocation.weight)}</span>
      <span>${formatCurrency(allocation.amount)}</span>
    `;
    target.appendChild(row);
  });
}

function renderTradePlan(target, trades) {
  target.innerHTML = "";
  trades.forEach((trade) => {
    const row = document.createElement("div");
    row.className = "row trade-row";
    row.innerHTML = `
      <strong>${trade.action} ${trade.ticker}</strong>
      <span>${trade.asset_class}</span>
      <span>${formatPercent(trade.weight)}</span>
      <span>${formatCurrency(trade.amount)}</span>
    `;
    target.appendChild(row);
  });
}

function renderWarnings(target, warnings) {
  target.innerHTML = "";
  (warnings || []).forEach((warning) => {
    const row = document.createElement("article");
    row.className = "warning-banner";
    row.textContent = warning;
    target.appendChild(row);
  });
}

function selectedSimulationTickers() {
  return Array.from(elements.simulationTickers.selectedOptions).map((option) => option.value);
}

function renderSimulation(result) {
  renderMetricBlock(elements.simulationSummary, [
    ["Bear value", formatCurrency(result.summary.bear_value), literacy.bear],
    ["Base value", formatCurrency(result.summary.base_value), literacy.base],
    ["Bull value", formatCurrency(result.summary.bull_value), literacy.bull],
    ["Base return", formatPercent(result.summary.base_return)],
    ["Average confidence", formatPercent(result.summary.average_confidence), literacy.confidence],
  ]);
  renderWarnings(elements.simulationWarnings, result.warnings);
  renderAllocationTable(elements.simulationClasses, result.class_allocations);
  renderAllocationTable(elements.simulationAssets, result.asset_allocations);
  renderTradePlan(elements.simulationTrades, result.trade_plan);
}

function renderRlAllocation(result) {
  renderMetricBlock(elements.rlSummary, [
    ["Base-case value", formatCurrency(result.summary.projected_value), literacy.base],
    ["Base-case profit", formatCurrency(result.summary.projected_profit)],
    ["Bear scenario value", formatCurrency(result.summary.downside_value), literacy.bear],
    ["Bull scenario value", formatCurrency(result.summary.upside_value), literacy.bull],
    ["Model-estimated daily return", formatPercent(result.summary.expected_daily_return)],
    ["Annualized volatility", formatPercent(result.summary.annualized_volatility), literacy.volatility],
  ]);
  renderAllocationTable(elements.rlClasses, result.class_allocations);
}

function renderBacktest(result) {
  const metrics = result.summary_metrics;
  renderMetricBlock(elements.backtestSummary, [
    ["Cumulative return", formatPercent(metrics.cumulative_return)],
    ["Annualized return", formatPercent(metrics.annualized_return)],
    ["Volatility", formatPercent(metrics.annualized_volatility), literacy.volatility],
    ["Sharpe ratio", formatNumber(metrics.sharpe_ratio, 2), literacy.sharpe],
    ["Max drawdown", formatPercent(metrics.max_drawdown), literacy.drawdown],
    ["Ending value", formatCurrency(metrics.ending_value)],
  ]);
  renderWarnings(
    elements.backtestWarnings,
    (result.warnings || []).filter((warning) => !warning.includes("artifacts were date-aligned")),
  );
}

async function refreshDiagnostics() {
  const [health, models] = await Promise.all([
    callApi("/api/health"),
    callApi("/api/models"),
  ]);
  elements.healthBlock.textContent = JSON.stringify(health, null, 2);
  elements.modelsBlock.textContent = JSON.stringify(models, null, 2);
}

async function loadUniverse() {
  state.universe = await callApi("/api/universe");
  renderUniverse(state.universe);
}

async function runMarketForecast() {
  setLoading(elements.marketTable);
  const result = await callApi("/api/forecasts/market", {
    method: "POST",
    body: JSON.stringify(dashboardPayload({ top_n: 10 })),
  });
  renderMarket(result);
}

async function runTickerForecast(ticker = elements.tickerSelect.value) {
  setLoading(elements.tickerMetrics);
  const result = await callApi("/api/forecasts/ticker", {
    method: "POST",
    body: JSON.stringify(
      dashboardPayload({
        ticker,
      }),
    ),
  });
  elements.tickerSelect.value = result.ticker;
  renderTickerForecast(result);
}

async function runSimulation() {
  setLoading(elements.simulationSummary);
  const selected = selectedSimulationTickers();
  const result = await callApi("/api/portfolio/simulations", {
    method: "POST",
    body: JSON.stringify({
      amount: Number(elements.amount.value),
      ...dashboardPayload({
        selected_tickers: selected.length > 0 ? selected : null,
      }),
    }),
  });
  renderSimulation(result);
}

async function runRlAllocation() {
  setLoading(elements.rlSummary);
  const result = await callApi("/api/inference", {
    method: "POST",
    body: JSON.stringify(allocationPayload()),
  });
  renderRlAllocation(result);
}

async function runBacktest() {
  setLoading(elements.backtestSummary);
  const result = await callApi("/api/backtests", {
    method: "POST",
    body: JSON.stringify({
      initial_amount: Number(elements.amount.value),
      risk: Number(elements.risk.value),
      window_size: Number(elements.windowSize.value),
      max_steps: Number(elements.horizon.value),
      strict_validation: false,
    }),
  });
  renderBacktest(result);
}

async function refreshDashboard() {
  elements.apiStatus.textContent = "Connecting to backend...";
  elements.apiStatus.className = "muted";
  await refreshDiagnostics();
  if (!state.universe) {
    await loadUniverse();
  }
  await Promise.all([runMarketForecast(), runTickerForecast(), runSimulation()]);
  elements.apiStatus.textContent = "Backend connected.";
  elements.apiStatus.className = "status-good";
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((node) => node.classList.remove("is-active"));
    document.querySelectorAll(".view").forEach((node) => node.classList.remove("is-active"));
    tab.classList.add("is-active");
    document.querySelector(`#${tab.dataset.tab}`).classList.add("is-active");
  });
});

elements.marketTable.addEventListener("click", (event) => {
  const button = event.target.closest("[data-view-ticker]");
  if (!button) return;
  document.querySelector('[data-tab="forecast"]').click();
  runTickerForecast(button.dataset.viewTicker).catch((error) => alert(error.message));
});

elements.risk.addEventListener("input", () => {
  elements.riskValue.textContent = Number(elements.risk.value).toFixed(2);
});

elements.saveApiBase.addEventListener("click", async () => {
  state.apiBase = elements.apiBase.value.replace(/\/$/, "");
  localStorage.setItem("stockify-api-base", state.apiBase);
  state.universe = null;
  await refreshDashboard().catch((error) => {
    elements.apiStatus.textContent = `Backend unavailable: ${error.message}`;
    elements.apiStatus.className = "status-bad";
  });
});

elements.refreshDashboard.addEventListener("click", () =>
  refreshDashboard().catch((error) => alert(error.message)),
);
elements.runTickerForecast.addEventListener("click", () =>
  runTickerForecast().catch((error) => alert(error.message)),
);
elements.runSimulation.addEventListener("click", () =>
  runSimulation().catch((error) => alert(error.message)),
);
elements.runRlAllocation.addEventListener("click", () =>
  runRlAllocation().catch((error) => alert(error.message)),
);
elements.runBacktest.addEventListener("click", () =>
  runBacktest().catch((error) => alert(error.message)),
);

elements.apiBase.value = state.apiBase;
elements.riskValue.textContent = Number(elements.risk.value).toFixed(2);
refreshDashboard().catch((error) => {
  elements.apiStatus.textContent = `Backend unavailable: ${error.message}`;
  elements.apiStatus.className = "status-bad";
});
