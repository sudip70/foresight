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
  marketIndices: document.querySelector("#marketIndices"),
  marketHighlights: document.querySelector("#marketHighlights"),
  macroSnapshot: document.querySelector("#macroSnapshot"),
  marketTable: document.querySelector("#marketTable"),
  tickerSelect: document.querySelector("#tickerSelect"),
  runTickerForecast: document.querySelector("#runTickerForecast"),
  forecastChart: document.querySelector("#forecastChart"),
  chartFallback: document.querySelector("#chartFallback"),
  tickerMetrics: document.querySelector("#tickerMetrics"),
  tickerProfile: document.querySelector("#tickerProfile"),
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
  refreshStatusBlock: document.querySelector("#refreshStatusBlock"),
};

const state = {
  apiBase:
    localStorage.getItem("foresight-api-base") ||
    localStorage.getItem("stockify-api-base") ||
    "http://localhost:8000",
  universe: null,
  chart: null,
  controllers: {},
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
const formatIndexValue = (value) =>
  isMissing(value)
    ? "Unavailable"
    : new Intl.NumberFormat("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      }).format(Number(value));
const formatSignedNumber = (value) => {
  if (isMissing(value)) return "Unavailable";
  const parsed = Number(value);
  return `${parsed >= 0 ? "+" : ""}${formatNumber(parsed, 2)}`;
};
const formatSignedPercent = (value) => {
  if (isMissing(value)) return "Unavailable";
  const parsed = Number(value);
  return `${parsed >= 0 ? "+" : ""}${formatPercent(parsed)}`;
};

function isMissing(value) {
  return value === null || value === undefined || value === "" || Number.isNaN(Number(value));
}

function formatCurrencyOptional(value) {
  return isMissing(value) ? "Unavailable" : formatCurrency(value);
}

function formatPercentOptional(value) {
  return isMissing(value) ? "Unavailable" : formatPercent(value);
}

function formatNumberOptional(value, digits = 2) {
  return isMissing(value) ? "Unavailable" : formatNumber(value, digits);
}

function formatLargeNumber(value) {
  if (isMissing(value)) return "Unavailable";
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 2,
  }).format(Number(value));
}

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

function setError(target, message) {
  target.innerHTML = `<article class="error-banner">${message}</article>`;
}

async function callApi(path, options = {}) {
  const response = await fetch(`${state.apiBase}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const payload = await response.text();
    let message = payload;
    try {
      const parsed = JSON.parse(payload);
      message = parsed.detail || parsed.message || payload;
    } catch {
      message = payload;
    }
    throw new Error(message || `Request failed: ${response.status}`);
  }
  return response.json();
}

async function callSectionApi(section, path, options = {}) {
  if (state.controllers[section]) {
    state.controllers[section].abort();
  }
  const controller = new AbortController();
  state.controllers[section] = controller;
  try {
    return await callApi(path, {
      ...options,
      signal: controller.signal,
    });
  } finally {
    if (state.controllers[section] === controller) {
      delete state.controllers[section];
    }
  }
}

function isAbort(error) {
  return error && error.name === "AbortError";
}

function dashboardPayload(extra = {}) {
  return {
    horizon_days: Number(elements.horizon.value),
    risk: Number(elements.risk.value),
    window_size: Number(elements.windowSize.value),
    strict_validation: true,
    ...extra,
  };
}

function allocationPayload() {
  return {
    amount: Number(elements.amount.value),
    risk: Number(elements.risk.value),
    duration: Number(elements.horizon.value),
    window_size: Number(elements.windowSize.value),
    strict_validation: true,
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

function renderMarketIndices(result) {
  const indices = result?.indices || [];
  if (!indices.length) {
    setError(elements.marketIndices, "Market index data unavailable.");
    return;
  }
  elements.marketIndices.innerHTML = indices
    .map((entry) => {
      const change = Number(entry.change || 0);
      const tone = change >= 0 ? "positive" : "negative";
      return `
        <article class="index-card">
          <div>
            <span>${entry.label || entry.symbol}</span>
            <strong>${formatIndexValue(entry.value)}</strong>
          </div>
          <div class="index-change ${tone}">
            <span>${formatSignedNumber(entry.change)}</span>
            <span>${formatSignedPercent(entry.change_percent)}</span>
          </div>
          <small>${entry.as_of_date || result.as_of_date || "Unavailable"}</small>
        </article>
      `;
    })
    .join("");
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
    <div class="freshness-list">
      <span>Market data as of ${forecast.data_as_of || forecast.latest_date || "Unavailable"}</span>
      <span>Forecast source: ${forecast.snapshot_used ? "stored daily snapshot" : "computed on request"}</span>
    </div>
  `;
}

function renderTickerProfile(profile, forecast) {
  if (!profile) {
    setError(elements.tickerProfile, "Company data unavailable.");
    return;
  }
  const fields = profile.fields || {};
  const rows = [
    ["Bid", formatCurrencyOptional(fields.bid)],
    ["Ask", formatCurrencyOptional(fields.ask)],
    ["Last sale", formatCurrencyOptional(fields.last_sale)],
    ["Open", formatCurrencyOptional(fields.open)],
    ["High", formatCurrencyOptional(fields.high)],
    ["Low", formatCurrencyOptional(fields.low)],
    ["Exchange", fields.exchange || "Unavailable"],
    ["Mkt cap", formatLargeNumber(fields.market_cap)],
    ["P/E ratio", formatNumberOptional(fields.pe_ratio, 2)],
    ["52W high", formatCurrencyOptional(fields.fifty_two_week_high)],
    ["52W low", formatCurrencyOptional(fields.fifty_two_week_low)],
    ["Volume", formatLargeNumber(fields.volume)],
    ["Avg vol", formatLargeNumber(fields.average_volume)],
    ["Margin req", formatPercentOptional(fields.margin_requirement)],
    ["Dividend freq.", fields.dividend_frequency || "Unavailable"],
    ["12-month yield", formatPercentOptional(fields.dividend_yield)],
    ["Ex-dividend date", fields.ex_dividend_date || "Unavailable"],
  ];
  elements.tickerProfile.innerHTML = rows
    .map(
      ([label, value]) => `
        <div class="profile-item">
          <span>${label}</span>
          <strong>${value}</strong>
        </div>
      `,
    )
    .join("");

  const tickerInfo = state.universe?.tickers?.find((entry) => entry.ticker === forecast.ticker);
  const warnings = [];
  if (tickerInfo && tickerInfo.min_history_days && tickerInfo.row_count < tickerInfo.min_history_days) {
    warnings.push(
      `Coverage warning: ${forecast.ticker} has ${tickerInfo.row_count} rows, below the preferred ${tickerInfo.min_history_days}.`,
    );
  }
  if (profile.as_of_date || profile.data_as_of) {
    warnings.push(
      `Profile data as of ${profile.as_of_date || "Unavailable"}; market data as of ${profile.data_as_of || "Unavailable"}.`,
    );
  }
  if (warnings.length > 0) {
    elements.tickerNarrative.insertAdjacentHTML(
      "beforeend",
      `<div class="freshness-list">${warnings.map((warning) => `<span>${warning}</span>`).join("")}</div>`,
    );
  }
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
  setLoading(elements.healthBlock);
  setLoading(elements.modelsBlock);
  setLoading(elements.refreshStatusBlock);
  const [healthResult, modelsResult, refreshResult] = await Promise.allSettled([
    callSectionApi("health", "/api/health"),
    callSectionApi("models", "/api/models"),
    callSectionApi("refreshStatus", "/api/data/refresh/status"),
  ]);
  elements.healthBlock.textContent =
    healthResult.status === "fulfilled"
      ? JSON.stringify(healthResult.value, null, 2)
      : `Health unavailable: ${healthResult.reason.message}`;
  elements.modelsBlock.textContent =
    modelsResult.status === "fulfilled"
      ? JSON.stringify(modelsResult.value, null, 2)
      : `Model metadata unavailable: ${modelsResult.reason.message}`;
  elements.refreshStatusBlock.textContent =
    refreshResult.status === "fulfilled"
      ? JSON.stringify(refreshResult.value, null, 2)
      : `Refresh status unavailable: ${refreshResult.reason.message}`;
}

async function loadUniverse() {
  state.universe = await callSectionApi("universe", "/api/universe");
  renderUniverse(state.universe);
}

async function runMarketForecast() {
  setLoading(elements.marketTable);
  setLoading(elements.marketHighlights);
  setLoading(elements.marketIndices);
  try {
    const [indexResult, marketResult] = await Promise.allSettled([
      callSectionApi("marketIndices", "/api/market/indices"),
      callSectionApi("market", "/api/forecasts/market", {
        method: "POST",
        body: JSON.stringify(dashboardPayload({ top_n: 10 })),
      }),
    ]);
    if (indexResult.status === "fulfilled") {
      renderMarketIndices(indexResult.value);
    } else {
      setError(elements.marketIndices, `Market index data unavailable: ${indexResult.reason.message}`);
    }
    if (marketResult.status === "rejected") {
      throw marketResult.reason;
    }
    renderMarket(marketResult.value);
  } catch (error) {
    if (isAbort(error)) return;
    setError(elements.marketTable, `Market forecast unavailable: ${error.message}`);
    setError(elements.marketHighlights, "Market highlights unavailable.");
    setError(elements.marketIndices, "Market index data unavailable.");
    throw error;
  }
}

async function runTickerForecast(ticker = elements.tickerSelect.value) {
  setLoading(elements.tickerMetrics);
  setLoading(elements.tickerProfile);
  elements.tickerNarrative.innerHTML = "";
  try {
    const [result, profile] = await Promise.all([
      callSectionApi("tickerForecast", "/api/forecasts/ticker", {
        method: "POST",
        body: JSON.stringify(
          dashboardPayload({
            ticker,
          }),
        ),
      }),
      callSectionApi("tickerProfile", `/api/tickers/${encodeURIComponent(ticker)}/profile`),
    ]);
    elements.tickerSelect.value = result.ticker;
    renderTickerForecast(result);
    renderTickerProfile(profile, result);
  } catch (error) {
    if (isAbort(error)) return;
    setError(elements.tickerMetrics, `Ticker forecast unavailable: ${error.message}`);
    setError(elements.tickerProfile, "Company data unavailable.");
    throw error;
  }
}

async function runSimulation() {
  setLoading(elements.simulationSummary);
  const selected = selectedSimulationTickers();
  try {
    const result = await callSectionApi("simulation", "/api/portfolio/simulations", {
      method: "POST",
      body: JSON.stringify({
        amount: Number(elements.amount.value),
        ...dashboardPayload({
          selected_tickers: selected.length > 0 ? selected : null,
        }),
      }),
    });
    renderSimulation(result);
  } catch (error) {
    if (isAbort(error)) return;
    setError(elements.simulationSummary, `Simulation unavailable: ${error.message}`);
    throw error;
  }
}

async function runRlAllocation() {
  setLoading(elements.rlSummary);
  try {
    const result = await callSectionApi("rlAllocation", "/api/inference", {
      method: "POST",
      body: JSON.stringify(allocationPayload()),
    });
    renderRlAllocation(result);
  } catch (error) {
    if (isAbort(error)) return;
    setError(elements.rlSummary, `RL allocation unavailable: ${error.message}`);
    throw error;
  }
}

async function runBacktest() {
  setLoading(elements.backtestSummary);
  try {
    const result = await callSectionApi("backtest", "/api/backtests", {
      method: "POST",
      body: JSON.stringify({
        initial_amount: Number(elements.amount.value),
        risk: Number(elements.risk.value),
        window_size: Number(elements.windowSize.value),
        max_steps: Number(elements.horizon.value),
        strict_validation: true,
      }),
    });
    renderBacktest(result);
  } catch (error) {
    if (isAbort(error)) return;
    setError(elements.backtestSummary, `Backtest unavailable: ${error.message}`);
    throw error;
  }
}

async function refreshDashboard() {
  elements.apiStatus.textContent = "Connecting to backend...";
  elements.apiStatus.className = "muted";
  await refreshDiagnostics();
  if (!state.universe) {
    await loadUniverse();
  }
  const results = await Promise.allSettled([
    runMarketForecast(),
    runTickerForecast(),
    runSimulation(),
  ]);
  const failed = results.filter((result) => result.status === "rejected");
  if (failed.length > 0) {
    throw failed[0].reason;
  }
  elements.apiStatus.textContent = "Backend connected.";
  elements.apiStatus.className = "status-good";
}

async function probeBackend() {
  elements.apiStatus.textContent = "Checking backend...";
  elements.apiStatus.className = "muted";
  try {
    const health = await callSectionApi("startupHealth", "/api/health");
    if (health.status !== "ok" || health.ready === false) {
      throw new Error(health.error || "Backend is not ready");
    }
    await refreshDashboard();
  } catch (error) {
    if (isAbort(error)) return;
    elements.apiStatus.textContent = `Set your backend URL, then click Use Backend. ${error.message}`;
    elements.apiStatus.className = "status-bad";
    setError(elements.marketTable, "Backend is not connected.");
    setError(elements.marketIndices, "Backend is not connected.");
    setError(elements.tickerMetrics, "Backend is not connected.");
    setError(elements.simulationSummary, "Backend is not connected.");
  }
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
  localStorage.setItem("foresight-api-base", state.apiBase);
  state.universe = null;
  await refreshDashboard().catch((error) => {
    elements.apiStatus.textContent = `Backend unavailable: ${error.message}`;
    elements.apiStatus.className = "status-bad";
  });
});

elements.refreshDashboard.addEventListener("click", () =>
  refreshDashboard().catch((error) => {
    if (isAbort(error)) return;
    elements.apiStatus.textContent = `Dashboard refresh failed: ${error.message}`;
    elements.apiStatus.className = "status-bad";
  }),
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
probeBackend();
