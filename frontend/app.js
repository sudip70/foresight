import {
  canonicalApiBase,
  defaultApiBase,
  MAX_FORECAST_HORIZON_DAYS,
  MAX_FORECAST_WINDOW_SIZE,
  MAX_RL_BACKTEST_STEPS,
} from "./api/endpoints.js";
import {
  callArtifactSectionApi,
  callSectionApi,
  isAbort,
  isBackendConnectionError,
} from "./api/client.js";
import { elements, glossary, state } from "./state/store.js";
import { renderForecastChart } from "./charts/forecastChart.js";
import { renderMarketIndexHistory } from "./charts/marketIndexChart.js";
import { renderMarket, renderMarketIndices, renderUniverse } from "./render/market.js";
import { renderTickerForecast, renderTickerProfile } from "./render/forecast.js";
import {
  renderBacktest,
  renderPortfolioLearning,
  renderRlAllocation,
  renderSimulation,
} from "./render/simulator.js";
import {
  renderDataHealthCards,
  renderGlossary,
  renderProjectStory,
} from "./render/diagnostics.js";
import {
  allocationSegmentAtPoint,
  allocationTooltipContent,
  hideHoverTooltip,
  iconSvg,
  sentimentTooltipContent,
  setBackendEmptyState,
  setError,
  setLoading,
  showHoverTooltip,
  showToast,
} from "./utils/dom.js";
import { escapeHtml } from "./utils/formatters.js";
import {
  allocationPayload,
  dashboardPayload,
  portfolioConstraintPayload,
  selectedHorizonDays,
  selectedSimulationTickers,
  selectedWindowSize,
} from "./utils/validation.js";

let appStarted = false;

function setLearnMode(enabled) {
  state.learnMode = Boolean(enabled);
  document.body.classList.toggle("learn-mode", state.learnMode);
  if (elements.learnModeToggle) {
    elements.learnModeToggle.classList.toggle("is-active", state.learnMode);
    elements.learnModeToggle.setAttribute("aria-pressed", String(state.learnMode));
    const badge = elements.learnModeToggle.querySelector(".toggle-badge");
    if (badge) badge.textContent = state.learnMode ? "On" : "Off";
  }
  localStorage.setItem("foresight-learn-mode", String(state.learnMode));
}

function setThemeMode(mode) {
  state.themeMode = mode === "dark" ? "dark" : "light";
  document.documentElement.dataset.theme = state.themeMode;
  document.body.classList.toggle("theme-dark", state.themeMode === "dark");
  if (elements.themeModeToggle) {
    const isDark = state.themeMode === "dark";
    elements.themeModeToggle.classList.toggle("is-active", isDark);
    elements.themeModeToggle.setAttribute("aria-pressed", String(isDark));
    const badge = elements.themeModeToggle.querySelector(".toggle-badge");
    if (badge) badge.textContent = isDark ? "Dark" : "Light";
  }
  localStorage.setItem("foresight-theme-mode", state.themeMode);

  if (state.currentForecast) {
    renderForecastChart(state.currentForecast);
  }
  if (state.marketIndexHistory) {
    renderMarketIndexHistory(state.marketIndexHistory);
  }
}

function syncWindowControls(value) {
  const parsed = Number(value);
  const nextValue =
    Number.isFinite(parsed) && parsed >= 2
      ? String(Math.min(Math.round(parsed), MAX_FORECAST_WINDOW_SIZE))
      : String(value || "");
  if (elements.windowSize && elements.windowSize.value !== nextValue) {
    elements.windowSize.value = nextValue;
  }
  if (elements.windowSizeInline && elements.windowSizeInline.value !== nextValue) {
    elements.windowSizeInline.value = nextValue;
  }
  state.loaded.market = false;
  state.loaded.forecast = false;
  state.loaded.simulator = false;
}

function syncHorizonControls(value) {
  const parsed = Number(value);
  const nextValue =
    Number.isFinite(parsed) && parsed >= 1
      ? String(Math.min(Math.round(parsed), MAX_FORECAST_HORIZON_DAYS))
      : String(value || "");
  if (elements.horizon && elements.horizon.value !== nextValue) {
    elements.horizon.value = nextValue;
  }
  if (elements.forecastHorizon && elements.forecastHorizon.value !== nextValue) {
    elements.forecastHorizon.value = nextValue;
  }
  state.loaded.market = false;
  state.loaded.forecast = false;
  state.loaded.simulator = false;
  renderPortfolioLearning();
}

function markSimulationInputsDirty() {
  state.loaded.simulator = false;
  state.backtest = null;
  renderPortfolioLearning();
}

async function refreshDiagnostics() {
  setLoading(elements.healthBlock);
  setLoading(elements.modelsBlock);
  setLoading(elements.refreshStatusBlock);
  if (elements.dataHealthCards) {
    setLoading(elements.dataHealthCards);
  }
  const [healthResult, modelsResult, refreshResult] = await Promise.allSettled([
    callSectionApi("health", "/api/health"),
    callArtifactSectionApi("models", "/api/models", { artifactAttempts: 4 }),
    callSectionApi("refreshStatus", "/api/data/refresh/status"),
  ]);
  const healthValue =
    healthResult.status === "fulfilled" && modelsResult.status === "fulfilled" && healthResult.value?.artifact_engine?.lazy_enabled
      ? {
          ...healthResult.value,
          error: healthResult.value.ready === false ? healthResult.value.error : null,
          artifact_engine: {
            ...healthResult.value.artifact_engine,
            status: "ready",
            error: null,
          },
        }
      : healthResult.status === "fulfilled"
        ? healthResult.value
        : null;
  elements.healthBlock.textContent =
    healthValue
      ? JSON.stringify(healthValue, null, 2)
      : `Health unavailable: ${healthResult.reason.message}`;
  elements.modelsBlock.textContent =
    modelsResult.status === "fulfilled"
      ? JSON.stringify(modelsResult.value, null, 2)
      : `Model metadata unavailable: ${modelsResult.reason.message}`;
  elements.refreshStatusBlock.textContent =
    refreshResult.status === "fulfilled"
      ? JSON.stringify(refreshResult.value, null, 2)
      : `Refresh status unavailable: ${refreshResult.reason.message}`;
  if (elements.dataHealthCards) {
    state.health = healthValue;
    state.models = modelsResult.status === "fulfilled" ? modelsResult.value : null;
    state.refreshStatus = refreshResult.status === "fulfilled" ? refreshResult.value : null;
    renderDataHealthCards();
  }
}

async function loadUniverse() {
  state.universe = await callSectionApi("universe", "/api/universe");
  renderUniverse(state.universe);
  renderDataHealthCards();
  setBackendEmptyState(false);
}

async function ensureUniverse() {
  if (!state.universe) {
    await loadUniverse();
  }
}

async function loadMarketIndexHistory() {
  const symbol = state.marketIndexSymbol || "SP500";
  const range = state.marketIndexRange || "1y";
  if (elements.marketIndexSummary) {
    setLoading(elements.marketIndexSummary);
  }
  if (elements.marketIndexChartFallback) {
    elements.marketIndexChartFallback.textContent = "Loading index history...";
  }
  try {
    const result = await callSectionApi(
      "marketIndexHistory",
      `/api/market/indices/${encodeURIComponent(symbol)}/history?range=${encodeURIComponent(range)}`,
    );
    renderMarketIndexHistory(result);
    return result;
  } catch (error) {
    if (isAbort(error)) return null;
    setError(elements.marketIndexSummary, `Market index history unavailable: ${error.message}`);
    if (elements.marketIndexChartFallback) {
      elements.marketIndexChartFallback.textContent = "";
    }
    if (state.marketIndexChart) {
      state.marketIndexChart.destroy();
      state.marketIndexChart = null;
    }
    throw error;
  }
}

async function runMarketForecast() {
  setLoading(elements.marketTable);
  setLoading(elements.marketHighlights);
  setLoading(elements.marketIndices);
  setLoading(elements.marketIndexSummary);
  if (elements.marketIndexChartFallback) {
    elements.marketIndexChartFallback.textContent = "Loading index history...";
  }
  setLoading(elements.marketInsightList);
  setLoading(elements.topOpportunities);
  setLoading(elements.sentimentReasons);
  setLoading(elements.marketLessonCards);
  let indexLoaded = false;
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
      indexLoaded = true;
      loadMarketIndexHistory().catch((error) => {
        if (!isAbort(error)) showToast(error.message, "error");
      });
    } else {
      setError(elements.marketIndices, `Market index data unavailable: ${indexResult.reason.message}`);
      setError(elements.marketIndexSummary, "Market index history unavailable.");
      if (elements.marketIndexChartFallback) {
        elements.marketIndexChartFallback.textContent = "";
      }
    }
    if (marketResult.status === "rejected") {
      throw marketResult.reason;
    }
    renderMarket(marketResult.value);
    setBackendEmptyState(false);
  } catch (error) {
    if (isAbort(error)) return;
    setError(elements.marketTable, `Market forecast unavailable: ${error.message}`);
    setError(elements.marketHighlights, "Market highlights unavailable.");
    if (!indexLoaded) {
      setError(elements.marketIndices, "Market index data unavailable.");
      setError(elements.marketIndexSummary, "Market index history unavailable.");
      if (elements.marketIndexChartFallback) {
        elements.marketIndexChartFallback.textContent = "";
      }
    }
    setError(elements.marketInsightList, "Market insights unavailable.");
    setError(elements.topOpportunities, "Top opportunities unavailable.");
    setError(elements.marketLessonCards, "Market lesson unavailable.");
    throw error;
  }
}

async function runTickerForecast(ticker = elements.tickerSelect.value) {
  setLoading(elements.tickerMetrics);
  if (elements.tickerDataBadges) setLoading(elements.tickerDataBadges);
  if (elements.forecastChange) setLoading(elements.forecastChange);
  setLoading(elements.tickerProfile);
  setLoading(elements.scenarioPathSummary);
  setLoading(elements.tickerInsights);
  setLoading(elements.forecastLessonContent);
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
    setBackendEmptyState(false);
  } catch (error) {
    if (isAbort(error)) return;
    setError(elements.tickerMetrics, `Ticker forecast unavailable: ${error.message}`);
    if (elements.tickerDataBadges) setError(elements.tickerDataBadges, "Data confidence unavailable.");
    if (elements.forecastChange) setError(elements.forecastChange, "Forecast change unavailable.");
    setError(elements.tickerProfile, "Company data unavailable.");
    setError(elements.scenarioPathSummary, "Scenario summary unavailable.");
    setError(elements.tickerInsights, "Ticker insights unavailable.");
    setError(elements.forecastLessonContent, "Forecast lesson unavailable.");
    throw error;
  }
}

async function runSimulation() {
  setLoading(elements.simulationSummary);
  setLoading(elements.simulationClasses);
  setLoading(elements.simulationAssets);
  setLoading(elements.simulationTrades);
  if (elements.benchmarkComparison) setLoading(elements.benchmarkComparison);
  if (elements.allocationWhy) setLoading(elements.allocationWhy);
  setLoading(elements.simulationWarnings);
  setLoading(elements.portfolioClassroom);
  const selected = selectedSimulationTickers();
  try {
    const result = await callSectionApi("simulation", "/api/portfolio/simulations", {
      method: "POST",
      body: JSON.stringify({
        amount: Number(elements.amount.value),
        ...dashboardPayload({
          selected_tickers: selected.length > 0 ? selected : null,
          ...portfolioConstraintPayload(),
        }),
      }),
    });
    renderSimulation(result);
    setBackendEmptyState(false);
  } catch (error) {
    if (isAbort(error)) return;
    setError(elements.simulationSummary, `Simulation unavailable: ${error.message}`);
    if (elements.benchmarkComparison) setError(elements.benchmarkComparison, "Benchmark comparison unavailable.");
    if (elements.allocationWhy) setError(elements.allocationWhy, "Allocation explanation unavailable.");
    setError(elements.simulationWarnings, "Simulation insights unavailable.");
    setError(elements.portfolioClassroom, "Portfolio classroom unavailable.");
    throw error;
  }
}

async function runRlAllocation() {
  setLoading(elements.rlSummary);
  try {
    const result = await callArtifactSectionApi("rlAllocation", "/api/inference", {
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
    const result = await callArtifactSectionApi("backtest", "/api/backtests", {
      method: "POST",
      body: JSON.stringify({
        initial_amount: Number(elements.amount.value),
        risk: Number(elements.risk.value),
        window_size: selectedWindowSize(),
        max_steps: Math.min(selectedHorizonDays(), MAX_RL_BACKTEST_STEPS),
        include_trade_log: false,
        strict_validation: true,
      }),
    });
    state.backtest = result;
    renderBacktest(result);
    renderDataHealthCards();
  } catch (error) {
    if (isAbort(error)) return;
    state.backtest = null;
    renderDataHealthCards();
    setError(elements.backtestSummary, `Backtest unavailable: ${error.message}`);
    throw error;
  }
}

function activeTabName() {
  return document.querySelector(".view.is-active")?.id || "market";
}

function resetLoadedViews() {
  state.loaded = {
    market: false,
    forecast: false,
    simulator: false,
    diagnostics: false,
  };
  state.backtest = null;
  state.marketIndexAsOf = null;
  state.marketIndexHistory = null;
}

async function refreshDiagnosticsInBackground() {
  if (state.loaded.diagnostics) return;
  try {
    await refreshDiagnostics();
    state.loaded.diagnostics = true;
  } catch (error) {
    if (isAbort(error)) return;
  }
}

async function refreshActiveView({ force = false } = {}) {
  const tabName = activeTabName();
  if (force) {
    state.loaded[tabName] = false;
  }
  if (tabName === "market" && !state.loaded.market) {
    await runMarketForecast();
    state.loaded.market = true;
  } else if (tabName === "forecast" && !state.loaded.forecast) {
    await ensureUniverse();
    const ticker = state.pendingForecastTicker || elements.tickerSelect.value;
    await runTickerForecast(ticker);
    state.pendingForecastTicker = null;
    state.loaded.forecast = true;
  } else if (tabName === "simulator" && !state.loaded.simulator) {
    await ensureUniverse();
    await runSimulation();
    state.loaded.simulator = true;
  } else if (tabName === "project") {
    await ensureUniverse();
    await refreshDiagnosticsInBackground();
    renderProjectStory();
  }
}

async function refreshDashboard() {
  elements.apiStatus.textContent = "Refreshing current view...";
  elements.apiStatus.className = "muted";
  setBackendEmptyState(false);
  try {
    await refreshActiveView({ force: true });
  } catch (error) {
    setBackendEmptyState(isBackendConnectionError(error));
    throw error;
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
    state.health = health;
    renderDataHealthCards();
    setBackendEmptyState(false);
  } catch (error) {
    if (isAbort(error)) return;
    elements.apiStatus.textContent = `Set your backend URL, then click Use Backend. ${error.message}`;
    elements.apiStatus.className = "status-bad";
    setError(elements.marketTable, "Backend is not connected.");
    setError(elements.marketIndices, "Backend is not connected.");
    setError(elements.marketIndexSummary, "Backend is not connected.");
    if (elements.marketIndexChartFallback) {
      elements.marketIndexChartFallback.textContent = "";
    }
    setError(elements.tickerMetrics, "Backend is not connected.");
    setError(elements.simulationSummary, "Backend is not connected.");
    setBackendEmptyState(true);
    return;
  }

  try {
    await refreshActiveView({ force: true });
    elements.apiStatus.textContent = "Backend connected.";
    elements.apiStatus.className = "status-good";
  } catch (error) {
    if (isAbort(error)) return;
    setBackendEmptyState(false);
    elements.apiStatus.textContent = `Backend connected, but dashboard refresh failed: ${error.message}`;
    elements.apiStatus.className = "status-bad";
  }
}

function switchTab(tabName) {
  document.querySelectorAll(".nav-item, .mobile-nav-item").forEach((n) => n.classList.remove("is-active"));
  document.querySelectorAll(".view").forEach((n) => n.classList.remove("is-active"));
  document.querySelectorAll(`[data-tab="${tabName}"]`).forEach((n) => n.classList.add("is-active"));
  const view = document.querySelector(`#${tabName}`);
  if (view) view.classList.add("is-active");
  closeMobileMenu();
  const shell = document.querySelector(".app-shell");
  shell?.scrollTo({ top: 0, left: 0, behavior: "auto" });
  window.scrollTo({ top: 0, left: 0, behavior: "auto" });
  if (tabName === "forecast" && state.chart) {
    requestAnimationFrame(() => state.chart.resize());
  }
  if (tabName === "market" && state.marketIndexChart) {
    requestAnimationFrame(() => state.marketIndexChart.resize());
  }
  refreshActiveView().catch((error) => {
    if (!isAbort(error)) showToast(error.message, "error");
  });
}

function setMobileMenu(open) {
  const sidebar = document.querySelector(".sidebar");
  sidebar?.classList.toggle("is-mobile-open", open);
  document.body.classList.toggle("mobile-sidebar-open", open);
  elements.mobileMenuToggle?.setAttribute("aria-expanded", String(open));
  if (elements.mobileMenuToggle) {
    elements.mobileMenuToggle.textContent = open ? "×" : "☰";
  }
}

function closeMobileMenu() {
  setMobileMenu(false);
}

document.querySelectorAll(".nav-item, .mobile-nav-item").forEach((btn) => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

elements.mobileMenuToggle?.addEventListener("click", (event) => {
  event.stopPropagation();
  const isOpen = document.querySelector(".sidebar")?.classList.contains("is-mobile-open");
  setMobileMenu(!isOpen);
});

document.addEventListener("click", (event) => {
  const sidebar = document.querySelector(".sidebar");
  if (!sidebar?.classList.contains("is-mobile-open")) return;
  if (sidebar.contains(event.target) || elements.mobileMenuToggle?.contains(event.target)) return;
  closeMobileMenu();
});

document.querySelectorAll(".range-button").forEach((button) => {
  if (button.dataset.indexRange) return;
  button.addEventListener("click", () => {
    document
      .querySelectorAll(".range-button:not([data-index-range])")
      .forEach((node) => node.classList.remove("is-active"));
    button.classList.add("is-active");
    state.chartRange = button.dataset.range;
    if (state.currentForecast) {
      renderForecastChart(state.currentForecast);
    }
  });
});

document.querySelectorAll("[data-index-range]").forEach((button) => {
  button.addEventListener("click", () => {
    document
      .querySelectorAll("[data-index-range]")
      .forEach((node) => node.classList.remove("is-active"));
    button.classList.add("is-active");
    state.marketIndexRange = button.dataset.indexRange;
    loadMarketIndexHistory().catch((error) => {
      if (!isAbort(error)) showToast(error.message, "error");
    });
  });
});

elements.marketIndexSelect?.addEventListener("change", () => {
  state.marketIndexSymbol = elements.marketIndexSelect.value || state.marketIndexSymbol;
  loadMarketIndexHistory().catch((error) => {
    if (!isAbort(error)) showToast(error.message, "error");
  });
});

const sentimentGauge = elements.sentimentGaugeArc?.closest(".sentiment-gauge");
sentimentGauge?.addEventListener("pointermove", (event) => {
  const content = sentimentTooltipContent();
  if (content) showHoverTooltip(event, content);
});
sentimentGauge?.addEventListener("pointerleave", hideHoverTooltip);

elements.simulationClasses.addEventListener("pointermove", (event) => {
  const donut = event.target.closest(".allocation-donut");
  if (!donut) {
    hideHoverTooltip();
    return;
  }
  const segment = allocationSegmentAtPoint(event, donut);
  if (segment) showHoverTooltip(event, allocationTooltipContent(segment));
});
elements.simulationClasses.addEventListener("pointerleave", hideHoverTooltip);

elements.marketTable.addEventListener("click", (event) => {
  const button = event.target.closest("[data-view-ticker]");
  if (!button) return;
  state.pendingForecastTicker = button.dataset.viewTicker;
  state.loaded.forecast = false;
  switchTab("forecast");
});

elements.risk.addEventListener("input", () => {
  elements.riskValue.textContent = Number(elements.risk.value).toFixed(2);
  state.loaded.market = false;
  state.loaded.forecast = false;
  state.loaded.simulator = false;
  renderPortfolioLearning();
});

elements.horizon.addEventListener("input", () => {
  syncHorizonControls(elements.horizon.value);
});
elements.forecastHorizon?.addEventListener("input", () => {
  syncHorizonControls(elements.forecastHorizon.value);
});
elements.windowSize?.addEventListener("input", () => {
  syncWindowControls(elements.windowSize.value);
});
elements.windowSizeInline?.addEventListener("input", () => {
  syncWindowControls(elements.windowSizeInline.value);
});
elements.amount.addEventListener("input", () => {
  markSimulationInputsDirty();
});
elements.tickerSelect.addEventListener("change", () => {
  state.loaded.forecast = false;
});
[
  elements.simulationTickers,
  elements.maxSinglePosition,
  elements.maxCryptoWeight,
  elements.minCashWeight,
  elements.preferStock,
  elements.preferEtf,
  elements.preferCrypto,
].forEach((control) => {
  control?.addEventListener("input", markSimulationInputsDirty);
  control?.addEventListener("change", markSimulationInputsDirty);
});

elements.learnModeToggle?.addEventListener("click", (e) => {
  e.preventDefault();
  setLearnMode(!state.learnMode);
});

elements.themeModeToggle?.addEventListener("click", (e) => {
  e.preventDefault();
  setThemeMode(state.themeMode === "dark" ? "light" : "dark");
});

document.querySelectorAll("[data-close-details]").forEach((button) => {
  button.addEventListener("click", (event) => {
    event.preventDefault();
    const details = button.closest("details");
    if (!details) return;
    details.open = false;
    details.querySelector("summary")?.focus();
  });
});

function startApp() {
  if (appStarted) return;
  appStarted = true;
  elements.apiBase.value = state.apiBase;
  elements.riskValue.textContent = Number(elements.risk.value).toFixed(2);
  syncHorizonControls(elements.horizon.value);
  syncWindowControls(elements.windowSize.value);
  setLearnMode(state.learnMode);
  setThemeMode(state.themeMode);
  renderPortfolioLearning();
  renderProjectStory();
  renderGlossary();
  probeBackend();
}

function acceptLegalDisclaimer() {
  elements.legalDisclaimer?.setAttribute("hidden", "");
  elements.legalDisclaimer?.setAttribute("aria-hidden", "true");
  document.body.classList.remove("disclaimer-pending");
  startApp();
  const activeNav = [...document.querySelectorAll(".nav-item.is-active, .mobile-nav-item.is-active")]
    .find((item) => item.offsetParent !== null);
  activeNav?.focus();
  maybeStartTour();
}

elements.acceptDisclaimer?.addEventListener("click", acceptLegalDisclaimer);

elements.saveApiBase.addEventListener("click", async () => {
  state.apiBase = canonicalApiBase(elements.apiBase.value) || defaultApiBase();
  elements.apiBase.value = state.apiBase;
  localStorage.setItem("foresight-api-base", state.apiBase);
  state.universe = null;
  resetLoadedViews();
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
elements.runTickerForecast.addEventListener("click", () => {
  runTickerForecast()
    .then(() => {
      state.loaded.forecast = true;
    })
    .catch((error) => showToast(error.message, "error"));
});
elements.runSimulation.addEventListener("click", () => {
  runSimulation()
    .then(() => {
      state.loaded.simulator = true;
    })
    .catch((error) => showToast(error.message, "error"));
});
elements.runRlAllocation.addEventListener("click", () =>
  runRlAllocation().catch((error) => showToast(error.message, "error")),
);
elements.runBacktest.addEventListener("click", () =>
  runBacktest().catch((error) => showToast(error.message, "error")),
);

startApp();
elements.acceptDisclaimer?.focus();

document.body.addEventListener("mouseover", (event) => {
  const chip = event.target.closest(".glossary-chip");
  if (chip) {
    const key = chip.dataset.term;
    const term = glossary[key];
    if (term) {
      showHoverTooltip(event, `<strong>${escapeHtml(term.title)}</strong><span>${escapeHtml(term.definition)}</span>`);
    }
  }
});

document.body.addEventListener("mouseout", (event) => {
  if (event.target.closest(".glossary-chip")) {
    hideHoverTooltip();
  }
});

// ── Command Palette ──
const cmdOverlay = document.getElementById("commandPalette");
const cmdInput = document.getElementById("cmdInput");
const cmdResults = document.getElementById("cmdResults");

function openCommandPalette() {
  if (!cmdOverlay) return;
  cmdOverlay.classList.add("is-open");
  cmdOverlay.setAttribute("aria-hidden", "false");
  cmdInput.value = "";
  cmdInput.focus();
  renderCmdResults("");
}

function closeCommandPalette() {
  if (!cmdOverlay) return;
  cmdOverlay.classList.remove("is-open");
  cmdOverlay.setAttribute("aria-hidden", "true");
}

function renderCmdResults(query) {
  if (!cmdResults) return;
  const q = query.toLowerCase().trim();
  const results = [];
  const tabs = [
    { label: "Market overview", tab: "market", icon: "market" },
    { label: "Ticker forecast", tab: "forecast", icon: "forecast" },
    { label: "Portfolio simulator", tab: "simulator", icon: "simulator" },
    { label: "About", tab: "project", icon: "project" },
  ];
  tabs.forEach((t) => {
    if (!q || t.label.toLowerCase().includes(q)) {
      results.push({ ...t, type: "Tab" });
    }
  });
  Object.entries(glossary).forEach(([key, term]) => {
    if (!q || term.title.toLowerCase().includes(q) || term.definition.toLowerCase().includes(q)) {
      results.push({ label: term.title, icon: "project", type: "Glossary", key });
    }
  });
  if (state.universe) {
    state.universe.tickers.forEach((entry) => {
      if (!q || entry.ticker.toLowerCase().includes(q)) {
        results.push({ label: entry.ticker, icon: "forecast", type: "Ticker", ticker: entry.ticker });
      }
    });
  }
  cmdResults.innerHTML = results.slice(0, 12).map((r) => `
    <div class="cmd-result-item" data-cmd-type="${escapeHtml(r.type)}" data-cmd-value="${escapeHtml(r.tab || r.key || r.ticker || "")}">
      ${iconSvg(r.icon, "cmd-result-icon")}
      <span>${escapeHtml(r.label)}</span>
      <span class="cmd-result-type">${escapeHtml(r.type)}</span>
    </div>
  `).join("") || `<div class="cmd-result-item"><span>No results</span></div>`;
}

cmdInput?.addEventListener("input", () => renderCmdResults(cmdInput.value));
cmdOverlay?.addEventListener("click", (e) => {
  if (e.target === cmdOverlay) closeCommandPalette();
});
cmdResults?.addEventListener("click", (e) => {
  const item = e.target.closest(".cmd-result-item");
  if (!item) return;
  const type = item.dataset.cmdType;
  const value = item.dataset.cmdValue;
  if (type === "Tab" && value) switchTab(value);
  if (type === "Ticker" && value) {
    switchTab("forecast");
    elements.tickerSelect.value = value;
    runTickerForecast(value).catch((err) => showToast(err.message, "error"));
  }
  if (type === "Glossary" && value) {
    switchTab("project");
    requestAnimationFrame(() => {
      const term = document.querySelector(`[data-glossary-key="${CSS.escape(value)}"]`);
      if (term) {
        term.scrollIntoView({ behavior: "smooth", block: "center" });
        term.classList.add("highlight-pulse");
        setTimeout(() => term.classList.remove("highlight-pulse"), 1500);
      } else {
        elements.glossaryList?.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  }
  closeCommandPalette();
});

document.getElementById("openCmdPalette")?.addEventListener("click", openCommandPalette);

// ── Keyboard Shortcuts ──
document.addEventListener("keydown", (e) => {
  if (document.body.classList.contains("disclaimer-pending")) return;
  if (cmdOverlay?.classList.contains("is-open")) {
    if (e.key === "Escape") { closeCommandPalette(); return; }
    return;
  }
  const tag = document.activeElement?.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
  if ((e.metaKey || e.ctrlKey) && e.key === "k") {
    e.preventDefault();
    openCommandPalette();
    return;
  }
  if (e.key === "Escape") closeMobileMenu();
  if (e.key === "1") switchTab("market");
  if (e.key === "2") switchTab("forecast");
  if (e.key === "3") switchTab("simulator");
  if (e.key === "4") switchTab("project");
  if (e.key.toLowerCase() === "l") setLearnMode(!state.learnMode);
  if (e.key.toLowerCase() === "d") setThemeMode(state.themeMode === "dark" ? "light" : "dark");
});

// ── Why Button Click Handler ──
document.body.addEventListener("click", (e) => {
  const btn = e.target.closest(".why-btn");
  if (!btn) return;
  const container = btn.closest(".metric, .why-container");
  if (!container) return;
  const slot = container.querySelector(".why-popover-slot");
  if (!slot) return;
  if (slot.innerHTML) {
    slot.innerHTML = "";
  } else {
    slot.innerHTML = `<div class="why-popover">${escapeHtml(btn.dataset.why)}</div>`;
  }
});

// ── Guided Tour (driver.js) ──
const TOUR_SEEN_KEY = "foresight-tour-completed";

function buildTourDriver() {
  const { driver } = window.driver.js;
  const isMobile = window.matchMedia("(max-width: 768px)").matches;
  const tabSelector = (tab) =>
    isMobile ? `.mobile-nav-item[data-tab="${tab}"]` : `.nav-item[data-tab="${tab}"]`;

  const side = isMobile ? "bottom" : "right";
  return driver({
    showProgress: true,
    animate: true,
    smoothScroll: true,
    overlayColor: "rgba(0, 0, 0, 0.65)",
    stagePadding: 6,
    stageRadius: 10,
    popoverClass: "foresight-tour-popover",
    nextBtnText: "Next",
    prevBtnText: "Back",
    doneBtnText: "Done",
    steps: [
      {
        element: tabSelector("market"),
        popover: {
          title: "Market Overview",
          description: "Start here. Live index data, ranked opportunities, and market sentiment update when the backend connects.",
          side,
          align: "start",
        },
      },
      {
        element: tabSelector("forecast"),
        popover: {
          title: "Ticker Forecast",
          description: "Pick any ticker to see bear, base, and bull scenario forecasts with confidence scoring.",
          side,
          align: "start",
        },
      },
      {
        element: tabSelector("simulator"),
        popover: {
          title: "Portfolio Simulator",
          description: "Set a dollar amount and risk level, then simulate a diversified portfolio with trade plans and benchmarks.",
          side,
          align: "start",
        },
      },
      {
        element: tabSelector("project"),
        popover: {
          title: "About",
          description: "Project details, architecture, glossary, and transparency notes about how Foresight works.",
          side,
          align: "start",
        },
      },
      {
        element: isMobile ? undefined : "#openCmdPalette",
        popover: {
          title: "Quick Search",
          description: "Search tickers, tabs, and glossary terms. You can also press Cmd+K anytime.",
          side: "right",
          align: "start",
        },
      },
      {
        element: isMobile ? undefined : "#learnModeToggle",
        popover: {
          title: "Learn Mode",
          description: "Toggle this to show educational cards, glossary terms, and explanations in every tab.",
          side: "right",
          align: "start",
        },
      },
      {
        element: isMobile ? undefined : "#themeModeToggle",
        popover: {
          title: "Theme",
          description: "Switch between dark and light mode to suit your preference.",
          side: "right",
          align: "start",
        },
      },
      {
        element: isMobile ? undefined : ".settings-menu",
        popover: {
          title: "Settings",
          description: "Configure the backend URL and estimator window. You can restart this tour from here.",
          side: "right",
          align: "start",
        },
      },
    ].filter((step) => step.element !== undefined),
    onDestroyed: () => {
      localStorage.setItem(TOUR_SEEN_KEY, "true");
    },
  });
}

function startGuidedTour() {
  if (!window.driver?.js?.driver) return;
  const tourDriver = buildTourDriver();
  tourDriver.drive();
}

function maybeStartTour() {
  if (localStorage.getItem(TOUR_SEEN_KEY)) return;
  setTimeout(startGuidedTour, 500);
}

document.getElementById("restartTour")?.addEventListener("click", () => {
  const details = document.querySelector(".settings-menu");
  if (details) details.open = false;
  startGuidedTour();
});
