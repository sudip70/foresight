const DEPLOYED_API_BASE = "https://stockify-backend-adc6.onrender.com";
const LOCAL_API_BASE = "http://localhost:8000";
const API_TIMEOUT_MS = 45_000;

function normalizeApiBase(value) {
  return String(value || "").trim().replace(/\/$/, "");
}

function queryApiBase() {
  try {
    return new URLSearchParams(window.location.search).get("apiBase");
  } catch {
    return "";
  }
}

function isLocalHost(host) {
  return !host || host === "localhost" || host === "127.0.0.1" || host === "::1" || host === "[::1]";
}

function isLocalApiBase(value) {
  try {
    return isLocalHost(new URL(value).hostname);
  } catch {
    return false;
  }
}

function defaultApiBase() {
  const host = window.location?.hostname || "";
  return isLocalHost(host) ? LOCAL_API_BASE : DEPLOYED_API_BASE;
}

function initialApiBase() {
  const explicitApiBase = queryApiBase() || window.FORESIGHT_API_BASE;
  if (explicitApiBase) return normalizeApiBase(explicitApiBase);

  const savedApiBase = normalizeApiBase(localStorage.getItem("foresight-api-base"));
  const host = window.location?.hostname || "";
  if (savedApiBase && (isLocalHost(host) || !isLocalApiBase(savedApiBase))) {
    return savedApiBase;
  }
  return defaultApiBase();
}

const elements = {
  apiBase: document.querySelector("#apiBase"),
  apiStatus: document.querySelector("#apiStatus"),
  saveApiBase: document.querySelector("#saveApiBase"),
  mobileMenuToggle: document.querySelector("#mobileMenuToggle"),
  amount: document.querySelector("#amount"),
  risk: document.querySelector("#risk"),
  riskValue: document.querySelector("#riskValue"),
  horizon: document.querySelector("#horizon"),
  forecastHorizon: document.querySelector("#forecastHorizon"),
  windowSize: document.querySelector("#windowSize"),
  refreshDashboard: document.querySelector("#refreshDashboard"),
  learnModeToggle: document.querySelector("#learnModeToggle"),
  themeModeToggle: document.querySelector("#themeModeToggle"),
  marketAsOf: document.querySelector("#marketAsOf"),
  marketIndices: document.querySelector("#marketIndices"),
  marketIndexSelect: document.querySelector("#marketIndexSelect"),
  marketIndexDateRange: document.querySelector("#marketIndexDateRange"),
  marketIndexSummary: document.querySelector("#marketIndexSummary"),
  marketIndexChart: document.querySelector("#marketIndexChart"),
  marketIndexChartFallback: document.querySelector("#marketIndexChartFallback"),
  marketHighlights: document.querySelector("#marketHighlights"),
  marketLessonCards: document.querySelector("#marketLessonCards"),
  macroSnapshot: document.querySelector("#macroSnapshot"),
  marketTable: document.querySelector("#marketTable"),
  sentimentGaugeArc: document.querySelector("#sentimentGaugeArc"),
  sentimentScore: document.querySelector("#sentimentScore"),
  sentimentLabel: document.querySelector("#sentimentLabel"),
  sentimentReasons: document.querySelector("#sentimentReasons"),
  marketInsightList: document.querySelector("#marketInsightList"),
  topOpportunities: document.querySelector("#topOpportunities"),
  dataHealthCards: document.querySelector("#dataHealthCards"),
  tickerSelect: document.querySelector("#tickerSelect"),
  runTickerForecast: document.querySelector("#runTickerForecast"),
  forecastDateRange: document.querySelector("#forecastDateRange"),
  forecastChart: document.querySelector("#forecastChart"),
  chartFallback: document.querySelector("#chartFallback"),
  tickerMetricTitle: document.querySelector("#tickerMetricTitle"),
  tickerMetrics: document.querySelector("#tickerMetrics"),
  forecastLessonContent: document.querySelector("#forecastLessonContent"),
  scenarioHorizonLabel: document.querySelector("#scenarioHorizonLabel"),
  scenarioPathSummary: document.querySelector("#scenarioPathSummary"),
  tickerInsights: document.querySelector("#tickerInsights"),
  tickerProfile: document.querySelector("#tickerProfile"),
  tickerAboutTitle: document.querySelector("#tickerAboutTitle"),
  tickerNarrative: document.querySelector("#tickerNarrative"),
  simulationTickers: document.querySelector("#simulationTickers"),
  runSimulation: document.querySelector("#runSimulation"),
  simulationSummary: document.querySelector("#simulationSummary"),
  simulationWarnings: document.querySelector("#simulationWarnings"),
  portfolioClassroom: document.querySelector("#portfolioClassroom"),
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
  projectStats: document.querySelector("#projectStats"),
  projectHighlights: document.querySelector("#projectHighlights"),
  projectTransparency: document.querySelector("#projectTransparency"),
  glossaryList: document.querySelector("#glossaryList"),
  footerDataAsOf: document.querySelector("#footerDataAsOf"),
};

const state = {
  apiBase: initialApiBase(),
  universe: null,
  health: null,
  models: null,
  refreshStatus: null,
  simulation: null,
  backtest: null,
  chart: null,
  chartRange: "6m",
  marketIndexChart: null,
  marketIndexRange: "1y",
  marketIndexSymbol: "SP500",
  marketIndexAsOf: null,
  marketIndexHistory: null,
  currentForecast: null,
  pendingForecastTicker: null,
  sentiment: null,
  classAllocationSegments: [],
  learnMode: localStorage.getItem("foresight-learn-mode") === "true",
  themeMode: localStorage.getItem("foresight-theme-mode") || "dark",
  controllers: {},
  loaded: {
    market: false,
    forecast: false,
    simulator: false,
    diagnostics: false,
  },
  progress: {
    actions: { market: false, forecast: false, simulation: false },
    level: 1,
  },
};

const progressLevelLabels = {
  1: "Level 1: Novice",
  2: "Level 2: Intermediate",
  3: "Level 3: Advanced",
};

const glossary = {
  bear: {
    title: "Bear scenario",
    definition: "A weaker outcome that estimates what could happen if price and volatility move against the asset.",
  },
  base: {
    title: "Base scenario",
    definition: "The central estimate. It is useful for comparison, but it is not a guaranteed target.",
  },
  bull: {
    title: "Bull scenario",
    definition: "A stronger outcome that estimates potential upside if conditions are favorable.",
  },
  volatility: {
    title: "Volatility",
    definition: "A measure of how much prices may swing. Higher volatility usually means wider scenario ranges.",
  },
  drawdown: {
    title: "Max drawdown",
    definition: "The largest historical fall from a previous high. It helps explain downside pain, not just average return.",
  },
  confidence: {
    title: "Confidence",
    definition: "A model-readiness score that falls when history is thin, volatility is high, or scenario bands are wide.",
  },
  sharpe: {
    title: "Sharpe ratio",
    definition: "A risk-adjusted return measure. Higher values mean more return per unit of volatility in the backtest.",
  },
  diversification: {
    title: "Diversification",
    definition: "Spreading exposure across assets so one holding does not dominate the portfolio outcome.",
  },
  cash: {
    title: "Cash buffer",
    definition: "A lower-risk sleeve the simulator can hold when downside risk or uncertainty is elevated.",
  },
  spread: {
    title: "Scenario spread",
    definition: "The distance between bear and bull outcomes. A wider spread means the model sees more uncertainty.",
  },
  freshness: {
    title: "Data freshness",
    definition: "The latest market date used by the model. Fresh data matters because forecasts start from recent prices.",
  },
};

const literacy = Object.fromEntries(
  Object.entries(glossary).map(([key, value]) => [key, value.definition]),
);

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

function formatDate(value) {
  if (!value) return "Unavailable";
  const normalized = /^\d{4}-\d{2}-\d{2}$/.test(String(value)) ? `${value}T00:00:00` : value;
  const parsed = new Date(normalized);
  if (Number.isNaN(parsed.getTime())) return value;
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(parsed);
}

function formatDateTime(value) {
  if (!value) return "Unavailable";
  if (/^\d{4}-\d{2}-\d{2}$/.test(String(value))) {
    return formatDate(value);
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(parsed);
}

function formatInteger(value) {
  if (isMissing(value)) return "--";
  return new Intl.NumberFormat("en-US").format(Number(value));
}

function classCoverageLabel(universe) {
  const groups = universe?.asset_classes || [];
  if (!groups.length) return "assets tracked";
  return groups
    .map((group) => `${formatInteger(group.tickers?.length || 0)} ${group.asset_class}`)
    .join(" / ");
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => {
    const entities = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#039;",
    };
    return entities[character];
  });
}

function hasProfileValue(value) {
  if (value === null || value === undefined) return false;
  if (typeof value === "string") return value.trim() !== "";
  if (typeof value === "number") return Number.isFinite(value);
  return true;
}

function profileText(value) {
  return hasProfileValue(value) ? String(value).trim() : "";
}

function signedPercentLabel(value) {
  if (isMissing(value)) return "Unavailable";
  const parsed = Number(value);
  return `${parsed >= 0 ? "+" : ""}${formatPercent(parsed)}`;
}

function assetClassNoun(value) {
  const normalized = String(value || "").toLowerCase();
  if (normalized === "etf") return "an ETF";
  if (normalized === "crypto") return "a crypto asset";
  if (normalized === "stock") return "a stock";
  return normalized ? `a ${normalized}` : "an asset";
}

function buildTickerAboutParagraph(profile, forecast) {
  const fields = profile?.fields || {};
  const ticker = forecast?.ticker || profile?.ticker || "this ticker";
  const name = profileText(profile?.display_name) || ticker;
  const assetClass = String(profile?.asset_class || forecast?.asset_class || "").toLowerCase();
  const sector = profileText(fields.sector);
  const industry = profileText(fields.industry);
  const exchange = profileText(fields.exchange);
  const country = profileText(fields.country);
  const categoryParts = [];

  if (assetClass === "stock") {
    if (sector) categoryParts.push(`in the ${sector} sector`);
    if (industry) categoryParts.push(`categorized under ${industry}`);
  } else if (assetClass === "etf") {
    if (industry) categoryParts.push(`with ${industry} exposure`);
    if (sector && sector.toLowerCase() !== "etf") categoryParts.push(`focused on ${sector}`);
  } else if (assetClass === "crypto") {
    if (industry) categoryParts.push(`categorized as ${industry}`);
    if (sector && sector.toLowerCase() !== "crypto") categoryParts.push(`in the ${sector} sector`);
  } else if (sector || industry) {
    if (sector) categoryParts.push(`in the ${sector} sector`);
    if (industry) categoryParts.push(`categorized under ${industry}`);
  }

  const category = categoryParts.length ? ` ${categoryParts.join(" and ")}` : "";
  const listingParts = [];
  if (exchange) listingParts.push(`trades on ${exchange}`);
  if (country) listingParts.push(`is listed in ${country}`);
  const listing = listingParts.length ? ` ${name} ${listingParts.join(" and ")}.` : "";
  const horizon = forecast?.horizon_days ? `${forecast.horizon_days}-day` : "current";
  const risk = profileText(forecast?.risk_label).toLowerCase() || "model-estimated";
  const confidence = profileText(forecast?.confidence_label).toLowerCase() || "available";

  return `${name} (${ticker}) is ${assetClassNoun(assetClass)}${category}.${listing} The current ${horizon} base scenario is ${signedPercentLabel(forecast?.returns?.base)} with ${risk} risk and ${confidence} confidence. This overview is for learning context and should be read alongside the scenario chart and profile metrics.`;
}

function renderTickerNarrative(profile, forecast, warnings = []) {
  const about = profile ? `<p class="about-summary">${escapeHtml(buildTickerAboutParagraph(profile, forecast))}</p>` : "";
  const warningBlock = warnings.length
    ? `<div class="freshness-list">${warnings.map((warning) => `<span>${escapeHtml(warning)}</span>`).join("")}</div>`
    : "";
  elements.tickerNarrative.innerHTML = `
    ${about}
    <p>${escapeHtml(forecast.plain_language)}</p>
    <p class="muted">${escapeHtml(forecast.literacy.bear_base_bull)}</p>
    <div class="freshness-list">
      <span>Market data as of ${escapeHtml(formatDate(forecast.data_as_of || forecast.latest_date))}</span>
      <span>Forecast source: ${forecast.snapshot_used ? "stored daily snapshot" : "computed on request"}</span>
    </div>
    ${warningBlock}
  `;
}

function toneForValue(value) {
  return Number(value || 0) >= 0 ? "positive" : "negative";
}

function hoverTooltip() {
  let tooltip = document.querySelector(".hover-tooltip");
  if (!tooltip) {
    tooltip = document.createElement("div");
    tooltip.className = "hover-tooltip";
    document.body.appendChild(tooltip);
  }
  return tooltip;
}

function moveHoverTooltip(event) {
  const tooltip = hoverTooltip();
  tooltip.style.left = `${event.clientX}px`;
  tooltip.style.top = `${event.clientY}px`;
}

function showHoverTooltip(event, content) {
  const tooltip = hoverTooltip();
  tooltip.innerHTML = content;
  moveHoverTooltip(event);
  tooltip.classList.add("is-visible");
}

function hideHoverTooltip() {
  const tooltip = document.querySelector(".hover-tooltip");
  if (tooltip) {
    tooltip.classList.remove("is-visible");
  }
}

function sentimentTooltipContent() {
  if (!state.sentiment) return "";
  return `
    <strong>Market sentiment: ${state.sentiment.label}</strong>
    <span>Score: ${state.sentiment.score}/100</span>
    <span>Base scenario average: ${signedPercentLabel(state.sentiment.baseAverage)}</span>
    <span>Bear scenario average: ${signedPercentLabel(state.sentiment.bearAverage)}</span>
    <span>Confidence average: ${formatPercent(state.sentiment.confidenceAverage)}</span>
  `;
}

function allocationSegmentAtPoint(event, donut) {
  const segments = state.classAllocationSegments || [];
  if (!segments.length) return null;
  const rect = donut.getBoundingClientRect();
  const centerX = rect.left + rect.width / 2;
  const centerY = rect.top + rect.height / 2;
  const dx = event.clientX - centerX;
  const dy = event.clientY - centerY;
  const angle = (Math.atan2(dy, dx) * (180 / Math.PI) + 450) % 360;
  const position = angle / 360;
  return segments.find((segment) => position >= segment.start && position <= segment.end) || segments.at(-1);
}

function allocationTooltipContent(segment) {
  if (!segment) return "";
  return `
    <strong>${segment.label}</strong>
    <span>${formatPercent(segment.weight)} of portfolio</span>
    <span>${formatCurrency(segment.amount || 0)}</span>
  `;
}

function sparklineSvg(change) {
  const positive = Number(change || 0) >= 0;
  const path = positive
    ? "M2 56 L10 50 L18 54 L27 39 L36 34 L45 38 L54 31 L63 36 L72 28 L81 22 L90 12 L102 6"
    : "M2 18 L11 16 L20 21 L29 18 L38 26 L47 24 L56 31 L65 33 L74 38 L83 40 L92 48 L102 55";
  return `
    <svg class="sparkline" viewBox="0 0 104 64" aria-hidden="true">
      <path class="${positive ? "positive-line" : "negative-line"}" d="${path}" />
    </svg>
  `;
}

function insight(message, tone = "info") {
  return `<article class="insight ${tone}"><span>${message}</span></article>`;
}

function metricCard(label, value, tooltip, subtitle = "", valueTone = "") {
  const tip = tooltip ? `<button class="why-btn" data-why="${label}: ${tooltip}" type="button">?</button>` : "";
  const detail = subtitle ? `<span class="metric-subtitle">${subtitle}</span>` : "";
  const toneClass = valueTone ? ` ${valueTone}` : "";
  return `
    <article class="metric">
      <h3>${label} ${tip}</h3>
      <strong class="animate-number${toneClass}">${value}</strong>
      ${detail}
      <div class="why-popover-slot"></div>
    </article>
  `;
}

function lessonCard(title, body, detail = "") {
  return `
    <article class="lesson-card">
      <strong>${title}</strong>
      <p>${body}</p>
      ${detail ? `<small>${detail}</small>` : ""}
    </article>
  `;
}

function termChip(key) {
  const term = glossary[key];
  if (!term) return "";
  return `<span class="glossary-chip" data-term="${key}">${term.title}</span>`;
}

function cssVar(name, fallback = "") {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

function riskPreferenceLabel(value) {
  const risk = Number(value || 0);
  if (risk >= 0.75) return "aggressive";
  if (risk >= 0.45) return "balanced";
  return "conservative";
}

function confidenceExplanation(confidence, label = "") {
  const value = Number(confidence || 0);
  if (value >= 0.7) {
    return `${label || "High"} confidence means the model sees relatively stable data and a narrower scenario band.`;
  }
  if (value >= 0.45) {
    return `${label || "Medium"} confidence means the forecast is usable for comparison, but the range still deserves attention.`;
  }
  return `${label || "Low"} confidence means the scenario is more uncertain, so the educational focus should be risk and assumptions.`;
}

function iconSvg(name, className = "ui-icon") {
  return `<svg class="${className}" aria-hidden="true"><use href="#icon-${name}"></use></svg>`;
}

function showToast(message, type = "info") {
  const container = document.getElementById("toastContainer");
  if (!container) return;
  const icons = { info: "info-circle", error: "x-circle", success: "check-circle" };
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `${iconSvg(icons[type] || icons.info, "toast-icon")}<span>${message}</span>`;
  container.appendChild(toast);
  setTimeout(() => { toast.style.opacity = "0"; toast.style.transition = "opacity 200ms"; setTimeout(() => toast.remove(), 250); }, 4000);
}

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

function renderGlossary() {
  if (!elements.glossaryList) return;
  elements.glossaryList.innerHTML = Object.values(glossary)
    .map(
      (term) => `
        <article class="glossary-item">
          <strong>${term.title}</strong>
          <span>${term.definition}</span>
        </article>
      `,
    )
    .join("");
}

function setLoading(target, message = "Loading...") {
  target.innerHTML = `
    <div class="skeleton">
      <div class="skeleton-text"></div>
      <div class="skeleton-text"></div>
      <div class="skeleton-text"></div>
    </div>
  `;
}

function setError(target, message) {
  target.innerHTML = `<article class="error-banner">${message}</article>`;
}

async function callApi(path, options = {}) {
  const headers = options.body ? { "Content-Type": "application/json" } : {};
  const response = await fetch(`${state.apiBase}${path}`, {
    ...options,
    headers: {
      ...headers,
      ...(options.headers || {}),
    },
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
  const { timeoutMs: requestTimeoutMs, ...requestOptions } = options;
  const controller = new AbortController();
  let timedOut = false;
  const timeoutMs = Number(requestTimeoutMs || API_TIMEOUT_MS);
  const timeoutId = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  state.controllers[section] = controller;
  try {
    return await callApi(path, {
      ...requestOptions,
      signal: controller.signal,
    });
  } catch (error) {
    if (timedOut && isAbort(error)) {
      throw new Error("Request timed out. Render may still be waking up; try again in a minute.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeoutId);
    if (state.controllers[section] === controller) {
      delete state.controllers[section];
    }
  }
}

function isAbort(error) {
  return error && error.name === "AbortError";
}

function selectedHorizonDays() {
  const parsed = Number(elements.forecastHorizon?.value || elements.horizon.value || 300);
  return Number.isFinite(parsed) && parsed >= 1 ? Math.round(parsed) : 300;
}

function syncHorizonControls(value) {
  const nextValue = String(value || "");
  if (elements.horizon.value !== nextValue) {
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

function dashboardPayload(extra = {}) {
  return {
    horizon_days: selectedHorizonDays(),
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
    duration: selectedHorizonDays(),
    window_size: Number(elements.windowSize.value),
    strict_validation: true,
  };
}

function renderUniverse(universe) {
  const assetClassOrder = { stock: 0, etf: 1, crypto: 2 };
  const orderedGroups = [...universe.asset_classes].sort(
    (left, right) =>
      (assetClassOrder[left.asset_class] ?? 99) - (assetClassOrder[right.asset_class] ?? 99),
  );
  const sortedGroups = orderedGroups.map((group) => ({
    ...group,
    tickers: [...(group.tickers || [])].sort((left, right) =>
      String(left.ticker || "").localeCompare(String(right.ticker || "")),
    ),
  }));
  const buildOptions = () =>
    sortedGroups
      .map(
        (group) => `
          <optgroup label="${escapeHtml(group.asset_class)}">
            ${group.tickers
              .map((entry) => `<option value="${escapeHtml(entry.ticker)}">${escapeHtml(entry.ticker)}</option>`)
              .join("")}
          </optgroup>
        `,
      )
      .join("");

  elements.tickerSelect.innerHTML = buildOptions();
  elements.simulationTickers.innerHTML = buildOptions();
  const firstTicker = sortedGroups.find((group) => group.tickers.length > 0)?.tickers[0];
  if (firstTicker) {
    elements.tickerSelect.value = firstTicker.ticker;
  }
}

function renderMacro(snapshot) {
  elements.macroSnapshot.innerHTML = "";
  const regimeLabel = ["Bull", "Normal", "Bear"][snapshot.global_regime] || "Normal";
  const regime = document.createElement("div");
  regime.className = "row two-column";
  regime.innerHTML = `<span>Market regime</span><strong class="pill">${regimeLabel}</strong>`;
  elements.macroSnapshot.appendChild(regime);

  snapshot.macro.slice(0, 6).forEach((entry) => {
    const row = document.createElement("div");
    row.className = "row two-column";
    row.innerHTML = `
      <span>${entry.name}</span>
      <strong>${formatNumber(entry.value, 2)}</strong>
    `;
    elements.macroSnapshot.appendChild(row);
  });
  elements.macroSnapshot.insertAdjacentHTML(
    "beforeend",
    `<div class="info-note">
      <strong>What is the market regime?</strong>
      <span>The model classifies current conditions as Normal, Bull, or Bear using macro and volatility signals.</span>
    </div>`,
  );
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
    renderMarketIndexOptions([]);
    return;
  }
  const indexDates = indices.map((entry) => entry.as_of_date).filter(Boolean);
  state.marketIndexAsOf = result.as_of_date || (indexDates.length ? indexDates.sort().at(-1) : null);
  renderProjectStory();
  renderMarketIndexOptions(indices);
  elements.marketIndices.innerHTML = indices
    .map((entry) => {
      const change = Number(entry.change || 0);
      const tone = toneForValue(change);
      return `
        <article class="index-card">
          <div>
            <span>${entry.label || entry.symbol}</span>
            <strong>${formatIndexValue(entry.value)}</strong>
            <div class="index-change ${tone}">
              <span>${formatSignedNumber(entry.change)}</span>
              <span>${formatSignedPercent(entry.change_percent)}</span>
            </div>
            <small>${formatDate(entry.as_of_date || result.as_of_date)}</small>
          </div>
          ${sparklineSvg(change)}
        </article>
      `;
    })
    .join("");
}

function renderMarketIndexOptions(indices) {
  if (!elements.marketIndexSelect) return;
  const available = indices.filter((entry) => entry?.symbol);
  if (!available.length) {
    elements.marketIndexSelect.innerHTML = "";
    return;
  }
  const selected = available.some((entry) => entry.symbol === state.marketIndexSymbol)
    ? state.marketIndexSymbol
    : available[0].symbol;
  state.marketIndexSymbol = selected;
  elements.marketIndexSelect.innerHTML = available
    .map(
      (entry) =>
        `<option value="${escapeHtml(entry.symbol)}">${escapeHtml(entry.label || entry.symbol)}</option>`,
    )
    .join("");
  elements.marketIndexSelect.value = selected;
}

function movingAverage(points, windowSize) {
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

function renderMarketIndexHistory(result) {
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

function renderMarketSentiment(result) {
  const entries = result.ranked_tickers || [];
  const baseAverage =
    entries.reduce((total, entry) => total + Number(entry.returns.base || 0), 0) /
    Math.max(entries.length, 1);
  const bearAverage =
    entries.reduce((total, entry) => total + Number(entry.returns.bear || 0), 0) /
    Math.max(entries.length, 1);
  const confidenceAverage =
    entries.reduce((total, entry) => total + Number(entry.confidence || 0), 0) /
    Math.max(entries.length, 1);
  const score = Math.round(
    Math.max(0, Math.min(100, 50 + baseAverage * 90 + confidenceAverage * 18 + bearAverage * 35)),
  );
  const label = score >= 70 ? "Bullish" : score >= 55 ? "Moderately bullish" : score >= 45 ? "Neutral" : "Cautious";
  elements.sentimentGaugeArc.style.setProperty("--gauge-deg", `${Math.round(score * 1.8)}deg`);
  state.sentiment = { score, label, baseAverage, bearAverage, confidenceAverage };
  elements.sentimentScore.textContent = score;
  elements.sentimentLabel.textContent = label;
  elements.sentimentReasons.innerHTML = [
    `Average base scenario is ${signedPercentLabel(baseAverage)} across the ranked universe.`,
    `Average bear scenario is ${signedPercentLabel(bearAverage)}, defining downside risk.`,
    `Model confidence averages ${formatPercent(confidenceAverage)} for this scan.`,
  ]
    .map((reason) => `<span>${reason}</span>`)
    .join("");
}

function renderTopOpportunities(result) {
  const topByBase = [...(result.ranked_tickers || [])]
    .sort((left, right) => Number(right.returns.base || 0) - Number(left.returns.base || 0))
    .slice(0, 5);
  elements.topOpportunities.innerHTML = `
    <div class="row table-head top-row">
      <span>Ticker</span>
      <span>Class</span>
      <span>Base return</span>
      <span>Confidence</span>
    </div>
    ${topByBase
      .map(
        (entry) => `
          <div class="row top-row">
            <strong>${entry.ticker}</strong>
            <span>${entry.asset_class}</span>
            <span class="${toneForValue(entry.returns.base)}">${formatPercent(entry.returns.base)}</span>
            <span>${entry.confidence_label}</span>
          </div>
        `,
      )
      .join("")}
  `;
}

function renderMarketInsights(result) {
  const ranked = result.ranked_tickers || [];
  const bestBase = result.highlights.best_base_case;
  const stable = result.highlights.most_stable;
  const averageVolatility =
    ranked.reduce((total, entry) => total + Number(entry.risk_metrics.annualized_volatility || 0), 0) /
    Math.max(ranked.length, 1);
  const confidenceAverage =
    ranked.reduce((total, entry) => total + Number(entry.confidence || 0), 0) /
    Math.max(ranked.length, 1);
  elements.marketInsightList.innerHTML = [
    insight(
      `Best base scenario is ${bestBase.ticker} at ${formatPercent(bestBase.returns.base)} over the next ${result.horizon_days} days.`,
      "good",
    ),
    insight(
      `Average volatility is ${formatPercent(averageVolatility)}, so scenario ranges may be wider than headline returns.`,
      "warn",
    ),
    insight(
      `Confidence is ${confidenceAverage >= 0.65 ? "high" : confidenceAverage >= 0.45 ? "medium" : "low"}. Most stable ticker in this scan is ${stable.ticker}.`,
      "info",
    ),
  ].join("");
}

function renderMarketLearning(result) {
  if (!elements.marketLessonCards) return;
  const ranked = result.ranked_tickers || [];
  const bestBase = result.highlights?.best_base_case;
  const downside = result.highlights?.highest_downside_risk;
  const confidenceAverage =
    ranked.reduce((total, entry) => total + Number(entry.confidence || 0), 0) /
    Math.max(ranked.length, 1);
  const baseAverage =
    ranked.reduce((total, entry) => total + Number(entry.returns?.base || 0), 0) /
    Math.max(ranked.length, 1);
  elements.marketLessonCards.innerHTML = [
    lessonCard(
      "What am I looking at?",
      `The market overview compares scenario estimates across the tracked universe. Focus on the relationship between ${termChip("base")}, ${termChip("bear")}, and ${termChip("confidence")}, not only the highest return.`,
    ),
    lessonCard(
      "Why sentiment is not a signal",
      `The sentiment score blends average base return (${signedPercentLabel(baseAverage)}) with downside and confidence. It is a summary of model conditions, not a buy or sell instruction.`,
    ),
    lessonCard(
      "How to compare rows",
      `${bestBase?.ticker || "The top row"} may have the highest base scenario, while ${downside?.ticker || "another asset"} may carry the largest downside. A beginner-friendly comparison always reads upside and downside together.`,
    ),
    lessonCard(
      "Confidence check",
      confidenceExplanation(confidenceAverage),
      "Use confidence as a prompt to inspect assumptions before interpreting a scenario.",
    ),
  ].join("");
}

function renderMarket(result) {
  renderMacro(result.macro_snapshot);
  renderMarketSentiment(result);
  renderTopOpportunities(result);
  renderMarketInsights(result);
  renderMarketLearning(result);
  if (elements.marketAsOf) {
    const dateText =
      state.marketIndexAsOf || result.macro_snapshot?.date || state.universe?.latest_date || "";
    elements.marketAsOf.textContent = `Live index performance and ranked opportunity scan${dateText ? ` - as of ${formatDate(dateText)}` : ""}.`;
  }
  if (elements.footerDataAsOf && (result.macro_snapshot?.date || state.universe?.latest_date)) {
    elements.footerDataAsOf.textContent = `Data as of ${formatDate(result.macro_snapshot?.date || state.universe.latest_date)}`;
  }

  const bestBase = compactForecast(result.highlights.best_base_case);
  const bestAdjusted = compactForecast(result.highlights.best_risk_adjusted);
  const downside = compactForecast(result.highlights.highest_downside_risk);
  const stable = compactForecast(result.highlights.most_stable);

  elements.marketHighlights.innerHTML = [
    metricCard("Best base case", `${bestBase.ticker} ${bestBase.base}`, literacy.base, "Highest projected base return"),
    metricCard("Best risk-adjusted", bestAdjusted.ticker, literacy.sharpe, "Sharpe-weighted score"),
    metricCard("Highest downside risk", `${downside.ticker} ${downside.bear}`, literacy.bear, "Bear scenario target"),
    metricCard("Most stable", `${stable.ticker} ${stable.volatility}`, literacy.volatility, "Lowest projected volatility"),
  ].join("");

  elements.marketTable.innerHTML = `
    <div class="row table-head market-row">
      <span>Ticker</span>
      <span>Class</span>
      <span>Base</span>
      <span>Bear</span>
      <span>Bull</span>
      <span>Conf.</span>
      <span></span>
    </div>
  `;

  result.ranked_tickers.forEach((entry) => {
    const row = document.createElement("div");
    row.className = "row market-row";
    row.innerHTML = `
      <strong>${entry.ticker}</strong>
      <span>${entry.asset_class}</span>
      <span class="${toneForValue(entry.returns.base)}">${formatPercent(entry.returns.base)}</span>
      <span class="${toneForValue(entry.returns.bear)}">${formatPercent(entry.returns.bear)}</span>
      <span class="${toneForValue(entry.returns.bull)}">${formatPercent(entry.returns.bull)}</span>
      <span>${entry.confidence_label}</span>
      <button class="small-button" data-view-ticker="${entry.ticker}">View</button>
    `;
    elements.marketTable.appendChild(row);
  });
  updateProgress("market");
}

function completedProgressCount() {
  return Object.values(state.progress.actions).filter(Boolean).length;
}

function progressLevelFor(completed, total) {
  if (completed >= total) return 3;
  if (completed >= Math.ceil(total * 0.66)) return 2;
  return 1;
}

function pulseProgressWidget() {
  const widget = document.querySelector("#learningProgress");
  if (!widget) return;
  widget.classList.remove("level-up");
  widget.offsetHeight;
  widget.classList.add("level-up");
  window.setTimeout(() => widget.classList.remove("level-up"), 900);
}

function launchLevelConfetti() {
  if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) {
    showToast("Level 3 reached", "success");
    return;
  }

  document.querySelector(".confetti-burst")?.remove();
  const widget = document.querySelector("#learningProgress");
  const rect = widget?.getBoundingClientRect();
  const originX = rect?.width ? rect.left + rect.width / 2 : window.innerWidth / 2;
  const originY = rect?.height ? rect.top + rect.height / 2 : 96;
  const colors = ["#21d59b", "#58a6ff", "#f5c542", "#ff6b8a", "#b78cff", "#ffffff"];
  const burst = document.createElement("div");
  burst.className = "confetti-burst";
  burst.setAttribute("aria-hidden", "true");
  burst.style.setProperty("--origin-x", `${originX}px`);
  burst.style.setProperty("--origin-y", `${originY}px`);

  for (let index = 0; index < 56; index += 1) {
    const piece = document.createElement("span");
    piece.className = "confetti-piece";
    piece.style.setProperty("--x", `${Math.round((Math.random() - 0.5) * 420)}px`);
    piece.style.setProperty("--y", `${Math.round(-120 - Math.random() * 300)}px`);
    piece.style.setProperty("--r", `${Math.round((Math.random() - 0.5) * 720)}deg`);
    piece.style.setProperty("--delay", `${Math.random() * 120}ms`);
    piece.style.setProperty("--confetti-color", colors[index % colors.length]);
    burst.appendChild(piece);
  }

  document.body.appendChild(burst);
  showToast("Level 3 reached", "success");
  window.setTimeout(() => burst.remove(), 1800);
}

function chartLabels(history, forecastPath) {
  const historyDates = history.map((point) => point.date);
  const forecastDates = forecastPath.map((point) => point.date);
  if (historyDates.at(-1) && historyDates.at(-1) === forecastDates[0]) {
    return historyDates.concat(forecastDates.slice(1));
  }
  return historyDates.concat(forecastDates);
}

function visibleHistory(history) {
  const rangeToPoints = {
    "1m": 22,
    "3m": 66,
    "6m": 132,
    "1y": 252,
  };
  const points = rangeToPoints[state.chartRange];
  return points ? history.slice(-points) : history;
}

function forecastSeries(history, forecastPath) {
  const startsAtHistoryEnd = history.at(-1)?.date === forecastPath[0]?.date;
  return Array(startsAtHistoryEnd ? history.length - 1 : history.length)
    .fill(null)
    .concat(forecastPath.map((point) => point.price));
}

function renderForecastChart(forecast) {
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

function renderForecastLearning(forecast) {
  if (!elements.forecastLessonContent) return;
  const spread = Number(forecast.returns.bull || 0) - Number(forecast.returns.bear || 0);
  const volatility = Number(forecast.risk_metrics.annualized_volatility || 0);
  const drawdown = Number(forecast.risk_metrics.max_historical_drawdown || 0);
  const dataAsOf = forecast.data_as_of || forecast.latest_date;
  const sourceLabel = forecast.snapshot_used ? "stored daily snapshot" : "computed on request";
  elements.forecastLessonContent.innerHTML = [
    lessonCard(
      "Start with the scenario range",
      `${termChip("bear")} is the weaker case, ${termChip("base")} is the central estimate, and ${termChip("bull")} is the stronger case. For ${forecast.ticker}, the base return is ${signedPercentLabel(forecast.returns.base)} over ${forecast.horizon_days} days.`,
    ),
    lessonCard(
      "Read confidence as model readiness",
      confidenceExplanation(forecast.confidence, forecast.confidence_label),
      `Current confidence: ${formatPercent(forecast.confidence)}.`,
    ),
    `<article class="risk-checklist">
      <strong>Risk checklist</strong>
      <span>${termChip("volatility")} ${formatPercent(volatility)} annualized</span>
      <span>${termChip("drawdown")} ${formatPercent(drawdown)} historical max drawdown</span>
      <span>${termChip("spread")} ${formatPercent(spread)} between bear and bull outcomes</span>
      <span>${termChip("freshness")} ${formatDate(dataAsOf)} · ${sourceLabel}</span>
    </article>`,
  ].join("");
}

function renderTickerForecast(forecast) {
  state.currentForecast = forecast;
  renderForecastChart(forecast);
  renderForecastLearning(forecast);
  elements.tickerMetricTitle.textContent = `Scenario metrics - ${forecast.ticker}`;
  elements.tickerAboutTitle.textContent = `About ${forecast.ticker}`;
  elements.scenarioHorizonLabel.textContent = `(${forecast.horizon_days} days)`;
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
  elements.scenarioPathSummary.innerHTML = [
    `<article class="scenario-tile bear">
      <span>Bear scenario</span>
      <strong>${formatCurrency(forecast.target_prices.bear)}</strong>
      <small class="${toneForValue(forecast.returns.bear)}">${signedPercentLabel(forecast.returns.bear)}</small>
    </article>`,
    `<article class="scenario-tile base">
      <span>Base scenario</span>
      <strong>${formatCurrency(forecast.target_prices.base)}</strong>
      <small class="${toneForValue(forecast.returns.base)}">${signedPercentLabel(forecast.returns.base)}</small>
    </article>`,
    `<article class="scenario-tile bull">
      <span>Bull scenario</span>
      <strong>${formatCurrency(forecast.target_prices.bull)}</strong>
      <small class="${toneForValue(forecast.returns.bull)}">${signedPercentLabel(forecast.returns.bull)}</small>
    </article>`,
  ].join("");
  elements.tickerInsights.innerHTML = [
    insight(
      `Base scenario suggests ${formatPercent(forecast.returns.base)} over the next ${forecast.horizon_days} days.`,
      forecast.returns.base >= 0 ? "good" : "warn",
    ),
    insight(
      `Volatility is ${formatPercent(forecast.risk_metrics.annualized_volatility)}, which drives wider scenario dispersion.`,
      "warn",
    ),
    insight(
      `Confidence is ${forecast.confidence_label.toLowerCase()}. Review assumptions and scenario drivers before acting.`,
      "info",
    ),
  ].join("");
  renderTickerNarrative(null, forecast);
  if (elements.footerDataAsOf) {
    elements.footerDataAsOf.textContent = `Data as of ${formatDate(forecast.data_as_of || forecast.latest_date)}`;
  }
}

function renderTickerProfile(profile, forecast) {
  if (!profile) {
    setError(elements.tickerProfile, "Company data unavailable.");
    return;
  }
  const fields = profile.fields || {};
  const rows = [
    { label: "Bid", value: formatCurrencyOptional(fields.bid), raw: fields.bid, optional: true },
    { label: "Ask", value: formatCurrencyOptional(fields.ask), raw: fields.ask, optional: true },
    { label: "Last sale", value: formatCurrencyOptional(fields.last_sale), raw: fields.last_sale },
    { label: "Exchange", value: fields.exchange || "Unavailable", raw: fields.exchange, optional: true },
    { label: "Mkt cap", value: formatLargeNumber(fields.market_cap), raw: fields.market_cap, optional: true },
    { label: "P/E ratio", value: formatNumberOptional(fields.pe_ratio, 2), raw: fields.pe_ratio, optional: true },
    {
      label: "52W high",
      value: formatCurrencyOptional(fields.fifty_two_week_high),
      raw: fields.fifty_two_week_high,
      optional: true,
    },
    {
      label: "52W low",
      value: formatCurrencyOptional(fields.fifty_two_week_low),
      raw: fields.fifty_two_week_low,
      optional: true,
    },
    { label: "Volume", value: formatLargeNumber(fields.volume), raw: fields.volume, optional: true },
    {
      label: "Dividend freq.",
      value: fields.dividend_frequency || "Unavailable",
      raw: fields.dividend_frequency,
      optional: true,
    },
    {
      label: "12-month yield",
      value: formatPercentOptional(fields.dividend_yield),
      raw: fields.dividend_yield,
      optional: true,
    },
  ].filter((row) => !row.optional || hasProfileValue(row.raw));
  elements.tickerProfile.innerHTML = rows
    .map(
      ({ label, value }) => `
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
  if (profile.source === "local_artifacts") {
    warnings.push("Market cap, P/E, bid/ask, and dividend fields require refreshed profile snapshots from Supabase/yfinance.");
  }
  renderTickerNarrative(profile, forecast, warnings);
}

function renderMetricBlock(target, entries) {
  target.innerHTML = entries
    .map(([label, value, tooltip, subtitle]) => metricCard(label, value, tooltip, subtitle))
    .join("");
}

function renderAllocationTable(target, allocations, options = {}) {
  const sourceRows = allocations || [];
  const rows = sourceRows.slice(0, options.limit || sourceRows.length);
  target.innerHTML = `
    <div class="row table-head asset-row">
      <span>${options.firstLabel || "Ticker"}</span>
      <span>${options.secondLabel || "Class"}</span>
      <span>Weight</span>
      <span>Value</span>
    </div>
    ${rows
      .map(
        (allocation) => `
          <div class="row asset-row">
            <strong>${allocation.ticker || allocation.asset_class}</strong>
            <span>${allocation.asset_class || ""}</span>
            <span>${formatPercent(allocation.weight)}</span>
            <span>${formatCurrency(allocation.amount || 0)}</span>
          </div>
        `,
      )
      .join("")}
  `;
}

function renderTradePlan(target, trades) {
  target.innerHTML = `
    <div class="row table-head trade-row">
      <span>Action</span>
      <span>Ticker</span>
      <span>Weight</span>
      <span>Amount</span>
    </div>
    ${(trades || [])
      .map((trade) => {
        const isHold = String(trade.action).includes("hold");
        const action = isHold ? "Hold" : "Buy";
        return `
          <div class="row trade-row">
            <span class="action-pill ${isHold ? "hold" : ""}">${action}</span>
            <strong>${trade.ticker}</strong>
            <span>${formatPercent(trade.weight)}</span>
            <span>${formatCurrency(trade.amount)}</span>
          </div>
        `;
      })
      .join("")}
  `;
}

function renderWarnings(target, warnings, tone = "warn") {
  target.innerHTML = (warnings || []).map((warning) => insight(warning, tone)).join("");
}

function selectedSimulationTickers() {
  return Array.from(elements.simulationTickers.selectedOptions).map((option) => option.value);
}

function renderClassAllocation(allocations) {
  const lookup = Object.fromEntries(
    (allocations || []).map((allocation) => [allocation.asset_class, Number(allocation.weight || 0)]),
  );
  const stock = lookup.stock || 0;
  const etf = stock + (lookup.etf || 0);
  const crypto = etf + (lookup.crypto || 0);
  const rows = [
    ["Stocks", lookup.stock || 0, "stock", allocations?.find((entry) => entry.asset_class === "stock")?.amount],
    ["ETFs", lookup.etf || 0, "etf", allocations?.find((entry) => entry.asset_class === "etf")?.amount],
    ["Crypto", lookup.crypto || 0, "crypto", allocations?.find((entry) => entry.asset_class === "crypto")?.amount],
    ["Cash", lookup.cash || 0, "cash", allocations?.find((entry) => entry.asset_class === "cash")?.amount],
  ].filter((row) => row[1] > 0.0001);
  let cursor = 0;
  state.classAllocationSegments = rows.map(([label, weight, key, amount]) => {
    const start = cursor;
    cursor += Number(weight || 0);
    return { label, weight, key, amount, start, end: cursor };
  });
  elements.simulationClasses.innerHTML = `
    <div class="allocation-view">
      <div class="allocation-donut" style="--stock: ${stock * 100}%; --etf: ${etf * 100}%; --crypto: ${crypto * 100}%;">
        <div><span>Total</span><strong>100%</strong></div>
      </div>
      <div class="allocation-legend">
        ${rows
          .map(
            ([label, weight, key]) => `
              <div class="legend-row">
                <span class="legend-dot ${key}"></span>
                <span>${label}</span>
                <strong>${formatPercent(weight)}</strong>
              </div>
            `,
          )
          .join("")}
      </div>
    </div>
  `;
}

function renderPortfolioLearning(result = state.simulation) {
  if (!elements.portfolioClassroom) return;
  const risk = Number(elements.risk.value || 0);
  const horizon = selectedHorizonDays();
  const amount = Number(elements.amount.value || 0);
  const riskLabel = riskPreferenceLabel(risk);
  const classRows = result?.class_allocations || [];
  const cashWeight = classRows.find((entry) => entry.asset_class === "cash")?.weight;
  const cryptoWeight = classRows.find((entry) => entry.asset_class === "crypto")?.weight;
  const spread =
    result?.summary
      ? Math.max(Number(result.summary.bull_return || 0) - Number(result.summary.bear_return || 0), 0)
      : null;

  elements.portfolioClassroom.innerHTML = [
    lessonCard(
      "Risk appetite",
      `Your current setting is ${risk.toFixed(2)}, which reads as a ${riskLabel} classroom profile. Higher risk can allow more volatile assets; lower risk usually favors more cash and steadier sleeves.`,
      `${termChip("volatility")} and ${termChip("drawdown")} are the key concepts to watch.`,
    ),
    lessonCard(
      "Cash buffer",
      result
        ? `The simulation currently holds ${formatPercent(cashWeight || 0)} in cash. Cash can reduce downside exposure, but it may also reduce upside in bull scenarios.`
        : "Run the simulation to see how much cash the model keeps aside for the selected risk and horizon.",
      termChip("cash"),
    ),
    lessonCard(
      "Scenario spread",
      result
        ? `The distance between simulated bear and bull returns is ${formatPercent(spread)}. A wider spread means the outcome is more uncertain.`
        : "After a simulation, this panel will explain how far apart the bear and bull outcomes are.",
      termChip("spread"),
    ),
    lessonCard(
      "Horizon and amount",
      `The classroom is modeling ${formatCurrency(amount)} over ${formatInteger(horizon)} days. Longer horizons give the model more time for compounding, but they also extend uncertainty.`,
      cryptoWeight ? `Crypto weight in the latest simulation: ${formatPercent(cryptoWeight)}.` : "",
    ),
  ].join("");
}

function renderSimulation(result) {
  state.simulation = result;
  elements.simulationSummary.innerHTML = [
    `<article class="scenario-tile bear">
      <span>Bear scenario</span>
      <strong>${formatCurrency(result.summary.bear_value)}</strong>
      <small class="${toneForValue(result.summary.bear_return)}">${signedPercentLabel(result.summary.bear_return)}</small>
    </article>`,
    `<article class="scenario-tile base">
      <span>Base scenario</span>
      <strong>${formatCurrency(result.summary.base_value)}</strong>
      <small class="${toneForValue(result.summary.base_return)}">${signedPercentLabel(result.summary.base_return)} · ${formatPercent(result.summary.average_confidence)} avg confidence</small>
    </article>`,
    `<article class="scenario-tile bull">
      <span>Bull scenario</span>
      <strong>${formatCurrency(result.summary.bull_value)}</strong>
      <small class="${toneForValue(result.summary.bull_return)}">${signedPercentLabel(result.summary.bull_return)}</small>
    </article>`,
    `<article class="scenario-tile">
      <span>Average confidence</span>
      <strong>${formatPercent(result.summary.average_confidence)}</strong>
    </article>`,
  ].join("");
  const volatilityProxy = Math.max(
    Math.abs(Number(result.summary.bull_return || 0) - Number(result.summary.bear_return || 0)) / 2,
    0,
  );
  elements.simulationWarnings.innerHTML = [
    insight(
      `Base scenario suggests ${formatPercent(result.summary.base_return)} over the next ${result.horizon_days} days with ${formatPercent(result.summary.average_confidence)} average confidence.`,
      "good",
    ),
    insight(`Scenario dispersion is ${formatPercent(volatilityProxy)}, based on bear and bull portfolio outcomes.`, "warn"),
    insight(`Diversified across ${result.asset_allocations.length} positions with risk-managed cash handling.`, "info"),
    ...(result.warnings || []).map((warning) => insight(warning, "info")),
  ].join("");
  renderClassAllocation(result.class_allocations);
  renderAllocationTable(elements.simulationAssets, result.asset_allocations, { limit: 10 });
  renderTradePlan(elements.simulationTrades, result.trade_plan);
  renderPortfolioLearning(result);
  if (elements.footerDataAsOf && state.universe?.latest_date) {
    elements.footerDataAsOf.textContent = `Data as of ${formatDate(state.universe.latest_date)}`;
  }
  renderDataHealthCards();
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
  renderAllocationTable(elements.rlClasses, result.class_allocations, {
    firstLabel: "Class",
    secondLabel: "",
  });
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

function latestDataDate(refresh = state.refreshStatus || {}) {
  const latestRun = refresh.latest_run || {};
  return (
    state.marketIndexAsOf ||
    refresh.market_index_refresh?.as_of_date ||
    refresh.latest_market_date ||
    state.universe?.latest_date ||
    latestRun.completed_at
  );
}

function latestDataSubtitle(refresh = state.refreshStatus || {}) {
  const latestRun = refresh.latest_run || {};
  if (state.marketIndexAsOf || refresh.market_index_refresh?.as_of_date) {
    return "Live market index refresh";
  }
  if (latestRun.mode) {
    return `${latestRun.mode} refresh`;
  }
  return "Latest available market date";
}

function renderDataHealthCards() {
  if (!elements.dataHealthCards) return;
  const health = state.health || {};
  const models = state.models || {};
  const refresh = state.refreshStatus || {};
  const latestRun = refresh.latest_run || {};
  const latestMarketDate = latestDataDate(refresh);
  const assetCount = refresh.asset_count ?? state.universe?.tickers?.length;
  const confidence = state.simulation?.summary?.average_confidence;
  const backtestMetrics = state.backtest?.summary_metrics || {};
  const sharpeRatio = backtestMetrics.sharpe_ratio;
  const backtestSubtitle = isMissing(sharpeRatio)
    ? "Sharpe ratio pending"
    : `${formatNumber(sharpeRatio, 2)} Sharpe ratio`;
  const latestSubtitle = latestDataSubtitle(refresh);
  const hasLiveIndexDate = Boolean(state.marketIndexAsOf || refresh.market_index_refresh?.as_of_date);

  elements.dataHealthCards.innerHTML = [
    `<article class="health-card">
      <span>Model status</span>
      <strong class="${health.ready === false ? "negative" : "positive"}">${health.ready === false ? "Degraded" : "Healthy"}</strong>
      <small>${health.error || "All systems operational"}</small>
    </article>`,
    `<article class="health-card">
      <span>Data freshness</span>
      <strong>${latestRun.completed_at && !hasLiveIndexDate ? formatDateTime(latestRun.completed_at) : formatDate(latestMarketDate)}</strong>
      <small>${latestSubtitle}</small>
    </article>`,
    `<article class="health-card">
      <span>Universe coverage</span>
      <strong>${formatInteger(assetCount)}</strong>
      <small>${classCoverageLabel(state.universe)}</small>
    </article>`,
    `<article class="health-card">
      <span>Backtest summary</span>
      <strong>${isMissing(confidence) ? "Pending" : formatPercent(confidence)}</strong>
      <small>${isMissing(confidence) ? models?.explainability?.method || "Scenario diagnostics pending" : `Avg confidence · ${backtestSubtitle}`}</small>
    </article>`,
  ].join("");
  renderProjectStory();
}

function renderProjectStory() {
  if (!elements.projectStats) return;
  const health = state.health || {};
  const models = state.models || {};
  const refresh = state.refreshStatus || {};
  const latestMarketDate = latestDataDate(refresh);
  const latestSubtitle = latestDataSubtitle(refresh);
  const assetCount = refresh.asset_count ?? state.universe?.tickers?.length;
  const featureGroups = models.feature_groups ? Object.keys(models.feature_groups).length : 0;
  const explainability = models.explainability?.method || "surrogate explainability";
  const modelSurface = models.feature_groups
    ? `model metadata, ${formatInteger(featureGroups)} feature groups, forecasts, simulations, backtests, and ${explainability}`
    : "Supabase-backed forecasts, portfolio simulations, refresh diagnostics, and optional artifact endpoints for local RL workflows";
  const statusLabel = state.health
    ? health.ready === false
      ? "Degraded"
      : "Healthy"
    : "Pending";
  const statusClass = state.health ? (health.ready === false ? "negative" : "positive") : "";

  elements.projectStats.innerHTML = [
    `<article><span>Tracked universe</span><strong>${formatInteger(assetCount)}</strong><small>${classCoverageLabel(state.universe)}</small></article>`,
    `<article><span>Latest data</span><strong>${formatDate(latestMarketDate)}</strong><small>${latestSubtitle}</small></article>`,
    `<article><span>Index history</span><strong>1M-5Y</strong><small>S&P 500, Nasdaq, TSX, and Dow charts</small></article>`,
    `<article><span>Backend status</span><strong class="${statusClass}">${statusLabel}</strong><small>FastAPI inference service</small></article>`,
  ].join("");

  if (elements.projectHighlights) {
    elements.projectHighlights.innerHTML = [
      lessonCard(
        "Full-stack product thinking",
        "The project connects a beginner-friendly interface to real backend APIs, Supabase market data, market index history, refresh jobs, and portfolio simulation logic.",
      ),
      lessonCard(
        "ML system design",
        `The backend exposes ${modelSurface}.`,
      ),
      lessonCard(
        "Market data UX",
        "The market overview now combines live index cards with selectable historical charts and moving-average context before showing model-ranked opportunities.",
      ),
      lessonCard(
        "Education-first UX",
        "Learn Mode reframes outputs as lessons: what the metric means, why it matters, and what risk to inspect next.",
      ),
    ].join("");
  }

  if (elements.projectTransparency) {
    elements.projectTransparency.innerHTML = [
      lessonCard(
        "Data freshness",
        `The app reports the latest live market/index date (${formatDate(latestMarketDate)}) so users know what data the page used.`,
        termChip("freshness"),
      ),
      lessonCard(
        "Historical index source",
        "Index charts are fetched through a FastAPI route backed by the configured market data provider and limited to the configured market index symbols.",
      ),
      lessonCard(
        "Model limitations",
        "The UI explicitly frames forecasts as educational estimates. It avoids promising returns and keeps scenario outputs separate from advice.",
      ),
      lessonCard(
        "Validation story",
        "The repo includes backend API tests, synthetic fixture artifacts, refresh diagnostics, and backtest metrics to support reliability conversations in interviews.",
      ),
    ].join("");
  }
  renderGlossary();
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
  if (elements.dataHealthCards) {
    state.health = healthResult.status === "fulfilled" ? healthResult.value : null;
    state.models = modelsResult.status === "fulfilled" ? modelsResult.value : null;
    state.refreshStatus = refreshResult.status === "fulfilled" ? refreshResult.value : null;
    renderDataHealthCards();
  }
}

async function loadUniverse() {
  state.universe = await callSectionApi("universe", "/api/universe");
  renderUniverse(state.universe);
  renderDataHealthCards();
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
    updateProgress("forecast");
  } catch (error) {
    if (isAbort(error)) return;
    setError(elements.tickerMetrics, `Ticker forecast unavailable: ${error.message}`);
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
        }),
      }),
    });
    renderSimulation(result);
    updateProgress("simulation");
  } catch (error) {
    if (isAbort(error)) return;
    setError(elements.simulationSummary, `Simulation unavailable: ${error.message}`);
    setError(elements.simulationWarnings, "Simulation insights unavailable.");
    setError(elements.portfolioClassroom, "Portfolio classroom unavailable.");
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
  await refreshActiveView({ force: true });
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
    await refreshActiveView({ force: true });
    elements.apiStatus.textContent = "Backend connected.";
    elements.apiStatus.className = "status-good";
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
    const emptyState = document.getElementById("marketEmptyState");
    if (emptyState) emptyState.classList.add("is-visible");
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
elements.amount.addEventListener("input", () => {
  state.loaded.simulator = false;
  state.backtest = null;
  renderPortfolioLearning();
});
elements.tickerSelect.addEventListener("change", () => {
  state.loaded.forecast = false;
});

elements.learnModeToggle?.addEventListener("click", (e) => {
  e.preventDefault();
  setLearnMode(!state.learnMode);
});

elements.themeModeToggle?.addEventListener("click", (e) => {
  e.preventDefault();
  setThemeMode(state.themeMode === "dark" ? "light" : "dark");
});

elements.saveApiBase.addEventListener("click", async () => {
  state.apiBase = normalizeApiBase(elements.apiBase.value) || defaultApiBase();
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

elements.apiBase.value = state.apiBase;
elements.riskValue.textContent = Number(elements.risk.value).toFixed(2);
syncHorizonControls(elements.horizon.value);
setLearnMode(state.learnMode);
setThemeMode(state.themeMode);
renderPortfolioLearning();
renderProjectStory();
renderGlossary();
probeBackend();

function updateProgress(action) {
  if (!state.progress.actions[action]) {
    state.progress.actions[action] = true;
  }
  const previousLevel = state.progress.level;
  const completed = completedProgressCount();
  const total = Object.keys(state.progress.actions).length;
  const nextLevel = progressLevelFor(completed, total);
  const bar = document.querySelector(".progress-bar");
  const text = document.querySelector(".progress-text");
  const level = document.querySelector(".progress-level");

  state.progress.level = nextLevel;
  if (bar) bar.style.width = `${(completed / total) * 100}%`;
  if (text) text.textContent = `${completed}/${total} tasks completed`;
  if (level) level.textContent = progressLevelLabels[nextLevel];

  if (nextLevel > previousLevel) {
    pulseProgressWidget();
  }
  if (previousLevel < 3 && nextLevel === 3) {
    launchLevelConfetti();
  }
}

document.body.addEventListener("mouseover", (event) => {
  const chip = event.target.closest(".glossary-chip");
  if (chip) {
    const key = chip.dataset.term;
    const term = glossary[key];
    if (term) {
      showHoverTooltip(event, `<strong>${term.title}</strong><span>${term.definition}</span>`);
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
    <div class="cmd-result-item" data-cmd-type="${r.type}" data-cmd-value="${r.tab || r.key || r.ticker || ""}">
      ${iconSvg(r.icon, "cmd-result-icon")}
      <span>${r.label}</span>
      <span class="cmd-result-type">${r.type}</span>
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
  }
  closeCommandPalette();
});

document.getElementById("openCmdPalette")?.addEventListener("click", openCommandPalette);

// ── Keyboard Shortcuts ──
document.addEventListener("keydown", (e) => {
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
  const metric = btn.closest(".metric");
  if (!metric) return;
  const slot = metric.querySelector(".why-popover-slot");
  if (!slot) return;
  if (slot.innerHTML) {
    slot.innerHTML = "";
  } else {
    slot.innerHTML = `<div class="why-popover">${btn.dataset.why}</div>`;
  }
});
