import { elements, state, glossary } from "../state/store.js";
import {
  classCoverageLabel,
  escapeHtml,
  formatDate,
  formatDateTime,
  formatInteger,
  formatNumber,
  formatPercent,
  formatSourceLabel,
  isMissing,
} from "../utils/formatters.js";
import { lessonCard, termChip } from "../utils/dom.js";

export function renderGlossary() {
  if (!elements.glossaryList) return;
  elements.glossaryList.innerHTML = Object.entries(glossary)
    .map(
      ([key, term]) => `
        <article class="glossary-item" data-glossary-key="${escapeHtml(key)}">
          <strong>${escapeHtml(term.title)}</strong>
          <span>${escapeHtml(term.definition)}</span>
        </article>
      `,
    )
    .join("");
}

export function latestDataDate(refresh = state.refreshStatus || {}) {
  const latestRun = refresh.latest_run || {};
  return (
    state.marketIndexAsOf ||
    refresh.market_index_refresh?.as_of_date ||
    refresh.latest_market_date ||
    state.universe?.latest_date ||
    latestRun.completed_at
  );
}

export function latestDataSubtitle(refresh = state.refreshStatus || {}) {
  const latestRun = refresh.latest_run || {};
  if (state.marketIndexAsOf || refresh.market_index_refresh?.as_of_date) {
    return "Live market index refresh";
  }
  if (latestRun.mode) {
    return `${latestRun.mode} refresh`;
  }
  return "Latest available market date";
}

export function renderFreshnessBar() {
  if (!elements.freshnessBar) return;
  const refresh = state.refreshStatus || {};
  const staleCount = (refresh.stale_tickers || []).length;
  const failedCount = (refresh.failed_items || []).length;
  const latestMarketDate =
    latestDataDate(refresh) ||
    state.currentForecast?.data_as_of ||
    state.currentForecast?.latest_date ||
    "";
  const source =
    refresh.source ||
    refresh.market_index_refresh?.source ||
    state.health?.market_data?.source ||
    state.universe?.source ||
    state.currentForecast?.source ||
    "pending";
  const statusClass = failedCount || staleCount ? "negative" : latestMarketDate ? "positive" : "pending";
  const statusLabel = failedCount
    ? `${formatInteger(failedCount)} failed refresh ${failedCount === 1 ? "item" : "items"}`
    : staleCount
      ? `${formatInteger(staleCount)} stale ${staleCount === 1 ? "ticker" : "tickers"}`
      : latestMarketDate
        ? "Fresh"
        : "Coverage pending";

  elements.freshnessStatus.className = `freshness-dot ${statusClass}`;
  elements.freshnessLatest.textContent = latestMarketDate
    ? `Latest market data ${formatDate(latestMarketDate)}`
    : "Waiting for backend";
  elements.freshnessSource.textContent = formatSourceLabel(source);
  elements.freshnessStale.textContent = statusLabel;
}

export function modelStatusDisplay(health = {}) {
  const artifactEngine = health.artifact_engine || {};
  const status = String(artifactEngine.status || "").toLowerCase();
  const error = String(artifactEngine.error || health.error || "");
  const lazyEnabled = artifactEngine.lazy_enabled === true;

  if (health.ready === false) {
    return {
      label: "Degraded",
      className: "negative",
      detail: error || "Backend health check is degraded.",
    };
  }

  if (lazyEnabled) {
    if (status === "ready") {
      return {
        label: "Healthy",
        className: "positive",
        detail: "Artifact engine ready; backtests and RL diagnostics are available.",
      };
    }
    if (status === "loading") {
      return {
        label: "Lazy loading",
        className: "pending",
        detail: "Market APIs are ready; the artifact engine is warming up in the background.",
      };
    }
    if (status === "not_started" || !status) {
      return {
        label: "Lazy loading",
        className: "pending",
        detail: "Market APIs are ready; the artifact engine loads automatically on first backtest or diagnostics request.",
      };
    }
    if (status === "error") {
      return {
        label: "Partial",
        className: "negative",
        detail: error || "Market APIs are ready, but the artifact engine failed to load.",
      };
    }
  }

  if (/artifact engine disabled|FORESIGHT_LOAD_ARTIFACT_ENGINE=false/i.test(error)) {
    return {
      label: "Limited",
      className: "pending",
      detail: "Market APIs are healthy. Deploy the lazy-loading backend update to enable backtests and RL diagnostics.",
    };
  }

  return {
    label: "Healthy",
    className: "positive",
    detail: error || "All systems operational.",
  };
}

export function renderDataHealthCards() {
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
  const modelStatus = modelStatusDisplay(health);

  elements.dataHealthCards.innerHTML = [
    `<article class="health-card">
      <span>Model status</span>
      <strong class="${modelStatus.className}">${escapeHtml(modelStatus.label)}</strong>
      <small>${escapeHtml(modelStatus.detail)}</small>
    </article>`,
    `<article class="health-card">
      <span>Data freshness</span>
      <strong>${latestRun.completed_at && !hasLiveIndexDate ? formatDateTime(latestRun.completed_at) : formatDate(latestMarketDate)}</strong>
      <small>${escapeHtml(latestSubtitle)}</small>
    </article>`,
    `<article class="health-card">
      <span>Universe coverage</span>
      <strong>${formatInteger(assetCount)}</strong>
      <small>${escapeHtml(classCoverageLabel(state.universe))}</small>
    </article>`,
    `<article class="health-card">
      <span>Backtest summary</span>
      <strong>${isMissing(confidence) ? "Pending" : formatPercent(confidence)}</strong>
      <small>${escapeHtml(isMissing(confidence) ? models?.explainability?.method || "Scenario diagnostics pending" : `Avg confidence · ${backtestSubtitle}`)}</small>
    </article>`,
  ].join("");
  renderProjectStory();
  renderFreshnessBar();
}

export function renderProjectStory() {
  if (!elements.projectStats) return;
  const health = state.health || {};
  const models = state.models || {};
  const refresh = state.refreshStatus || {};
  const latestMarketDate = latestDataDate(refresh);
  const latestSubtitle = latestDataSubtitle(refresh);
  const assetCount = refresh.asset_count ?? state.universe?.tickers?.length;
  const featureGroups = models.feature_groups ? Object.keys(models.feature_groups).length : 0;
  const explainability = escapeHtml(models.explainability?.method || "surrogate explainability");
  const modelSurface = models.feature_groups
    ? `model metadata, ${formatInteger(featureGroups)} feature groups, forecasts, simulations, backtests, and ${explainability}`
    : "Supabase-backed forecasts, portfolio simulations, refresh diagnostics, and lazily loaded artifact endpoints for RL workflows";
  const statusLabel = state.health
    ? health.ready === false
      ? "Degraded"
      : "Healthy"
    : "Pending";
  const statusClass = state.health ? (health.ready === false ? "negative" : "positive") : "";

  elements.projectStats.innerHTML = [
    `<article><span>Tracked universe</span><strong>${formatInteger(assetCount)}</strong><small>${escapeHtml(classCoverageLabel(state.universe))}</small></article>`,
    `<article><span>Latest data</span><strong>${formatDate(latestMarketDate)}</strong><small>${escapeHtml(latestSubtitle)}</small></article>`,
    `<article><span>Index history</span><strong>1M-5Y</strong><small>S&P 500, Nasdaq, TSX, and Dow charts</small></article>`,
    `<article><span>Backend status</span><strong class="${statusClass}">${statusLabel}</strong><small>FastAPI inference service</small></article>`,
  ].join("");

  if (elements.projectHighlights) {
    elements.projectHighlights.innerHTML = [
      lessonCard(
        "Full-stack product architecture",
        "The project connects a static frontend, modular FastAPI routers, Supabase market data, scheduled refresh jobs, market index history, and portfolio simulation logic.",
      ),
      lessonCard(
        "Maintainable frontend",
        "The JavaScript is split into API, state, chart, render, and utility modules, while CSS is organized into base, layout, forms, components, charts, responsive, and theme layers.",
      ),
      lessonCard(
        "ML system design",
        `The backend exposes ${modelSurface}. Route groups are separated for health, market data, forecasts, portfolio simulation, inference, and diagnostics.`,
      ),
      lessonCard(
        "Education-first UX",
        "Learn Mode, compact simulator rationale, benchmark comparisons, and clear data-freshness indicators explain what the metric means, why it matters, and what risk to inspect next.",
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
        "The UI frames forecasts as educational estimates, keeps scenario outputs separate from advice, and shows allocation rationale without presenting it as a recommendation.",
      ),
      lessonCard(
        "Validation story",
        "The repo includes backend API tests, syntax checks for split frontend modules, CSS integrity checks, synthetic fixture artifacts, refresh diagnostics, and backtest metrics to support deployment confidence.",
      ),
    ].join("");
  }
  renderGlossary();
}
