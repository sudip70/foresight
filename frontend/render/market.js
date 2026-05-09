import { elements, literacy, state } from "../state/store.js";
import {
  confidenceExplanation,
  escapeHtml,
  formatDate,
  formatIndexValue,
  formatNumber,
  formatPercent,
  formatSignedNumber,
  formatSignedPercent,
  formatSourceLabel,
  signedPercentLabel,
  toneForValue,
} from "../utils/formatters.js";
import {
  badge,
  insight,
  lessonCard,
  metricCard,
  setError,
  sparklineSvg,
  termChip,
} from "../utils/dom.js";
import { renderFreshnessBar, renderProjectStory } from "./diagnostics.js";

export function renderUniverse(universe) {
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

export function renderMacro(snapshot) {
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
      <span>${escapeHtml(entry.name)}</span>
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

export function compactForecast(entry) {
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

export function freshnessLabelForDate(value) {
  if (!value) return "Unknown freshness";
  const normalized = /^\d{4}-\d{2}-\d{2}$/.test(String(value)) ? `${value}T00:00:00` : value;
  const parsed = new Date(normalized);
  if (Number.isNaN(parsed.getTime())) return "Unknown freshness";
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const asOf = new Date(parsed);
  asOf.setHours(0, 0, 0, 0);
  const daysOld = Math.max(Math.floor((today - asOf) / 86_400_000), 0);
  if (daysOld <= 5) return "Fresh";
  if (daysOld <= 10) return "Delayed";
  return "Stale";
}

export function dataQualityBadges(entry = {}) {
  const quality = entry.data_quality || {};
  const asOf = quality.as_of_date || entry.data_as_of || entry.latest_date;
  const freshness = quality.freshness_label || freshnessLabelForDate(asOf);
  const freshnessTone =
    freshness === "Fresh" ? "positive" : freshness === "Stale" ? "negative" : "pending";
  const source = formatSourceLabel(quality.source || entry.source);
  const sourceLabel = entry.snapshot_used || quality.snapshot_used ? "Snapshot" : "Computed";
  return [
    badge(freshness, freshnessTone),
    badge(sourceLabel, entry.snapshot_used || quality.snapshot_used ? "info" : "neutral"),
    source ? badge(source, "neutral") : "",
    asOf ? badge(formatDate(asOf), "neutral") : "",
  ].join("");
}

export function forecastChangeChip(entry = {}) {
  const change = entry.forecast_change || {};
  if (!change.available) {
    return badge("No prior refresh", "neutral");
  }
  const tone = change.direction === "up" ? "positive" : change.direction === "down" ? "negative" : "neutral";
  return badge(`${formatSignedPercent(change.base_return_delta)} vs prior`, tone);
}

export function renderMarketIndices(result) {
  const indices = result?.indices || [];
  if (!indices.length) {
    setError(elements.marketIndices, "Market index data unavailable.");
    renderMarketIndexOptions([]);
    renderFreshnessBar();
    return;
  }
  const indexDates = indices.map((entry) => entry.as_of_date).filter(Boolean);
  state.marketIndexAsOf = result.as_of_date || (indexDates.length ? indexDates.sort().at(-1) : null);
  renderProjectStory();
  renderFreshnessBar();
  renderMarketIndexOptions(indices);
  elements.marketIndices.innerHTML = indices
    .map((entry) => {
      const change = Number(entry.change || 0);
      const tone = toneForValue(change);
      return `
        <article class="index-card">
          <div>
            <span>${escapeHtml(entry.label || entry.symbol)}</span>
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

export function renderMarketIndexOptions(indices) {
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

export function renderMarketSentiment(result) {
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

export function renderTopOpportunities(result) {
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
            <strong>${escapeHtml(entry.ticker)}</strong>
            <span>${escapeHtml(entry.asset_class)}</span>
            <span class="${toneForValue(entry.returns.base)}">${formatPercent(entry.returns.base)}</span>
            <span>${escapeHtml(entry.confidence_label)}</span>
          </div>
        `,
      )
      .join("")}
  `;
}

export function renderMarketInsights(result) {
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

export function renderMarketLearning(result) {
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
      `${escapeHtml(bestBase?.ticker || "The top row")} may have the highest base scenario, while ${escapeHtml(downside?.ticker || "another asset")} may carry the largest downside. A beginner-friendly comparison always reads upside and downside together.`,
    ),
    lessonCard(
      "Confidence check",
      confidenceExplanation(confidenceAverage),
      "Use confidence as a prompt to inspect assumptions before interpreting a scenario.",
    ),
  ].join("");
}

export function renderMarket(result) {
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
      <span>Change</span>
      <span></span>
    </div>
  `;

  result.ranked_tickers.forEach((entry) => {
    const row = document.createElement("div");
    row.className = "row market-row";
    row.innerHTML = `
      <strong>${escapeHtml(entry.ticker)}</strong>
      <span>${escapeHtml(entry.asset_class)}</span>
      <span class="${toneForValue(entry.returns.base)}">${formatPercent(entry.returns.base)}</span>
      <span class="${toneForValue(entry.returns.bear)}">${formatPercent(entry.returns.bear)}</span>
      <span class="${toneForValue(entry.returns.bull)}">${formatPercent(entry.returns.bull)}</span>
      <span>${escapeHtml(entry.confidence_label)}</span>
      <span>${forecastChangeChip(entry)}</span>
      <button class="small-button" data-view-ticker="${escapeHtml(entry.ticker)}">View</button>
    `;
    elements.marketTable.appendChild(row);
  });
  renderFreshnessBar();
}
