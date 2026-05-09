import { elements, literacy, state } from "../state/store.js";
import {
  buildTickerAboutParagraph,
  confidenceExplanation,
  escapeHtml,
  formatCurrency,
  formatCurrencyOptional,
  formatDate,
  formatLargeNumber,
  formatNumberOptional,
  formatPercent,
  formatPercentOptional,
  formatSignedNumber,
  formatSignedPercent,
  hasProfileValue,
  signedPercentLabel,
  toneForValue,
} from "../utils/formatters.js";
import { insight, lessonCard, metricCard, setError, termChip } from "../utils/dom.js";
import { renderForecastChart } from "../charts/forecastChart.js";
import { dataQualityBadges } from "./market.js";

export function renderTickerNarrative(profile, forecast, warnings = []) {
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

export function renderForecastLearning(forecast) {
  if (!elements.forecastLessonContent) return;
  const spread = Number(forecast.returns.bull || 0) - Number(forecast.returns.bear || 0);
  const volatility = Number(forecast.risk_metrics.annualized_volatility || 0);
  const drawdown = Number(forecast.risk_metrics.max_historical_drawdown || 0);
  const dataAsOf = forecast.data_as_of || forecast.latest_date;
  const sourceLabel = forecast.snapshot_used ? "stored daily snapshot" : "computed on request";
  elements.forecastLessonContent.innerHTML = [
    lessonCard(
      "Start with the scenario range",
      `${termChip("bear")} is the weaker case, ${termChip("base")} is the central estimate, and ${termChip("bull")} is the stronger case. For ${escapeHtml(forecast.ticker)}, the base return is ${signedPercentLabel(forecast.returns.base)} over ${forecast.horizon_days} days.`,
    ),
    lessonCard(
      "Read confidence as model readiness",
      escapeHtml(confidenceExplanation(forecast.confidence, forecast.confidence_label)),
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

export function renderForecastChange(forecast) {
  if (!elements.forecastChange || !elements.tickerDataBadges) return;
  elements.tickerDataBadges.innerHTML = dataQualityBadges(forecast);
  const change = forecast.forecast_change || {};
  if (!change.available) {
    elements.forecastChange.innerHTML = `
      <div class="change-card">
        <span>Since last refresh</span>
        <strong>No prior snapshot</strong>
        <small>Run or store another forecast snapshot to compare model drift.</small>
      </div>
    `;
    return;
  }
  const tone = change.direction === "up" ? "positive" : change.direction === "down" ? "negative" : "neutral";
  elements.forecastChange.innerHTML = `
    <div class="change-card">
      <span>Since ${formatDate(change.previous_as_of)}</span>
      <strong class="${tone}">${formatSignedPercent(change.base_return_delta)}</strong>
      <small>Base target ${formatSignedNumber(change.base_target_delta)} · confidence ${formatSignedPercent(change.confidence_delta)}</small>
    </div>
  `;
}

export function renderTickerForecast(forecast) {
  state.currentForecast = forecast;
  renderForecastChart(forecast);
  renderForecastLearning(forecast);
  renderForecastChange(forecast);
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

export function renderTickerProfile(profile, forecast) {
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
          <span>${escapeHtml(label)}</span>
          <strong>${escapeHtml(value)}</strong>
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
