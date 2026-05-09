import { elements, literacy, state } from "../state/store.js";
import {
  escapeHtml,
  assetClassNoun,
  formatCurrency,
  formatDate,
  formatInteger,
  formatNumber,
  formatPercent,
  formatSignedPercent,
  riskPreferenceLabel,
  signedPercentLabel,
  toneForValue,
} from "../utils/formatters.js";
import { badge, insight, lessonCard, metricCard, termChip } from "../utils/dom.js";
import { selectedHorizonDays } from "../utils/validation.js";
import { renderDataHealthCards } from "./diagnostics.js";

export function renderMetricBlock(target, entries) {
  target.innerHTML = entries
    .map(([label, value, tooltip, subtitle]) => metricCard(label, value, tooltip, subtitle))
    .join("");
}

export function renderAllocationTable(target, allocations, options = {}) {
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
            <strong>${escapeHtml(allocation.ticker || allocation.asset_class)}</strong>
            <span>${escapeHtml(allocation.asset_class || "")}</span>
            <span>${formatPercent(allocation.weight)}</span>
            <span>${formatCurrency(allocation.amount || 0)}</span>
          </div>
        `,
      )
      .join("")}
  `;
}

export function renderTradePlan(target, trades) {
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
            <strong>${escapeHtml(trade.ticker)}</strong>
            <span>${formatPercent(trade.weight)}</span>
            <span>${formatCurrency(trade.amount)}</span>
          </div>
        `;
      })
      .join("")}
  `;
}

export function renderWarnings(target, warnings, tone = "warn") {
  target.innerHTML = (warnings || []).map((warning) => insight(warning, tone)).join("");
}

function forecastLookup(result) {
  const entries = [...(result?.source_forecasts || []), ...(result?.asset_allocations || [])];
  return Object.fromEntries(entries.map((entry) => [entry.ticker, entry]));
}

function forecastBaseReturn(entry) {
  return Number(entry?.returns?.base ?? entry?.base_return ?? 0);
}

function derivedBenchmarkRows(result) {
  const amount = Number(result?.amount || 0);
  const portfolioBaseReturn = Number(result?.summary?.base_return || 0);
  const byTicker = forecastLookup(result);
  const cash = byTicker.CASH;
  const rows = [];
  if (cash) {
    const baseReturn = forecastBaseReturn(cash);
    rows.push({
      label: "Cash return",
      ticker: "CASH",
      available: true,
      base_return: baseReturn,
      base_value: amount * (1 + baseReturn),
      delta_vs_portfolio_base: baseReturn - portfolioBaseReturn,
      note: "Cash sleeve used by the simulator.",
    });
  }

  [
    ["SPY", "S&P 500 ETF"],
    ["QQQ", "Nasdaq 100 ETF"],
  ].forEach(([ticker, label]) => {
    const forecast = byTicker[ticker];
    if (!forecast) return;
    const baseReturn = forecastBaseReturn(forecast);
    rows.push({
      label,
      ticker,
      available: true,
      base_return: baseReturn,
      base_value: amount * (1 + baseReturn),
      delta_vs_portfolio_base: baseReturn - portfolioBaseReturn,
      confidence: forecast.confidence,
    });
  });

  return rows;
}

function allocationRole(assetClass) {
  return {
    stock: "Growth sleeve",
    etf: "Diversifier",
    crypto: "High-volatility sleeve",
    cash: "Downside buffer",
  }[assetClass] || "Scenario sleeve";
}

function derivedAllocationExplanations(result) {
  return (result?.asset_allocations || []).map((allocation) => {
    const ticker = allocation.ticker;
    const assetClass = allocation.asset_class;
    const baseReturn = Number(allocation.base_return || 0);
    const bearReturn = Number(allocation.bear_return || 0);
    const confidence = Number(allocation.confidence || 0);
    const why =
      ticker === "CASH"
        ? "Cash is included to cushion downside scenarios and keep part of the portfolio stable."
        : `${ticker} is included as ${assetClassNoun(assetClass)} exposure with ${formatSignedPercent(baseReturn)} base return, ${formatSignedPercent(bearReturn)} bear return, and ${formatPercent(confidence)} confidence.`;
    return {
      ticker,
      asset_class: assetClass,
      weight: allocation.weight,
      role: allocationRole(assetClass),
      why,
      base_return: baseReturn,
      confidence,
    };
  });
}

export function renderBenchmarkComparison(result) {
  if (!elements.benchmarkComparison) return;
  const rows = result?.benchmark_comparison?.length
    ? result.benchmark_comparison
    : derivedBenchmarkRows(result);
  if (!rows.length) {
    elements.benchmarkComparison.innerHTML = `<div class="muted">Run a simulation to compare against cash, SPY, and QQQ when available.</div>`;
    return;
  }
  elements.benchmarkComparison.innerHTML = `
    <div class="row table-head benchmark-row">
      <span>Benchmark</span>
      <span>Base return</span>
      <span>Base value</span>
      <span>Vs portfolio</span>
    </div>
    ${rows
      .map((row) => {
        if (!row.available) {
          return `
            <div class="row benchmark-row unavailable">
              <strong>${escapeHtml(row.label || row.ticker)}</strong>
              <span>Unavailable</span>
              <span>--</span>
              <span>${escapeHtml(row.note || "Not enough benchmark data")}</span>
            </div>
          `;
        }
        return `
          <div class="row benchmark-row">
            <strong>${escapeHtml(row.label || row.ticker)}</strong>
            <span class="${toneForValue(row.base_return)}">${formatPercent(row.base_return)}</span>
            <span>${formatCurrency(row.base_value)}</span>
            <span class="${toneForValue(row.delta_vs_portfolio_base)}">${formatSignedPercent(row.delta_vs_portfolio_base)}</span>
          </div>
        `;
      })
      .join("")}
  `;
}

export function renderAllocationWhy(result) {
  if (!elements.allocationWhy) return;
  const explanations = result?.allocation_explanations?.length
    ? result.allocation_explanations
    : derivedAllocationExplanations(result);
  if (!explanations.length) {
    elements.allocationWhy.innerHTML = `<div class="muted">Run a simulation to see allocation explanations.</div>`;
    return;
  }
  elements.allocationWhy.innerHTML = explanations
    .slice()
    .sort((left, right) => Number(right.weight || 0) - Number(left.weight || 0))
    .slice(0, 4)
    .map(
      (entry) => `
        <article class="why-allocation-card">
          <div>
            <strong>${escapeHtml(entry.ticker)}</strong>
            <span>${escapeHtml(entry.role || entry.asset_class)}</span>
          </div>
          <p>${escapeHtml(entry.why || "Scenario-ranked allocation.")}</p>
          <div class="why-allocation-meta">
            ${badge(formatPercent(entry.weight), "info")}
            ${badge(`Base ${formatSignedPercent(entry.base_return)}`, toneForValue(entry.base_return))}
          </div>
        </article>
      `,
    )
    .join("");
}

export function renderClassAllocation(allocations) {
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

export function renderPortfolioLearning(result = state.simulation) {
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

export function renderSimulation(result) {
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
  renderBenchmarkComparison(result);
  renderAllocationWhy(result);
  renderPortfolioLearning(result);
  if (elements.footerDataAsOf && state.universe?.latest_date) {
    elements.footerDataAsOf.textContent = `Data as of ${formatDate(state.universe.latest_date)}`;
  }
  renderDataHealthCards();
}

export function renderRlAllocation(result) {
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

export function renderBacktest(result) {
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
