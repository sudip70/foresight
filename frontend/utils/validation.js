import { MAX_FORECAST_HORIZON_DAYS, MAX_FORECAST_WINDOW_SIZE } from "../api/endpoints.js";
import { elements } from "../state/store.js";

export function selectedHorizonDays() {
  const parsed = Number(elements.forecastHorizon?.value || elements.horizon.value || 300);
  if (!Number.isFinite(parsed) || parsed < 1) return 300;
  return Math.min(Math.round(parsed), MAX_FORECAST_HORIZON_DAYS);
}

export function selectedWindowSize() {
  const parsed = Number(elements.windowSizeInline?.value || elements.windowSize.value || 60);
  if (!Number.isFinite(parsed) || parsed < 2) return 60;
  return Math.min(Math.round(parsed), MAX_FORECAST_WINDOW_SIZE);
}

export function dashboardPayload(extra = {}) {
  return {
    horizon_days: selectedHorizonDays(),
    risk: Number(elements.risk.value),
    window_size: selectedWindowSize(),
    strict_validation: true,
    ...extra,
  };
}

export function allocationPayload() {
  return {
    amount: Number(elements.amount.value),
    risk: Number(elements.risk.value),
    duration: selectedHorizonDays(),
    window_size: selectedWindowSize(),
    strict_validation: true,
  };
}

export function percentConstraintValue(input, fallbackPercent) {
  const parsed = Number(input?.value ?? fallbackPercent);
  const percent = Number.isFinite(parsed) ? Math.max(0, Math.min(parsed, 100)) : fallbackPercent;
  return percent / 100;
}

export function selectedPreferredAssetClasses() {
  return [
    elements.preferStock?.checked ? "stock" : null,
    elements.preferEtf?.checked ? "etf" : null,
    elements.preferCrypto?.checked ? "crypto" : null,
  ].filter(Boolean);
}

export function portfolioConstraintPayload() {
  const preferred = selectedPreferredAssetClasses();
  return {
    max_single_position_weight: percentConstraintValue(elements.maxSinglePosition, 25),
    max_crypto_weight: percentConstraintValue(elements.maxCryptoWeight, 35),
    min_cash_weight: percentConstraintValue(elements.minCashWeight, 2),
    preferred_asset_classes: preferred.length ? preferred : null,
  };
}

export function selectedSimulationTickers() {
  return Array.from(elements.simulationTickers.selectedOptions).map((option) => option.value);
}
