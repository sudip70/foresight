export const DEPLOYED_API_BASE = "https://foresight-backend-a5qx.onrender.com";

export const LOCAL_API_BASE = "http://localhost:8000";

export const API_TIMEOUT_MS = 45_000;

export const MAX_FORECAST_HORIZON_DAYS = 730;

export const MAX_FORECAST_WINDOW_SIZE = 756;

export const MAX_RL_BACKTEST_STEPS = 120;

export const RETIRED_API_BASES = new Set([
  "https://stockify-backend-adc6.onrender.com",
]);

export function normalizeApiBase(value) {
  return String(value || "").trim().replace(/\/$/, "");
}

export function canonicalApiBase(value) {
  const normalized = normalizeApiBase(value);
  return RETIRED_API_BASES.has(normalized) ? DEPLOYED_API_BASE : normalized;
}

export function queryApiBase() {
  try {
    return new URLSearchParams(window.location.search).get("apiBase");
  } catch {
    return "";
  }
}

export function isLocalHost(host) {
  return !host || host === "localhost" || host === "127.0.0.1" || host === "::1" || host === "[::1]";
}

export function isLocalApiBase(value) {
  try {
    return isLocalHost(new URL(value).hostname);
  } catch {
    return false;
  }
}

export function defaultApiBase() {
  const host = window.location?.hostname || "";
  return isLocalHost(host) ? LOCAL_API_BASE : DEPLOYED_API_BASE;
}

export function initialApiBase() {
  const explicitApiBase = queryApiBase() || window.FORESIGHT_API_BASE;
  if (explicitApiBase) return canonicalApiBase(explicitApiBase);

  const savedApiBase = canonicalApiBase(localStorage.getItem("foresight-api-base"));
  if (savedApiBase) {
    localStorage.setItem("foresight-api-base", savedApiBase);
  }
  const host = window.location?.hostname || "";
  if (savedApiBase && (isLocalHost(host) || !isLocalApiBase(savedApiBase))) {
    return savedApiBase;
  }
  return defaultApiBase();
}
