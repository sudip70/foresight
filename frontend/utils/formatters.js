export const formatCurrency = (value) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(Number(value || 0));

export const formatPercent = (value) => `${(Number(value || 0) * 100).toFixed(2)}%`;

export const formatNumber = (value, digits = 2) => Number(value || 0).toFixed(digits);

export const formatIndexValue = (value) =>
  isMissing(value)
    ? "Unavailable"
    : new Intl.NumberFormat("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      }).format(Number(value));

export const formatSignedNumber = (value) => {
  if (isMissing(value)) return "Unavailable";
  const parsed = Number(value);
  return `${parsed >= 0 ? "+" : ""}${formatNumber(parsed, 2)}`;
};

export const formatSignedPercent = (value) => {
  if (isMissing(value)) return "Unavailable";
  const parsed = Number(value);
  return `${parsed >= 0 ? "+" : ""}${formatPercent(parsed)}`;
};

export function isMissing(value) {
  return value === null || value === undefined || value === "" || Number.isNaN(Number(value));
}

export function formatCurrencyOptional(value) {
  return isMissing(value) ? "Unavailable" : formatCurrency(value);
}

export function formatPercentOptional(value) {
  return isMissing(value) ? "Unavailable" : formatPercent(value);
}

export function formatNumberOptional(value, digits = 2) {
  return isMissing(value) ? "Unavailable" : formatNumber(value, digits);
}

export function formatLargeNumber(value) {
  if (isMissing(value)) return "Unavailable";
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 2,
  }).format(Number(value));
}

export function formatDate(value) {
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

export function formatDateTime(value) {
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

export function formatInteger(value) {
  if (isMissing(value)) return "--";
  return new Intl.NumberFormat("en-US").format(Number(value));
}

export function classCoverageLabel(universe) {
  const groups = universe?.asset_classes || [];
  if (!groups.length) return "assets tracked";
  return groups
    .map((group) => `${formatInteger(group.tickers?.length || 0)} ${group.asset_class}`)
    .join(" / ");
}

export function formatSourceLabel(value) {
  return String(value || "unknown")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

export function escapeHtml(value) {
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

export function cssClassToken(value, fallback = "") {
  const token = String(value || "");
  return /^[a-z0-9_-]+$/i.test(token) ? token : fallback;
}

export function hasProfileValue(value) {
  if (value === null || value === undefined) return false;
  if (typeof value === "string") return value.trim() !== "";
  if (typeof value === "number") return Number.isFinite(value);
  return true;
}

export function profileText(value) {
  return hasProfileValue(value) ? String(value).trim() : "";
}

export const signedPercentLabel = formatSignedPercent;

export function assetClassNoun(value) {
  const normalized = String(value || "").toLowerCase();
  if (normalized === "etf") return "an ETF";
  if (normalized === "crypto") return "a crypto asset";
  if (normalized === "stock") return "a stock";
  return normalized ? `a ${normalized}` : "an asset";
}

export function buildTickerAboutParagraph(profile, forecast) {
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

export function toneForValue(value) {
  return Number(value || 0) >= 0 ? "positive" : "negative";
}

export function riskPreferenceLabel(value) {
  const risk = Number(value || 0);
  if (risk >= 0.75) return "aggressive";
  if (risk >= 0.45) return "balanced";
  return "conservative";
}

export function confidenceExplanation(confidence, label = "") {
  const value = Number(confidence || 0);
  if (value >= 0.7) {
    return `${label || "High"} confidence means the model sees relatively stable data and a narrower scenario band.`;
  }
  if (value >= 0.45) {
    return `${label || "Medium"} confidence means the forecast is usable for comparison, but the range still deserves attention.`;
  }
  return `${label || "Low"} confidence means the scenario is more uncertain, so the educational focus should be risk and assumptions.`;
}
