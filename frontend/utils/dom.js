import { glossary, state } from "../state/store.js";
import {
  cssClassToken,
  escapeHtml,
  formatCurrency,
  formatPercent,
  signedPercentLabel,
} from "./formatters.js";

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

export function showHoverTooltip(event, content) {
  const tooltip = hoverTooltip();
  tooltip.innerHTML = content;
  moveHoverTooltip(event);
  tooltip.classList.add("is-visible");
}

export function hideHoverTooltip() {
  const tooltip = document.querySelector(".hover-tooltip");
  if (tooltip) {
    tooltip.classList.remove("is-visible");
  }
}

export function sentimentTooltipContent() {
  if (!state.sentiment) return "";
  return `
    <strong>Market sentiment: ${escapeHtml(state.sentiment.label)}</strong>
    <span>Score: ${escapeHtml(String(state.sentiment.score))}/100</span>
    <span>Base scenario average: ${signedPercentLabel(state.sentiment.baseAverage)}</span>
    <span>Bear scenario average: ${signedPercentLabel(state.sentiment.bearAverage)}</span>
    <span>Confidence average: ${formatPercent(state.sentiment.confidenceAverage)}</span>
  `;
}

export function allocationSegmentAtPoint(event, donut) {
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

export function allocationTooltipContent(segment) {
  if (!segment) return "";
  return `
    <strong>${escapeHtml(segment.label)}</strong>
    <span>${formatPercent(segment.weight)} of portfolio</span>
    <span>${formatCurrency(segment.amount || 0)}</span>
  `;
}

export function sparklineSvg(change) {
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

export function insight(message, tone = "info") {
  return `<article class="insight ${cssClassToken(tone, "info")}"><span>${escapeHtml(message)}</span></article>`;
}

export function metricCard(label, value, tooltip, subtitle = "", valueTone = "") {
  const safeLabel = escapeHtml(label);
  const safeTooltip = escapeHtml(tooltip);
  const tip = tooltip ? `<button class="why-btn" data-why="${safeLabel}: ${safeTooltip}" type="button">?</button>` : "";
  const detail = subtitle ? `<span class="metric-subtitle">${escapeHtml(subtitle)}</span>` : "";
  const toneClass = valueTone ? ` ${cssClassToken(valueTone)}` : "";
  return `
    <article class="metric">
      <h3>${safeLabel} ${tip}</h3>
      <strong class="animate-number${toneClass}">${escapeHtml(value)}</strong>
      ${detail}
      <div class="why-popover-slot"></div>
    </article>
  `;
}

export function lessonCard(title, body, detail = "") {
  return `
    <article class="lesson-card">
      <strong>${escapeHtml(title)}</strong>
      <p>${body}</p>
      ${detail ? `<small>${detail}</small>` : ""}
    </article>
  `;
}

export function termChip(key) {
  const term = glossary[key];
  if (!term) return "";
  return `<span class="glossary-chip" data-term="${key}">${term.title}</span>`;
}

export function cssVar(name, fallback = "") {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

export function iconSvg(name, className = "ui-icon") {
  return `<svg class="${className}" aria-hidden="true"><use href="#icon-${name}"></use></svg>`;
}

export function showToast(message, type = "info") {
  const container = document.getElementById("toastContainer");
  if (!container) return;
  const icons = { info: "info-circle", error: "x-circle", success: "check-circle" };
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `${iconSvg(icons[type] || icons.info, "toast-icon")}<span>${escapeHtml(message)}</span>`;
  container.appendChild(toast);
  setTimeout(() => { toast.style.opacity = "0"; toast.style.transition = "opacity 200ms"; setTimeout(() => toast.remove(), 250); }, 4000);
}

export function setLoading(target) {
  target.innerHTML = `
    <div class="skeleton">
      <div class="skeleton-text"></div>
      <div class="skeleton-text"></div>
      <div class="skeleton-text"></div>
    </div>
  `;
}

export function setError(target, message) {
  target.innerHTML = `<article class="error-banner">${escapeHtml(message)}</article>`;
}

export function setBackendEmptyState(visible) {
  const emptyState = document.getElementById("marketEmptyState");
  emptyState?.classList.toggle("is-visible", Boolean(visible));
}

export function badge(label, tone = "neutral") {
  return `<span class="data-badge ${tone}">${escapeHtml(label)}</span>`;
}
