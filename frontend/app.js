const elements = {
  apiBase: document.querySelector("#apiBase"),
  apiStatus: document.querySelector("#apiStatus"),
  saveApiBase: document.querySelector("#saveApiBase"),
  risk: document.querySelector("#risk"),
  riskValue: document.querySelector("#riskValue"),
  amount: document.querySelector("#amount"),
  duration: document.querySelector("#duration"),
  windowSize: document.querySelector("#windowSize"),
  runInference: document.querySelector("#runInference"),
  runExplanations: document.querySelector("#runExplanations"),
  runBacktest: document.querySelector("#runBacktest"),
  summaryCards: document.querySelector("#summaryCards"),
  classAllocations: document.querySelector("#classAllocations"),
  assetAllocations: document.querySelector("#assetAllocations"),
  macroSnapshot: document.querySelector("#macroSnapshot"),
  explanationCards: document.querySelector("#explanationCards"),
  healthBlock: document.querySelector("#healthBlock"),
  modelsBlock: document.querySelector("#modelsBlock"),
  backtestSummary: document.querySelector("#backtestSummary"),
  equityCurve: document.querySelector("#equityCurve"),
};

const state = {
  apiBase: localStorage.getItem("stockify-api-base") || "http://localhost:8000",
};

const formatCurrency = (value) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(value);

const formatPercent = (value) => `${(value * 100).toFixed(2)}%`;

function requestPayload() {
  return {
    amount: Number(elements.amount.value),
    risk: Number(elements.risk.value),
    duration: Number(elements.duration.value),
    window_size: Number(elements.windowSize.value),
    strict_validation: false,
  };
}

async function callApi(path, options = {}) {
  const response = await fetch(`${state.apiBase}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const payload = await response.text();
    throw new Error(payload || `Request failed: ${response.status}`);
  }
  return response.json();
}

function renderSummary(summary) {
  elements.summaryCards.innerHTML = "";
  const cards = [
    ["Expected daily return", formatPercent(summary.expected_daily_return)],
    ["Annualized return", formatPercent(summary.annualized_return)],
    ["Portfolio variance", summary.portfolio_variance.toFixed(6)],
    ["Annualized volatility", formatPercent(summary.annualized_volatility)],
  ];
  cards.forEach(([label, value]) => {
    const card = document.createElement("article");
    card.className = "summary-card";
    card.innerHTML = `<h3>${label}</h3><strong>${value}</strong>`;
    elements.summaryCards.appendChild(card);
  });
}

function renderAllocations(target, allocations) {
  const wrapper = document.createElement("div");
  wrapper.className = "table";
  allocations.forEach((allocation) => {
    const label = allocation.ticker || allocation.asset_class;
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `
      <strong>${label}</strong>
      <span>${formatPercent(allocation.weight)}</span>
      <span>${formatCurrency(allocation.amount)}</span>
    `;
    wrapper.appendChild(row);
  });
  target.innerHTML = "";
  target.appendChild(wrapper);
}

function renderMacro(snapshot) {
  elements.macroSnapshot.innerHTML = "";
  snapshot.macro.forEach((entry) => {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `
      <strong>${entry.name}</strong>
      <span>${entry.normalized_value.toFixed(4)}</span>
      <span>Regime ${snapshot.global_regime}</span>
    `;
    elements.macroSnapshot.appendChild(row);
  });
}

function renderExplanationTargets(targets) {
  elements.explanationCards.innerHTML = "";
  targets.forEach((target) => {
    const container = document.createElement("article");
    container.className = "driver-list";
    if (!target.available) {
      container.innerHTML = `
        <h3>${target.target}</h3>
        <p class="status-warn">Unavailable. Surrogate fidelity ${target.fidelity.toFixed(3)} is below threshold.</p>
      `;
      elements.explanationCards.appendChild(container);
      return;
    }

    const groupedRows = Object.entries(target.grouped_contributions)
      .map(
        ([group, value]) =>
          `<div class="row"><strong>${group}</strong><span>${value.toFixed(4)}</span><span>${value >= 0 ? "positive" : "negative"}</span></div>`,
      )
      .join("");

    const positive = target.top_positive_drivers
      .map((driver) => `<li>${driver.group}: ${driver.value.toFixed(4)}</li>`)
      .join("");
    const negative = target.top_negative_drivers
      .map((driver) => `<li>${driver.group}: ${driver.value.toFixed(4)}</li>`)
      .join("");

    container.innerHTML = `
      <h3>${target.target}</h3>
      <p class="muted">Surrogate fidelity: ${target.fidelity.toFixed(3)}</p>
      <p>${target.plain_language ?? "No narrative available."}</p>
      <div class="drivers">
        <div class="driver-list">
          <h4>Top positive drivers</h4>
          <ul>${positive}</ul>
        </div>
        <div class="driver-list">
          <h4>Top negative drivers</h4>
          <ul>${negative}</ul>
        </div>
      </div>
      <div class="stack">${groupedRows}</div>
    `;
    elements.explanationCards.appendChild(container);
  });
}

function renderBacktest(result) {
  elements.backtestSummary.innerHTML = "";
  Object.entries(result.summary_metrics).forEach(([label, value]) => {
    const card = document.createElement("article");
    card.className = "summary-card";
    const display =
      label.includes("return") || label.includes("drawdown")
        ? formatPercent(value)
        : label.includes("value")
          ? formatCurrency(value)
          : value.toFixed(4);
    card.innerHTML = `<h3>${label.replaceAll("_", " ")}</h3><strong>${display}</strong>`;
    elements.backtestSummary.appendChild(card);
  });

  const maxValue = Math.max(...result.equity_curve.map((point) => point.value));
  elements.equityCurve.innerHTML = "";
  result.equity_curve.slice(-20).forEach((point) => {
    const row = document.createElement("div");
    row.className = "curve-row";
    row.innerHTML = `
      <span>Step ${point.step}</span>
      <div class="curve-bar" style="width:${(point.value / maxValue) * 100}%"></div>
    `;
    elements.equityCurve.appendChild(row);
  });
}

async function refreshDiagnostics() {
  elements.apiStatus.textContent = "Connecting to backend…";
  try {
    const [health, models] = await Promise.all([
      callApi("/api/health"),
      callApi("/api/models"),
    ]);
    elements.healthBlock.textContent = JSON.stringify(health, null, 2);
    elements.modelsBlock.textContent = JSON.stringify(models, null, 2);
    elements.apiStatus.textContent = "Backend connected.";
    elements.apiStatus.className = "status-good";
  } catch (error) {
    elements.apiStatus.textContent = `Backend unavailable: ${error.message}`;
    elements.apiStatus.className = "status-bad";
    elements.healthBlock.textContent = "";
    elements.modelsBlock.textContent = "";
  }
}

async function runInference() {
  const payload = requestPayload();
  const result = await callApi("/api/inference", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  renderSummary(result.summary);
  renderAllocations(elements.classAllocations, result.class_allocations);
  renderAllocations(elements.assetAllocations, result.asset_allocations);
  renderMacro(result.latest_snapshot);
}

async function runExplanations() {
  const payload = requestPayload();
  const result = await callApi("/api/explanations", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  renderExplanationTargets(result.targets);
}

async function runBacktest() {
  const payload = {
    initial_amount: Number(elements.amount.value),
    risk: Number(elements.risk.value),
    window_size: Number(elements.windowSize.value),
    max_steps: 120,
    strict_validation: false,
  };
  const result = await callApi("/api/backtests", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  renderBacktest(result);
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((node) => node.classList.remove("is-active"));
    document.querySelectorAll(".view").forEach((node) => node.classList.remove("is-active"));
    tab.classList.add("is-active");
    document.querySelector(`#${tab.dataset.tab}`).classList.add("is-active");
  });
});

elements.risk.addEventListener("input", () => {
  elements.riskValue.textContent = Number(elements.risk.value).toFixed(2);
});

elements.saveApiBase.addEventListener("click", async () => {
  state.apiBase = elements.apiBase.value.replace(/\/$/, "");
  localStorage.setItem("stockify-api-base", state.apiBase);
  await refreshDiagnostics();
});

elements.runInference.addEventListener("click", () =>
  runInference().catch((error) => alert(error.message)),
);
elements.runExplanations.addEventListener("click", () =>
  runExplanations().catch((error) => alert(error.message)),
);
elements.runBacktest.addEventListener("click", () =>
  runBacktest().catch((error) => alert(error.message)),
);

elements.apiBase.value = state.apiBase;
elements.riskValue.textContent = Number(elements.risk.value).toFixed(2);
refreshDiagnostics();
