import { API_TIMEOUT_MS } from "./endpoints.js";
import { state } from "../state/store.js";
import { showToast } from "../utils/dom.js";

export async function callApi(path, options = {}) {
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
    const error = new Error(message || `Request failed: ${response.status}`);
    error.status = response.status;
    error.retryAfter = Number(response.headers.get("Retry-After") || 0);
    if (response.status === 429) {
      showToast("Too many requests — please wait a moment before retrying.", "error");
    }
    throw error;
  }
  return response.json();
}

export async function callSectionApi(section, path, options = {}) {
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

export function isAbort(error) {
  return error && error.name === "AbortError";
}

export function wait(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

export function isArtifactEngineLoadingError(error) {
  return error?.status === 503 && /artifact engine|foresight engine is loading/i.test(String(error.message || ""));
}

export async function callArtifactSectionApi(section, path, options = {}) {
  const attempts = Number(options.artifactAttempts || 8);
  const { artifactAttempts, ...requestOptions } = options;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      return await callSectionApi(section, path, requestOptions);
    } catch (error) {
      if (!isArtifactEngineLoadingError(error) || attempt === attempts - 1) {
        throw error;
      }
      if (attempt === 0) {
        showToast("Artifact engine is warming up. Retrying automatically.", "info");
      }
      const retryAfter = Number(error.retryAfter || 0);
      await wait(Math.max(4000, Math.min(15000, retryAfter * 1000 || 6000)));
    }
  }
  throw new Error("Artifact engine did not finish loading in time.");
}

export function isBackendConnectionError(error) {
  const message = String(error?.message || error || "");
  return /failed to fetch|networkerror|load failed|request timed out|backend is not connected/i.test(message);
}
