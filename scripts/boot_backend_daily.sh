#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DEFAULT_BACKEND_URL="https://foresight-backend-a5qx.onrender.com"
BACKEND_URL="${FORESIGHT_BACKEND_BOOT_URL:-${FORESIGHT_API_BASE:-$DEFAULT_BACKEND_URL}}"
BACKEND_URL="${BACKEND_URL%/}"
HEALTH_URL="${BACKEND_URL}/api/health"
ATTEMPTS="${FORESIGHT_BACKEND_BOOT_ATTEMPTS:-6}"
SLEEP_SECONDS="${FORESIGHT_BACKEND_BOOT_SLEEP_SECONDS:-20}"
TIMEOUT_SECONDS="${FORESIGHT_BACKEND_BOOT_TIMEOUT_SECONDS:-60}"

if [[ "${FORESIGHT_BACKEND_BOOT_DRY_RUN:-false}" == "true" ]]; then
  echo "Would boot backend by checking ${HEALTH_URL}"
  exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to boot the backend." >&2
  exit 1
fi

for attempt in $(seq 1 "$ATTEMPTS"); do
  echo "Booting backend (${attempt}/${ATTEMPTS}): ${HEALTH_URL}"
  status_code="$(curl -sS --max-time "$TIMEOUT_SECONDS" -o /dev/null -w "%{http_code}" "$HEALTH_URL" || true)"

  if [[ "$status_code" =~ ^2[0-9][0-9]$ ]]; then
    echo "Backend is awake. Health check returned HTTP ${status_code}."
    exit 0
  fi

  if [[ "$attempt" -lt "$ATTEMPTS" ]]; then
    echo "Health check returned HTTP ${status_code}; retrying in ${SLEEP_SECONDS}s."
    sleep "$SLEEP_SECONDS"
  fi
done

echo "Backend did not become healthy after ${ATTEMPTS} attempts." >&2
exit 1
