#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${SUPABASE_URL:?Set SUPABASE_URL before running the Supabase refresh.}"
: "${SUPABASE_SERVICE_ROLE_KEY:?Set SUPABASE_SERVICE_ROLE_KEY before running the Supabase refresh.}"

MODE="${STOCKIFY_REFRESH_MODE:-incremental}"
LOOKBACK_DAYS="${STOCKIFY_REFRESH_LOOKBACK_DAYS:-10}"
FRESHNESS_DAYS="${STOCKIFY_REFRESH_FRESHNESS_DAYS:-10}"
HORIZONS="${STOCKIFY_FORECAST_HORIZONS:-30,90,180,300}"
WINDOW_SIZE="${STOCKIFY_FORECAST_WINDOW_SIZE:-60}"
PROVIDER="${STOCKIFY_MARKET_DATA_PROVIDER:-yfinance}"

args=(
  "--mode" "$MODE"
  "--lookback-days" "$LOOKBACK_DAYS"
  "--freshness-days" "$FRESHNESS_DAYS"
  "--horizons" "$HORIZONS"
  "--window-size" "$WINDOW_SIZE"
  "--provider" "$PROVIDER"
)

if [[ -n "${STOCKIFY_REFRESH_START_DATE:-}" ]]; then
  args+=("--start-date" "$STOCKIFY_REFRESH_START_DATE")
fi

if [[ -n "${STOCKIFY_REFRESH_END_DATE:-}" ]]; then
  args+=("--end-date" "$STOCKIFY_REFRESH_END_DATE")
fi

if [[ "${STOCKIFY_REFRESH_DRY_RUN:-false}" == "true" ]]; then
  args+=("--dry-run")
fi

python offline/supabase_refresh.py "${args[@]}" "$@"
