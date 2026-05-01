#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${SUPABASE_URL:?Set SUPABASE_URL before running the Supabase refresh.}"
: "${SUPABASE_SERVICE_ROLE_KEY:?Set SUPABASE_SERVICE_ROLE_KEY before running the Supabase refresh.}"

MODE="${FORESIGHT_REFRESH_MODE:-${STOCKIFY_REFRESH_MODE:-incremental}}"
LOOKBACK_DAYS="${FORESIGHT_REFRESH_LOOKBACK_DAYS:-${STOCKIFY_REFRESH_LOOKBACK_DAYS:-10}}"
FRESHNESS_DAYS="${FORESIGHT_REFRESH_FRESHNESS_DAYS:-${STOCKIFY_REFRESH_FRESHNESS_DAYS:-10}}"
HORIZONS="${FORESIGHT_FORECAST_HORIZONS:-${STOCKIFY_FORECAST_HORIZONS:-30,90,180,300}}"
WINDOW_SIZE="${FORESIGHT_FORECAST_WINDOW_SIZE:-${STOCKIFY_FORECAST_WINDOW_SIZE:-60}}"
PROVIDER="${FORESIGHT_MARKET_DATA_PROVIDER:-${STOCKIFY_MARKET_DATA_PROVIDER:-yfinance}}"
START_DATE="${FORESIGHT_REFRESH_START_DATE:-${STOCKIFY_REFRESH_START_DATE:-}}"
END_DATE="${FORESIGHT_REFRESH_END_DATE:-${STOCKIFY_REFRESH_END_DATE:-}}"
DRY_RUN="${FORESIGHT_REFRESH_DRY_RUN:-${STOCKIFY_REFRESH_DRY_RUN:-false}}"

args=(
  "--mode" "$MODE"
  "--lookback-days" "$LOOKBACK_DAYS"
  "--freshness-days" "$FRESHNESS_DAYS"
  "--horizons" "$HORIZONS"
  "--window-size" "$WINDOW_SIZE"
  "--provider" "$PROVIDER"
)

if [[ -n "$START_DATE" ]]; then
  args+=("--start-date" "$START_DATE")
fi

if [[ -n "$END_DATE" ]]; then
  args+=("--end-date" "$END_DATE")
fi

if [[ "$DRY_RUN" == "true" ]]; then
  args+=("--dry-run")
fi

python offline/supabase_refresh.py "${args[@]}" "$@"
