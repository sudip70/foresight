create extension if not exists pgcrypto;

create table if not exists public.asset_universe (
  ticker text primary key,
  asset_class text not null check (asset_class in ('stock', 'crypto', 'etf')),
  display_name text,
  exchange text,
  currency text default 'USD',
  sector text,
  industry text,
  country text,
  provider_symbol text not null,
  benchmark_group text,
  min_history_days integer not null default 252,
  active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.market_ohlcv_daily (
  ticker text not null references public.asset_universe(ticker) on delete cascade,
  date date not null,
  open numeric,
  high numeric,
  low numeric,
  close numeric not null,
  adjusted_close numeric,
  volume numeric,
  provider text not null,
  ingested_at timestamptz not null default now(),
  primary key (ticker, date)
);

create index if not exists idx_market_ohlcv_daily_date
  on public.market_ohlcv_daily (date desc);

create table if not exists public.asset_profile_snapshots (
  ticker text not null references public.asset_universe(ticker) on delete cascade,
  as_of_date date not null,
  market_cap numeric,
  pe_ratio numeric,
  fifty_two_week_high numeric,
  fifty_two_week_low numeric,
  average_volume numeric,
  volume numeric,
  dividend_yield numeric,
  dividend_frequency text,
  ex_dividend_date date,
  bid numeric,
  ask numeric,
  last_sale numeric,
  day_open numeric,
  day_high numeric,
  day_low numeric,
  exchange text,
  margin_requirement numeric,
  raw_payload jsonb,
  ingested_at timestamptz not null default now(),
  primary key (ticker, as_of_date)
);

create table if not exists public.macro_observations (
  date date primary key,
  vix numeric,
  federal_funds_rate numeric,
  treasury_10y numeric,
  unemployment_rate numeric,
  cpi_all_items numeric,
  recession_indicator numeric,
  provider text not null,
  ingested_at timestamptz not null default now()
);

create table if not exists public.forecast_snapshots (
  ticker text not null references public.asset_universe(ticker) on delete cascade,
  as_of_date date not null,
  horizon_days integer not null,
  window_size integer not null,
  method_version text not null,
  latest_price numeric not null,
  bear_target numeric not null,
  base_target numeric not null,
  bull_target numeric not null,
  bear_return numeric not null,
  base_return numeric not null,
  bull_return numeric not null,
  volatility numeric not null,
  drawdown numeric not null,
  confidence numeric not null,
  confidence_label text not null,
  forecast_paths_json jsonb not null,
  created_at timestamptz not null default now(),
  primary key (ticker, as_of_date, horizon_days, window_size, method_version)
);

create index if not exists idx_forecast_snapshots_lookup
  on public.forecast_snapshots (horizon_days, window_size, method_version, as_of_date desc);

create table if not exists public.refresh_runs (
  id uuid primary key default gen_random_uuid(),
  provider text not null,
  status text not null check (status in ('running', 'completed', 'partial', 'failed')),
  started_at timestamptz not null default now(),
  finished_at timestamptz,
  requested_start_date date,
  requested_end_date date,
  rows_inserted integer not null default 0,
  rows_updated integer not null default 0,
  error text,
  metadata jsonb
);

create table if not exists public.refresh_run_items (
  run_id uuid not null references public.refresh_runs(id) on delete cascade,
  ticker text not null,
  stage text not null,
  status text not null check (status in ('completed', 'failed', 'skipped')),
  rows_written integer not null default 0,
  error text,
  created_at timestamptz not null default now(),
  primary key (run_id, ticker, stage)
);

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists set_asset_universe_updated_at on public.asset_universe;
create trigger set_asset_universe_updated_at
before update on public.asset_universe
for each row execute function public.set_updated_at();

notify pgrst, 'reload schema';
