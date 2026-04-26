create table if not exists public.market_index_snapshots (
  symbol text not null,
  as_of_date date not null,
  label text not null,
  display_name text,
  provider_symbol text not null,
  value numeric not null,
  previous_close numeric,
  change numeric,
  change_percent numeric,
  day_open numeric,
  day_high numeric,
  day_low numeric,
  volume numeric,
  currency text default 'USD',
  provider text not null,
  raw_payload jsonb,
  display_order integer not null default 0,
  ingested_at timestamptz not null default now(),
  primary key (symbol, as_of_date)
);

create index if not exists idx_market_index_snapshots_latest
  on public.market_index_snapshots (as_of_date desc, display_order asc);

notify pgrst, 'reload schema';
