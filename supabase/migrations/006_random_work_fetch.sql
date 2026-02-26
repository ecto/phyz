-- RPC function to fetch pending work units in random order.
-- PostgREST doesn't support ORDER BY random(), so we use an RPC endpoint.
-- This ensures different volunteers get different subsets of work.
create or replace function fetch_pending_work(p_limit int default 1024)
returns setof work_units
language sql stable
as $$
  select *
  from work_units
  where status = 'pending'
    and (params->>'level')::int < 3
  order by (params->>'level')::int desc, random()
  limit p_limit;
$$;
