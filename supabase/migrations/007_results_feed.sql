-- RPC function to fetch recent results with contributor names and work unit
-- context. Used for the pageload viz feed and incremental realtime updates.
create or replace function recent_results_feed(
  p_limit int default 100,
  p_after timestamptz default null
)
returns table(
  result_id uuid,
  contributor_id uuid,
  contributor_name text,
  coupling_g2 double precision,
  result_data jsonb,
  submitted_at timestamptz
)
language sql stable
as $$
  select
    r.id,
    r.contributor_id,
    coalesce(c.display_name, 'anonymous'),
    (wu.params->>'coupling_g2')::double precision,
    r.result,
    r.submitted_at
  from results r
  join work_units wu on wu.id = r.work_unit_id
  left join contributors c on c.id = r.contributor_id
  where (p_after is null or r.submitted_at > p_after)
  order by r.submitted_at desc
  limit p_limit;
$$;
