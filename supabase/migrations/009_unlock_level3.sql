-- Unlock level 3 work units for distribution.
-- The original fetch_pending_work had `level < 3` as a safety guard
-- while levels 0-1 were being validated. Those are now complete,
-- so remove the cap to let level 3 units flow to volunteers.
create or replace function fetch_pending_work(p_limit int default 1024)
returns setof work_units
language sql stable
as $$
  select *
  from work_units
  where status = 'pending'
  order by (params->>'level')::int desc, random()
  limit p_limit;
$$;
