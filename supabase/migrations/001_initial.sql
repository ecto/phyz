-- phyz@home: distributed quantum gravity computation
-- Schema for work unit distribution and result collection

-- Work units: the parameter grid
create table work_units (
  id uuid primary key default gen_random_uuid(),
  params jsonb not null,                -- {triangulation, edge_lengths, g_squared, partition}
  completed_count int not null default 0,
  consensus_result jsonb,               -- set when 2 results agree
  status text not null default 'pending'
    check (status in ('pending', 'complete', 'disputed')),
  created_at timestamptz default now()
);

-- Results: individual contributions
create table results (
  id uuid primary key default gen_random_uuid(),
  work_unit_id uuid references work_units(id) not null,
  contributor_id uuid not null,
  result jsonb not null,                -- {s_ee, s_shannon, s_distillable, a_cut, energy, walltime_ms}
  submitted_at timestamptz default now()
);

-- Contributors: anonymous browser identities
create table contributors (
  id uuid primary key default gen_random_uuid(),
  fingerprint text unique not null,
  total_units int not null default 0,
  last_seen_at timestamptz default now(),
  created_at timestamptz default now()
);

-- Indexes
create index idx_work_units_status on work_units(status);
create index idx_results_work_unit on results(work_unit_id);
create index idx_results_contributor on results(contributor_id);

-- Enable realtime
alter publication supabase_realtime add table results;
alter publication supabase_realtime add table work_units;

-- RLS
alter table work_units enable row level security;
create policy "anyone can read work_units" on work_units for select using (true);

alter table results enable row level security;
create policy "anyone can read results" on results for select using (true);
create policy "anyone can insert results" on results for insert with check (true);

alter table contributors enable row level security;
create policy "anyone can read contributors" on contributors for select using (true);
create policy "anyone can insert contributors" on contributors for insert with check (true);
create policy "anyone can update own contributor" on contributors for update using (true);

-- Consensus trigger: on result insert, check if 2 results agree
create or replace function check_consensus() returns trigger as $$
declare
  v_other results;
  v_tolerance float := 1e-10;
begin
  -- Find another result for the same work unit
  select * into v_other from results
    where work_unit_id = NEW.work_unit_id
    and id != NEW.id
    limit 1;

  if v_other.id is not null then
    -- Check if s_ee values agree within fp tolerance
    if abs((NEW.result->>'s_ee')::float - (v_other.result->>'s_ee')::float) < v_tolerance then
      -- Consensus! Mark work unit complete with this result
      update work_units
        set status = 'complete',
            completed_count = 2,
            consensus_result = NEW.result
        where id = NEW.work_unit_id;
    else
      -- Disagreement — need a 3rd result (status stays 'pending')
      update work_units
        set completed_count = completed_count + 1
        where id = NEW.work_unit_id;
    end if;
  else
    -- First result for this unit
    update work_units
      set completed_count = 1
      where id = NEW.work_unit_id;
  end if;

  -- Update contributor stats
  update contributors
    set total_units = total_units + 1, last_seen_at = now()
    where id = NEW.contributor_id;

  return NEW;
end;
$$ language plpgsql;

create trigger on_result_insert
  after insert on results
  for each row execute function check_consensus();
