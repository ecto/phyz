-- Migration 008: Fix check_consensus trigger
--
-- The trigger was running as SECURITY INVOKER (the anon role),
-- but work_units only has a SELECT RLS policy — no UPDATE policy.
-- So all UPDATEs inside the trigger silently matched zero rows.
--
-- Fix: SECURITY DEFINER so the trigger runs as the postgres owner.

CREATE OR REPLACE FUNCTION check_consensus() RETURNS trigger
  SECURITY DEFINER
  SET search_path = public
AS $$
DECLARE
  v_other results;
  v_tolerance float := 1e-10;
  v_new_energy float;
  v_other_energy float;
BEGIN
  SELECT * INTO v_other FROM results
    WHERE work_unit_id = NEW.work_unit_id
    AND id != NEW.id
    LIMIT 1;

  IF v_other.id IS NOT NULL THEN
    v_new_energy := (NEW.result->>'ground_state_energy')::float;
    v_other_energy := (v_other.result->>'ground_state_energy')::float;
    IF abs(v_new_energy - v_other_energy) < v_tolerance THEN
      UPDATE work_units
        SET status = 'complete',
            completed_count = 2,
            consensus_result = NEW.result
        WHERE id = NEW.work_unit_id;
    ELSE
      UPDATE work_units
        SET completed_count = completed_count + 1
        WHERE id = NEW.work_unit_id;
    END IF;
  ELSE
    UPDATE work_units
      SET completed_count = 1
      WHERE id = NEW.work_unit_id;
  END IF;

  UPDATE contributors
    SET total_units = total_units + 1, last_seen_at = now()
    WHERE id = NEW.contributor_id;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Backfill: retroactively compute consensus for existing results.
-- For each work unit with 2+ results where energies match, mark complete.
WITH consensus AS (
  SELECT DISTINCT ON (r.work_unit_id)
    r.work_unit_id,
    r.result AS sample_result
  FROM results r
  INNER JOIN (
    SELECT work_unit_id, count(*) AS n
    FROM results
    GROUP BY work_unit_id
    HAVING count(*) >= 2
      AND max((result->>'ground_state_energy')::float)
        - min((result->>'ground_state_energy')::float) < 1e-10
  ) matched ON matched.work_unit_id = r.work_unit_id
  ORDER BY r.work_unit_id, r.submitted_at DESC
)
UPDATE work_units wu
  SET status = 'complete',
      completed_count = 2,
      consensus_result = c.sample_result
FROM consensus c
WHERE wu.id = c.work_unit_id;

-- Also set completed_count for work units that have results but no consensus
UPDATE work_units wu
  SET completed_count = sub.n
FROM (
  SELECT work_unit_id, count(*) AS n
  FROM results
  GROUP BY work_unit_id
) sub
WHERE wu.id = sub.work_unit_id
  AND wu.status != 'complete'
  AND wu.completed_count = 0;
