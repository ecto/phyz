-- Show anonymous contributors in the leaderboard with a generated name
CREATE OR REPLACE VIEW leaderboard AS
  SELECT
    COALESCE(auth_id, id) AS player_id,
    COALESCE(display_name, 'anon-' || LEFT(id::text, 6)) AS name,
    SUM(total_units) AS units,
    MAX(last_seen_at) AS last_active
  FROM contributors
  WHERE total_units > 0
  GROUP BY COALESCE(auth_id, id), COALESCE(display_name, 'anon-' || LEFT(id::text, 6))
  ORDER BY units DESC
  LIMIT 50;
