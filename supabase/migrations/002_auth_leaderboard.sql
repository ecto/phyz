-- Auth + leaderboard support for phyz@home
-- Adds optional email auth (Supabase Auth) and public leaderboard

ALTER TABLE contributors ADD COLUMN auth_id uuid UNIQUE;
ALTER TABLE contributors ADD COLUMN display_name text;
CREATE INDEX idx_contributors_total ON contributors(total_units DESC);

-- Allow authenticated users to update their own contributor row
CREATE POLICY "auth users can update own contributor"
  ON contributors FOR UPDATE
  USING (auth_id = auth.uid());

-- Leaderboard view: merges multiple devices per auth_id
CREATE VIEW leaderboard AS
  SELECT
    COALESCE(auth_id, id) AS player_id,
    display_name AS name,
    SUM(total_units) AS units,
    MAX(last_seen_at) AS last_active
  FROM contributors
  WHERE display_name IS NOT NULL
  GROUP BY COALESCE(auth_id, id), display_name
  ORDER BY units DESC
  LIMIT 50;
