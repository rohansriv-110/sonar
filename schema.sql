CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS tracks (
  id           BIGSERIAL PRIMARY KEY,
  jamendo_id   TEXT UNIQUE NOT NULL,
  title        TEXT,
  artist       TEXT,
  tags         TEXT[],
  duration_sec REAL
);

CREATE TABLE IF NOT EXISTS track_embeddings (
  track_id   BIGINT REFERENCES tracks(id) ON DELETE CASCADE,
  model_name TEXT NOT NULL,
  embedding  vector(512) NOT NULL,
  PRIMARY KEY (track_id, model_name)
);
