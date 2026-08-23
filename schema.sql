CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS tracks (
  id           BIGSERIAL PRIMARY KEY,
  jamendo_id   TEXT UNIQUE NOT NULL,
  title        TEXT,
  artist       TEXT,
  tags         TEXT[],
  duration_sec REAL,
  embedding    vector(512)
);
