DROP TABLE IF EXISTS track_embeddings, tracks;
CREATE TABLE tracks (
  track_id  TEXT PRIMARY KEY,
  path      TEXT,
  tags      TEXT[],
  embedding vector(512)
);
CREATE INDEX ON tracks USING hnsw (embedding vector_cosine_ops);
