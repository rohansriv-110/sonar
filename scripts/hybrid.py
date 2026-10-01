import numpy as np, psycopg
from pgvector.psycopg import register_vector
from rank_bm25 import BM25Okapi

QUERIES = ["dark trap beat with heavy 808s", "nighttime jazz piano beat",
  "energetic electronic beat with punchy drums", "sad lo-fi hip hop with soft guitar",
  "chill relaxing beat with smooth synth pads", "epic cinematic beat with strings",
  "happy upbeat funk beat with bass guitar", "dreamy ambient beat with reverb piano",
  "aggressive rock beat with distorted guitar", "romantic slow beat with acoustic guitar"]
QVECS = np.load('data/phase2/phase2_queries.npy')

conn = psycopg.connect('postgresql://postgres:sonar@localhost:5433/postgres')
register_vector(conn)
rows = conn.execute('SELECT track_id, tag_text FROM tracks').fetchall()
ids = [r[0] for r in rows]
bm25 = BM25Okapi([(r[1] or '').split() for r in rows])

def vector_top(qvec, n=50):
    return [r[0] for r in conn.execute(
        'SELECT track_id FROM tracks ORDER BY embedding <=> %s LIMIT %s', (qvec, n))]

def keyword_top(text, n=50):
    s = bm25.get_scores(text.lower().split())
    return [ids[i] for i in s.argsort()[::-1][:n] if s[i] > 0]

def rrf(*lists, k=60):
    score = {}
    for lst in lists:
        for rank, tid in enumerate(lst, 1):
            score[tid] = score.get(tid, 0) + 1 / (k + rank)
    return sorted(score, key=score.get, reverse=True)

def hybrid(i, n=10):
    return rrf(vector_top(QVECS[i]), keyword_top(QUERIES[i]))[:n]


if __name__ == '__main__':
    i = 1   # "nighttime jazz piano beat"
    tags = dict(rows)
    print('VECTOR:', [tags[t] for t in vector_top(QVECS[i])[:10]])
    print('HYBRID:', [tags[t] for t in hybrid(i)])

    import json
    out = [{"id": i + 1, "prompt": QUERIES[i],
            "vector": vector_top(QVECS[i])[:10], "hybrid": hybrid(i)}
           for i in range(10)]
    json.dump(out, open('data/phase2/phase3_runs.json', 'w'), indent=1)
    uniq = {t for r in out for t in r['vector'] + r['hybrid']}
    print('tracks to label:', len(uniq))


