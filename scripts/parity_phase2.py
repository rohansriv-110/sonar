import csv, time, numpy as np, psycopg
from pgvector.psycopg import register_vector

embs = np.load('data/phase2/phase2_embs.npy')
qs = np.load('data/phase2/phase2_queries.npy')
ids = [r['track_id'] for r in csv.DictReader(open('data/phase2/phase2_tracks.csv', encoding='utf-8'))]

with psycopg.connect('postgresql://postgres:sonar@localhost:5433/postgres') as conn:
    register_vector(conn)
    cur = conn.cursor()
    ok = True
    for i, q in enumerate(qs, 1):
        brute = {ids[j] for j in np.argsort(-(embs @ q))[:10]}     # exact top-10
        t0 = time.perf_counter()
        cur.execute('SELECT track_id FROM tracks ORDER BY embedding <=> %s LIMIT 10', (q,))
        ms = (time.perf_counter() - t0) * 1000
        hnsw = {r[0] for r in cur.fetchall()}
        overlap = len(brute & hnsw)
        ok &= overlap >= 9 and ms < 50
        print(f'q{i:02d}  overlap {overlap}/10  {ms:.1f} ms')
    print('PASS' if ok else 'FAIL')