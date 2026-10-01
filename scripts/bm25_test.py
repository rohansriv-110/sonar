import psycopg
from rank_bm25 import BM25Okapi

with psycopg.connect('postgresql://postgres:sonar@localhost:5433/postgres') as conn:
    rows = conn.execute('SELECT track_id, tag_text FROM tracks').fetchall()

ids = [r[0] for r in rows]
docs = [(r[1] or '').split() for r in rows]
bm25 = BM25Okapi(docs)

q = 'nighttime jazz piano beat'.split()
scores = bm25.get_scores(q)
for i in scores.argsort()[::-1][:10]:
    print(f'{scores[i]:.2f}  {ids[i]}  {rows[i][1]}')
print('empty-tag tracks:', sum(1 for d in docs if not d))