import csv, numpy as np, psycopg
from pgvector.psycopg import register_vector

embs = np.load('data/phase2/phase2_embs.npy')
rows = list(csv.DictReader(open('data/phase2/phase2_tracks.csv', encoding='utf-8')))
assert len(rows) == len(embs) == 2000

with psycopg.connect('postgresql://postgres:sonar@localhost:5432/postgres') as conn:
    register_vector(conn)
    with conn.cursor() as cur:
        cur.executemany(
            'INSERT INTO tracks (track_id, path, tags, embedding) VALUES (%s, %s, %s, %s)',
            [(r['track_id'], r['path'], r['tags'].split('|') if r['tags'] else [], e)
             for r, e in zip(rows, embs)])
        cur.execute('SELECT count(*) FROM tracks')
        print(cur.fetchone()[0])