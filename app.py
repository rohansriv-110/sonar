from contextlib import asynccontextmanager
import time, torch, psycopg
from fastapi import FastAPI, HTTPException
from pgvector.psycopg import register_vector
from rank_bm25 import BM25Okapi
from transformers import ClapModel, ClapProcessor

S = {}

@asynccontextmanager
async def lifespan(app):
    name = 'laion/larger_clap_music_and_speech'
    S['model'] = ClapModel.from_pretrained(name).eval()
    S['proc'] = ClapProcessor.from_pretrained(name)
    S['conn'] = psycopg.connect('postgresql://postgres:sonar@localhost:5433/postgres')
    register_vector(S['conn'])
    rows = S['conn'].execute('SELECT track_id, tag_text FROM tracks').fetchall()
    S['ids'], S['tags'] = [r[0] for r in rows], dict(rows)
    S['bm25'] = BM25Okapi([(r[1] or '').split() for r in rows])
    yield
    S['conn'].close()

app = FastAPI(lifespan=lifespan)

def embed(text):
    with torch.no_grad():
        t = S['model'].get_text_features(**S['proc'](text=[text], return_tensors='pt', padding=True))
        t = t if torch.is_tensor(t) else t.pooler_output
    return torch.nn.functional.normalize(t, dim=-1)[0].numpy()

def rrf(*lists, k=60):
    score = {}
    for lst in lists:
        for rank, tid in enumerate(lst, 1):
            score[tid] = score.get(tid, 0) + 1 / (k + rank)
    return sorted(score, key=score.get, reverse=True)

@app.get('/search')
def search(q: str, k: int = 10):
    if not q.strip():
        raise HTTPException(400, 'q must not be empty')
    k = max(1, min(k, 50))
    t0 = time.perf_counter()
    vec = [r[0] for r in S['conn'].execute(
        'SELECT track_id FROM tracks ORDER BY embedding <=> %s LIMIT 50', (embed(q),))]
    s = S['bm25'].get_scores(q.lower().split())
    kw = [S['ids'][i] for i in s.argsort()[::-1][:50] if s[i] > 0]
    top = rrf(vec, kw)[:k]
    return {'q': q, 'ms': round((time.perf_counter() - t0) * 1000),
            'results': [{'track_id': t, 'tags': S['tags'][t]} for t in top]}