import json, psycopg

STOP = {'with', 'beat', 'and', 'the', 'a'}
runs = json.load(open('data/phase2/phase3_runs.json'))
with psycopg.connect('postgresql://postgres:sonar@localhost:5433/postgres') as conn:
    tags = {r[0]: set((r[1] or '').split()) for r in conn.execute('SELECT track_id, tag_text FROM tracks')}

def words(prompt):
    w = prompt.lower().replace('-', '').split()
    w += [a + b for a, b in zip(w, w[1:])]          # "hip hop" -> "hiphop"
    return {x for x in w if x not in STOP}

tot = {'vector': 0, 'hybrid': 0}
for r in runs:
    q = words(r['prompt'])
    hits = {k: sum(1 for t in r[k] if tags[t] & q) for k in tot}
    for k in tot: tot[k] += hits[k]
    print(f"q{r['id']:02d}  vector {hits['vector']}/10  hybrid {hits['hybrid']}/10  {r['prompt']}")
print(f"TOTAL  vector {tot['vector']}/100  hybrid {tot['hybrid']}/100")