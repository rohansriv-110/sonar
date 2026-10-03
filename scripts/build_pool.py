import json, csv, os

runs = json.load(open('data/phase2/phase3_runs.json'))
paths = {r['track_id']: r['path'] for r in csv.DictReader(open('data/phase2/phase2_tracks.csv'))}

os.makedirs('eval', exist_ok=True)
rows = []
for r in runs:
    for tid in dict.fromkeys(r['vector'] + r['hybrid']):   # union, keeps order, no dupes
        rows.append([r['id'], r['prompt'], tid, paths[str(tid)]])

with open('eval/pool.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['prompt_id', 'prompt', 'track_id', 'path'])
    w.writerows(rows)

uniq = sorted({row[3] for row in rows})
open('eval/pool_paths.txt', 'w').write('\n'.join(uniq))
print('pairs to label:', len(rows), '| unique clips:', len(uniq))
