import csv, json, os, time
from judge import judge

RUNS = "data/phase2/phase3_runs.json"
TRACKS = "data/phase2/phase2_tracks.csv"
OUT = "data/llm_labels.json"

# track_id -> readable tags, e.g. "genre: ambient"
tags = {}
with open(TRACKS, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        tags[row["track_id"]] = [t.replace("---", ": ") for t in row["tags"].split("|")]

runs = json.load(open(RUNS, encoding="utf-8"))

# load existing labels so re-runs skip finished work
labels = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else []
done = {(l["prompt_id"], l["track_id"]) for l in labels}

for run in runs:
    pool = list(dict.fromkeys(run["vector"] + run["hybrid"]))   # dedupe, keep order
    for tid in pool:
        if (run["id"], tid) in done:
            continue
        score, reason = judge(run["prompt"], tags[tid])
        labels.append({"prompt_id": run["id"], "track_id": tid,
                       "score": score, "reason": reason})
        json.dump(labels, open(OUT, "w", encoding="utf-8"), indent=2)
        print(f"P{run['id']} {tid}: {score}")
        time.sleep(4.5)   # stay under 15 requests/min

print(f"Done: {len(labels)} labels")
