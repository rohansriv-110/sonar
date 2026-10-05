import json, math

RUNS = "data/phase2/phase3_runs.json"
LABELS = "data/llm_labels.json"
OUT = "data/eval_phase4.json"
K = 10
REL = 2          # score needed to count as "relevant" for recall/MRR

runs = json.load(open(RUNS, encoding="utf-8"))
labels = json.load(open(LABELS, encoding="utf-8"))

# (prompt_id, track_id) -> score
grade = {(l["prompt_id"], l["track_id"]): l["score"] for l in labels}

def recall_at_k(pid, ranked):
    relevant = {t for (p, t), s in grade.items() if p == pid and s >= REL}
    if not relevant:
        return None                      # skip prompts with no relevant tracks
    hits = sum(1 for t in ranked[:K] if t in relevant)
    return hits / len(relevant)

def mrr(pid, ranked):
    for i, t in enumerate(ranked[:K], start=1):
        if grade.get((pid, t), 0) >= REL:
            return 1 / i
    return 0.0

def ndcg_at_k(pid, ranked):
    dcg = sum((2 ** grade.get((pid, t), 0) - 1) / math.log2(i + 2)
              for i, t in enumerate(ranked[:K]))
    ideal = sorted((s for (p, _), s in grade.items() if p == pid), reverse=True)[:K]
    idcg = sum((2 ** s - 1) / math.log2(i + 2) for i, s in enumerate(ideal))
    return dcg / idcg if idcg else 0.0

results = {}
for system in ["vector", "hybrid"]:
    rec, rr, nd = [], [], []
    for run in runs:
        r = recall_at_k(run["id"], run[system])
        if r is not None:
            rec.append(r)
        rr.append(mrr(run["id"], run[system]))
        nd.append(ndcg_at_k(run["id"], run[system]))
    results[system] = {
        "recall@10": round(sum(rec) / len(rec), 3),
        "MRR": round(sum(rr) / len(rr), 3),
        "nDCG@10": round(sum(nd) / len(nd), 3),
    }

for system, m in results.items():
    print(f"{system:7} | recall@10 {m['recall@10']} | MRR {m['MRR']} | nDCG@10 {m['nDCG@10']}")

json.dump(results, open(OUT, "w", encoding="utf-8"), indent=2)