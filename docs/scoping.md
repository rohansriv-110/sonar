# Sonar — Scoping Doc (v0)

_Decisions made by Rohan, 24 Sep 2026. Update this file whenever evidence changes a decision._

## 1. Problem
Rappers and singers looking for a beat can't find one that matches the vibe in their head, because beat stores only let you filter by genre and BPM.

## 2. Input / Output
- **Input:** a free-text prompt (e.g. "dark lo-fi with jazzy piano")
- **Output:** top 10 instrumental beats ranked by cosine similarity, each with an audio preview and its tags

## 3. Latency
- **Target:** p95 < 3s from search to results
- Cold starts excluded for now

## 4. Budget
- ₹0/month by default
- Up to ₹500/month **only if** cold starts break the p95 < 3s target

## 5. Success metric
- **Metric:** nDCG@10 against hand labels
- **Target:** MVP baseline + 0.15 (stretch goal: 0.91)
- The CLAP score is NOT a quality metric: the model can't grade itself

## 6. Does AI make sense?
Tag search fails because users describe vibes in free text and tags are sparse; CLAP works because it listens to the audio directly.

## 7. MVP data
- ~1000 instrumental tracks from MTG-Jamendo (`raw_30s`), low-quality audio, ~3 GB (2 tar archives, stored in Google Drive)
- "Instrumental" = no `instrument---voice` tag (sparse, so some vocal tracks slip through)
- Embedding runs on Colab (free T4); embeddings saved to Drive
- License: non-commercial research use only

## 8. Labeling rubric
- Binary: 0 = miss, 1 = match
- **Rule:** a beat is a 1 if it matches the prompt's **mood** and at least one **instrument or genre** it names, judged by listening to the first **15 seconds**
- Unsure = 0

## MVP log
- run_001: CLAP larger_clap_music_and_speech, first 10s per track, 1102 tracks, ~30 ms/query on T4
- Labeled so far: prompt 1 → [4], prompt 2 → [4]

## Parked ideas (only add if evals justify them)
- LLM query decomposition: split the prompt into tags for hybrid search
- Better instrumental filter: CLAP zero-shot "vocals vs instrumental"

## Out of scope for v1
- Reference-beat upload as input
- Accounts / login
- Purchase links