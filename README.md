# sonar

Text-to-audio music search using CLAP joint embeddings — find songs by how they *sound*, not by their tags.

Type `"melancholy lo-fi with warm analog hiss"` and rank real audio. No genre labels, no metadata matching, no keyword search. The query and the waveform are compared directly in a shared vector space.

> **Status: Phase 1 of 6 complete.** The embedding core works end-to-end on a single track. Corpus indexing, retrieval evaluation, and the agent layer are not built yet. See the roadmap below.

---

## Why this is hard

Text search is easy because text arrives pre-chunked into meaningful units — a token like `"melancholy"` already carries meaning. Audio does not. A 30-second track is ~1.3 million floats, each one an instantaneous air-pressure reading that means nothing on its own. Pitch, rhythm, and timbre only exist as patterns spanning thousands of samples.

The pipeline that closes that gap:

```
mp3
 └─► 1,323,000 raw samples @ 44.1kHz
      └─► resample to 48kHz              (CLAP's training rate)
           └─► mel spectrogram, 128 bands  (frequency content over time,
                                             spaced how human ears hear)
                └─► CLAP audio encoder
                     └─► 512-dim vector, L2-normalized
```

Meanwhile the text side runs through CLAP's text encoder into the *same* 512-dim space. The two encoders were trained contrastively on hundreds of thousands of (audio, caption) pairs, so matching pairs land close together. Cosine similarity between a sentence and a song becomes a meaningful number.

---

## Current result

10 seconds of a Brahms violin recording, scored against four candidate descriptions:

| Query | Cosine similarity |
|---|---|
| a solo violin playing a romantic classical melody | **0.3996** |
| upbeat funk with a slap bass groove | 0.1009 |
| aggressive heavy metal with distorted electric guitars | 0.0581 |
| ambient lo-fi hip hop for studying | 0.0555 |

The correct description scores ~4× the runner-up. The model was given no tags, no filename, and no metadata — only the waveform.

Two things worth noting:

- **Absolute scores are compressed.** CLAP's audio and text embeddings occupy slightly offset regions of the shared space (the "modality gap"), so a correct match at 0.40 is normal. Rank, never threshold on an absolute value.
- **The errors are musically sensible.** Funk outscoring metal is not noise — funk and classical share acoustic instrumentation and clean, undistorted tone. Coherent failure modes are evidence of a real representation.

---

## Stack

| Component | Choice |
|---|---|
| Joint embedding model | `laion/larger_clap_music_and_speech` (194M params) |
| Audio I/O and DSP | `librosa`, `soundfile` |
| Inference | PyTorch, CUDA (Colab T4) |
| Corpus | MTG-Jamendo — 55k CC-licensed tracks, 195 tags |

---

## Running it

`notebooks/01_clap_first_vector.ipynb` runs top to bottom on a free Colab T4. It downloads a sample track, plots its mel spectrogram, embeds 10 seconds of audio, embeds four text queries, and scores them against each other.

```bash
pip install -r requirements.txt
```

The heavy work runs on a rented GPU by design. Embedding a corpus is a batch job, not a laptop job — the same code runs on CUDA, Apple MPS, or CPU by swapping the device string.

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | CLAP embedding pipeline, cross-modal similarity | ✅ done |
| 2 | Corpus indexing — MTG-Jamendo subset into pgvector with HNSW | ⬜ |
| 3 | Hybrid retrieval — dense audio vectors + BM25 over tags, fused with RRF | ⬜ |
| 4 | Evaluation harness — recall@10, MRR, nDCG against held-out tags | ⬜ |
| 5 | Agent layer — LangGraph query planner and critic, Langfuse tracing | ⬜ |
| 6 | Deployment — Docker Compose, GitHub Actions CI, live demo | ⬜ |

Phase 4 is the one that matters most. Retrieval systems that were never measured are demos, not engineering — and a search engine that feels good on five hand-picked queries routinely falls apart on the sixth.

An open question for phase 4: whether a music-specialised joint embedding model outperforms general-purpose CLAP on this corpus. That comparison, run against a real eval set, is the point of building the harness.

---

## Notes

Audio files and computed embeddings are gitignored. MTG-Jamendo is distributed under Creative Commons licenses; see the [dataset repository](https://github.com/MTG/mtg-jamendo-dataset) for terms and download instructions.
