# Roadmap

Each milestone ends with a **git tag** (`v0.1`, `v0.2`, …) and an entry in `CHANGELOG.md`,
so every working state of the project is saved and can be checked out again.

Target: ~4 weeks part-time. The "MVP cut" marks what to keep if time runs short.

---

## v0.0 — Repo + architecture ✅
- [x] Repo, folder layout, `.gitignore`, `uv` project
- [x] `docs/ARCHITECTURE.md`, this roadmap

## v0.1 — Data spike (Week 1, days 1–2) · MVP ✅
- [x] `fetch_lexglue.py` — download + print label stats
- [x] `fetch_tosdr.py` — throttled, cached, resumable fetcher
- [x] Notebook/script: inspect 100 raw points by hand
- [x] Stats: #points, #with quotes, case distribution, rating distribution, #services
- [x] Check ToS;DR data licence / attribution terms
- [x] Decision log (`docs/DECISIONS.md`) and concepts guide (`docs/CONCEPTS.md`)
- [x] Documents fetch (`fetch_tosdr documents`): 4,655 docs, finished during v0.2

## v0.2 — Dataset v1 (Week 1, days 3–7) · MVP ✅
- [x] `configs/taxonomy.yaml`: 222 cases → 24 categories (rare cases dropped instead of `other`, D24)
- [x] `clean.py`, `taxonomy.py`, `chunk.py`, `prompt.py`, `build_dataset.py` (negatives + hash split inside)
- [x] Unit tests, plus checks on the built files: no group in two splits, valid JSON, quotes inside chunks
- [x] Manual review of negatives → similarity filter (D31)
- [ ] Push processed splits to private HF dataset repo (moved to v0.4: needs an HF account; only Kaggle needs it)

## v0.3 — Baselines + eval harness (Week 2) · MVP
- [x] `eval/` harness: one predictions format → all metrics → `results/eval/*.json` + `results/RESULTS.md`
- [x] B0 keyword baseline, B0b TF-IDF + logistic regression, `ref_always_flag` reference
- [x] Kaggle notebook + bundle/import scripts (Mac can't run the heavy jobs, D35)
- [x] B1/B2 base model zero/few-shot: run `notebooks/kaggle_baselines.ipynb` on Kaggle
- [x] B3 DeBERTa-v3-small classifier: same notebook (first run NaN → fp32 fix and rerun, D36)
- [x] Import results, write up findings (`results/baselines_v0.3.md`), tag v0.3

## v0.4 — First fine-tune (Week 2–3) · MVP
- [x] ~~Push processed splits to a private HF dataset repo~~ → data comes from the private Kaggle dataset (D37)
- [x] `training/train.py` (QLoRA, transformers + peft, completion-only loss, unit-tested masking) +
      `notebooks/kaggle_train.ipynb` (timing / full modes)
- [x] Budget: timing run → 4.6 h/epoch on 2×T4 with DDP vs 6.9 h with Unsloth on one → 1 epoch, all data (D37)
- [x] Full run on Kaggle (4.3 h; val loss 0.129 → 0.049); adapter comes back as `fine-print-adapter.zip`
      (HF Hub push moves to v0.7). Two failed starts fixed: eval OOM, torchao (D37)
- [x] Evaluate → first full results table (`results/finetune_v0.4.md`): macro-F1 0.571 vs DeBERTa 0.635,
      precision 0.93, recall 0.68, severity 0.85

## v0.5 — Error analysis + iteration (Week 3)
- [ ] Read ≥50 errors (start with missed positives: model miss vs missing label), write `results/error_analysis_v1.md`
- [ ] Cheapest recall fix first: threshold on the probability of the empty answer, tuned on val
- [ ] Fix the largest error group (rebalance / more negatives / prompt / threshold)
- [ ] Retrain (1–2 runs) and compare
- [ ] LexGLUE external evaluation

## v0.6 — Inference pipeline (Week 4) · MVP
- [ ] Merge adapter → GGUF Q4_K_M; run on Mac with `llama-cpp-python`
- [ ] `inference/`: chunk → predict → validate → aggregate → grade
- [ ] Latency measurement
- [ ] (optional) DeBERTa pre-filter

## v0.7 — Demo (Week 4) · MVP
- [ ] Gradio app: paste text → highlighted clauses + grade + explanations
- [ ] Precomputed examples for ~10 well-known services
- [ ] Deploy to HF Spaces (free CPU)

## v1.0 — Writeup & release (Week 4)
- [ ] README: problem, architecture diagram, results vs baselines, error analysis, limitations
- [ ] HF model card + dataset card (with attribution)
- [ ] 2-minute demo video / GIF
- [ ] (optional) browser extension
