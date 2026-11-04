# Roadmap

Each milestone ends with a **git tag** (`v0.1`, `v0.2`, …) and an entry in `CHANGELOG.md`,
so every working state of the project is saved and can be checked out again.

Target: ~4 weeks part-time. The "MVP cut" marks what to keep if time runs short.

---

## v0.0 — Repo + architecture ✅
- [x] Repo, folder layout, `.gitignore`, `uv` project
- [x] `docs/ARCHITECTURE.md`, this roadmap

## v0.1 — Data spike (Week 1, days 1–2) · MVP
- [ ] `fetch_lexglue.py` — download + print label stats
- [ ] `fetch_tosdr.py` — throttled, cached, resumable fetcher
- [ ] Notebook/script: inspect 100 raw points by hand
- [ ] Stats: #points, #with quotes, case distribution, rating distribution, #services
- [ ] Check ToS;DR data licence / attribution terms

## v0.2 — Dataset v1 (Week 1, days 3–7) · MVP
- [ ] `configs/taxonomy.yaml` — cases → ~20–25 categories + `other`
- [ ] `clean.py`, `negatives.py`, `split.py` (by service), `format.py`
- [ ] Unit tests: no service appears in two splits; every assistant message is valid JSON
- [ ] Push processed splits to private HF dataset repo, record version

## v0.3 — Baselines + eval harness (Week 2) · MVP
- [ ] `eval/` harness: one predictions format → all metrics → `results/<run>.json`
- [ ] B0 keyword baseline
- [ ] B1/B2 base model zero/few-shot (run on Kaggle or Mac via llama.cpp)
- [ ] B3 DeBERTa-v3-small classifier (Kaggle)

## v0.4 — First fine-tune (Week 2–3) · MVP
- [ ] `notebooks/train_kaggle.ipynb` (QLoRA, Unsloth, completion-only loss)
- [ ] Adapter + checkpoints pushed to HF Hub
- [ ] Evaluate → first full results table

## v0.5 — Error analysis + iteration (Week 3)
- [ ] Read ≥50 errors, group them, write `results/error_analysis_v1.md`
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
