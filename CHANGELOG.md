# Changelog

One entry per tagged milestone (see `docs/ROADMAP.md`).

## v1.0 — 2026-11-16 · Writeup & release
- Whole-service grading on held-out companies (`eval/service_grades.py`, `notebooks/kaggle_services.py`, D43):
  9,086 chunks of 109 val/test services on 2×T4; D41 rule on the model's flags: test exact 0.46, within one 0.88
  (ToS;DR's own flags, same services: 0.58 / 1.00; majority 0.29). Errors near balanced (17 kind / 11 harsh); the
  gap is missed blockers (0.31 vs 0.60 per service). Refit on model flags overfits (test 0.37): D41 rule kept
- `run_knob --shard i/n --threshold t` for parallel GPU shards with the empty-answer threshold
- Public labels dataset `rupp15/fine-print-tos-labels` (31,362 human-reviewed points, company split, CC BY-SA 3.0;
  no document text, D44)
- Release README (example flag, architecture, results, error analysis, limitations), demo GIF, model card with
  dataset link and service-grade numbers; demo and app caveats updated with whole-service results

## v0.7 — 2026-11-15 · Demo (MVP)
- Public model `rupp15/fine-print-gguf` (Q4_K_M, model card, Qwen licence + NOTICE, "Built with Qwen")
- Public demo: free Static Space `rupp15/fine-print`, 10 services precomputed (8 unseen in training) on 3 parallel
  Kaggle CPU kernels (`inference/examples.py`, `notebooks/kaggle_examples.py`); flags, quotes, both grades
- First end-to-end grades: exact 1/10, within one 5/10, more generous than ToS;DR on 8/10; stated in the demo (D42)
- Gradio app (`app/app.py`): paste text → highlights, flags, grade; runs locally. HF now needs PRO for CPU Gradio
  Spaces; live version planned on free ZeroGPU (account must be 30 days old)

## v0.6 — 2026-11-13 · Inference pipeline (MVP)
- Merged v2 adapter → GGUF Q4_K_M, 1.93 GB, on a free Kaggle CPU kernel (`notebooks/kaggle_gguf.py`, D40)
- Empty-answer threshold inside llama.cpp via logits (`inference/gguf.py`); repeat penalty over the full context,
  hand-built ChatML tested against HF's template
- **`ft_qwen3b_v2_gguf` test-500 macro-F1 0.652** (full model 0.658), recall 0.91, severity 0.857, quotes 99.5%;
  scored on Kaggle CPU (`notebooks/kaggle_gguf_eval.py`) after the Mac overheated. Mac vs Kaggle: 76/82 identical
- `inference/pipeline.py`: document → chunks → flags with document spans → merged → A–E grade
- Grade rule fitted on ToS;DR grades (`eval/grade_fit.py`, D41): good − bad − 2·blocker, exact 0.58 / within one
  grade 1.00 on 45 test services (majority guess 0.33)
- Latency (M1): 1.5 s empty chunk, 5.4 s flagged chunk, ~45 s–2.5 min per median document. DeBERTa pre-filter skipped

## v0.5 — 2026-11-13 · Error analysis + iteration
- Error analysis v1 (`results/error_analysis_v1.md`): 56% of misses are empty answers; `transparency` labels are
  noisy (35% headings); LexGLUE quote gap is a lowercasing artefact, not hallucination
- Empty-answer threshold (D38): P(` []`) at the decision token, t tuned on val → v1 macro-F1 0.571 → 0.626,
  recall 0.68 → 0.87 (`eval/run_knob.py`, `eval/knob.py`)
- Parity check found Qwen's hidden `repetition_penalty` 1.05 in every `generate()` result so far; now explicit in
  all runners (`prompt.REPETITION_PENALTY`, `eval/diag_knob.py`)
- Second epoch from the v1 adapter (D39, `train.py --init-adapter`, 4.8 h, val loss 0.049 → 0.044):
  **`ft_qwen3b_v2_tuned` test macro-F1 0.662** (DeBERTa 0.635), recall 0.91, severity 0.854; ≥ DeBERTa on 18/24
  categories. Limitation: LexGLUE 0.466 (v1 0.589, DeBERTa 0.559). Findings: `results/finetune_v0.5.md`

## v0.4 — 2026-11-12 · First fine-tune
- `training/train.py`: QLoRA (NF4 + LoRA r 16), completion-only masking (unit-tested), DDP on both T4s, time budget
- `notebooks/kaggle_train.ipynb`: timing / full / eval modes; timing run picked DDP 2×T4 (4.6 h/epoch) over
  Unsloth 1×T4 (6.9 h) → 1 epoch on all data (D37)
- Training: 2,383 steps, 4.3 h, val loss 0.129 → 0.049. Fixed on the way: eval OOM (batch 2), torchao vs PEFT
- `ft_qwen3b_v1` on full test: precision **0.933**, recall 0.681, cat macro-F1 0.571 (DeBERTa 0.635), severity
  **0.851** (best), flags 7% of harmless chunks (DeBERTa 27%), 99.95% valid JSON, 99.2% faithful quotes.
  Findings: `results/finetune_v0.4.md`. Data from the Kaggle dataset; HF Hub push deferred to v0.7

## v0.3 — 2026-11-11 · Baselines + eval harness
- `eval/`: one predictions format → JSON validity, detection P/R/F1, category macro-F1, severity, quote
  faithfulness, flag rate per example kind → `results/eval/*.json` + generated `results/RESULTS.md` (D32)
- Baselines (D33): always-flag reference, keywords, TF-IDF + LR, Qwen2.5-3B zero-/3-shot, DeBERTa-v3-small
- Heavy jobs on Kaggle's free 2×T4, driven from the Mac with the Kaggle CLI (D35, D36)
- DeBERTa first run went NaN (transformers v5 loads fp16 weights; AdamW breaks) → fp32 + NaN guard (D36)
- Results (test-500, cat macro-F1): DeBERTa **0.634**, TF-IDF 0.601, keywords 0.365, Qwen 3-shot 0.331,
  Qwen zero-shot 0.224. Base Qwen's severity accuracy (0.29–0.30) is below the majority guess (0.575).
  Bar for the fine-tune: DeBERTa's ≈0.63 plus explanations and faithful quotes. Findings: `results/baselines_v0.3.md`

## v0.2 — 2026-11-06 · Dataset v1
- Documents fetch finished: 4,655 docs via `/document/v1` (`/document/v2` returns 404), none missing
- `clean`: 301,107 points → 31,520 clean (D30); `taxonomy`: 222 cases → 24 categories (D24)
- `chunk.py` + `prompt.py` shared with inference (D25, D27); LexGLUE is evaluation-only (D28)
- `build_dataset`: windows around quotes, hard + easy negatives, service hash split (D26, D29)
- Manual review found missing labels in negatives (22% easy, 50% hard) → TF-IDF similarity filter (D31)
- Dataset v1: **38,143 / 4,773 / 4,024** train/val/test examples, ~42% negatives, ~18M train tokens/epoch
- Tests: 22, including checks on the built files

## v0.1 — 2026-11-05 · Data spike
- `fetch_lexglue`: LexGLUE unfair_tos downloaded (5,532 / 2,275 / 1,607; ~89% fair)
- `fetch_tosdr`: polite cached fetcher; found the undocumented quote fields and the 400-for-empty-case quirk
- Full ToS;DR points fetch: 301,107 points → **32,349 usable** (approved, quoted, English) across 247 cases
- Found that ~79% of points are unreviewed DocBot suggestions → excluded (D07)
- `stats`: `results/data_stats.json` + 100-quote manual review (labels good; HTML needs cleaning)
- Licensing checked: ToS;DR CC BY-SA 3.0; AI policy → our grade is an estimate (D20)
- Docs: `DATA.md`, `DECISIONS.md` (D01–D23), `CONCEPTS.md`, docs index, `CLAUDE.md`
- Documents fetch (4,655 docs) started; finishes outside this tag

## v0.0 — 2026-11-04
- Repository created with folder layout, `uv` project and configs
- Architecture (`docs/ARCHITECTURE.md`) and roadmap (`docs/ROADMAP.md`)
- Verified data access: LexGLUE `unfair_tos` loads; ToS;DR API is rate-limited (needs throttled fetcher)
