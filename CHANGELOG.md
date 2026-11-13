# Changelog

One entry per tagged milestone (see `docs/ROADMAP.md`).

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
