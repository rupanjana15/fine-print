# Fine Print 🔍

**A fine-tuned 3B-parameter LLM that reads Terms of Service and flags the clauses that matter —
in plain English, with severity and the exact quote.** Runs locally; built entirely on free resources.

> ⚠️ Not legal advice.

## Status (2026-11-13)

| Milestone | State |
|---|---|
| v0.0 Repo + architecture | ✅ done |
| v0.1 Data spike: 301k ToS;DR points fetched, 32k usable | ✅ done |
| v0.2 Dataset v1: 38,143 / 4,773 / 4,024 train/val/test, split by company | ✅ done |
| v0.3 Baselines + eval harness (heavy runs on Kaggle's free GPUs) | ✅ done |
| v0.4 First fine-tune: QLoRA, 1 epoch, 4.3 h on 2×T4 | ✅ done |
| v0.5 Error analysis, empty-answer threshold, 2nd epoch | ✅ done |
| v0.6 Inference pipeline on the Mac: 4-bit GGUF (1.9 GB), A–E grade | ✅ done |
| v0.7 Demo on HF Spaces | ⏳ next |

**Results** (full test set, 4,024 chunks; [all results](results/RESULTS.md) ·
[v0.5 findings](results/finetune_v0.5.md) · [fine-tune findings](results/finetune_v0.4.md) · [baseline findings](results/baselines_v0.3.md)):

| System | Precision | Recall | Category macro-F1 | Severity acc | Flags harmless chunks |
|---|---|---|---|---|---|
| Always flag (reference) | 0.647 | 1.000 | 0.006 | 0.584 | 100% |
| Keyword rules | 0.693 | 0.919 | 0.371 | 0.664 | 74% |
| TF-IDF + logistic regression | 0.862 | 0.815 | 0.617 | 0.675 | 21% |
| DeBERTa-v3-small classifier | 0.846 | 0.881 | 0.635 | 0.678 | 27% |
| Fine-tuned Qwen2.5-3B, greedy (v0.4) | **0.933** | 0.681 | 0.571 | 0.851 | **7%** |
| + empty-answer threshold (v0.5) | 0.869 | 0.871 | 0.626 | 0.842 | 21% |
| **+ 2nd epoch (v0.5, ours: `ft_qwen3b_v2_tuned`)** | 0.870 | 0.909 | **0.662** | **0.854** | 22% |

Base Qwen 3B (500-chunk subset) → fine-tuned: category macro-F1 0.33 → 0.58, severity 0.30 → 0.82, harmless chunks
flagged 71% → 5%, valid JSON 100%, quotes copied exactly 99.6%. The fine-tune is the most precise and best at
severity, and every flag comes with an explanation and a quote, but it is conservative: it misses about a third of
risky chunks. v0.5 added an empty-answer threshold tuned on validation (recall 0.87, macro-F1 0.626) and a second
epoch: the model now beats DeBERTa on category macro-F1 (0.662 vs 0.635) while keeping far better severity
(0.85 vs 0.68). Known limitation: on LexGLUE (older, lowercased contracts, different labels) the second epoch
scores 0.466 vs DeBERTa 0.559 and v1's 0.589.
The shipped model is the 4-bit GGUF of v2 (1.9 GB, llama.cpp): test-500 macro-F1 0.652 vs 0.658 for the full
model, ~1.5–5 s per chunk on an M1 Mac. The A–E grade is a counting rule fitted on ToS;DR's human grades: exact on
58% of held-out services, never more than one grade off (D40, D41).
Plan: [docs/ROADMAP.md](docs/ROADMAP.md) · changes: [CHANGELOG.md](CHANGELOG.md).

## How it works

```
ToS text → chunks → fine-tuned 3B model → {"flags":[{category, severity, explanation, quote}]} → highlighted doc + A–E grade
```

- **Data:** human-labelled clauses from [ToS;DR](https://tosdr.org) + [LexGLUE unfair_tos](https://huggingface.co/datasets/coastalcph/lex_glue) as an external test set
- **Model:** Qwen2.5-3B-Instruct, QLoRA fine-tuned on a free Kaggle GPU
- **Serving:** 4-bit GGUF via llama.cpp — runs on a laptop and on a free Hugging Face Space

Full design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · Why each choice: [docs/DECISIONS.md](docs/DECISIONS.md) ·
ML concepts explained: [docs/CONCEPTS.md](docs/CONCEPTS.md) · All docs: [docs/README.md](docs/README.md)

## Repo layout

```
configs/            data, taxonomy and training settings
src/fineprint/
  data/             fetch → stats → clean → taxonomy → build_dataset
  baselines/        always-flag, keywords, TF-IDF, DeBERTa
  training/         helpers used by the Kaggle notebook
  eval/             eval sets, one scorer, Mac (GGUF) and GPU (HF) runners
  inference/        chunk → predict → validate → aggregate → grade
notebooks/          Kaggle notebooks (baselines now, training next)
app/                Gradio demo
data/               (gitignored) raw / interim / processed
results/            metrics JSON + error analyses (committed)
tests/
docs/               architecture, roadmap
```

## Setup (Mac)

```bash
uv sync
```

## Results

See the status table above and `results/`.

## Acknowledgements

Labels come from the ToS;DR volunteer community. LexGLUE / CLAUDETTE by Chalkidis et al. and Lippi et al.
