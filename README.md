# Fine Print 🔍

**A fine-tuned 3B-parameter LLM that reads Terms of Service and flags the clauses that matter —
in plain English, with severity and the exact quote.** Runs locally; built entirely on free resources.

> ⚠️ Not legal advice.

## Status (2026-11-12)

| Milestone | State |
|---|---|
| v0.0 Repo + architecture | ✅ done |
| v0.1 Data spike: 301k ToS;DR points fetched, 32k usable | ✅ done |
| v0.2 Dataset v1: 38,143 / 4,773 / 4,024 train/val/test, split by company | ✅ done |
| v0.3 Baselines + eval harness (heavy runs on Kaggle's free GPUs) | ✅ done |
| v0.4 First fine-tune: QLoRA, 1 epoch, 4.3 h on 2×T4 | ✅ done |
| v0.5 Error analysis + iteration | ⏳ threshold done (macro-F1 0.626); retrain decision open |

**Results** (full test set, 4,024 chunks; [all results](results/RESULTS.md) ·
[v0.5 findings](results/finetune_v0.5.md) · [fine-tune findings](results/finetune_v0.4.md) · [baseline findings](results/baselines_v0.3.md)):

| System | Precision | Recall | Category macro-F1 | Severity acc | Flags harmless chunks |
|---|---|---|---|---|---|
| Always flag (reference) | 0.647 | 1.000 | 0.006 | 0.584 | 100% |
| Keyword rules | 0.693 | 0.919 | 0.371 | 0.664 | 74% |
| TF-IDF + logistic regression | 0.862 | 0.815 | 0.617 | 0.675 | 21% |
| DeBERTa-v3-small classifier | 0.846 | **0.881** | **0.635** | 0.678 | 27% |
| Fine-tuned Qwen2.5-3B, greedy (v0.4) | **0.933** | 0.681 | 0.571 | **0.851** | **7%** |
| **+ empty-answer threshold (v0.5, ours)** | 0.869 | 0.871 | 0.626 | 0.842 | 21% |

Base Qwen 3B (500-chunk subset) → fine-tuned: category macro-F1 0.33 → 0.58, severity 0.30 → 0.82, harmless chunks
flagged 71% → 5%, valid JSON 100%, quotes copied exactly 99.6%. The fine-tune is the most precise and best at
severity, and every flag comes with an explanation and a quote, but it is conservative: it misses about a third of
risky chunks. v0.5's empty-answer threshold (tuned on validation, no retraining) lifts recall to 0.87 and
macro-F1 to 0.626, within 0.01 of DeBERTa, and beats it on LexGLUE (0.589 vs 0.559).
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
