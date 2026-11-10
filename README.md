# Fine Print 🔍

**A fine-tuned 3B-parameter LLM that reads Terms of Service and flags the clauses that matter —
in plain English, with severity and the exact quote.** Runs locally; built entirely on free resources.

> ⚠️ Not legal advice.

## Status (2026-11-10)

| Milestone | State |
|---|---|
| v0.0 Repo + architecture | ✅ done |
| v0.1 Data spike: 301k ToS;DR points fetched, 32k usable | ✅ done |
| v0.2 Dataset v1: 38,143 / 4,773 / 4,024 train/val/test, split by company | ✅ done |
| v0.3 Baselines + eval harness | 🟡 cheap baselines scored; GPU baselines ready to run on Kaggle ([guide](docs/KAGGLE.md)) |
| v0.4 First fine-tune (QLoRA on Kaggle) | ⏳ next |

**Baselines so far** (full test set; [all results](results/RESULTS.md)):

| System | Detection F1 | Category macro-F1 | Flags harmless chunks |
|---|---|---|---|
| Always flag (reference) | 0.786 | 0.006 | 100% |
| Keyword rules | 0.790 | 0.371 | 74% |
| TF-IDF + logistic regression | **0.838** | **0.617** | 21% |

Base Qwen 3B (zero/few-shot) and DeBERTa run next on Kaggle's free GPUs. The fine-tuned model has to beat
TF-IDF, not just keywords. Full plan: [docs/ROADMAP.md](docs/ROADMAP.md) · changes: [CHANGELOG.md](CHANGELOG.md).

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

_Coming in v0.4._

## Acknowledgements

Labels come from the ToS;DR volunteer community. LexGLUE / CLAUDETTE by Chalkidis et al. and Lippi et al.
