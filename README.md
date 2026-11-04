# Fine Print 🔍

**A fine-tuned 3B-parameter LLM that reads Terms of Service and flags the clauses that matter —
in plain English, with severity and the exact quote.** Runs locally; built entirely on free resources.

> ⚠️ Not legal advice.

**Status:** v0.0 — architecture & repo set up. See [docs/ROADMAP.md](docs/ROADMAP.md).

## How it works

```
ToS text → chunks → fine-tuned 3B model → {"flags":[{case, severity, explanation, quote}]} → highlighted doc + A–E grade
```

- **Data:** human-labelled clauses from [ToS;DR](https://tosdr.org) + [LexGLUE unfair_tos](https://huggingface.co/datasets/coastalcph/lex_glue) as an external test set
- **Model:** Qwen2.5-3B-Instruct, QLoRA fine-tuned on a free Kaggle GPU
- **Serving:** 4-bit GGUF via llama.cpp — runs on a laptop and on a free Hugging Face Space

Full design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Repo layout

```
configs/            data, taxonomy and training settings
src/fineprint/
  data/             fetch → clean → taxonomy → negatives → split → format
  baselines/        keyword, zero/few-shot, DeBERTa
  training/         helpers used by the Kaggle notebook
  eval/             one harness, all metrics
  inference/        chunk → predict → validate → aggregate → grade
notebooks/          Kaggle/Colab training notebooks
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
