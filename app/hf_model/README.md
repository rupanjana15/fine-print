---
license: other
license_name: qwen-research
license_link: LICENSE
base_model: Qwen/Qwen2.5-3B-Instruct
language: [en]
pipeline_tag: text-generation
tags: [gguf, legal, terms-of-service, privacy, qlora, llama-cpp]
---

# Fine Print (4-bit GGUF)

**Built with Qwen.** A fine-tune of Qwen2.5-3B-Instruct that reads a chunk of a Terms of Service or privacy policy
and returns the clauses that matter as JSON: category (24 categories), severity (good / neutral / bad / blocker),
a plain-English explanation and the exact quote. **Not legal advice.** Demo: [rupp15/fine-print](https://huggingface.co/spaces/rupp15/fine-print).

**This file is modified from the original Qwen model:** QLoRA adapter (r 16) trained for 2 epochs on ~38k chunks of
human-reviewed ToS;DR points, merged into the weights, then converted to GGUF and quantized to Q4_K_M with llama.cpp
(1.93 GB). See `NOTICE`.

## Results (held-out companies, test-500 subset)
| System | Detect P | Detect R | Category macro-F1 | Severity acc | Quotes found verbatim |
|---|---|---|---|---|---|
| DeBERTa-v3-small classifier (baseline, full test set) | 0.846 | 0.881 | 0.635 | 0.678 | – |
| Fine Print, full precision on GPU | 0.837 | 0.910 | 0.658 | 0.845 | 99.2% |
| **This file (Q4_K_M, llama.cpp)** | 0.835 | 0.910 | **0.652** | 0.857 | 99.5% |

Known limitation: on LexGLUE `unfair_tos` (older, lowercased contracts, a different labelling scheme) it scores
0.466 macro-F1, below the DeBERTa baseline (0.559).

## How to use
The model expects the exact prompt from the project (`src/fineprint/prompt.py`) and an **empty-answer threshold**:
read P(`" []"`) vs P(`' [{"'`) after `{"flags":` and answer `{"flags": []}` when P(empty) ≥ 0.9, else continue
greedily with repeat penalty 1.05 over the whole context. Reference implementation: `src/fineprint/inference/gguf.py`
(llama-cpp-python 0.3.36) in the [Fine Print Space](https://huggingface.co/spaces/rupp15/fine-print/tree/main/src/fineprint).

## Licence and credits
- Base model: Qwen2.5-3B-Instruct, **Qwen Research License, non-commercial use only** (`LICENSE`).
  Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved.
- Training data: [ToS;DR](https://tosdr.org), CC BY-SA 3.0; the labels are published as
  [rupp15/fine-print-tos-labels](https://huggingface.co/datasets/rupp15/fine-print-tos-labels). Grades produced with this model are estimates, not
  ToS;DR's human-reviewed ratings.
