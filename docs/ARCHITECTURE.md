# Architecture — Fine Print

Fine Print reads a Terms of Service or privacy policy and flags risky clauses in plain English,
with a severity rating and the exact quote it is based on. It is a **fine-tuned ~3B-parameter LLM**
that runs locally (laptop CPU/GPU) — no paid API at inference time.

> Constraint for the whole project: **every resource is free** (Claude Code Pro is the only paid
> tool, used for writing code — never for generating training data).

---

## 1. System overview

```mermaid
flowchart LR
    subgraph DATA["① Data layer (Mac)"]
        A[ToS;DR API<br/>services · points · cases] -->|throttled, cached| R[(data/raw<br/>JSON cache)]
        L[LexGLUE unfair_tos<br/>HF datasets] --> R
        R --> B[build_dataset<br/>clean · taxonomy · negatives · company split]
        B --> P[(data/processed<br/>train/val/test .jsonl)]
    end

    subgraph TRAIN["② Training (Kaggle free GPU)"]
        P -->|HF Hub private dataset| T[QLoRA fine-tune<br/>Unsloth + TRL]
        T --> AD[LoRA adapter]
        AD --> M[merge + GGUF Q4_K_M<br/>~2 GB]
    end

    subgraph EVAL["③ Evaluation (Mac / Kaggle)"]
        P --> E[eval harness]
        BL[Baselines<br/>keyword · zero-shot · few-shot · DeBERTa] --> E
        M --> E
        E --> RS[(results/*.json<br/>committed to git)]
    end

    subgraph SERVE["④ Inference + demo"]
        M --> I[pipeline<br/>chunk → classify → aggregate → grade]
        I --> UI[Gradio app<br/>HF Spaces free CPU]
    end
```

Four layers, each with its own folder under `src/fineprint/` and clear file-based handoffs
between them (JSONL datasets, model files, results JSON). Any layer can be re-run alone.

---

## 2. The core task (what the model learns)

**Input:** one chunk of a legal document (~100–300 tokens).
**Output:** strict JSON — zero or more flags.

```json
{"flags": [
  {"case": "shares_data_with_third_parties",
   "severity": "bad",
   "explanation": "They can share your personal data with advertisers.",
   "quote": "share your personal information with third-party advertising partners"}
]}
```

Harmless chunk → `{"flags": []}`. Most real chunks are harmless, so learning to say *nothing*
is as important as learning to flag.

**Why per-chunk and not whole-document?** A full ToS is 5k–30k tokens. A 3B model degrades
badly on long context and it would not fit training memory on a T4. Chunking keeps every
example short, trainable, and lets us point at exact spans in the UI.

**Severity scale** (inherited from ToS;DR point ratings):

| Label | Meaning | UI colour |
|---|---|---|
| `good` | Protects the user | green |
| `neutral` | Worth knowing | grey |
| `bad` | Against the user's interest | orange |
| `blocker` | Serious — a reason to avoid the service | red |

---

## 3. Data layer — `src/fineprint/data/`

### Sources

| Source | What it gives | Role | Access |
|---|---|---|---|
| **ToS;DR** (`api.tosdr.org`) | *Points*: quoted clause + *case* (e.g. "This service can sell your data") + rating + service | Main training data; case titles are already plain-English explanations | Public REST API, **rate-limited** |
| **LexGLUE `unfair_tos`** (CLAUDETTE) | ~9.4k sentences from 50 ToS, 8 unfair-clause labels | External test set + clean negatives | `datasets.load_dataset("coastalcph/lex_glue", "unfair_tos")` |

Verified 2026-11-04: LexGLUE loads (5,532 / 2,275 / 1,607 train/val/test). The ToS;DR API returned
HTTP 429 on the first request, so the fetcher **must** be polite.

### Pipeline stages

| Stage | Module | Output | Notes |
|---|---|---|---|
| Fetch | `fetch_tosdr.py` | `data/raw/tosdr/*.json` | Disk cache = resumable; 1 req / N s; exponential backoff on 429 |
| Fetch | `fetch_lexglue.py` | `data/raw/lexglue/` | One-shot download |
| Clean | `clean.py` | `data/interim/points.parquet` | Keep approved points with quote text; normalise whitespace; drop duplicates |
| Taxonomy | `configs/taxonomy.yaml` | case → category map | Collapse hundreds of ToS;DR cases into ~20–25 categories + `other` |
| Negatives | `negatives.py` | harmless chunks | LexGLUE "fair" sentences + document text not covered by any point |
| Split | `split.py` | train/val/test | **By service (company), never random** — see §7 |
| Format | `format.py` | `data/processed/*.jsonl` | Chat-template messages: system · user(chunk) · assistant(JSON) |

### Processed example (one JSONL line)

```json
{"id": "tosdr-pt-12345", "service": "example.com", "source": "tosdr",
 "messages": [
   {"role": "system", "content": "<task instructions>"},
   {"role": "user", "content": "<chunk text>"},
   {"role": "assistant", "content": "{\"flags\": [...]}"}
 ]}
```

### Data versioning
Raw and processed data are **not** committed to git (too large, third-party content). Instead:
- `configs/data.yaml` records the dataset version string,
- processed splits are pushed to a **private Hugging Face dataset repo** (free) tagged with that version,
- `results/` files record which dataset version they were computed on.

---

## 4. Training — `src/fineprint/training/` + `notebooks/`

| Choice | Value | Why |
|---|---|---|
| Base model | `Qwen2.5-3B-Instruct` (alt: `Llama-3.2-3B-Instruct`) | Strong at JSON at 3B size; swappable via `configs/train.yaml` |
| Method | **QLoRA** — 4-bit frozen base + LoRA adapters (r=16, α=32, all linear layers) | Fits a free 16 GB T4 |
| Library | Unsloth + TRL `SFTTrainer` | ~2× faster, less memory on free GPUs |
| Loss | **Completion-only** (mask system+user tokens) | Model learns to answer, not to echo legal text |
| Hyper-params (start) | lr 2e-4, 2 epochs, max_len 1024, eff. batch 16, cosine schedule | Sensible QLoRA defaults; tuned in v0.4 |
| Compute | Kaggle 2×T4 (≈30 GPU-h/week), Colab T4 as backup | Free |
| Checkpoints | Pushed to private HF model repo every N steps | Kaggle sessions die at ~12 h |

The training notebook pulls code from this GitHub repo and data from the HF dataset repo, so the
notebook itself stays thin and the logic is versioned here.

**Why not train on the Mac?** M1 / 8 GB unified memory cannot hold a 3B model + optimizer state +
activations comfortably. The Mac is used for everything else.

---

## 5. Evaluation — `src/fineprint/eval/`, `src/fineprint/baselines/`

Every system — baseline or fine-tuned — writes predictions in the **same JSON format**, and one
harness scores them all into `results/<run_id>.json`. That makes the comparison table automatic.

### Systems compared

| # | System | Question it answers |
|---|---|---|
| B0 | Keyword / regex rules | Is this task trivially solvable? |
| B1 | Base 3B, zero-shot | What does the model know before training? |
| B2 | Base 3B, few-shot | How far does prompting alone get? |
| B3 | DeBERTa-v3-small multi-label classifier | Is a generative LLM even needed for *detection*? |
| FT | Fine-tuned 3B (QLoRA) | Our model |
| (ref) | ToS;DR DocBot (123 per-case binary classifiers) | Related work — discussed, not re-run |

### Metrics

| Metric | Level | Detects |
|---|---|---|
| Detection P / R / F1 | chunk | crying wolf vs missing risks |
| Category macro-F1 | flag | getting the *type* right, rare types included |
| Severity accuracy (+ confusion matrix) | flag | bad vs blocker mix-ups |
| JSON validity rate | output | can the app trust the output? |
| Quote faithfulness | flag | % of quotes that literally appear in the input (hallucination check) |
| LexGLUE F1 (mapped 8 labels) | external | generalisation beyond ToS;DR |
| Latency (s / chunk, Mac + CPU Space) | system | is the demo usable? |

Plus a manual **error analysis** of ≥50 mistakes per major version, saved in `results/`.

---

## 6. Inference & demo — `src/fineprint/inference/`, `app/`

```mermaid
flowchart LR
    D[Document text / URL] --> C[clean + chunk<br/>paragraph-aware, ~200 tok, small overlap]
    C --> G[model on each chunk<br/>llama.cpp GGUF Q4]
    G --> V[validate JSON<br/>check quote ∈ chunk]
    V --> AG[aggregate<br/>dedupe same case across chunks]
    AG --> S[grade A–E<br/>from severities]
    S --> UI[highlighted document<br/>+ flag list]
```

- **Runtime:** `llama-cpp-python` with the GGUF Q4_K_M file (~2 GB). Same file on the Mac and on
  HF Spaces — what you test is what you ship.
- **Speed plan for the free CPU Space** (2 vCPU): precomputed results for ~10 well-known services
  (instant), cap on pasted length, and (optional, v0.6) a DeBERTa pre-filter so the LLM only runs
  on chunks likely to be risky.
- **Grade:** simple, transparent rule (e.g. any `blocker` → E; count of `bad` thresholds) — documented
  in the UI so it is not a black box.
- Disclaimer on every page: *not legal advice*.

---

## 7. Key design decisions

| Decision | Alternative rejected | Reason |
|---|---|---|
| Split by **service** | Random split | Companies reuse boilerplate; random split leaks near-duplicates into test and inflates scores |
| One generative model with explanations | 123 binary classifiers (DocBot style) | One small artefact; produces explanation + quote; tests whether a 3B LLM can match specialist classifiers |
| Labels from human ToS;DR data | Synthetic data from a big LLM | Free, real, and avoids Claude terms (no training on Claude outputs) |
| ~20–25 merged categories | All ToS;DR cases | Long tail has too few examples to learn |
| GGUF + llama.cpp | Transformers on CPU | 4× smaller, much faster on CPU and Apple Silicon |
| Results JSON in git | Paid experiment tracker | Free, versioned with the code that produced it |

---

## 8. Free resource map

| Need | Resource | Limit to watch |
|---|---|---|
| Code + versions | GitHub (private repo) | — |
| Local dev | MacBook Air M1, 8 GB | Run ≤3B Q4 models only; no training |
| Python env | `uv` + Python 3.12 | — |
| GPU training | Kaggle notebooks (2×T4) | ~30 h/week, 12 h/session |
| Backup GPU | Google Colab free T4 | Session limits vary |
| Data + model storage | Hugging Face Hub (private repos) | — |
| Demo hosting | HF Spaces, CPU basic | 2 vCPU / 16 GB, sleeps when idle |
| Coding assistant | Claude Code (Pro) | Not used to generate training data |

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| ToS;DR API throttling / changes | Slow cached fetcher; LexGLUE fallback; ask team@tosdr.org for a dump (they plan to publish datasets) |
| ToS;DR data licence / attribution | Verify in v0.1; attribute in README + model card; don't redistribute raw text publicly |
| Noisy / inconsistent labels | Merge categories, report as limitation, inspect in error analysis |
| Kaggle session timeouts | Checkpoint to HF Hub |
| FT does not beat DeBERTa on detection | Still a valid finding; LLM's edge is explanation + quote — report honestly |
| Slow demo on free CPU | Precomputed examples, length cap, pre-filter |
