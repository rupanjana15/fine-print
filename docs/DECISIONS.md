# Decision log

Every significant choice in this project, **why we made it, and why not the alternatives**.
Each entry ends with a one-line **"Explain it"** answer you can use in an interview or demo.

Format: *Context → Decision → Alternatives considered → Consequences → Explain it.*
New decisions are appended with the next number; superseded ones are marked, never deleted.

| # | Decision | Status |
|---|---|---|
| [D01](#d01--project-tos-red-flag-detector) | Project: ToS red-flag detector | ✅ |
| [D02](#d02--model-size-23b-parameters) | Model size: 2–3B parameters | ✅ |
| [D03](#d03--base-model-qwen25-3b-instruct) | Base model: Qwen2.5-3B-Instruct | ✅ (revisit in v0.4) |
| [D04](#d04--fine-tuning-method-qlora) | Fine-tuning method: QLoRA | ✅ |
| [D05](#d05--compute-kaggle-for-training-mac-for-everything-else) | Compute: Kaggle for training, Mac for the rest | ✅ |
| [D06](#d06--training-data-human-labels-from-tosdr) | Training data: human labels from ToS;DR | ✅ |
| [D07](#d07--only-human-approved-points-docbot-rejections-as-hard-negatives) | Only human-approved points; DocBot rejections as hard negatives | ✅ |
| [D08](#d08--data-collection-throttled-cached-api-fetch-by-case) | Data collection: throttled, cached API fetch by case | ✅ |
| [D09](#d09--external-test-set-lexglue-unfair_tos) | External test set: LexGLUE unfair_tos | ✅ |
| [D10](#d10--task-framing-chunk-level-generation-of-json) | Task framing: chunk-level generation of JSON | ✅ |
| [D11](#d11--output-includes-the-exact-quote) | Output includes the exact quote | ✅ |
| [D12](#d12--merge-cases-into-2025-categories) | Merge cases into ~20–25 categories | ✅ (numbers in v0.2) |
| [D13](#d13--about-half-the-training-examples-are-negatives) | About half the examples are negatives | ✅ (tune in v0.5) |
| [D14](#d14--split-by-service-company-never-randomly) | Split by service, never randomly | ✅ |
| [D15](#d15--english-only-for-v1) | English only for v1 | ✅ |
| [D16](#d16--four-baselines-before-claiming-anything) | Four baselines before claiming anything | ✅ |
| [D17](#d17--metrics) | Metrics | ✅ |
| [D18](#d18--serving-4-bit-gguf-with-llamacpp) | Serving: 4-bit GGUF with llama.cpp | ✅ |
| [D19](#d19--demo-gradio-on-hugging-face-spaces-free-cpu) | Demo: Gradio on HF Spaces (free CPU) | ✅ |
| [D20](#d20--our-grade-is-an-estimate-never-an-official-tosdr-grade) | Our grade is an estimate, not ToS;DR's | ✅ |
| [D21](#d21--versioning-git-tags-for-code-hf-hub-for-data-and-models) | Versioning: git tags + HF Hub | ✅ |
| [D22](#d22--python-tooling-uv-python-312-config-in-yaml) | Tooling: uv, Python 3.12, YAML configs | ✅ |
| [D23](#d23--how-claude-code-is-used) | How Claude Code is used | ✅ |

---

## D01 — Project: ToS red-flag detector

**Context.** Needed a portfolio project that fine-tunes a 2–3B LLM using only free resources.
Shortlist was: corporate-speak decoder, ToS red-flag detector, workout form coach.

**Decision.** Build "Fine Print": a model that flags risky clauses in Terms of Service / privacy policies.

**Alternatives considered.**

| Option | Why not (for now) |
|---|---|
| Corporate-speak decoder | No existing dataset (all synthetic); output quality is subjective, so hard to prove the fine-tune helped with numbers |
| Workout form coach | Two systems (computer vision + LLM); most of the hard work is in pose estimation, not the LLM, and it's 5–8 weeks of work |

**Why ToS wins.** Thousands of **real human labels already exist** (ToS;DR); the task has
**objective metrics** (precision/recall); the demo is instantly understandable; it has a real-world use.

**Explain it:** *"I picked a task where real human labels exist and success is measurable, so I could
prove the fine-tune works instead of just showing nice examples."*

---

## D02 — Model size: 2–3B parameters

**Context.** The brief was a 2–3B model. The size also has to fit free GPUs and run on an 8 GB laptop.

**Decision.** ~3B parameters.

**Alternatives considered.**

| Option | Why not |
|---|---|
| <1B (e.g. 0.5B) | Weak at following a strict JSON format and writing fluent explanations |
| 7–8B | Training on a free 16 GB T4 is tight and slow; a 4-bit 7B (~4.5 GB) barely fits an 8 GB Mac alongside the OS and is slow on a free CPU Space |
| Encoder-only classifier (BERT/DeBERTa) as the main model | Can detect but cannot *explain* in natural language; kept as a **baseline** (D16) |

**Explain it:** *"3B is the sweet spot: big enough to write good JSON and explanations, small enough to
train for free and run on a laptop."*

---

## D03 — Base model: Qwen2.5-3B-Instruct

**Context.** Several good open ~3B instruct models exist.

**Decision.** Start with **Qwen2.5-3B-Instruct**; keep **Llama-3.2-3B-Instruct** as a drop-in
alternative (`configs/train.yaml`). Re-check newer releases before training in v0.4.

**Alternatives considered.**

| Option | Notes |
|---|---|
| Llama-3.2-3B-Instruct | Strong, but HF access is gated (must accept the licence); fine as alternative |
| Gemma 2B-class | Good quality; slightly smaller; own licence terms |
| Phi-3.5/4-mini (~3.8B) | Strong reasoning but above the 3B target and heavier to serve |
| SmolLM 3B-class | Fully open, but generally weaker at structured output in our size class |
| A base (non-instruct) model | Would need more data to learn instruction following; instruct models already know chat format and JSON |

**Why Qwen2.5-3B.** Known for strong structured-output (JSON) ability at this size, long context
support, and first-class Unsloth support (fast, low-memory training on free GPUs).

**Caveat to verify before release:** Qwen2.5-3B is published under the *Qwen Research License*
(non-commercial), unlike some other Qwen sizes. Fine for a portfolio project; if commercial use ever
matters, switch to an Apache-2.0 model.

**Explain it:** *"I chose an instruct model that's good at JSON and well supported by free training tools,
and kept the choice swappable in config so I can compare base models."*

---

## D04 — Fine-tuning method: QLoRA

**Context.** We need to change the model's behaviour cheaply on a single free GPU.

**Decision.** **QLoRA**: load the base model in 4-bit (frozen), train small LoRA adapter matrices
(rank 16) on all linear layers. Loss only on the assistant's answer.

**Alternatives considered.**

| Option | Why not |
|---|---|
| Full fine-tuning | Updating all 3B weights + Adam optimizer state needs ~40+ GB of GPU memory; a T4 has 16 GB |
| LoRA in 16-bit | Possible on a T4 for 3B but tight; QLoRA leaves headroom for longer sequences/bigger batches with little quality loss |
| Prompt engineering only (no training) | That's our baseline B1/B2: small models drift from the format and miss categories; the project is to show fine-tuning beats it |
| RAG (retrieve similar labelled clauses at inference) | Helps a big model; a 3B model still struggles with the format, and it adds an index to serve. Could be a later experiment |

**Explain it:** *"QLoRA freezes a 4-bit copy of the model and trains about 1% extra parameters, so a 3B
model fine-tunes on a free 16 GB GPU with almost the same quality as full fine-tuning."*

---

## D05 — Compute: Kaggle for training, Mac for everything else

**Context.** Hardware is a MacBook Air M1 with 8 GB unified memory. Budget: free.

**Decision.** Train on **Kaggle** (free 2×T4, ~30 GPU-hours/week, 12 h sessions), Colab as backup.
Data processing, evaluation code and running the final 4-bit model happen on the Mac.

**Alternatives considered.**

| Option | Why not |
|---|---|
| Train on the Mac (MLX) | 8 GB must hold the OS, model, activations and optimizer state; a 3B LoRA would swap heavily and be very slow or crash |
| Colab free as primary | Works, but GPU availability and session length are less predictable than Kaggle's weekly quota |
| Paid cloud GPU | Violates the free-only constraint |

**Consequence.** Sessions can die, so checkpoints are pushed to the Hugging Face Hub during training.

**Explain it:** *"I split the work by what each machine is good at: free cloud GPUs for training, my
laptop for data, evaluation and running the quantized model."*

---

## D06 — Training data: human labels from ToS;DR

**Context.** Need (clause → risk label) examples.

**Decision.** Use **ToS;DR** points: a human-quoted clause, the *case* it supports (whose title is
already a plain-English explanation), and a severity rating. Licence CC BY-SA 3.0 (see `DATA.md`).

**Alternatives considered.**

| Option | Why not |
|---|---|
| Generate synthetic labels with Claude | Anthropic's terms restrict using outputs to train other models; Pro rate limits make it slow; synthetic labels inherit the teacher's mistakes |
| Generate with a local open model | Lower quality than human labels and the 8 GB Mac runs a 7B teacher slowly; kept as a fallback only |
| Label clauses myself | Thousands of examples needed; weeks of work, single annotator bias |
| LexGLUE only | Only ~1k positive labels and 8 categories, all about unfairness; no privacy topics, no "good" clauses. Used as external test (D09) |
| CUAD (contract clause dataset) | Commercial contracts, not consumer ToS; different domain |

**Explain it:** *"ToS;DR volunteers had already quoted and rated tens of thousands of clauses, so I got
real human labels for free, and their case titles double as plain-English explanations."*

---

## D07 — Only human-approved points; DocBot rejections as hard negatives

**Context.** The v0.1 data spike showed ~80% of ToS;DR points are `pending` with analysis
"Created by Docbot version v3", i.e. **unreviewed ML predictions**.

**Decision.** Train only on `status == approved`. Use DocBot suggestions that humans **declined**
as **hard negatives** for that case (v0.2). Skip human-declined and other statuses.

**Alternatives considered.**

| Option | Why not |
|---|---|
| Use all points | 80% would be another model's guesses; we'd be distilling DocBot's errors, not learning from humans |
| Use pending points with high `ml_score` | Still unreviewed; would also make evaluation circular |
| Ignore declined points entirely | Wastes the most informative negatives: text a model thought matched but a human said doesn't |

**Explain it:** *"I found most of the dataset was unreviewed machine output, so I trained only on
human-approved labels, and used the clauses humans rejected as tricky negative examples."*

---

## D08 — Data collection: throttled, cached API fetch by case

**Context.** ToS;DR has a public API with an incomplete spec, a strict rate limit, and no public
bulk dump (DocBot uses a private database).

**Decision.** Fetch `/point/v1?case_id=X` page by page for every case id (there is no case-list
endpoint, so ids are scanned), then `/document/v2` for each document behind an approved point.
Every response is cached to disk; 2 s between requests; exponential backoff on HTTP 429.

**Alternatives considered.**

| Option | Why not |
|---|---|
| `/service/v3` per service | ~10k services, and its points have **no quote text** |
| `/point/v1?id=` per point | Hundreds of thousands of requests |
| Scrape edit.tosdr.org | Case list needs a login; scraping HTML is fragile and less polite |
| Ask ToS;DR for a dump | Good idea and still an option (team@tosdr.org), but we'd be blocked waiting for a reply |

**Consequences.** The full fetch takes hours, but it's resumable (stop anytime, re-run continues).

**Explain it:** *"I reverse-engineered which endpoint actually returns the quotes, then wrote a polite
fetcher that caches everything, so rate limits and interruptions never lose work."*

---

## D09 — External test set: LexGLUE unfair_tos

**Decision.** Evaluate on LexGLUE `unfair_tos` (CLAUDETTE) in addition to the ToS;DR test split; also
use its "fair" sentences as clean negatives.

**Why.** A different dataset, annotated by different people with a different scheme, shows whether
the model **generalises** or just memorised ToS;DR's style. Its 8 labels map onto some of our categories.

**Alternative rejected.** Evaluating only on the ToS;DR test split: it can't reveal annotation-style overfitting.

**Explain it:** *"I tested on a second, independently labelled dataset to show the model generalises
beyond the data it was trained on."*

---

## D10 — Task framing: chunk-level generation of JSON

**Decision.** The model reads one ~200-token chunk and outputs `{"flags":[...]}` (possibly empty).
Documents are split into chunks; results are aggregated.

**Alternatives considered.**

| Option | Why not |
|---|---|
| Whole-document summarisation | ToS are 5k–30k tokens; 3B models degrade on long input; doesn't fit T4 memory; can't point to exact clauses |
| Pure multi-label classification | No explanations; that's our DeBERTa baseline |
| Span extraction (token tagging) | Finds clauses but needs a separate step to name and explain them |
| 123 binary classifiers (DocBot's approach) | 123 models to train and serve; no explanations; our single model is the comparison |

**Explain it:** *"I framed it as short-chunk → structured JSON, which keeps training examples small,
lets one model do detection, categorisation and explanation, and lets the UI highlight exact text."*

---

## D11 — Output includes the exact quote

**Decision.** Each flag contains `quote`, a verbatim span from the input.

**Why.** (1) The UI can highlight it. (2) It makes **hallucination measurable**: if the quote isn't in
the input, the flag is invalid ("quote faithfulness" metric). (3) Users can verify claims themselves.

**Alternative rejected.** Explanation only: there'd be no way to check what the model was looking at.

**Explain it:** *"Requiring a verbatim quote turns hallucination into something I can measure and
filter: if the quote isn't in the text, the flag is dropped."*

---

## D12 — Merge cases into ~20–25 categories

**Context.** ToS;DR has hundreds of cases; the long tail has very few approved examples
(partial data: 76 cases, only 39 with ≥100 examples).

**Decision.** Map cases to ~20–25 categories (+ `other`) in `configs/taxonomy.yaml`, choosing by
example count and meaning (e.g. several "data shared with third parties" cases → one category).
Final numbers come from the full v0.1 stats.

**Alternative rejected.** Keep every case: rare classes would be learned poorly and make macro-F1 noisy.

**Explain it:** *"I merged hundreds of fine-grained labels into about 20 categories so every class has
enough examples to learn from."*

---

## D13 — About half the training examples are negatives

**Decision.** ~55% of examples have `{"flags": []}` (`configs/data.yaml`). Sources: LexGLUE fair
sentences, document text not covered by any point, DocBot-declined clauses.

**Why.** Real documents are mostly harmless boilerplate. A model trained only on positives learns to
flag *everything*. The exact ratio is tuned in v0.5 against precision/recall.

**Explain it:** *"Most of a real ToS is harmless, so the model has to learn when to say nothing,
otherwise it cries wolf on every paragraph."*

---

## D14 — Split by service (company), never randomly

**Decision.** All examples from one service go to exactly one of train/val/test.
Service comes from `/document/v2` `service_id` (not the source URL, which ~40% of points lack).

**Why.** Companies reuse boilerplate, and one company's clauses resemble each other. A random split puts
near-duplicates in both train and test, which **inflates scores** (data leakage).

**Alternative rejected.** Random split: easier, but the numbers would overstate real-world performance.

**Explain it:** *"I split by company so the test set contains companies the model has never seen;
a random split would leak near-identical boilerplate and inflate the scores."*

---

## D15 — English only for v1

**Decision.** Filter to English quotes (simple stop-word heuristic in v0.1, proper language ID in v0.2).

**Why.** Most labels are English; mixing languages with few examples each dilutes a 3B model and
complicates evaluation. Multilingual is a possible later extension.

**Explain it:** *"I scoped v1 to English to keep the data consistent; multilingual is a clear next step."*

---

## D16 — Four baselines before claiming anything

**Decision.** Compare the fine-tuned model against:
B0 keyword rules · B1 base model zero-shot · B2 base model few-shot · B3 DeBERTa-v3-small classifier.

| Baseline | The question it answers |
|---|---|
| B0 keywords | Is the task trivial? |
| B1 zero-shot | What does the model already know? |
| B2 few-shot | Can prompting alone do it, making fine-tuning unnecessary? |
| B3 DeBERTa | Is a generative LLM even needed for *detection*? |

**Why.** Without baselines, "82% F1" means nothing. If DeBERTa matches us on detection, that's an honest
finding; our model's added value is then explanation + quote.

**Explain it:** *"I benchmarked against simple and strong baselines so every improvement is measured,
not assumed."*

---

## D17 — Metrics

| Metric | Why this one |
|---|---|
| Detection precision / recall / F1 | Trade-off between crying wolf and missing risks |
| Category **macro**-F1 | Averages over classes so rare categories count as much as common ones (micro-F1 would hide failures on rare ones) |
| Severity accuracy + confusion matrix | Shows *which* severities get confused (bad vs blocker matters most) |
| JSON validity rate | The app breaks if the output can't be parsed |
| Quote faithfulness | Measures hallucination (D11) |
| LexGLUE F1 | Generalisation (D09) |
| Latency | Whether the demo is usable on free hardware |

**Explain it:** *"Each metric maps to a failure I care about: false alarms, missed risks, rare classes,
broken output, hallucination, and speed."*

---

## D18 — Serving: 4-bit GGUF with llama.cpp

**Decision.** Merge the LoRA adapter into the base model, convert to **GGUF Q4_K_M** (~2 GB),
run with `llama-cpp-python` on the Mac and on the demo server.

**Alternatives considered.**

| Option | Why not |
|---|---|
| Transformers in fp16 on CPU | ~6 GB, slow on CPU, won't fit comfortably in 8 GB |
| MLX | Great on Apple Silicon but Mac-only; the demo server is Linux |
| ONNX Runtime | Workable, but LLM support/quantization is less mature than llama.cpp |
| Hosted inference API | Not free at useful volumes; defeats "runs locally" |

**Explain it:** *"Quantizing to 4-bit GGUF shrinks the model about 3× so the same file runs fast on my laptop
and on a free CPU server: what I test is what I ship."*

---

## D19 — Demo: Gradio on Hugging Face Spaces (free CPU)

**Decision.** Gradio app on a free HF Space (2 vCPU, 16 GB RAM). Precomputed results for ~10 well-known
services, a length cap on pasted text, optional DeBERTa pre-filter to skip harmless chunks.

**Alternatives.** Streamlit Cloud (lower memory), Vercel/Netlify (no Python model serving),
paid GPU Space (not free), browser extension first (more work; kept optional for v1.0).

**Explain it:** *"Free CPU hosting is slow for an LLM, so I precompute popular examples and filter
chunks before the LLM runs."*

---

## D20 — Our grade is an estimate, never an official ToS;DR grade

**Decision.** The UI labels the A–E grade as a "Fine Print estimate", explains the rule, and links to
tosdr.org for the human-reviewed rating.

**Why.** ToS;DR's AI policy says they never publish grades without human review; presenting a model's
output as their grade would be misleading and disrespect the data source. Also: *not legal advice*.

**Explain it:** *"The model assists, it doesn't judge: I label its grade as an automated estimate and
point to the human-reviewed one."*

---

## D21 — Versioning: git tags for code, HF Hub for data and models

**Decision.** Each milestone = git tag (`v0.1`…) + `CHANGELOG.md` entry, pushed to a private GitHub repo.
Raw/processed data and model weights go to private Hugging Face repos, with the version recorded in
`configs/`. Metrics JSON is committed to `results/`.

**Alternatives.** Data in git (too big; third-party text); DVC (extra tooling for a solo project);
W&B/MLflow (fine, but results-in-git is free and lives next to the code that produced it).

**Explain it:** *"Every result can be traced to a tagged commit and a dataset version, so any experiment
can be reproduced."*

---

## D22 — Python tooling: uv, Python 3.12, config in YAML

**Decision.** `uv` for environments/dependencies, Python pinned to 3.12, settings in `configs/*.yaml`.

**Why.** The Mac's default Python is 3.14, which ML libraries (and Kaggle) may not support yet; 3.12 is
widely supported. `uv` is fast and produces a lockfile (`uv.lock`) for reproducible installs.
YAML configs keep magic numbers out of code, so experiments change config, not code.

**Explain it:** *"Pinned Python and a lockfile make the environment reproducible; configs make
experiments a settings change rather than a code change."*

---

## D23 — How Claude Code is used

**Decision.** Claude Code (Pro) helps write and review code, debug, and design evaluations.
It is **not** used to generate training data or labels.

**Why.** Anthropic's terms restrict using model outputs to train other models, and human labels are
better anyway. All design decisions are documented here so the author can explain and defend each one.

**Explain it:** *"I used an AI coding assistant for engineering speed, but the data is human-labelled and
every design decision is mine to explain."*
