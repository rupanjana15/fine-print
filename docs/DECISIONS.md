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
| [D24](#d24--taxonomy-tosdr-topics--24-categories-severity-per-case) | Taxonomy: ToS;DR topics → 24 categories, severity per case | ✅ |
| [D25](#d25--training-inputs-are-document-windows-not-bare-quotes) | Training inputs are document windows, not bare quotes | ✅ |
| [D26](#d26--where-negatives-come-from-and-how-much-to-trust-them) | Where negatives come from, and how much to trust them | ✅ |
| [D27](#d27--one-short-shared-prompt-and-a-fixed-output-schema) | One short shared prompt and a fixed output schema | ✅ |
| [D28](#d28--lexglue-is-never-used-for-training) | LexGLUE is never used for training | ✅ (amends D09, D13) |
| [D29](#d29--split-by-a-stable-hash-of-the-service) | Split by a stable hash of the service | ✅ (implements D14) |
| [D30](#d30--cleaning-rules-and-language-detection) | Cleaning rules and language detection | ✅ (implements D15) |

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

## D05 — Compute: Kaggle for training, Mac for everything else (amended by D37)

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

**Decision.** Evaluate on LexGLUE `unfair_tos` (CLAUDETTE) in addition to the ToS;DR test split.
~~Also use its "fair" sentences as clean negatives.~~ *Amended by D28: LexGLUE is not used for training at all.*

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

**Context.** ToS;DR has hundreds of cases; the long tail has very few approved examples.
Full v0.1 data: 247 usable cases, but only 111 with ≥100 examples and 46 with fewer than 20.
Many cases overlap in meaning (e.g. "provided as is" / "no warranty" / "no guarantee of accuracy").

**Decision.** Map cases to ~20–25 categories (+ `other`) in `configs/taxonomy.yaml`, choosing by
example count and meaning (e.g. several "data shared with third parties" cases → one category).
Final numbers come from the full v0.1 stats.

**Alternative rejected.** Keep every case: rare classes would be learned poorly and make macro-F1 noisy.

**Explain it:** *"I merged hundreds of fine-grained labels into about 20 categories so every class has
enough examples to learn from."*

---

## D13 — About half the training examples are negatives

**Decision.** ~55% of examples have `{"flags": []}` (`configs/data.yaml`). Sources: document text not
covered by any point, and DocBot-declined clauses. *(LexGLUE fair sentences were dropped as a source
by D28; how negatives are chosen and filtered is detailed in D26.)*

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

## D16 — Four baselines before claiming anything (extended by D33)

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

## D17 — Metrics (implemented in D32)

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

## D21 — Versioning: git tags for code, HF Hub for data and models (amended by D37)

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

---

## D24 — Taxonomy: ToS;DR topics → 24 categories, severity per case

**Context.** 247 usable cases (after cleaning). ToS;DR already groups cases into ~27 numbered
*topics* (cookies, third parties, arbitration…). Topic 31 is a grab bag (3.7k examples mixing
liability, age limits, boilerplate); topics 35, 38 and 40 are tiny.

**Decision.** (`configs/taxonomy.yaml`, `src/fineprint/data/taxonomy.py`)
- Category = ToS;DR topic by default, with **case overrides** that split topic 31 into
  `liability_and_indemnity`, `eligibility`, `general_legal_terms`, `acceptable_use`;
  tiny topics merged (35, 38 → `changes_to_terms`; 40 logs → `data_retention_and_deletion`).
- **Severity stays per case**, not per category: "notified 30 days before changes" (good) and
  "changed without notice" (bad) are both `changes_to_terms`.
- **Explanation = the case title**, a plain-English sentence written by ToS;DR curators.
- Drop cases with <10 examples (24 cases, 110 points) and the deprecated case 278.
- Result: **31,362 points, 222 cases, 24 categories**, smallest 378 (business_transfer),
  largest 3,030 (tracking_cookies). See `results/taxonomy_report.json`.

**Alternatives considered.**

| Option | Why not |
|---|---|
| My own categories from scratch | Harder to defend; ToS;DR's topics are curated by domain experts |
| Topics exactly as-is | Topic 31 would be a meaningless "misc" class; tiny topics too small to learn |
| Category = case (222 classes) | Many cases have <50 examples; macro-F1 over 222 classes would be noise. The case is still learned through the explanation text |
| Severity per category | Loses the most useful signal: same topic, opposite meaning |
| Keep neutral boilerplate out (e.g. severability) | Those clauses exist in every document; if unlabeled they'd become false "negatives". The UI can hide neutral flags instead |

**Explain it:** *"I built on the expert topics ToS;DR already uses, split the one catch-all topic,
and kept severity per clause, so a category says what it's about and severity says whether it's good or bad."*

---

## D25 — Training inputs are document windows, not bare quotes

**Context.** At prediction time the model reads ~200-token chunks of a whole document. A ToS;DR quote is
only the exact clause.

**Decision.** For every approved quote, find it in the cleaned document text and take a ~200-token
window of whole sentences around it (`window_around` in `src/fineprint/chunk.py`). The quote's position in
the window is randomised. Every approved quote fully inside the window becomes a flag. If the quote isn't
in the current document (the document changed since annotation), the bare quote is used instead.
The **same chunking code** is used by inference later.

**Measured.** On the first 186 documents, 99% of quotes were found in the document text (2,736 vs 30).

**Alternatives considered.**

| Option | Why not |
|---|---|
| Bare quotes as inputs | Train/test mismatch: the model would only see perfectly cut clauses, never surrounding noise |
| Fixed chunks, label whatever falls inside | Quotes get cut at chunk borders and lose their label |
| Quote always centred | The model could learn "flag the middle sentence" as a shortcut |
| Use `quote_start`/`quote_end` offsets | They point into the raw HTML; string search on cleaned text is simpler and robust |

**Explain it:** *"I train on the same kind of chunks the model sees in production, real document windows
with the clause somewhere inside, so there's no gap between training and use."*

---

## D26 — Where negatives come from, and how much to trust them (amended by D31)

**Context.** "Not annotated" ≠ "harmless": volunteers don't annotate every sentence. A sample review
in v0.2 found an "easy negative" that was arguably third-party sharing, and a DocBot-declined
"hard negative" that clearly *was* a liability clause (declined because an approved point already covered it).

**Decision.** (`src/fineprint/data/build_dataset.py`)
- **Easy negatives:** chunks only from services ToS;DR marks **comprehensively reviewed** (494 of 10,301),
  excluding any chunk that overlaps an approved quote **or a pending DocBot suggestion**.
- **Hard negatives:** windows around DocBot suggestions a human declined, **only if the document has no
  approved point for that case** (2,737 of 13,442 declined suggestions excluded by this rule).
- At most half of negatives are hard; overall ~55% negatives (D13).
- Remaining noise is **accepted, measured and documented**: a manual review of sampled negatives (v0.2)
  and a note that test-set precision is a *lower bound* (some "false positives" are unannotated true flags).

**Alternatives considered.**

| Option | Why not |
|---|---|
| Negatives from any document | Far more unannotated risky clauses |
| Filter negatives with risk keywords ("share", "liability"…) | Model would learn "keyword ⇒ flag", a shortcut that breaks on harmless mentions |
| No hard negatives | Loses the examples that teach fine distinctions |
| All declined suggestions as hard negatives | Many are duplicates of approved points → wrong labels |

**Explain it:** *"Unlabelled text isn't automatically harmless, so I took negatives only from fully reviewed
services, excluded anything near an unreviewed suggestion, and filtered out rejections that were really duplicates."*

---

## D27 — One short shared prompt and a fixed output schema

**Decision.** `src/fineprint/prompt.py` defines the system prompt (~220 tokens: the task, the 24 category
names, 4 severities, JSON rules) and the canonical answer format
`{"flags":[{"category","severity","explanation","quote"}]}` with fixed key order. Dataset building,
training and inference all import it.

**Why.**
- **Short:** the prompt is processed for every chunk. On a free CPU, prompt length is a big part of latency.
  Category *meanings* are learned from 30k examples, not from descriptions in the prompt.
- **Shared:** a mismatch between the training prompt and the inference prompt silently degrades a fine-tuned model.
- **Fixed key order:** consistent targets are easier to learn and to parse.
- `category` replaced the earlier `case` field: cases are too many to predict as labels (D24).

**Alternatives.** Long prompt with category descriptions (slower, little gain after fine-tuning);
no system prompt (works after fine-tuning, but the base-model baselines B1/B2 need the instructions,
and keeping the same prompt makes the comparison fair).

**Explain it:** *"One prompt file is shared by data building, training and inference, so they can never
drift apart, and it's short because it's paid for on every chunk."*

---

## D28 — LexGLUE is never used for training

**Context.** D09/D13 planned to use LexGLUE's "fair" sentences as negatives.

**Decision.** LexGLUE is used **only** for evaluation.

**Why.** Training on LexGLUE's train split, even just its negatives, means the model has seen that
dataset's annotators, documents and sentence style, so the "external" test would no longer be fully
external. ToS;DR documents already provide plenty of negatives.

**Explain it:** *"I kept the external test set completely unseen, so it measures generalisation honestly."*

---

## D29 — Split by a stable hash of the service

**Decision.** Group key = `service:<service_id>` (from the document); fallback `domain:<source domain>`,
then `doc:<id>`. Split = MD5 hash of the group key → bucket: 10% test, 10% val, 80% train.

**Why a hash, not a random shuffle:** a service always lands in the same split, even when the data is
rebuilt with more documents. With a shuffle, adding data would move services between train and test,
and old results would no longer be comparable.

**Alternatives.** scikit-learn `GroupShuffleSplit` (correct grouping, but assignments change when the data
changes); stratifying by category (hard with multi-label groups; the hash split is already close to proportional).

**Explain it:** *"Hashing the company name decides its split, so a company never moves between train and test,
even when I rebuild the dataset."*

---

## D30 — Cleaning rules and language detection

**Decision.** (`src/fineprint/data/clean.py`)
- Strip HTML tags (block tags → space), partial tags at quote edges, HTML entities; collapse whitespace.
- Keep quotes of 20–1,500 characters (shorter: fragments; longer: whole sections, bigger than a chunk).
- Language: `langdetect` (seeded for reproducibility); for text under 40 characters, where it's unreliable,
  the stop-word heuristic.
- Drop exact duplicates of (quote, case); keep the same quote under different cases as multiple flags.
- Result: 301,107 → **31,520** clean points (`results/clean_report.json`).

**Alternatives.** BeautifulSoup for HTML (heavier; quotes are fragments with broken tags, which regex handles
fine); fastText language ID (needs a model download; langdetect is enough for a yes/no English check).

**Explain it:** *"Cleaning is a few explicit, tested rules, each with a reason and a count of what it removed."*

---

## D31 — Filter negatives by similarity to known risky quotes

**Context.** The manual review of the first full build (DATA.md, "Manual review of negatives") found that
22% of easy negatives and 50% of hard negatives contained a clearly flag-worthy clause. ToS;DR records a
case about once per service, so repeats of a known clause are left unlabelled even on comprehensively
reviewed services. Training on them as `{"flags": []}` teaches the model to stay silent on real risks.

**Decision.** Before balancing, compute for every negative the highest TF-IDF cosine similarity between any
of its sentences and any approved quote (from every service). Drop it if ≥ `max_quote_similarity` (0.6,
`configs/data.yaml`). TF-IDF: word 1–2-grams, English stop words removed, sublinear term frequency.

**Result.** 4,335 easy and 1,313 hard negatives removed; clearly noisy easy negatives 22% → ~10% on the
review sample; negative share of train 48.6% → 42.5%.

**Alternatives.**
| Option | Why not (now) |
|---|---|
| Accept the noise | 1 in 5 "harmless" examples would teach the model to miss risks |
| Drop all hard negatives | They're the noisiest, but the filter keeps the cleaner ~60% that still teach fine distinctions |
| Embedding similarity (sentence-transformers) | Better at paraphrases, but a model download and slower; TF-IDF already separates the sample well. Possible upgrade in v0.5 |
| Use the baseline/fine-tuned model to find missed flags and relabel | Good later step (v0.5), but relabelling by a model risks training on its own mistakes; would need manual checks |
| Lower threshold (e.g. 0.45) | Removes more noise but also many clean negatives (on the sample, about as many clean as noisy below 0.6) |

**Consequences.** Fewer negatives than the 55% target (D26). The filter uses labels from all splits, which is
fine: it only *removes* doubtful negatives, it never adds information to the inputs. Remaining noise makes
measured precision on our test set a lower bound and pushes the model towards under-flagging; check recall
per category in v0.5.

**Explain it:** *"Volunteers label each issue once per company, so a repeated risky clause looks 'harmless'.
I found that by reading samples, then dropped negatives that look too much like known risky clauses, which
halved the noise."*

---

## D32 — One evaluation harness, fixed eval sets, an "always flag" reference

**Context.** v0.3 compares several very different systems (regexes, a linear classifier, DeBERTa, a
prompted LLM, later the fine-tuned LLM). The comparison is only fair if they're scored identically.

**Decision.**
- Every system writes the **same output**: the model's JSON string per example, saved as
  `data/predictions/<system>/<set>.jsonl`. One scorer (`src/fineprint/eval/metrics.py`) produces
  `results/eval/<system>__<set>.json`, and `results/RESULTS.md` is regenerated from those files.
- **Eval sets** (`src/fineprint/eval/sets.py`): `test` (4,024), `test-500` (fixed random 500 of test, seed 0,
  for slow systems), `lexglue` (1,607, labels mapped to 5 of our categories). Each result stores a
  fingerprint of the set, so a rebuilt dataset can't be silently compared with old numbers.
- **Parsing:** `json_valid` is strict (what the app needs); every other metric uses a lenient parse
  (finds a `{"flags": [...]}` object anywhere), so a stray code fence doesn't zero a system's quality scores.
  Unknown categories are dropped and counted.
- **On LexGLUE** only the 5 mapped categories count; flags in other categories (e.g. cookies) are ignored,
  since LexGLUE simply has no labels for them.
- **Reference row `ref_always_flag`:** flags every chunk. On our test set, 65% of chunks have flags, so it
  gets detection **F1 0.786** for free. Detection F1 alone is therefore not a useful headline, and the table
  also shows precision and the **flag rate on harmless chunks** (easy and hard negatives).

**Alternatives.** Per-system ad-hoc scripts (easy to make subtly different); scoring only on `test`
(LLMs on a CPU or free GPU are too slow for 4k examples × several systems); an experiment tracker
like W&B (unnecessary: JSON files in git are versioned with the code, D21).

**Consequences.** `test-500` numbers have more noise than full `test` (±~3 points on F1). Compare systems
on the same set only; the table groups rows by set.

**Explain it:** *"Every system produces the same JSON and goes through one scorer, and I include a dumb
'always flag' row, which showed that detection F1 alone would have flattered weak systems."*

---

## D33 — Baseline line-up, as built

| System | What it is | Where it runs |
|---|---|---|
| `ref_always_flag` | Flags every chunk with the most common category | Mac, instant |
| `b0_keywords` | Hand-written regex per category, written from general ToS knowledge, not tuned on our data | Mac, instant |
| `b0b_tfidf_logreg` | TF-IDF 1–2-grams → one logistic regression per category; threshold tuned on val | Mac, ~1.5 min |
| `b1_qwen3b_zeroshot` | Base Qwen2.5-3B-Instruct, our prompt only | Kaggle GPU (D35) |
| `b2_qwen3b_fewshot` | Same + 3 fixed train examples (D34) | Kaggle GPU |
| `b3_deberta_small` | DeBERTa-v3-small, sigmoid per category, threshold tuned on val | Kaggle GPU |

**Added beyond D16: B0b TF-IDF + logistic regression.** It's the standard cheap text classifier and costs
nothing to run, and it turned out to be strong (test detection F1 0.838, category macro-F1 0.617,
LexGLUE F1 0.615). Without it we'd only know the LLM beats regexes, which is a low bar.

**Shared post-processing for classifiers (B0b, B3):** severity = the category's most common severity in
train; explanation = the category description; quote = the chunk sentence with the highest score for that
category. So classifier vs LLM differences come from the model, not the output wrapper. Their quotes are
always faithful by construction; it's the LLMs whose quotes need checking.

**Explain it:** *"I added a TF-IDF classifier because a baseline should be the best cheap thing, not the
easiest thing to beat."*

---

## D34 — How the base model is prompted (B1/B2)

**Decision.**
- Same system prompt as training and inference (`prompt.py`, D27): the base model gets exactly what the
  fine-tuned model will get.
- **Greedy decoding** (no sampling), so runs are repeatable.
- **No JSON grammar** for baselines: we measure whether the model *produces* valid JSON on its own. (A
  grammar can be switched on for the app later: `--json-grammar` in the Mac runner.)
- **Few-shot (B2):** 3 fixed train examples of 300–700 characters, one with no flags, one with one flag, one
  with two, picked with seed 0 (`run_gguf.pick_shots`), placed as earlier chat turns. From train only, so no
  eval data leaks into the prompt.
- Base model licence: Qwen2.5-3B uses the **Qwen Research License** (free for non-commercial use, which
  covers this portfolio project). If the project ever became commercial, Llama-3.2-3B is the swap (D03).

**First observations (Mac smoke test, 10 chunks):** the model often wraps its answer in a list, invents
categories (`account_ownership`), flags almost every chunk, and writes long pretty-printed JSON
(~10 s per chunk on the M1). That's exactly what fine-tuning should fix: format, taxonomy, and when to stay quiet.

**Explain it:** *"The baseline sees the same prompt as my fine-tuned model, decoded greedily and without
help, so the comparison isolates what fine-tuning adds."*

---

## D35 — Heavy jobs run on Kaggle (two GPUs in parallel); the Mac does light work only

**Context.** On the M1 / 8 GB, the base model takes ~10 s per chunk (≈7 h for B1 + B2) and DeBERTa training
was killed by memory pressure. The Mac couldn't be used for anything else meanwhile.

**Decision.**
- **Mac:** data pipeline, cheap baselines, scoring, docs. Nothing that needs a GPU or > a few GB of RAM.
- **Kaggle (free, 2×T4):** one notebook, `notebooks/kaggle_baselines.ipynb`, runs two job queues at once,
  one per GPU: GPU 0 = B1; GPU 1 = DeBERTa then B2. This is **job-level parallelism** (two independent
  jobs), simpler and more robust than splitting one model across GPUs.
- LLM baselines on Kaggle use **Hugging Face transformers in fp16, batched** (`eval/run_hf.py`) instead of
  the 4-bit GGUF: much faster on a GPU, and fp16 is the model's full quality, which is the fairer baseline.
- **Moving data:** `python -m fineprint.kaggle_bundle` zips code + processed data (16 MB) → uploaded once as a
  *private* Kaggle dataset. Results come back as `fine-print-results.zip`; `python -m fineprint.kaggle_results`
  re-scores them on the Mac and checks the eval-set fingerprints match.

**Alternatives.**
| Option | Why not |
|---|---|
| Run overnight on the Mac | Blocks the laptop for ~7 h; DeBERTa doesn't fit in memory alongside other work |
| Smaller eval subsets on the Mac | Noisier numbers, still slow |
| HF Hub dataset + GitHub clone in the notebook | Needs tokens stored as Kaggle secrets (private repo); a zip upload is simpler for now. Revisit in v0.4 for training |
| Colab | Single GPU, session limits less predictable; Kaggle gives 2 GPUs and background "commit" runs |
| Paid APIs / cloud GPUs | Not free |

**Consequences.** B1/B2 latency on Kaggle is not demo latency; demo latency is measured on the Mac/Space with
the GGUF in v0.6. Kaggle's fp16 vs the shipped Q4 GGUF may differ slightly; the fine-tuned model will be
scored both ways in v0.6 to quantify that.

**Explain it:** *"My laptop couldn't handle it, so I split the work: the Mac does data and scoring, and one
Kaggle notebook runs the heavy jobs on two free GPUs at the same time."*

## D36 — DeBERTa trains in fp32; Kaggle is driven from the Mac with the Kaggle CLI

**Context.** First Kaggle run (2026-11-10): B1/B2 finished, but B3 DeBERTa scored 0 everywhere. Its loss was 0.73
at step 0 and `nan` from then on, so it predicted no flags. Reproduced on the Mac's CPU in 2 steps: transformers
v5 loads weights in the **checkpoint's** dtype by default, and deberta-v3's checkpoint is fp16. Training pure fp16
weights with AdamW breaks on the first update (its ε = 1e-8 rounds to 0 in fp16 → divide by zero → NaN).
Also, the run was launched by hand-holding through the Kaggle web UI, which was slow and error-prone.

**Decision.**
- `baselines/deberta.py` loads the model with `dtype=torch.float32`, and stops with an error on any
  non-finite loss instead of saving a broken model and writing empty predictions.
- Inference-only LLM runs (`run_hf.py`) stay fp16 on purpose (D35): no optimizer, no underflow problem.
- Kaggle is driven with the official **Kaggle CLI** (dev dependency) from the Mac: `kaggle datasets version`
  uploads the bundle, `kaggle kernels push --accelerator NvidiaTeslaT4` starts the notebook on 2×T4,
  `kaggle kernels status/output` fetch the result. Staging folders live in `dist/kaggle/` (gitignored).
  The API token sits in `~/.kaggle/access_token` on the Mac only.

**Alternatives.**
| Option | Why not |
|---|---|
| Mixed precision (fp32 master weights + fp16 compute, `torch.autocast` + `GradScaler`) | Faster on T4, but more code; DeBERTa-small in fp32 trains in ~10 min, so not worth it here. Will matter for the LLM fine-tune (QLoRA handles it) |
| Pin transformers < 5 | Hides the cause, blocks upgrades |
| Raise AdamW ε | Still fp16 weights: tiny updates round away; a hack |
| A Kaggle MCP connector | None exists; the CLI already does upload / run / download |
| Keep using the web UI | Manual steps for every rerun |

**Consequences.** One extra Kaggle run (DeBERTa only, ~15 min). Any future training script must set its dtype
explicitly rather than rely on library defaults. Reruns are a few commands, no clicking.

**Explain it:** *"DeBERTa came out at zero because the new library loaded it in half precision, and training in
half precision divides by zero on the first step; I force full precision and make NaN losses crash loudly."*

---

## D37 — First fine-tune: plain transformers + peft on both T4s, 1 epoch, data from the Kaggle dataset

**Context.** v0.4 trains the first adapter. Dataset v1 has 38,143 train examples, mean 418 tokens (p99 717;
18 examples over 1,024 tokens), ~16M tokens per epoch, of which the answer is ~18%. Kaggle gives 2×T4 for at most
12 h per session and ~30 GPU-hours a week. The plan (D04, configs/train.yaml) said Unsloth + TRL, 2 epochs, and
data + checkpoints on the HF Hub.

**Decision.**
- **Our own small script** `training/train.py` on the Hugging Face `Trainer` + PEFT + bitsandbytes (QLoRA: NF4
  base, LoRA r 16 / α 32 on all linear layers, lr 2e-4 cosine, 3% warmup, effective batch 16, fp16 mixed
  precision, gradient checkpointing, length-grouped batches). Completion-only masking is done by us
  (`tokenize_example`, unit-tested), not by TRL's `SFTTrainer`. Over-long examples are dropped, never cut.
- **Backend chosen by measurement:** a 30-step timing run on Kaggle (`results/training/timing-*.json`):
  | Backend | GPUs | s/step | tokens/s | h/epoch |
  |---|---|---|---|---|
  | transformers + peft, DDP (`torchrun`) | 2 | 6.96 | 965 | **4.6** |
  | Unsloth | 1 | 10.35 | 649 | 6.9 |
  Unsloth is faster per GPU, but its free version uses one GPU; plain DDP on both wins. Unsloth stays as a
  `--backend` option.
- **1 epoch on all the data** (not 2): 2 epochs ≈ 9.2 h + evaluation would brush the 12 h session limit. The
  roadmap rule was to subsample only if an epoch took > ~5 h. Whether a 2nd epoch helps is a v0.5 experiment.
- **Data comes from the private Kaggle dataset** (the same bundle as the baselines) instead of an HF dataset
  repo, and the adapter comes back as a notebook output (`fine-print-adapter.zip`, kept in `data/models/`,
  gitignored). This avoids storing an HF write token in Kaggle for now. A time budget (`max_hours: 10.5`) stops
  and saves before the session limit, so a long run still returns an adapter. *Amends D05's "checkpoints to the
  HF Hub" and D21 for now; the HF push comes with the demo (v0.7), which needs an HF token anyway.*
- **Evaluation in the same notebook**, with the same runner and prompt as B1 (`run_hf.py --adapter`, fp16 base
  + merged LoRA, greedy): `test-500` + full `test` on GPU 0, `lexglue` on GPU 1.

**Alternatives.**
| Option | Why not |
|---|---|
| Unsloth on one GPU (the original plan) | Measured 1.5× slower here than DDP on two GPUs |
| TRL `SFTTrainer` | Its completion-only options changed across versions; 20 lines of our own masking are explicit and tested |
| 2 epochs in one session | ~10.5 h with eval; no margin for a slow node |
| 2 epochs over two sessions (resume from checkpoint) | Works, but doubles the bookkeeping before we know a 2nd epoch helps |
| Subsample to ~20k examples | Not needed at 4.6 h/epoch; less data = less coverage of rare categories |
| Truncate long examples | Would teach cut-off JSON; dropping 18 of 38k costs nothing |

**Consequences.** One run ≈ 6 h of the weekly GPU quota. The fine-tuned model is scored on exactly the same sets
and metrics as the baselines (`ft_qwen3b_v1`). The 4-bit GGUF used for serving is scored separately in v0.6.

**Explain it:** *"I wrote a small QLoRA training script, timed it against Unsloth on Kaggle, and used plain
data-parallel training on both free GPUs because it was faster; one epoch of all 38k examples takes about
4.6 hours, and the prompt tokens are masked so the model only learns to write the answer."*
