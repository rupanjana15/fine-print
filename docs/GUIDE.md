# Fine Print — the whole project, explained

*A personal guide: what was built, every decision behind it, the numbers, and how to explain each part in your own words.*

Version 1.1 · November 2026 · repo `rupanjana15/fine-print`

---

## How to use this guide

This guide is written so you can **explain and defend every part of Fine Print** in an interview, a demo or a
write-up. It is the short, readable path through the project's documentation; the full details stay in the repo:

| If you want… | Read |
|---|---|
| The story and the reasoning (this guide) | `docs/GUIDE.md` |
| Every decision in full, with all alternatives | `docs/DECISIONS.md` (D01–D47) |
| Plain-language ML terms | `docs/CONCEPTS.md` |
| The data, its quirks and licences | `docs/DATA.md` |
| How the parts fit together | `docs/ARCHITECTURE.md` |
| The numbers | `results/RESULTS.md` and the `results/*.md` write-ups |

**Reading plan.** Part 1 (the pitch) and Part 2 (the story) first: they give you the thread. Part 4 (the decision
cards) is the reference you come back to. Before an interview, reread Part 5 (numbers), Part 7 (limitations) and
Part 8 (questions).

Each decision card ends with a **"Say it"** line: one or two sentences you can say out loud. They are written in
the first person because they are yours to say.

---

## Part 1 — The project in one page

### The 20-second version
> Fine Print is a 3-billion-parameter language model I fine-tuned to read Terms of Service and flag the clauses that
> matter, with a severity, a plain-English explanation and the exact quote it's based on. It beats a specialist
> classifier on unseen companies, runs on a laptop, and everything was built on free resources.

### The 2-minute version

1. **The problem.** Nobody reads Terms of Service. ToS;DR volunteers read them by hand and record what matters
   ("you waive your right to a class action"), but they can't cover every service, and terms change.
2. **The data.** I used ToS;DR's human-reviewed points as labels: 31k clauses, each with a quote, a category and a
   severity. I found that 79% of their database is unreviewed machine output, so I trained only on human-approved
   points.
3. **The model.** Qwen2.5-3B-Instruct, fine-tuned with QLoRA on Kaggle's free GPUs. It reads a ~200-token chunk
   and writes JSON: zero or more flags, each with category, severity, explanation and a verbatim quote.
4. **The proof.** Split by company, so the test set is companies the model never saw. Compared against five
   baselines. Final model: category macro-F1 **0.662** vs **0.635** for a DeBERTa classifier, severity accuracy
   **0.85** vs 0.68, quotes found word for word **99.6%** of the time.
5. **The key trick.** The first model was precise but too quiet. Instead of retraining, I read the model's
   probability of answering "nothing here" and tuned a cut-off on validation data, like a classifier threshold.
   Recall went from 0.68 to 0.87 at zero training cost.
6. **Shipping it.** Merged and compressed to a 1.9 GB 4-bit file that runs on an 8 GB laptop; a static demo on
   Hugging Face; a Chrome extension that highlights flagged clauses on the page, with the model running locally.
7. **Honesty.** I measured the weak parts too: the A–E grade is right or one grade off for ~9 in 10 unseen
   companies but exact for only about half, and I measured which fixes were worth building before building them.

### Fine Print at a glance

| | |
|---|---|
| Task | Chunk of a legal document → JSON flags (category, severity, explanation, quote) |
| Base model | Qwen2.5-3B-Instruct (Qwen Research License, non-commercial) |
| Fine-tuning | QLoRA: 4-bit frozen base + LoRA rank 16, 2 epochs, Kaggle 2×T4, ~9 h total |
| Data | 31,362 human-approved ToS;DR points, 24 categories, 4 severities; 47k training chunks |
| Evaluation | Split by company; 5 baselines; one scorer; external test set (LexGLUE) |
| Result | Macro-F1 0.662 (DeBERTa 0.635), severity 0.854, recall 0.909, precision 0.870 |
| Shipped model | Merged, 4-bit GGUF, 1.93 GB, llama.cpp; macro-F1 0.652 |
| Speed (M1 laptop) | ~1.5 s per harmless chunk, ~5 s per flagged chunk; a median document ~45 s–2.5 min |
| Products | Static demo (HF Space), local Gradio app, Chrome extension + local server, public model and labels |
| Cost | Nothing (Claude Code Pro was the only paid tool, used for coding, never for data) |

---

## Part 2 — The story, in the order it happened

This is the thread to tell. Each step had a problem, a decision, and a number that justified it.

### v0.0 — Choosing the project (D01–D05, D21–D23)
The brief: fine-tune a 2–3B model using free resources only. Three ideas were on the table: a corporate-speak
decoder, a Terms of Service red-flag detector, and a workout form coach.

**I picked ToS because real human labels already existed and success was measurable.** The decoder would have
needed synthetic labels and subjective judging; the coach was mostly computer vision. Then the core choices:
3B parameters (big enough for good JSON, small enough for free GPUs and an 8 GB laptop), Qwen2.5-3B-Instruct
(strong at structured output), QLoRA (trains on a free 16 GB GPU), Kaggle for training and the Mac for
everything else.

Two rules were set on day one and never broken: **free resources only**, and **no AI-generated training data or
labels**. Every design choice gets a numbered decision entry, so it can be explained later.

### v0.1 — The data spike: most of the data wasn't human (D06–D09)
ToS;DR has a public API with an incomplete spec, a strict rate limit and no bulk download. I found which
endpoint actually returns the quotes (`/point/v1?case_id=`), scanned case ids because there is no case list, and
wrote a polite fetcher: 2 seconds between requests, exponential backoff on HTTP 429, every response cached so the
hours-long download could be stopped and resumed.

**301,107 points came back, and 79% were "pending": unreviewed suggestions created by ToS;DR's own bot, DocBot.**
Training on them would have meant copying another model's guesses. So: train only on human-approved points
(37,609), and use the suggestions humans *rejected* as tricky negative examples. After requiring a quote and
English text: 32,349 usable points. Blockers, the most serious severity, are only 2.4% of them, a fact that comes
back at the end.

LexGLUE's `unfair_tos` (a different dataset, different annotators) was set aside as an external test set.

### v0.2 — Building the dataset (D10–D15, D24–D31)
The task was framed as **one ~200-token chunk in → JSON out**. Whole documents are 5k–30k tokens, too long for a
3B model and for training memory; chunks also let the app point at exact text. Every flag must carry a **verbatim
quote**, which makes hallucination measurable: if the quote isn't in the text, the flag is wrong.

Key steps and what each one fixed:

- **Cleaning** (D30): strip HTML, keep English, 20–1,500 characters, dedupe → 31,520 points.
- **Taxonomy** (D24): 222 ToS;DR cases → **24 categories**, built on ToS;DR's own topics, with the one catch-all
  topic split up. Severity stays per case, because "notified 30 days before changes" (good) and "changed without
  notice" (bad) are the same category with opposite meaning. The explanation is the case title, already plain
  English. Result: **31,362 points**.
- **Training inputs are real document windows** (D25): each quote is found in its document and a ~200-token window
  of whole sentences is cut around it, with the quote at a random position, so training looks like production.
  98.7% of quotes were found.
- **Negatives** (D13, D26): real documents are mostly harmless, so the model must learn to say nothing. Easy
  negatives come only from services ToS;DR marks as fully reviewed; hard negatives are DocBot suggestions humans
  declined.
- **A noise problem found by reading** (D31): a manual review of 60 negatives showed 22% of "harmless" chunks
  actually contained a risky clause. Cause: volunteers record each issue once per company, so a repeated clause
  looks unlabelled. Fix: drop negatives that look too similar (TF-IDF cosine ≥ 0.6) to any known risky quote.
  Noise fell from 22% to ~10%.
- **Split by company with a stable hash** (D14, D29): a random split would put near-identical boilerplate from one
  company in both train and test and inflate scores. Hashing the company id means a company never moves between
  splits, even when the data is rebuilt.

Final dataset: **38,143 / 4,773 / 4,024** chunks (train / val / test), about 40% negatives.

### v0.3 — Baselines first, so the fine-tune has something to beat (D16, D17, D32–D36)
Before training anything, every alternative was measured with **one scorer** and the same output format:

| System | Category macro-F1 (test) | What it showed |
|---|---|---|
| Always flag (reference) | 0.006 | Gets detection F1 0.79 for free, so detection F1 alone means little |
| Keyword rules | 0.371 | The task isn't trivial |
| TF-IDF + logistic regression | 0.617 | Cheap and strong: most signal is vocabulary |
| Qwen 3B zero-shot | 0.224* | Breaks JSON 20% of the time, flags 79% of harmless chunks |
| Qwen 3B few-shot | 0.331* | Format fixed, but categories and severity still poor |
| **DeBERTa-v3-small** | **0.635** | **The bar to beat** |

\*on the 500-chunk subset.

The "always flag" row was the important insight: because 65% of test chunks contain a flag, a system that flags
everything scores 0.79 detection F1 without reading anything. So the headline metric became **category
macro-F1**, plus precision and how often harmless text gets flagged.

Two engineering problems on the way: the Mac was far too slow (10 s per chunk; DeBERTa killed by memory pressure),
so heavy jobs moved to Kaggle, driven from the Mac with the Kaggle CLI; and DeBERTa first scored zero because the
new transformers version loaded it in half precision, where the optimizer divides by zero on the first step (D36).

### v0.4 — The first fine-tune: precise but quiet (D37)
A small, explicit training script (Hugging Face Trainer + PEFT, my own unit-tested masking so the loss covers only
the answer). Timed against Unsloth on Kaggle: plain data-parallel training on both T4 GPUs was faster (4.6 h per
epoch vs 6.9 h), so it won. One epoch on all data in 4.3 hours.

Two failed starts fixed along the way: running out of GPU memory at the first validation check (the loss over a
151k-token vocabulary needed 3.6 GB at once), and a preinstalled library on Kaggle that broke the adapter loading.

**Result:** the best precision (0.93) and best severity accuracy (0.85) of any system, and only 7% false alarms,
but recall of only 0.68 and macro-F1 0.571, below DeBERTa.

### v0.5 — Reading the errors, a free fix, and a second epoch (D38, D39)
The error analysis counted every miss: **56% were empty answers**, the model saying "nothing here" on a chunk
with a labelled clause. It knew the categories but stayed silent when unsure.

**The empty-answer threshold (D38).** Every answer starts `{"flags":`, and the next token is either ` []` (nothing)
or ` [{"` (a flag follows). One forward pass gives the probability of "nothing"; if it's below a cut-off, the
model's best non-empty answer is used instead. The cut-off was tuned on validation data (t = 0.9), exactly like
DeBERTa's threshold. **Recall 0.68 → 0.87, macro-F1 0.571 → 0.626, no retraining.**

Checking it, a hidden setting surfaced: Qwen ships a repetition penalty of 1.05 that the library applies even in
greedy decoding, and since ` []` appears in the prompt, it was quietly pushing the model to flag more. It is now
set explicitly everywhere.

**A second epoch (D39)**, continued from the first adapter at half the learning rate (mimicking a two-epoch
schedule), with a rule fixed beforehand: keep it only if it scores higher on validation. It did: **test macro-F1
0.662 vs DeBERTa 0.635**, ties or beats DeBERTa on 18 of 24 categories, severity 0.854. The cost: on LexGLUE (the
external set) it fell from 0.589 to 0.466, since it fitted ToS;DR's labelling style more closely. That is reported
as a limitation, not hidden, and v1 wasn't brought back because choosing on test data would be cheating.

### v0.6 — Making it run on a laptop (D40, D41)
The adapter was merged into the base model and compressed to a **4-bit GGUF file (1.93 GB)** for llama.cpp. The
threshold had to survive the move, so it was re-implemented by reading the raw scores (logits) at that one
position. **Macro-F1 0.652 vs 0.658 at full precision**; flag/no-flag answers agree on 487 of 500 chunks.
On the M1: ~1.5 s per harmless chunk, ~5 s per flagged chunk.

The scoring of the 4-bit file moved to a free Kaggle CPU after the laptop overheated: a pattern that repeated.

**The A–E grade (D41).** A counting rule: good +1, bad −1, blocker −2 per distinct issue, four cut-offs to A–E,
fitted by dynamic programming on ToS;DR's human grades of training companies. With ToS;DR's own flags it is exact
for 58% of unseen companies and never more than one grade off.

### v0.7 — The demo, and the grade's first real test (D42)
The plan was a Gradio app on Hugging Face's free CPU. That turned out to need a paid account, and the free GPU
tier needs an account at least 30 days old. So the public demo became a **free static page** with 10 well-known
services analysed in advance on Kaggle CPUs (8 of them companies the model never trained on). The interactive
app runs locally; the live version waits for the free GPU tier (~mid-December 2026).

Running real documents end to end gave the first honest grade check: **the model's grade was more generous than
ToS;DR's for 8 of 10 services.** Instead of hiding it, the demo says so next to the grade.

### v1.0 — Release, and measuring the grade properly (D43, D44)
The README, a demo GIF, the model card and the public model went out. The labels (not the documents, which the
companies own) were published as a dataset, with the same company split.

**Whole-service grading (D43).** All documents of 109 unseen companies, 9,086 chunks, on Kaggle GPUs:

- The grade is **exact for 46%** and **within one grade for 88%** of test companies (always guessing the most
  common grade: 29%). ToS;DR's own flags with the same rule: 58% / 100%.
- Over whole services the errors are **balanced** (17 too kind, 11 too harsh): the demo's generosity was mostly
  from reading one document instead of the whole service.
- The real gap: the model finds **about half as many blockers** (0.31 per company vs 0.60). A blocker counts
  double, so the worst companies drift up a grade.
- Refitting the rule on the model's own flags overfitted (test 0.37), so the original rule stayed.

### v1.1 — Measure before you build (D45–D47)

- **Blockers (D45):** before spending GPU time, a ceiling check: give the model ToS;DR's true blocker counts and
  re-grade. Even a perfect fix lifts test from 0.51 to 0.56 exact, about two companies of 45. A realistic fix is
  maybe one, which 45 companies can't distinguish from noise. Not built.
- **Browser extension (D46):** a Chrome extension that sends the page to a small server on your own laptop, runs
  the same pipeline, and highlights the flagged quotes on the page. Nothing leaves the machine, nothing costs
  money. Tried on Apple's media terms: 84 chunks, ~3 minutes, grade D.
- **Pre-filter (D47):** would running the fast DeBERTa first and skipping "harmless" chunks speed things up? It
  would save ~23% of the time but lose 4.8% of correct flags, because the chunks it skips are already the fast
  ones. Not built.
- **Apple check:** on Apple's terms, all 18 quotes were found word for word; 3 of ToS;DR's 4 points on that
  document matched exactly; of 2 blockers, one matched a real Apple blocker and one was a false alarm.

---

## Part 3 — How it works, end to end

```
 ① DATA (Mac)                     ② TRAINING (Kaggle 2×T4)          ③ EVALUATION
 ToS;DR API ─▶ cache ─▶ clean     dataset zip ─▶ QLoRA, 2 epochs    one scorer for every system
   ─▶ 24 categories ─▶ windows      ─▶ LoRA adapter                  test · test-500 · LexGLUE
   ─▶ negatives ─▶ company split    ─▶ empty-answer threshold 0.9    whole-service grades
   ─▶ train/val/test .jsonl         ─▶ merge ─▶ 4-bit GGUF 1.9 GB

 ④ INFERENCE (laptop or CPU): document ─▶ clean ─▶ chunks (~200 tokens, small overlap)
   ─▶ model per chunk (threshold) ─▶ check JSON + find quote ─▶ merge duplicates ─▶ A–E grade
   Used by: static demo (precomputed) · local Gradio app · Chrome extension via a local server
```

**Shared code is the backbone.** `chunk.py` (how text is cut), `prompt.py` (the instructions and output format)
and `clean_text` are used by dataset building, training *and* inference. If training and production chunked or
prompted differently, the model would silently get worse ("train/serve skew"). Every step reads files written by
the previous one, so any step can be rerun on its own.

**What one answer looks like** (a real output on Uber's terms, a company the model never trained on):
```json
{"flags": [{"category": "dispute_resolution", "severity": "bad",
            "explanation": "You waive your right to a class action",
            "quote": "You acknowledge and agree that you and Uber are each waiving the right to a trial by jury or to participate as a plaintiff or class member in any purported class action or representative proceeding."}]}
```

**Severity scale** (ToS;DR's): good (protects you) · neutral (worth knowing) · bad (against your interest) ·
blocker (a reason to avoid the service).

---

## Part 4 — Every decision, as a card

Each card: the question, what was chosen, the main alternative and why not, and the line to say. Full entries
with every alternative are in `docs/DECISIONS.md`.

### Scoping and setup

#### D01 · Which project?
**Chose:** a ToS red-flag detector. **Instead of:** a corporate-speak decoder (no real labels, subjective
quality) or a workout coach (mostly computer vision).
> I picked a task where real human labels exist and success is measurable, so I could prove the fine-tune works
> instead of just showing nice examples.

#### D02 · How big a model?
**Chose:** ~3B parameters. **Instead of:** <1B (weak at strict JSON), 7–8B (tight on a free T4, barely fits an
8 GB laptop), an encoder classifier (can't explain; kept as a baseline).
> 3B is the sweet spot: big enough to write good JSON and explanations, small enough to train for free and run on
> a laptop.

#### D03 · Which base model?
**Chose:** Qwen2.5-3B-Instruct, with Llama-3.2-3B as a swappable alternative. **Why:** strong structured output,
an instruct model already knows chat and JSON. **Catch:** its licence is non-commercial, fine for a portfolio.
> I chose an instruct model that's good at JSON and kept the choice swappable in config.

#### D04 · How to fine-tune?
**Chose:** QLoRA: the base frozen in 4-bit, small LoRA adapters (rank 16) trained on all linear layers, loss on the
answer only. **Instead of:** full fine-tuning (~40+ GB of GPU memory; a T4 has 16), prompting only (that's the
baseline it beat).
> QLoRA freezes a 4-bit copy of the model and trains about 1% extra parameters, so a 3B model fine-tunes on a free
> 16 GB GPU with almost the same quality as full fine-tuning.

#### D05 · Where does what run?
**Chose:** Kaggle's free GPUs for heavy jobs, the Mac for data, scoring and the final 4-bit model. **Instead of:**
training on the 8 GB Mac (would swap or crash) or paid GPUs (breaks the rule).
> I split the work by what each machine is good at: free cloud GPUs for training, my laptop for data, evaluation
> and running the compressed model.

#### D21, D22 · Versioning and tooling
**Chose:** git tags per milestone, results as JSON committed next to the code, data and weights outside git; `uv`
with Python 3.12 and a lockfile; settings in YAML. **Instead of:** DVC or an experiment tracker (extra tooling for
a solo project).
> Every result traces to a tagged commit and a dataset fingerprint, and experiments are a config change, not a
> code change.

#### D23 · How the AI coding assistant was used
**Chose:** Claude Code for writing and reviewing code, debugging and designing evaluations; **never** for training
data or labels. Every design decision is written down so it can be explained.
> I used an AI coding assistant for engineering speed, but the data is human-labelled and every design decision is
> documented and mine to explain.

### Data

#### D06 · Where do labels come from?
**Chose:** ToS;DR's points: a quoted clause, the case it supports (whose title is a plain-English explanation) and
a severity. **Instead of:** synthetic labels from a big LLM (terms forbid training on its output; inherits its
mistakes), labelling by hand (weeks, one annotator's bias).
> ToS;DR volunteers had already quoted and rated tens of thousands of clauses, so I got real human labels for free,
> and their case titles double as explanations.

#### D07 · Which points to trust?
**Chose:** only `approved` points; DocBot suggestions that humans declined become hard negatives. **Why:** 79% of
points were unreviewed DocBot output.
> I found most of the dataset was unreviewed machine output, so I trained only on human-approved labels and used
> the clauses humans rejected as tricky negatives.

#### D08 · How to collect it?
**Chose:** fetch by case id (the only endpoint with quotes), 2 s between requests, backoff on 429, everything
cached and resumable. **Instead of:** per-service endpoint (no quotes), scraping (fragile), waiting for a dump.
> I worked out which endpoint actually returns the quotes and wrote a polite fetcher that caches everything, so
> rate limits and interruptions never lose work.

#### D09, D28 · An external test set
**Chose:** LexGLUE `unfair_tos`, used **only** for evaluation, never for training (not even its negatives).
> I kept a second, independently labelled dataset completely unseen, so it measures generalisation honestly.

#### D10 · What exactly does the model do?
**Chose:** one ~200-token chunk → JSON flags. **Instead of:** whole-document summaries (too long for 3B; can't point
at clauses), a pure classifier (no explanations), 123 binary classifiers like DocBot.
> Short chunk to structured JSON keeps examples small, lets one model detect, categorise and explain, and lets the
> app highlight the exact text.

#### D11 · Why a quote?
**Chose:** every flag includes a verbatim quote. **Why:** the UI can highlight it, users can check it, and
hallucination becomes measurable.
> Requiring a verbatim quote turns hallucination into something I can measure and filter: if the quote isn't in the
> text, the flag is dropped.

#### D12, D24 · How many categories?
**Chose:** 222 cases → 24 categories built on ToS;DR's topics, the catch-all topic split, tiny ones merged;
severity kept per case; explanation = case title; cases with <10 examples dropped.
> I built on the expert topics ToS;DR already uses and kept severity per clause, so a category says what it's
> about and severity says whether it's good or bad.

#### D13, D26, D31 · Negatives, and the noise in them
**Chose:** ~40% negatives; easy ones only from fully reviewed services, hard ones from declined suggestions; drop
any negative too similar to a known risky quote (TF-IDF ≥ 0.6). **Found by reading 60 samples:** 22% of
"harmless" chunks weren't. After the filter: ~10%.
> Volunteers label each issue once per company, so a repeated risky clause looks harmless. I found that by reading
> samples, then dropped negatives that look too much like known risky clauses, which halved the noise.

#### D14, D29 · How to split?
**Chose:** by company, using a stable hash (80/10/10). **Instead of:** a random split (leaks boilerplate, inflates
scores) or a shuffle-based group split (assignments change when data is rebuilt).
> I split by company so the test set contains companies the model has never seen, and hashing the company id
> means a company never moves between train and test.

#### D15, D30 · Cleaning and language
**Chose:** English only; a few explicit, tested rules (HTML stripped, 20–1,500 characters, dedupe), each with a count
of what it removed. 301,107 → 31,520 points.
> Cleaning is a handful of explicit rules, each with a reason and a count of what it removed.

#### D25 · What do training inputs look like?
**Chose:** real document windows around each quote, quote at a random position, same chunker as production.
**Instead of:** bare quotes (train/test mismatch), fixed chunks (cuts quotes), always centred (a shortcut to learn).
> I train on the same kind of chunks the model sees in production, so there's no gap between training and use.

#### D27 · The prompt
**Chose:** one short shared prompt (~220 tokens) and a fixed JSON key order, imported by data building, training
and inference. **Why short:** it's paid for on every chunk.
> One prompt file is shared by data, training and inference, so they can never drift apart.

#### D44 · What to publish?
**Chose:** the labels only (31,362 points, same split, CC BY-SA), not the training chunks. **Why:** the chunks are
large parts of companies' documents.
> I released the human-reviewed labels with the same company split, under ToS;DR's licence, but not the documents
> themselves, because the companies own that text.

### Evaluation

#### D16, D33 · Baselines
**Chose:** always-flag reference, keyword rules, TF-IDF + logistic regression, the base model zero- and few-shot,
DeBERTa-v3-small. TF-IDF was added because a baseline should be the best cheap thing.
> I benchmarked against simple and strong baselines so every improvement is measured, not assumed; and I added a
> TF-IDF classifier because a baseline should be the best cheap thing, not the easiest to beat.

#### D17, D32 · Metrics and one scorer
**Chose:** detection precision/recall, **category macro-F1** (rare categories count equally), severity accuracy,
JSON validity, quote faithfulness, LexGLUE, latency; every system writes the same JSON and goes through one scorer;
eval sets are fingerprinted.
> Each metric maps to a failure I care about, and the 'always flag' row showed that detection F1 alone would have
> flattered weak systems.

#### D34 · How the base model was prompted
**Chose:** the same prompt as the fine-tune, greedy decoding, no JSON grammar, few-shot examples from train only.
> The baseline sees exactly what my fine-tuned model sees, so the comparison isolates what fine-tuning adds.

#### D35, D36 · Running heavy jobs
**Chose:** Kaggle notebooks, two independent jobs on two GPUs at once, driven from the Mac with the Kaggle CLI;
DeBERTa trained in full precision after it produced NaN.
> My laptop couldn't handle it, so the Mac does data and scoring and Kaggle runs the heavy jobs on two free GPUs;
> DeBERTa came out at zero because half precision made the optimizer divide by zero, so I forced full precision.

### Training

#### D37 · The first fine-tune
**Chose:** my own small script (Trainer + PEFT), data-parallel on both T4s (timed: 4.6 h/epoch vs Unsloth's 6.9 h),
1 epoch, loss on the answer only, over-long examples dropped not cut.
> I timed my script against Unsloth and used plain data-parallel training on both free GPUs because it was faster,
> and the prompt tokens are masked so the model only learns to write the answer.

#### D38 · The empty-answer threshold
**Chose:** read P(empty) at the first decision point, answer with the model's best non-empty output when P(empty) <
0.9, tuned on validation. Also: the hidden repetition penalty made explicit.
> My model was precise but too quiet, so instead of retraining I measured how likely it was to say 'nothing here'
> and tuned a cut-off on validation, like a classifier threshold. Recall went from 68% to 87% at zero training
> cost.

#### D39 · A second epoch
**Chose:** continue from the v1 adapter at half the learning rate, new data order; keep it only if validation
improves. It did (0.625 → 0.652 on val).
> I continued for a second epoch at half the learning rate, re-tuned the threshold, and kept whichever version did
> better on validation, even though the external score went down.

### Serving and the grade

#### D18, D40 · Shipping format
**Chose:** merge the adapter, convert to 4-bit GGUF (Q4_K_M, 1.93 GB), run with llama.cpp; re-implement the threshold
from the logits; same decoding settings as on the GPU. **Cost:** macro-F1 0.658 → 0.652.
> I folded the adapter into the model, compressed it to 4 bits, and re-implemented the threshold by reading the
> logits directly. The 2 GB file loses 0.006 macro-F1 and runs on a laptop.

#### D20, D41 · The A–E grade
**Chose:** a transparent counting rule (good +1, bad −1, blocker −2; cuts A≥7, B≥2, C≥0, D≥−6), fitted on ToS;DR's
human grades of training companies; always labelled "Fine Print estimate", never ToS;DR's grade.
> Each flag adds or removes points and four cut-offs turn points into a grade. I fitted the cut-offs on ToS;DR's
> human grades; with their flags it's exact 58% of the time on unseen companies and never more than one grade off.

#### D19 → D42 · The demo
**Planned:** Gradio on a free Hugging Face CPU (D19). **Changed to:** a free static page of 10 precomputed services,
because free CPU apps now need a paid account and the free GPU tier needs a 30-day-old account. The grade's
generosity is stated next to it.
> Hugging Face stopped hosting free CPU apps, so the demo shows ten services precomputed for free; when I first ran
> whole documents the grade came out too generous, so the demo says so instead of hiding it.

#### D43 · Grading whole services
**Did:** every document of 109 unseen companies through the model. **Found:** exact 46%, within one 88%; errors
balanced; half as many blockers as ToS;DR; refit overfits, so the rule stays.
> On 109 companies the model never saw, the grade is right or one off for 9 in 10, exact for half. The gap wasn't
> the rule, it was missed blockers.

### Follow-ups

#### D45 · Fix blockers?
**Measured the ceiling first:** even perfect blocker counts fix ~2 of 45 test companies. **Not built.**
> Before spending GPU time I checked the ceiling: even a perfect fix would change about two companies, which I
> couldn't tell apart from noise. So I wrote the measurement down and left it.

#### D46 · Browser extension
**Chose:** a Chrome extension plus a small local server (standard library, only answers the extension) running the
same pipeline; quotes highlighted on the page. **Instead of:** a hosted API (not free; text leaves the machine) or
the model in the browser (1.9 GB per profile).
> The extension sends the page to a small server on my own laptop and highlights the quotes it returns. Nothing
> goes to the cloud, nothing costs money, and web pages can't use the server.

#### D47 · A pre-filter for speed?
**Measured:** ~23% faster, 4.8% fewer correct flags, because the chunks it can safely skip are already the fast
ones. **Not built.**
> Skipping chunks with a small classifier would save a quarter of the time but drop 1 in 20 correct flags, so I
> measured it from saved predictions and didn't build it.

---

## Part 5 — Numbers to remember

| What | Number |
|---|---|
| ToS;DR points fetched / unreviewed bot output | 301,107 / 79% |
| Human-approved points used / categories / cases | 31,362 / 24 / 222 |
| Training chunks (train / val / test) | 38,143 / 4,773 / 4,024 |
| Negatives that weren't harmless, before → after filter | 22% → ~10% |
| DeBERTa (the bar): macro-F1 / severity | 0.635 / 0.678 |
| Final model: macro-F1 / severity / precision / recall | **0.662** / **0.854** / 0.870 / 0.909 |
| Categories where it ties or beats DeBERTa | 18 of 24 |
| Base model → fine-tuned (500-chunk subset): macro-F1 | 0.33 (few-shot) → 0.66 |
| Valid JSON / quotes found verbatim | 100% / 99.6% |
| First model: precision / recall → after threshold | 0.93 / 0.68 → 0.87 / 0.87 |
| Threshold | 0.9 on P(empty), tuned on validation |
| Training time | 4.3 h + 4.8 h on 2×T4; 4.6 h/epoch vs 6.9 h Unsloth |
| 4-bit file: size / macro-F1 | 1.93 GB / 0.652 (full: 0.658) |
| Speed on M1 | 1.5 s per harmless chunk, 5.4 s per flagged; median document ~45 s–2.5 min |
| LexGLUE (external) | v1 0.589 (DeBERTa 0.559) → v2 0.466 |
| Grade with ToS;DR's flags (unseen companies) | exact 58%, within one 100% |
| Grade with the model's flags, whole services | exact 46%, within one 88% (majority guess 29%) |
| Blockers per company: model vs ToS;DR | 0.31 vs 0.60 |
| Pre-filter trade-off | −23% time, −4.8% correct flags |
| Money spent | None (Claude Code Pro aside) |

---

## Part 6 — Concepts in one line each

| Term | One line | Where it matters |
|---|---|---|
| Fine-tuning / SFT | Keep training a pretrained model on input → correct output pairs for one task | The whole project |
| LoRA | Train two small matrices beside each big one instead of the big one; ~1% of parameters | D04 |
| QLoRA | LoRA on a 4-bit frozen base, so a 3B model trains on a 16 GB GPU | D04 |
| Completion-only loss | Only the answer tokens count in the loss (prompt tokens get label −100) | D37 |
| Epoch, learning rate, warmup, cosine | One pass over the data; step size; start small, end small | D37, D39 |
| DDP | Each GPU trains on different examples and they average gradients | D37 |
| Overfitting | Memorising training data; watched with validation loss | D39 |
| Data leakage | Test information sneaking into training; prevented by the company split | D14 |
| Hard negative | A harmless example that looks risky; teaches fine distinctions | D07 |
| Label noise | Wrong labels; here mostly missing ones; reduced, measured, accounted for | D26, D31 |
| TF-IDF + cosine | Word-weight vectors and how similar two are; cheap similarity | D31, D33 |
| Precision / recall / F1 | Crying wolf vs missing risks, and their balance | D17 |
| Macro-F1 | F1 per class, averaged, so rare classes count as much as common ones | D17 |
| Threshold tuned on validation | Choose the cut-off on val, never on test | D38 |
| Greedy decoding | Always take the most likely next token: repeatable | D34 |
| Repetition penalty | Lowers the score of tokens already in the text; was hidden in the defaults | D38 |
| Logits | Raw scores for each possible next token, before probabilities | D40 |
| Quantization / GGUF / llama.cpp | Fewer bits per weight; llama.cpp's file format; a fast C++ engine | D40 |
| Merging an adapter | Add the LoRA update into the weights once: an ordinary model | D40 |
| Ordinal cut-points | Ordered grades from one score; best cuts found exactly by dynamic programming | D41 |
| Ceiling (oracle) analysis | Pretend one part is perfect to see the most a fix could give | D45 |
| Train/serve skew | Training and production differ silently; prevented by shared code | D25, D27 |

---

## Part 7 — What went wrong, and what each taught

| Problem | What happened | Fix | Lesson |
|---|---|---|---|
| Most data was bot output | 79% of points were unreviewed DocBot suggestions | Approved only; rejections as hard negatives | Check who wrote the labels |
| "Harmless" wasn't | 22% of negatives held a risky clause | Similarity filter, ~10% left | Read samples, don't trust "unlabelled" |
| Laptop too slow, too hot | 10 s per chunk; DeBERTa killed; overheating on long runs | Kaggle GPUs and CPUs via the CLI | Match the job to the machine |
| DeBERTa scored zero | Half-precision weights + AdamW → NaN on step 1 | Full precision; NaN now crashes loudly | Never trust library dtype defaults |
| GPU out of memory | Validation loss over a 151k vocabulary needed 3.6 GB | Smaller eval batch, allocator setting | First eval step is a memory peak |
| Kaggle library clash | Preinstalled `torchao` broke adapter loading | Uninstall it first; eval-only mode | Pin or remove what you don't use |
| Model too quiet | Recall 0.68, 56% of misses were empty answers | Empty-answer threshold | Count error types before retraining |
| Hidden setting | Qwen's repetition penalty applied silently | Made explicit everywhere | Write down every decoding setting |
| External score dropped | LexGLUE 0.589 → 0.466 after epoch 2 | Reported, not hidden | Don't choose models on test data |
| Hosting changed | Free CPU apps need a paid plan now | Static demo; free GPU tier later | Have a plan B for "free" |
| Grade too generous | 8 of 10 demo grades kinder than ToS;DR | Stated in the demo; measured in D43 | Show the weakness next to the number |
| Quote matching on pages | `<em>` inside a sentence broke highlighting | Block-aware text index | Test with real HTML |

---

## Part 8 — Limitations, and honest answers

- **The grade is the weakest part.** Exact for about half of unseen companies, within one grade for ~9 in 10. The
  worst companies tend to look a grade better because the model finds half as many blockers. One page is less
  reliable than a whole service.
- **LexGLUE fell with the second epoch** (0.589 → 0.466): the model fits ToS;DR's labelling style. On a different
  annotation scheme it raises more false alarms in dispute resolution and jurisdiction.
- **The labels are ToS;DR's.** The model learns what volunteers chose to record; points they never wrote down look
  like false alarms; the 24-category mapping is ours.
- **English only**, and a clause must fit in one ~200-token chunk; cross-section meanings are missed.
- **Slow on CPU:** minutes per long document on a laptop; the free GPU demo is the fix.
- **Not legal advice.** It reads text, not how a company behaves.
- **Measured recall understates real recall:** about 40% of sampled misses were boundary cases or doubtful labels.

---

## Part 9 — Questions you may be asked

**Why fine-tune instead of prompting a big model?**
I measured it: the base 3B model few-shot reached 0.33 macro-F1 and 30% severity accuracy; fine-tuned, 0.66 and
85%. A big hosted model would cost money per page and send the text off the machine, and the project rule was free
and local.

**Why not just use DeBERTa, which is 20× smaller?**
On detection they're close; the fine-tune is better on categories (0.662 vs 0.635) and far better on severity
(0.85 vs 0.68), and it writes an explanation and quotes its evidence, which a classifier can't. I kept DeBERTa as
the bar to beat.

**How do you know it isn't memorising?**
The split is by company, with a stable hash, so the test set is companies it never saw. And I tested on LexGLUE,
a separately annotated dataset, where I report the drop honestly.

**What was the hardest bug?**
The hidden repetition penalty. The threshold at 0.5 should have reproduced the greedy answers but agreed on only
79%. A diagnostic run showed the library applied Qwen's default penalty even in greedy mode, and ` []` was in the
prompt, so the model was being nudged to flag. I made the setting explicit everywhere.

**Why is the threshold legitimate and not cheating?**
It was tuned on validation only, the same procedure DeBERTa's threshold used, then reported once on test.

**How did you handle label noise?**
Reduce, measure, account: negatives only from fully reviewed services, a similarity filter found by reading samples,
and I treat test precision as a lower bound because some "false alarms" are real but unlabelled clauses.

**What would you do with more time or money?**
Run the live demo on a GPU; more reviewed blocker examples (the measured gap); a dataset v2 that drops the noisy
`transparency` category; multilingual data.

**Why didn't you fix the blockers?**
I measured the ceiling first: even perfect blocker counts change about two of 45 test companies. That's below what
45 companies can measure, so I documented it rather than spend 9 GPU hours.

**How much of this did AI write?**
I used Claude Code as a coding assistant for writing and reviewing code and running the jobs. The direction, the
choices between alternatives and the final calls were made in the conversation and each one is documented with
its reasoning; no AI output was used as training data or labels. *(Answer this honestly and in your own words.)*

**How does the extension stay safe?**
The server binds only to localhost and refuses any request whose origin is a normal web page, so a site can't use
your CPU through it; the page text never leaves the machine.

---

## Part 10 — What's next

- **Live "paste your own" demo on Hugging Face's free GPU tier** once the account is 30 days old (~2026-12-15);
  the extension can use it as a faster backend.
- **More reviewed blockers** if ToS;DR's data grows: the one measured gap in the grade.
- **Dataset v2:** drop or merge `transparency` (a third of its labels are section headings).
- **Multilingual** labels, and a Chrome Web Store listing if a $5 fee ever becomes acceptable.

---

## Appendix — Repo map and commands

```
configs/            data, taxonomy, training, grade rule, demo examples
src/fineprint/
  data/             fetch → stats → clean → taxonomy → build_dataset → export_labels
  baselines/        always-flag, keywords, TF-IDF, DeBERTa
  training/         QLoRA training (runs on Kaggle)
  eval/             eval sets, one scorer, threshold, error analysis, grade fit, service grades, pre-filter
  inference/        GGUF runner, document pipeline, grade, demo examples, local server
  chunk.py prompt.py   shared by data, training and inference
notebooks/          Kaggle jobs (training, evaluation, GGUF, demo examples, service grades, Apple check)
app/                local Gradio app, static demo page, model and dataset cards
extension/          Chrome extension
results/            metrics JSON and findings (committed)
docs/               this guide, decisions, concepts, data, architecture, roadmap, Kaggle
```

| Task | Command |
|---|---|
| Install | `uv sync --all-extras` |
| Tests / lint | `uv run pytest -q` · `uv run ruff check .` |
| Rebuild the dataset | `stats` → `clean` → `taxonomy` → `build_dataset` (`uv run python -m fineprint.data.<step>`) |
| Analyse a file locally | `uv run --extra eval python -m fineprint.inference.pipeline --model <gguf> --file terms.txt` |
| Local app | `uv run --extra app python app/app.py` |
| Extension server | `FINEPRINT_MODEL=data/models/fine-print-v2-q4_k_m.gguf uv run --extra eval python -m fineprint.inference.server` |
| Rebuild this PDF | `uv run --with markdown --with pypdf --with reportlab python docs/guide/build.py` |

**Links.** Demo: huggingface.co/spaces/rupp15/fine-print · Model: huggingface.co/rupp15/fine-print-gguf ·
Labels: huggingface.co/datasets/rupp15/fine-print-tos-labels

*Credits: labels from the ToS;DR volunteer community (CC BY-SA 3.0); LexGLUE / CLAUDETTE; base model Qwen2.5-3B-Instruct,
Qwen Research License. Built with Qwen. Not legal advice.*
