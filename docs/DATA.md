# Data sources — findings & licensing

Notes from the v0.1 data spike (2026-11-04). Numbers for the fetched data live in
`results/data_stats.json` (regenerate with `uv run python -m fineprint.data.stats`).

## ToS;DR API (`https://api.tosdr.org`)

The published OpenAPI spec (`tosdr/docs/api.json`) is **incomplete** — real responses carry more
fields. What actually works:

| Endpoint | Returns | Quotes? | Use |
|---|---|---|---|
| `/point/v1/?case_id=X&page=N` | All points for one case (any status), paginated | ✅ `quote_text`, `quote_start`, `quote_end` | **Main source** |
| `/point/v1/?id=X` | One point | ✅ | Debugging only |
| `/service/v3/?page=N` | Service list, 500/page, ~10.3k services | — | Names, URLs, ratings |
| `/service/v3/?id=X` | Service + its approved points + document list | ❌ no quotes | Document → service mapping (v0.2, if needed) |
| `/document/v1/?id=X` | Full document text (HTML) + `service_id`, wrapped in `parameters` | — | Service for the split + negatives. (`/document/v2` in the spec returns 404 Route Not Found) |

Quirks:
- **Rate limit:** headers say `X-RateLimit-Limit: 5` per 5 s, but short bursts still return
  `429 {"error_msg":"You are doing this too much"}`. The fetcher waits 2 s between calls and backs off
  30 s → 10 min on 429.
- A case with no points returns **400** `"Invalid page number, must be 1 <= page <= 0"`, not an empty list.
- `status` includes `approved`, `declined`, and others — **only `approved` points are used**.
- Quotes can contain HTML fragments (e.g. `</li>\n</`) and non-English text → cleaned/filtered in v0.2.
- Points don't include `service_id`, and ~40% of approved points have no `source` URL either.
  Every point does have a `document_id`, so the service comes from `/document/v2` (`fetch_tosdr documents`).
- Severity comes from `case.classification`: `good` / `neutral` / `bad` / `blocker`.
- There is no public case list (edit.tosdr.org/cases needs a login), so case ids are scanned.
  Real ids start around 120.

### Full fetch results (v0.1, 2026-11-05)

Case ids 1–822 scanned (stopped after 300 empty ids in a row; last real case was 522).

| Count | Value |
|---|---|
| Points fetched | **301,107** across 257 cases |
| Status | pending 239,135 · approved 37,609 · declined 24,262 · other ~100 |
| Approved with a quote | 35,606 |
| **Usable** (approved + quote + English) | **32,349** across 247 cases |
| Severity of usable | neutral 14,608 · bad 8,831 · good 8,142 · **blocker 768 (2.4%)** |
| Quote length (chars) | median 189 · 10th pct 73 · 90th pct 487 · max 4,388 |
| Cases with ≥20 / 50 / 100 / 200 / 500 examples | 201 / 156 / 111 / 62 / 5 |
| Distinct source domains / documents | 2,320 / 4,328 (42% of usable points have no source URL) |
| Most examples from one domain | 77 (khanacademy.org): no single company dominates |

What this means for v0.2:
- **Plenty of data.** 32k positives is ~3× the early estimate, so the ~20–25 category plan (D12) is comfortable.
  111 cases already have ≥100 examples.
- **`blocker` is rare (2.4%).** The most important severity has the fewest examples. Options for v0.2:
  oversample blockers, or evaluate severity with a confusion matrix and per-class recall so this stays visible.
- **Many cases are near-duplicates in meaning** (e.g. 286 "provided as is", 287 "no warranty…",
  288 "no guarantee of accuracy"), which can be merged naturally into categories.
- **Good diversity:** the largest domain contributes only 77 examples, so the company split won't be lopsided.

### Point statuses: who wrote the label?

Measured on the first ~103k points (cases up to 200). The full fetch shows the same pattern (79% pending).

| Status | Count | `ml_score` set? | What it is | Use |
|---|---|---|---|---|
| `pending` | ~82k | yes | **"Created by Docbot version v3"**: unreviewed ML suggestions | ❌ never train on these (another model's guesses) |
| `approved` | ~12.5k | mostly no | Human-written or human-approved | ✅ training labels |
| `declined` (DocBot) | ~4.7k | yes | DocBot suggestions a human **rejected** | 💡 hard negatives for that case (v0.2) |
| `declined` (human) | ~3.6k | no | Rejected human submissions (often duplicates) | ❌ ambiguous, skip |

Sample review of 100 approved quotes: labels match the quotes well. Cleaning needed: HTML tags and
entities (`<p>`, `<li>`, `&nbsp;`), occasional fragments that start mid-sentence, mixed case
("WE MAY TERMINATE…"). Median quote is ~200 characters, which fits the ~200-token chunk design.

ToS;DR's own **DocBot** (github.com/tosdr/DocBot, GPL-3.0) trains 123 per-case binary classifiers
from an internal DB dump that isn't public. It is related work, not a data source.

## Dataset v1 (v0.2, 2026-11-06)

Built by `build_dataset.py` from all 4,655 approved-point documents (4,641 with text; 0 failed fetches).
Full numbers: `results/dataset_report.json`.

| Split | Examples | With flags | Hard negatives | Easy negatives | Negative share | Services/groups |
|---|---|---|---|---|---|---|
| train | 38,143 | 21,933 | 1,067 | 15,143 | 42.5% | 2,053 |
| val | 4,773 | 2,967 | 142 | 1,664 | 37.8% | 249 |
| test | 4,024 | 2,604 | 96 | 1,324 | 35.3% | 276 |

- 30,966 of 31,362 labelled quotes (98.7%) were found in their document; 331 weren't (document changed),
  65 had no document → "bare" examples.
- 1.52 flags per positive example (a window often contains neighbouring approved quotes).
- Flags per category (train): largest `tracking_cookies` 3,219, smallest `business_transfer` 361.
- Size: ~470 tokens per example incl. the ~220-token prompt → **~18M tokens per epoch**.
  Rough estimate at ~2k tokens/s for QLoRA on one T4: ~2.5 h per epoch, which fits a 12 h Kaggle session.
- The 55% negative target is **not reached**: easy negatives can only come from the documents we have,
  and the similarity filter (below) removes a quarter of them. We accept ~40% rather than pad with
  lower-quality negatives.

### Manual review of negatives: missing labels (2026-11-06)

Read 40 random easy negatives and 20 hard negatives from the first build and judged whether each contains
a clause ToS;DR has a case for.

| Sample | Clearly flag-worthy | Arguably | Before filter | After TF-IDF filter (≥ 0.6 dropped) |
|---|---|---|---|---|
| Easy negatives | 9 / 40 | 6 / 40 | **22%** clearly noisy | 3 / 29 kept = **10%** |
| Hard negatives | 10 / 20 | — | **50%** | 4 / 12 kept = **33%** |

**Cause:** ToS;DR records a case roughly **once per service**. If a service's privacy policy already has
a "governed by German law" point, the same clause in its terms is left unannotated. So "comprehensively
reviewed" means *every case was checked*, not *every sentence was labelled*. Hard-negative windows also
often contain *other* risky clauses next to the declined suggestion.

**Mitigation (D31):** drop negatives where any sentence has TF-IDF cosine ≥ 0.6 to any approved quote
(4,335 easy and 1,313 hard negatives removed). Remaining noise (~10% of easy, ~1/3 of hard negatives)
is a known limitation: it pushes the model towards **under-flagging**, so expect recall to be lower than
precision. The threshold was chosen on a 60-example sample, so it is a rough estimate, to be revisited in
v0.5 error analysis.

## LexGLUE `unfair_tos` (CLAUDETTE)

`coastalcph/lex_glue`, config `unfair_tos`: 5,532 / 2,275 / 1,607 sentences (train / val / test),
~89% have no label ("fair"), median ~26 words. 8 labels: Limitation of liability, Unilateral
termination, Unilateral change, Content removal, Contract by using, Choice of law, Jurisdiction,
Arbitration. Used **only** as the external test set, never for training (D28).

## Licensing & obligations

| Source | Licence | What we must do |
|---|---|---|
| ToS;DR points, cases, analyses | **CC BY-SA 3.0** (contribution terms, `tosdr/docs/site/tosdr-terms/tosdr.org-terms-of-contribution.md`) | Credit ToS;DR in README, dataset card and model card; release any derived dataset under CC BY-SA |
| Quoted clauses (`quote_text`) | Copyright of each company | Short excerpts for research; don't republish full documents; keep raw data out of git |
| LexGLUE / CLAUDETTE | CC BY 4.0 for LexGLUE (see its dataset card) | Cite Chalkidis et al. 2022 and Lippi et al. 2019 |

ToS;DR's AI policy (`ai-policy.md`) is about their own internal use: they never publish grades
without human review. So **our A–E grade must never be presented as an official ToS;DR grade.**
The UI calls it a "Fine Print estimate" and links to tosdr.org for the human-reviewed rating.
