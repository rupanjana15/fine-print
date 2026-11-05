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
| `/document/v2/?id=X` | Full document text + `service_id` | — | Negatives from uncovered text (v0.2) |

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

### Point statuses: who wrote the label?

From the first ~103k points (cases up to 200):

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

## LexGLUE `unfair_tos` (CLAUDETTE)

`coastalcph/lex_glue`, config `unfair_tos`: 5,532 / 2,275 / 1,607 sentences (train / val / test),
~89% have no label ("fair"), median ~26 words. 8 labels: Limitation of liability, Unilateral
termination, Unilateral change, Content removal, Contract by using, Choice of law, Jurisdiction,
Arbitration. Used as the external test set and as a source of negatives.

## Licensing & obligations

| Source | Licence | What we must do |
|---|---|---|
| ToS;DR points, cases, analyses | **CC BY-SA 3.0** (contribution terms, `tosdr/docs/site/tosdr-terms/tosdr.org-terms-of-contribution.md`) | Credit ToS;DR in README, dataset card and model card; release any derived dataset under CC BY-SA |
| Quoted clauses (`quote_text`) | Copyright of each company | Short excerpts for research; don't republish full documents; keep raw data out of git |
| LexGLUE / CLAUDETTE | CC BY 4.0 for LexGLUE (see its dataset card) | Cite Chalkidis et al. 2022 and Lippi et al. 2019 |

ToS;DR's AI policy (`ai-policy.md`) is about their own internal use: they never publish grades
without human review. So **our A–E grade must never be presented as an official ToS;DR grade.**
The UI calls it a "Fine Print estimate" and links to tosdr.org for the human-reviewed rating.
