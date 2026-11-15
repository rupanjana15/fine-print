---
license: cc-by-sa-3.0
language: [en]
pretty_name: Fine Print ToS labels
size_categories: [10K<n<100K]
task_categories: [text-classification]
tags: [legal, terms-of-service, privacy, tosdr]
configs:
  - config_name: default
    data_files:
      - {split: train, path: train.parquet}
      - {split: validation, path: validation.parquet}
      - {split: test, path: test.parquet}
---

# Fine Print ToS labels

31,362 human-reviewed clauses from Terms of Service and privacy policies, each with a **category** (24),
a **severity** (good / neutral / bad / blocker), a plain-English **explanation** and the exact **quote**. These are
the labels the [Fine Print model](https://huggingface.co/rupp15/fine-print-gguf) was trained and tested on.
Demo: [rupp15/fine-print](https://huggingface.co/spaces/rupp15/fine-print).

## Source and licence

Every row is an **approved** point from [ToS;DR](https://tosdr.org) ("Terms of Service; Didn't Read"), written and
reviewed by its volunteers, used under **CC BY-SA 3.0**; this derived dataset is released under the same licence.
Please credit ToS;DR. Each row links to its point (`tosdr_url`). Unreviewed points were excluded, and no labels
were written or changed by an AI system.

Quotes are short excerpts of documents owned by the companies that wrote them; the full documents are not included
(follow `document_url`).

## What we changed
- Kept only approved points with a quote, in English, deduplicated, 20–1,500 characters, quote HTML stripped.
- Mapped ToS;DR's 222 kept cases onto **24 categories** (e.g. `dispute_resolution`, `tracking_cookies`,
  `content_license`); cases with fewer than 10 examples were dropped. `explanation` is the case title, `case_id`
  the original case.
- **Split by company**: every point of a service is in the same split (hash of the service id), so test measures
  unseen companies.

| Split | Points |
|---|---|
| train | 25,030 |
| validation | 3,382 |
| test | 2,950 |

Severity: neutral 14,152 · bad 8,655 · good 7,808 · blocker 747.

## Fields
`point_id`, `tosdr_url`, `service_id`, `service`, `document`, `document_url`, `quote`, `category`, `severity`,
`explanation`, `case_id`.

## Limitations
- Labels reflect what ToS;DR volunteers chose to record; a clause without a point is not necessarily harmless.
- 35% of `transparency` quotes are section headings rather than clauses (label noise).
- The model's training *chunks* (windows of the documents around each quote, plus negatives) are not included, to
  avoid republishing whole documents; they can be rebuilt from ToS;DR with the project's code.
