# Error analysis v1: fine-tuned model `ft_qwen3b_v1` on the test set (2026-11-12)

Counts: `results/error_analysis_ft_qwen3b_v1__test.json`, made by `python -m fineprint.eval.errors` (no judgement,
only gold labels + saved predictions). Reading sample: 60 random misses (seed 0), in
`data/interim/error_sample_ft_qwen3b_v1__test.jsonl` (gitignored: it holds ToS text).

> **Who judged what.** The counts are mechanical. The 60-case reading below was done by Claude (the coding
> assistant) and should be spot-checked by the author; it is used only to choose what to try next. None of it is
> used as a label or to change training data (CLAUDE.md, D38).

## The unit
A (chunk, gold category) pair, the same as category macro-F1. Test has 3,277 of them; the model gets **49%**
(1,604) and misses 1,673. DeBERTa catches 663 of those misses.

## Mechanical findings

| Miss type | Count | Share of misses |
|---|---|---|
| **empty**: model returned `{"flags": []}` for the chunk | **944** | **56%** |
| partial: flagged the chunk, but not this category | 413 | 25% |
| confused: flagged a different category instead | 316 | 19% |

- **More gold flags per chunk → lower recall:** 1 flag 0.55, 2 flags 0.42, 3+ flags 0.33. The model writes fewer
  flags than the labels have (partial misses).
- **By gold severity:** neutral 0.59, blocker 0.56, bad 0.46, good 0.34. Under-flagging is general, worst for
  "good" (pro-user) points.
- **Weakest categories:** transparency 0.08, advertising 0.11, third_party_sharing 0.21, personal_data_use 0.26.
  DeBERTa is weak on them too (0.27–0.46), so they are hard for any model.
- **Heading-like gold quotes** (under 60 characters, or ending in ":" or "?"): 35% of transparency labels vs 0–18% for
  every other category. Many transparency "clauses" are section titles such as "What Information Do You Collect
  About Me and Why?": label noise, not model error.
- **Top confusions** are neighbouring privacy categories (third_party_sharing → tracking_cookies, tracking_cookies →
  personal_data_use / data_collection) and liability ↔ warranty.
- False alarms on harmless chunks are rare (7%) and spread over many categories: not a problem to fix now.

## Reading 60 misses (Claude's judgement, see note above)

| Group | n | What it looks like | Examples |
|---|---|---|---|
| **Real miss**: a clear clause of that category, model silent | **19** (17 of them *empty*) | warranty disclaimer, recurring billing, ML use of data, account removal at sole discretion, binding arbitration with a 5-day opt-out, irrevocable content licence, feedback licence | #7, #10, #11, #19, #56, #31, #59 |
| **Category boundary**: the model flagged a defensible neighbour, or a second clause in the chunk | 17 | liability vs warranty, advertising vs cookies, analytics vs sharing, general terms vs changes to terms | #12, #28, #5, #17 |
| **Gold label doubtful**: heading, scope sentence or taxonomy mis-mapping | 15 | headings as transparency, "you may need an account" as transparency, a refund clause as data retention, a liability clause as personal data use | #13, #47, #6, #33, #44 |
| **Low value**: procedural or neutral text | 9 | DMCA notice procedure, "anonymous statistics" | #0, #21, #41 |

## What this means
1. **The biggest fixable group is "empty" answers on clear clauses.** 56% of all misses are empty answers, and 17 of
   the 19 real misses in the sample are empty. The model knows the categories but stays silent when unsure.
2. **Measured recall (0.49 per category, 0.68 per chunk) understates real recall.** About 40% of the sampled misses
   are boundary cases or doubtful labels. That is a measurement limit of the ToS;DR labels, and DeBERTa's numbers have
   the same limit.
3. **Taxonomy:** `transparency` is mostly not learnable from the text (headings, scope sentences). Merge or drop it
   in dataset v2, as a *rule-based* taxonomy change (D24 style), not by relabelling examples by hand.

## LexGLUE quotes: a scoring artefact, not hallucination
The fine-tune's quote faithfulness on LexGLUE was 65% vs 99% on our test set. LexGLUE text is lowercased and
pre-tokenised ("you agree that you will not :"), while the model writes normal case and spacing. Ignoring case,
punctuation and spacing, **99.5%** of its LexGLUE quotes appear in the chunk (base 3-shot model: 97.2%). The metric
stays as defined (whitespace-normalised, D32) so earlier rows remain comparable; this note explains the gap.

## Next (v0.5)
1. **Empty-answer threshold** (targets group 1): one forward pass gives P(empty answer). Accept the flags whenever
   P(empty) is below a threshold tuned on val for macro-F1, the same way DeBERTa's threshold was tuned (D38).
2. Report macro-F1 with and without `transparency` so the comparison with DeBERTa is visible both ways.
3. Retrain only if (1) doesn't close most of the gap; GPU quota resets weekly.
