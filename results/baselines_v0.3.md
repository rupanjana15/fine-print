# v0.3 baseline findings (2026-11-11)

Numbers: `results/RESULTS.md` (generated). Systems: docs/ARCHITECTURE.md §5, D33. Dataset v1 (D31).
Qwen runs use `test-500` (random 500 of test); cheap baselines and DeBERTa also ran on the full test set.

## Headline (test-500, so every system is on the same examples)

| System | Detect F1 | Cat macro-F1 | Severity acc | JSON valid | Flags harmless / hard-negative chunks |
|---|---|---|---|---|---|
| Always flag (reference) | 0.750 | 0.005 | 0.575 | 1.000 | 100% / 100% |
| Keyword rules (B0) | 0.758 | 0.365 | 0.673 | 1.000 | 77% / 77% |
| TF-IDF + LR (B0b) | 0.812 | 0.601 | 0.649 | 1.000 | 27% / 69% |
| Qwen2.5-3B zero-shot (B1) | 0.669 | 0.224 | 0.290 | 0.796 | 79% / 85% |
| Qwen2.5-3B 3-shot (B2) | 0.784 | 0.331 | 0.302 | 0.996 | 71% / 54% |
| **DeBERTa-v3-small (B3)** | **0.843** | **0.634** | 0.652 | 1.000 | 26% / 69% |

Full test (4,024): DeBERTa 0.863 / 0.635, TF-IDF 0.838 / 0.617.

## What this tells us

1. **The bar for the fine-tune is DeBERTa: category macro-F1 ≈ 0.63.** A 140M-parameter classifier trained
   for 40 min on a T4 beats everything else. The fine-tuned 3B model has to match it on categories *and* add what
   a classifier can't: a written explanation and a faithful quote per flag.
2. **TF-IDF is almost as good (0.617 vs 0.635)** at ~1/20 of DeBERTa's latency. Most of the signal is
   vocabulary; the hard part is the long tail of small categories.
3. **The base Qwen 3B is not usable as-is.** Zero-shot it breaks the JSON format on 20% of chunks, flags
   79% of harmless chunks and gets severity right 29% of the time — *worse than always guessing the most
   common severity (0.575)*. Three examples in the prompt fix the format (99.6% valid) and raise recall, but
   categories (0.331) and severity (0.302) stay poor. This is the case for fine-tuning: the model can read, it
   doesn't know our taxonomy or our severity scale.
4. **Hard negatives are hard for everyone.** Every trained system still flags ~55–70% of hard negatives
   (chunks from the same documents, near but not on a labelled quote). Some of these are probably real but
   unlabelled cases that the D31 filter didn't catch; v0.5 error analysis should read a sample before
   treating this as model error.
5. **LexGLUE (other annotators, other label scheme, 5 categories) is much harder.** Best is TF-IDF 0.615 detection
   F1; DeBERTa 0.582; Qwen ~0.32. Qwen's quote faithfulness drops to 5–20% there, most likely because LexGLUE text
   is lowercased and pre-tokenized and the model "tidies" quotes when copying them. To check in v0.5.

## Cost and speed (Kaggle 2×T4, free)
- Qwen fp16, batch 8: ~1.7 s per chunk on test-500, ~0.8–0.9 s on LexGLUE (shorter chunks).
- DeBERTa: 2 epochs in 2,360 s (fp32, D36); inference ~0.02 s per chunk incl. sentence scoring for quotes.
- First DeBERTa run produced NaN from step 1 (fp16 weights + AdamW); fixed and rerun (D36).
