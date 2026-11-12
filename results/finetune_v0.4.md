# v0.4 first fine-tune: findings (2026-11-12)

Model `ft_qwen3b_v1`: Qwen2.5-3B-Instruct + QLoRA adapter (r 16), 1 epoch on all 38,125 train examples, transformers
+ peft on 2×T4 (D37). Scored with the same runner, prompt and metrics as the B1/B2 baselines (fp16 base + merged
adapter, greedy). Numbers: `results/RESULTS.md`; training curve: `results/training/ft_qwen3b_v1.json`.

## Training
- 2,383 steps in 4.3 h (6.5 s/step, ~1,030 tokens/s). Train loss 0.084; val loss 0.129 (step 200) → 0.049 (end),
  still falling slightly, flat over the last ~400 steps. No overfitting sign in 1 epoch.

## Results (full test set, 4,024 chunks)

| System | Detect P | Detect R | Detect F1 | Cat macro-F1 | Severity acc | Flags harmless / hard neg |
|---|---|---|---|---|---|---|
| TF-IDF + LR | 0.862 | 0.815 | 0.838 | 0.617 | 0.675 | 21% / 67% |
| DeBERTa-v3-small | 0.846 | **0.881** | **0.863** | **0.635** | 0.678 | 27% / 63% |
| **Fine-tuned Qwen 3B** | **0.933** | 0.681 | 0.787 | 0.571 | **0.851** | **7% / 36%** |

On test-500 (where the base model was also run): base 3-shot → fine-tuned = macro-F1 0.331 → **0.582**, severity
0.302 → **0.818**, valid JSON 99.6% → **100%**, faithful quotes 91% → **99.6%**, harmless chunks flagged 71% → **5%**.

## What this tells us
1. **Fine-tuning works, and it transformed the model.** Every metric improved by a lot over the base model; output
   is always valid JSON and the quotes are copied exactly (99%+), so explanations come with real evidence.
2. **Best severity of any system (0.85 vs 0.68)** and by far the fewest false alarms (7% of harmless chunks vs
   21–27%). Severity accuracy is measured on correctly-detected flags, so it partly benefits from the model only
   flagging what it's sure of.
3. **It did not reach the DeBERTa bar on categories (0.571 vs 0.635).** The gap is almost all **recall**: it flags
   68% of the chunks that contain a risky clause vs DeBERTa's 88%, at much higher precision (0.93 vs 0.85). The model
   is conservative. Likely causes, to test in v0.5:
   - Missing labels in the training positives (a window can contain a risky clause ToS;DR never recorded for that
     service; see D26/D31) teach the model to stay quiet.
   - DeBERTa's threshold was tuned on val for macro-F1; the LLM has no threshold. A comparable knob: score the
     probability of the empty answer `{"flags": []}` and only accept it above a tuned level.
   - Only 1 epoch; val loss was still edging down.
4. **LexGLUE (other annotators):** detection F1 0.562, close to DeBERTa (0.582), TF-IDF still best (0.615). Quote
   faithfulness falls to 65% there: LexGLUE text is lowercased and pre-tokenised, and the model "repairs" it while
   copying. Check in v0.5 whether this is a scoring artefact (compare after normalising case and spacing).
5. **Speed:** ~1.0 s per chunk on a T4 (fp16, batch 8). Serving speed on the Mac (4-bit GGUF) is measured in v0.6.

## Decision for v0.5
Error analysis first: read ≥50 missed positives to split "model miss" from "label noise", then try the cheapest
recall fix (empty-answer threshold) before retraining.
