# v0.5: empty-answer threshold on the fine-tune (2026-11-12)

Systems: `ft_qwen3b_v1` (v0.4, greedy) and `ft_qwen3b_v1_tuned` (same adapter, empty-answer threshold t = 0.9
tuned on val, D38). Scores: `results/eval/`, the val curve: `results/knob_ft_qwen3b_v1.json`.

## Test (4,024 chunks)
| System | Detect P | Detect R | Cat macro-F1 | macro-F1 w/o transparency | Severity acc | False alarms on negatives |
|---|---|---|---|---|---|---|
| DeBERTa-v3-small | 0.846 | 0.881 | **0.635** | **0.649** | 0.678 | 27% |
| Fine-tune, greedy (v0.4) | **0.933** | 0.681 | 0.571 | 0.591 | **0.851** | **7%** |
| **Fine-tune + threshold** | 0.869 | 0.871 | 0.626 | 0.642 | 0.842 | 21% |

LexGLUE (other source, never trained on): **0.589** macro-F1 vs DeBERTa 0.559 and greedy 0.510.

## Val curve (macro-F1)
t = 0.5: 0.401 · 0.7: 0.563 · 0.8: 0.605 · **0.9: 0.625** · 0.95: 0.621 · 0.99: 0.589 · 1.0 (always flag): 0.572.
Wide, flat optimum between 0.85 and 0.95, so the choice isn't fragile.

## Per category (test F1, tuned vs DeBERTa)
- Ties or beats DeBERTa on 12 of 24: biggest wins identity_and_accounts (0.44 vs 0.32), payments_and_refunds
  (0.57 vs 0.49), content_ownership, security, account_termination.
- Behind on: advertising (0.29 vs 0.42), data_collection (0.53 vs 0.62), dispute_resolution (0.69 vs 0.76),
  liability_and_indemnity (0.75 vs 0.81), government_requests (0.49 vs 0.53).

## Parity check and the repetition penalty
At t = 0.5 the threshold agreed with v0.4 greedy answers on only 79% of test chunks. Diagnostic
(`results/knob_parity_diag.json`): Qwen's default `repetition_penalty = 1.05` was applied by `generate()` all
along and pushes greedy decoding toward flagging; the threshold uses raw probabilities. Batch size and dtype play
no part. Now explicit in every runner (D38).

## Second epoch: `ft_qwen3b_v2` (D39)
Continued from the v1 adapter for one more epoch (peak lr 1e-4, new data order), 4.8 h on 2×T4; val loss
0.0493 → 0.0437. Same threshold procedure; t = 0.9 again.

| System | Val | Test | Test w/o transparency | LexGLUE | Detect P / R | Severity |
|---|---|---|---|---|---|---|
| DeBERTa | – | 0.635 | 0.649 | 0.559 | 0.846 / 0.881 | 0.678 |
| v1 + threshold | 0.625 | 0.626 | 0.642 | **0.589** | 0.869 / 0.871 | 0.842 |
| **v2 + threshold** | **0.652** | **0.662** | **0.675** | 0.466 | 0.870 / **0.909** | **0.854** |

- **Chosen: v2**, by the rule fixed before the run (higher on val).
- Beats DeBERTa on test and ties or beats it on 18 of 24 categories; behind on data_collection and advertising.
- **Limitation: LexGLUE fell to 0.466**, and v2 is below v1 there at every threshold (val-free check:
  t 0.7/0.8/0.9/0.95 → v1 0.487/0.574/0.589/0.574, v2 0.450/0.472/0.466/0.479). Most of it is extra false
  alarms in dispute_resolution and jurisdiction. LexGLUE has 172 positives over 5 categories, lowercased,
  pre-tokenised text and its own annotation scheme. v1 stays saved for comparison.
- Quote faithfulness on LexGLUE rose from 67% to 94% (v2 copies the lowercased text more literally).
