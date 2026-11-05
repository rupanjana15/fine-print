# Fine Print — working rules

Fine-tuned ~3B LLM that flags risky Terms-of-Service clauses. Start with `docs/README.md`.

## Documentation is part of every change
The author must be able to explain every part of this project, so:
- **Any design choice** (data, model, training, eval, serving, tooling) gets a numbered entry in
  `docs/DECISIONS.md`: context → decision → alternatives with reasons → consequences → "Explain it" line.
  Superseded decisions are marked, never deleted.
- **New ML concept** used for the first time → add a plain-language section to `docs/CONCEPTS.md`.
- **Data findings** (counts, quirks, licence facts) → `docs/DATA.md`; aggregate numbers → `results/`.
- Architecture changes → `docs/ARCHITECTURE.md`. Milestone progress → `docs/ROADMAP.md` checkboxes.
- Each milestone: `CHANGELOG.md` entry, git tag `vX.Y`, push.

## Constraints
- Free resources only (Claude Code Pro is the one paid tool).
- Never use Claude (or any assistant) output as training data or labels.
- Never train on unreviewed ToS;DR points (`status != approved`); see D07.
- Raw/processed data and model weights stay out of git (`.gitignore`); they go to HF Hub.

## Commands
- `uv sync` / `uv run pytest -q`
- `uv run python -m fineprint.data.fetch_lexglue`
- `uv run python -m fineprint.data.fetch_tosdr {points|documents|services}` (slow, resumable)
- `uv run python -m fineprint.data.stats`
- Dataset pipeline, in order: `fineprint.data.stats` → `fineprint.data.clean` → `fineprint.data.taxonomy`
  → `fineprint.data.build_dataset` (all fast, local; rerun after any fetch or config change)
- `chunk.py` and `prompt.py` are shared by data, training and inference: change them deliberately (D25, D27).
