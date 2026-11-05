# Changelog

One entry per tagged milestone (see `docs/ROADMAP.md`).

## v0.1 — 2026-11-05 · Data spike
- `fetch_lexglue`: LexGLUE unfair_tos downloaded (5,532 / 2,275 / 1,607; ~89% fair)
- `fetch_tosdr`: polite cached fetcher; found the undocumented quote fields and the 400-for-empty-case quirk
- Full ToS;DR points fetch: 301,107 points → **32,349 usable** (approved, quoted, English) across 247 cases
- Found that ~79% of points are unreviewed DocBot suggestions → excluded (D07)
- `stats`: `results/data_stats.json` + 100-quote manual review (labels good; HTML needs cleaning)
- Licensing checked: ToS;DR CC BY-SA 3.0; AI policy → our grade is an estimate (D20)
- Docs: `DATA.md`, `DECISIONS.md` (D01–D23), `CONCEPTS.md`, docs index, `CLAUDE.md`
- Documents fetch (4,655 docs) started; finishes outside this tag

## v0.0 — 2026-11-04
- Repository created with folder layout, `uv` project and configs
- Architecture (`docs/ARCHITECTURE.md`) and roadmap (`docs/ROADMAP.md`)
- Verified data access: LexGLUE `unfair_tos` loads; ToS;DR API is rate-limited (needs throttled fetcher)
