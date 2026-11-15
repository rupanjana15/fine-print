# Whole-service grades (v1.0, decision D43)

**Question.** In the demo, Fine Print's grade was more generous than ToS;DR's for 8 of 10 services (D42). Is the
grade itself the problem, or the fact that the demo reads one document while ToS;DR grades the whole service?

**Method.** Every document of the 109 comprehensively-reviewed val/test services (companies never seen in
training) was cut into chunks (9,086 in all) and run through the v2 model with the 0.9 empty-answer threshold on
Kaggle's 2×T4 (70 min per GPU, `notebooks/kaggle_services.py`). Flags were pooled per service and graded with the
D41 rule, unchanged. Full precision on GPU instead of the shipped 4-bit file: ~9,000 chunks would take days on a
CPU, and the two agree on 487/500 flag/no-flag answers (D40). Numbers: [service_grades.json](service_grades.json).
Reproduce: `python -m fineprint.eval.service_grades prepare|analyse`.

## Results

4,991 of the 9,086 chunks (55%) got at least one flag.

| Test services (unseen companies) | n | Exact | Within one grade |
|---|---|---|---|
| Majority guess | 52 | 0.29 | — |
| **D41 rule on the model's flags** | 52 | **0.46** | **0.88** |
| — same 45 services that have ToS;DR points: model's flags | 45 | 0.51 | 0.89 |
| — same 45 services: ToS;DR's own flags (upper bound for the rule) | 45 | 0.58 | 1.00 |
| Rule refitted on the model's flags of val services | 52 | 0.37 | 0.88 |

Validation services (57) behave the same: 0.51 exact, 0.95 within one with the D41 rule.

**Direction of the errors (test, D41 rule).** 17 services graded too kindly, 11 too harshly, 24 exact. The demo's
"8 of 10 too generous" was mostly the single-document effect: over whole services the errors are close to
balanced. The pattern that remains is at the bottom: of 10 E-rated test services, the model gives 3 an E, 5 a D
and 2 a C.

**Mean flags per service (distinct issues, test).**

| | good | neutral | bad | blocker |
|---|---|---|---|---|
| Model (52 services) | 5.65 | 8.08 | 5.25 | **0.31** |
| ToS;DR's own points (45 services) | 6.27 | 7.44 | 5.40 | **0.60** |

Neutral-vs-bad calibration, suspected from the demo (D42), is not the main problem at service level: good, neutral
and bad counts are close to ToS;DR's. **Blockers are**: the model finds about half as many, and a blocker counts
double in the rule, so the worst services drift up a grade. This matches the clause-level evaluation, where
blocker is the rarest severity (747 of 31k labels).

**Refitting does not help.** Fitting the rule's weights and cuts on the model's own flags of the 57 val services
raised val accuracy to 0.60 but dropped test to 0.37: with ~50 services per split it fits noise. The D41 rule,
fitted on 344 train services' human flags, stays.

## Takeaways
- The grade is right or one off for ~9 in 10 unseen services when the model reads the whole service, against 29%
  for always guessing the most common grade. It is exact for about half.
- Most of the gap to the rule's upper bound (0.58 / 1.00) comes from missed blockers, not from the rule.
- Next lever: blocker recall (more blocker examples or a blocker-specific threshold), not a new grade rule.
