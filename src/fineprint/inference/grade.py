"""Document grade A–E from the flags, with a small transparent rule (decisions D20, D41).

score = w_good · #good − #bad − w_blocker · #blocker, counting distinct (category, severity) pairs, so the same
issue found in several chunks counts once. Cut-points on the score give A–E. Weights and cut-points are fitted
on ToS;DR's human grades of *train* services (`eval/grade_fit.py` → configs/grade.yaml) and checked on val/test
services. It is always shown as a "Fine Print estimate", never as ToS;DR's grade (D20).
"""

from fineprint.config import load_config

GRADES = ("A", "B", "C", "D", "E")


def counts(flags: list[dict]) -> dict[str, int]:
    """Distinct (category, severity) pairs per severity."""
    pairs = {(f["category"], f["severity"]) for f in flags}
    return {s: sum(1 for _, sev in pairs if sev == s) for s in ("good", "neutral", "bad", "blocker")}


def score(c: dict[str, int], w_good: float, w_blocker: float) -> float:
    return w_good * c["good"] - c["bad"] - w_blocker * c["blocker"]


def grade_from_score(s: float, cuts: list[float]) -> str:
    """cuts: 4 descending score cut-points; score >= cuts[0] → A, ..., below cuts[3] → E."""
    for g, cut in zip(GRADES, cuts):
        if s >= cut:
            return g
    return GRADES[-1]


def grade(flags: list[dict], params: dict | None = None) -> dict:
    p = params or load_config("grade")
    c = counts(flags)
    s = score(c, p["w_good"], p["w_blocker"])
    return {"grade": grade_from_score(s, p["cuts"]), "score": s, "counts": c}
