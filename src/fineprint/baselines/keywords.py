"""B0 — keyword rules. Answers "is this task trivially solvable with regexes?" (decision D33).

Rules are hand-written from general knowledge of ToS wording, not tuned on our data. For each
sentence of the input, every category whose pattern matches gives a flag (first matching sentence
per category). Severity = the category's most common severity in *train*; explanation = the
category description from the taxonomy (B0 can't write case-specific explanations).

Usage:  uv run python -m fineprint.baselines.keywords [--sets test lexglue]
"""

import argparse
import json
import re
import time
from collections import Counter, defaultdict

from fineprint.chunk import split_sentences
from fineprint.config import data_path, load_config
from fineprint.eval.score import score_run, write_predictions
from fineprint.eval.sets import load_eval_set
from fineprint.prompt import format_target

SYSTEM = "b0_keywords"

PATTERNS = {
    "tracking_cookies": r"\bcookies?\b|web beacons?|pixel tags?|tracking (technolog|pixel)|fingerprint|local storage|\bsdks?\b",
    "identity_and_accounts": r"real name|legal name|verify your identity|proof of identity|government[- ]issued id|one account per",
    "content_ownership": r"you (retain|keep|own) (all )?(ownership|rights)|own(s|ership of)? (the |all )?(content|feedback|suggestions)|feedback.{0,60}without (compensation|obligation)",
    "government_requests": r"law enforcement|government (request|agenc|authorit)|subpoena|court order|legal process",
    "content_license": r"(worldwide|perpetual|irrevocable|royalty[- ]free|sublicensable|non-exclusive).{0,80}licen[cs]e|licen[cs]e.{0,80}(worldwide|perpetual|irrevocable|royalty[- ]free)",
    "liability_and_indemnity": r"limitation of liability|not (be )?liable|no liability|indemnif|hold (us )?harmless|consequential damages|aggregate liability",
    "eligibility": r"(at least|under(?: the age of)?|older than|over) (the age of )?1[3-8]\b|years of age|minimum age|personal,? non-commercial|age of majority",
    "acceptable_use": r"you (may|must|agree) not|prohibited|shall not (use|post|upload)|acceptable use|not (allowed|permitted) to|reverse engineer|scrap(e|ing)\b",
    "general_legal_terms": r"severab|entire agreement|no waiver|waiver of any|headings|force majeure|assign(ment)? (this|these|the) (agreement|terms)",
    "warranty_and_availability": r"\bas is\b|as available|without warrant|no warrant|disclaim(s|er)? .{0,40}warrant|uninterrupted|error[- ]free",
    "user_rights_and_control": r"right to (access|rectif|correct|eras|delete|object|data portab|withdraw|opt[- ]out)|opt[- ]out|export your (data|content)|request (a )?copy",
    "transparency": r"we will (notify|inform) you|contact us|data protection officer|in plain language|transparen",
    "changes_to_terms": r"(change|modify|amend|update|revise)[sd]? (these|this|the|our) (terms|agreement|policy|privacy)|continued use|without (prior )?notice",
    "business_transfer": r"merger|acquisition|bankruptcy|sale of (all|some|substantially)|change of control|reorgani[sz]ation",
    "personal_data_use": r"(sell|rent|use|process)s? (your )?personal (data|information)|we do not sell|for (any|other) purposes?|aggregated?|anonymi[sz]ed",
    "dispute_resolution": r"arbitrat|class action|class[- ]wide|jury trial|small claims",
    "jurisdiction_and_compliance": r"governed by the laws?|governing law|jurisdiction|exclusive(ly)? (venue|courts?)|\bgdpr\b|\bccpa\b|california residents",
    "data_retention_and_deletion": r"retain|retention|delete (your )?(account|data|content)|deletion|backup copies|after (you )?(close|terminate|delete)",
    "account_termination_and_moderation": r"(suspend|terminate|disable|remove|delete)s? (your )?(account|access|content|service)|at any time,? (for any reason|with or without)|sole discretion",
    "third_party_sharing": r"third[- ]part(y|ies)|share (your |this )?(personal )?(information|data)|partners|affiliates|service providers",
    "security": r"\bsecur(e|ity)\b|encrypt|breach|unauthori[sz]ed access|password",
    "payments_and_refunds": r"refund|non-refundable|\bfees?\b|subscription|billing|price|charge(s|d)? (you|your)|auto(matic(ally)?)?[- ]renew",
    "data_collection": r"\bip address|location (data|information)|geolocation|biometric|voice|device (information|identifiers?)|browsing (history|data)|we collect",
    "advertising": r"advertis|marketing|promotional|targeted ads|interest[- ]based|ad network",
}
COMPILED = {c: re.compile(p, re.IGNORECASE) for c, p in PATTERNS.items()}


def train_flags():
    with open(data_path("processed") / "train.jsonl") as f:
        for line in f:
            yield from json.loads(json.loads(line)["messages"][2]["content"])["flags"]


def majority_severity() -> dict[str, str]:
    """Most common severity per category in train (shared with the TF-IDF baseline)."""
    counts = defaultdict(Counter)
    for f in train_flags():
        counts[f["category"]][f["severity"]] += 1
    return {c: counts[c].most_common(1)[0][0] for c in counts}


def predict(text: str, severity: dict[str, str], descriptions: dict[str, str]) -> str:
    flags = []
    sentences = [text[s:e] for s, e in split_sentences(text, 800)]
    for cat, rx in COMPILED.items():
        hit = next((s for s in sentences if rx.search(s)), None)
        if hit:
            flags.append({"category": cat, "severity": severity.get(cat, "neutral"),
                          "explanation": descriptions[cat], "quote": hit})
    return format_target(flags)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["test", "test-500", "lexglue"])
    args = ap.parse_args()
    assert set(PATTERNS) == set(load_config("taxonomy")["categories"])
    severity = majority_severity()
    descriptions = load_config("taxonomy")["categories"]
    for name in args.sets:
        outputs, extra = {}, {}
        for ex in load_eval_set(name):
            t = time.perf_counter()
            outputs[ex["id"]] = predict(ex["input"], severity, descriptions)
            extra[ex["id"]] = {"seconds": time.perf_counter() - t}
        write_predictions(SYSTEM, name, outputs, extra)
        r = score_run(SYSTEM, name, notes="hand-written regexes, one flag per matching category")
        print(name, r["detection"], "macro-F1", r["category_macro_f1"])


if __name__ == "__main__":
    main()
