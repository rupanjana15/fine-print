"""Document pipeline and grade rule, with a fake model (no GGUF needed)."""

import json

from fineprint.inference.grade import grade, grade_from_score
from fineprint.inference.pipeline import aggregate, analyse_document, locate

PARAMS = {"w_good": 1.0, "w_blocker": 2.0, "cuts": [7.0, 2.0, 0.0, -6.0]}


def test_grade_rule():
    assert grade_from_score(7, PARAMS["cuts"]) == "A"
    assert grade_from_score(0, PARAMS["cuts"]) == "C"
    assert grade_from_score(-7, PARAMS["cuts"]) == "E"
    flags = [{"category": "dispute_resolution", "severity": "blocker"}] * 3 + [
        {"category": "data_collection", "severity": "bad"}]
    r = grade(flags, PARAMS)  # the repeated blocker counts once: -1 - 2 = -3 → D
    assert r["counts"]["blocker"] == 1 and r["score"] == -3 and r["grade"] == "D"


def test_locate_ignores_whitespace():
    assert locate("may  sell\nyour data", "We may sell your data.") == (3, 21)
    assert locate("not there", "We may sell your data.") is None


def test_aggregate_merges_overlapping_same_category():
    a = {"category": "x", "quote": "q", "span": (0, 10)}
    b = {"category": "x", "quote": "q", "span": (5, 12)}
    c = {"category": "y", "quote": "q", "span": (5, 12)}
    assert aggregate([a, b, c]) == [a, c]


def test_analyse_document_with_fake_model():
    text = "We may sell your personal data to anyone. " * 3 + "Be nice to each other. " * 60

    def fake(chunk):
        flags = ([{"category": "third_party_sharing", "severity": "bad", "explanation": "e",
                   "quote": "We may sell your personal data to anyone."}] if "sell" in chunk else [])
        return {"output": json.dumps({"flags": flags}), "p_empty": 0.1 if flags else 0.99}

    r = analyse_document(text, fake)
    assert r["n_chunks"] >= 2
    assert len(r["flags"]) == 1 and r["flags"][0]["quote_found"]
    assert r["counts"]["bad"] == 1 and r["grade"] in "ABCDE"
