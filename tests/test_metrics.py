import json

from fineprint.eval.metrics import parse_output, score

CATS = {"advertising", "dispute_resolution", "tracking_cookies"}


def flag(cat, sev="bad", quote="We use cookies."):
    return {"category": cat, "severity": sev, "explanation": "x", "quote": quote}


def test_parse_strict_and_lenient():
    ok = parse_output('{"flags": []}', CATS)
    assert ok["strict"] and ok["recovered"] and ok["flags"] == []
    fenced = parse_output('```json\n{"flags": [{"category": "advertising", "severity": "bad"}]}\n```', CATS)
    assert not fenced["strict"] and fenced["recovered"] and fenced["flags"][0]["category"] == "advertising"
    bad = parse_output("I think this is fine.", CATS)
    assert not bad["recovered"] and bad["flags"] == []


def test_parse_drops_unknown_category():
    p = parse_output(json.dumps({"flags": [flag("made_up"), flag("advertising")]}), CATS)
    assert [f["category"] for f in p["flags"]] == ["advertising"] and p["invalid_flags"] == 1


def test_score_detection_category_severity_quotes():
    examples = [
        {"id": "a", "kind": "positive", "input": "We use cookies.", "flags": [flag("tracking_cookies")]},
        {"id": "b", "kind": "hard_negative", "input": "Hello there.", "flags": []},
        {"id": "c", "kind": "positive", "input": "Disputes go to arbitration.",
         "flags": [flag("dispute_resolution", "blocker", "Disputes go to arbitration.")]},
    ]
    outputs = {
        "a": json.dumps({"flags": [flag("tracking_cookies", "bad")]}),            # right
        "b": json.dumps({"flags": [flag("advertising", quote="not in text")]}),  # false alarm
        "c": '{"flags": []}',                                                    # miss
    }
    s = score(examples, outputs)
    assert s["detection"]["tp"] == 1 and s["detection"]["fp"] == 1 and s["detection"]["fn"] == 1
    assert s["severity_accuracy"] == 1.0
    assert s["quote_faithfulness"] == 0.5
    assert s["flag_rate_by_kind"]["hard_negative"] == 1.0
    # gold categories: tracking_cookies (F1 1) and dispute_resolution (F1 0) -> macro 0.5
    assert s["category_macro_f1"] == 0.5


def test_label_space_ignores_other_categories():
    examples = [{"id": "a", "kind": "negative", "input": "x", "flags": []}]
    s = score(examples, {"a": json.dumps({"flags": [flag("advertising")]})},
              label_space={"dispute_resolution"})
    assert s["detection"]["fp"] == 0
