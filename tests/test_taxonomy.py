import pandas as pd

from fineprint.data.taxonomy import (
    assign_category,
    explanation_from_title,
    label_points,
    load_taxonomy,
)


def test_config_is_consistent():
    tax = load_taxonomy()  # raises if anything maps to an undefined category
    assert len(tax["categories"]) >= 20
    assert set(tax["lexglue_labels"].values()) <= set(tax["categories"])


def test_case_override_beats_topic():
    tax = load_taxonomy()
    assert assign_category(293, 31, tax) == "liability_and_indemnity"  # override
    assert assign_category(280, 52, tax) == "acceptable_use"           # topic default
    assert assign_category(1, 999, tax) is None


def test_explanation_cleanup():
    assert explanation_from_title(" Terms may be changed  any time ") == "Terms may be changed any time"


def test_label_points_drops_rare_and_deprecated():
    tax = load_taxonomy()
    rows = [{"case_id": 117, "topic_id": 42, "case_title": "You waive your right to a class action",
             "severity": "bad"}] * 12
    rows += [{"case_id": 278, "topic_id": 55, "case_title": "[DEPRECATED] x", "severity": "neutral"}] * 12
    rows += [{"case_id": 516, "topic_id": 25, "case_title": "rare", "severity": "blocker"}]
    out, report = label_points(pd.DataFrame(rows), tax)
    assert set(out["case_id"]) == {117}
    assert out["category"].iloc[0] == "dispute_resolution"
    assert report["removed_dropped_cases"] == 12
