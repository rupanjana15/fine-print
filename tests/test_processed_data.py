"""Checks on the real built dataset; skipped until `build_dataset` has been run."""

import json

import pytest

from fineprint.config import data_path

SPLITS = ("train", "val", "test")
files = {s: data_path("processed") / f"{s}.jsonl" for s in SPLITS}
pytestmark = pytest.mark.skipif(not all(f.exists() for f in files.values()),
                                reason="processed dataset not built")


def rows(split):
    with open(files[split]) as f:
        return [json.loads(line) for line in f]


def test_no_group_in_two_splits():
    groups = {s: {r["group"] for r in rows(s)} for s in SPLITS}
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]


def test_answers_are_valid_json_with_quotes_in_chunk():
    for s in SPLITS:
        for r in rows(s):
            chunk, answer = r["messages"][1]["content"], json.loads(r["messages"][2]["content"])
            assert (len(answer["flags"]) > 0) == (r["kind"] in ("positive", "bare"))
            for flag in answer["flags"]:
                assert flag["quote"] in chunk
