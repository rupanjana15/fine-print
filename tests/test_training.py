"""Completion-only masking: the loss must see the answer (and the end-of-turn token), never the prompt."""

import json

import pytest

from fineprint.training.train import tokenize_example

FLAGS = [{"category": "data_selling", "severity": "bad", "explanation": "They can sell your data.",
          "quote": "We may sell your personal data."}]
EX = {"id": "t-1", "input": "Section 4. We may sell your personal data. Section 5. Be nice.", "flags": FLAGS}


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    except OSError:
        pytest.skip("Qwen tokenizer not available offline")


def test_only_answer_is_trained(tok):
    r = tokenize_example(tok, EX, max_len=1024)
    trained = [t for t, lab in zip(r["input_ids"], r["labels"]) if lab != -100]
    text = tok.decode(trained)
    assert text.startswith('{"flags": [')
    assert json.loads(text.split("<|im_end|>")[0])["flags"] == FLAGS
    assert "<|im_end|>" in text  # the model must learn to stop
    assert "Section 5" not in text  # no prompt tokens leak into the loss
    assert len(r["input_ids"]) == len(r["labels"]) == len(r["attention_mask"])


def test_too_long_is_dropped_not_truncated(tok):
    assert tokenize_example(tok, EX, max_len=50) is None
