"""The llama.cpp path must build exactly the prompt the model was trained on (D27, D40)."""

import pytest

from fineprint.inference.gguf import chatml
from fineprint.prompt import build_messages


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    except OSError:
        pytest.skip("Qwen tokenizer not available offline")


@pytest.mark.parametrize("chunk", ["We may share your data with partners.", "Line one.\n\nLine \"two\"."])
def test_chatml_matches_training_template(tok, chunk):
    messages = build_messages(chunk)
    assert chatml(messages) == tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = build_messages(chunk, [])
    assert chatml(full, add_generation_prompt=False) == tok.apply_chat_template(full, tokenize=False)
