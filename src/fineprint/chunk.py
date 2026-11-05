"""Split documents into model-sized chunks. Shared by dataset building and inference, so the
model is trained on exactly the kind of chunks it will see at prediction time (decision D25).

Sizes are in characters (~4 chars per token for English) to avoid loading a tokenizer here.
"""

import random
import re
from functools import lru_cache

CHARS_PER_TOKEN = 4

# Sentence end: . ! ? ; followed by space and an uppercase letter, digit, quote or bracket.
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9\"“(\[])")


@lru_cache(maxsize=16)  # window_around is called once per quote on the same document
def split_sentences(text: str, max_chars: int) -> list[tuple[int, int]]:
    """Return (start, end) offsets of sentences; overly long sentences are split at spaces."""
    spans, start = [], 0
    for m in _SENTENCE_END.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))

    out = []
    for s, e in spans:
        while e - s > max_chars:
            cut = text.rfind(" ", s, s + max_chars)
            cut = cut if cut > s else s + max_chars
            out.append((s, cut))
            s = cut + 1
        if e > s:
            out.append((s, e))
    return out


def chunk_text(text: str, target_tokens: int = 200, overlap_tokens: int = 30) -> list[tuple[int, int]]:
    """Greedily pack whole sentences into ~target-size chunks with a small sentence overlap.

    Returns (start, end) character offsets into `text`.
    """
    target, overlap = target_tokens * CHARS_PER_TOKEN, overlap_tokens * CHARS_PER_TOKEN
    sents = split_sentences(text, max_chars=target)
    chunks, i = [], 0
    while i < len(sents):
        j = i
        while j + 1 < len(sents) and sents[j + 1][1] - sents[i][0] <= target:
            j += 1
        chunks.append((sents[i][0], sents[j][1]))
        if j + 1 >= len(sents):
            break
        # step back over trailing sentences that fit in the overlap budget
        k = j
        while k > i and sents[j][1] - sents[k][0] <= overlap:
            k -= 1
        i = max(k + 1, i + 1) if k < j else j + 1
    return chunks


def window_around(text: str, start: int, end: int, target_tokens: int = 200,
                  rng: random.Random | None = None) -> tuple[int, int]:
    """A ~target-size window of whole sentences that fully contains text[start:end].

    Sentences are added left/right in random order so the quote isn't always in the middle
    (the model must not learn "the flag is always the centre sentence").
    """
    rng = rng or random.Random(0)
    target = target_tokens * CHARS_PER_TOKEN
    sents = split_sentences(text, max_chars=target)
    lo = next((i for i, (s, e) in enumerate(sents) if e > start), 0)
    hi = next((i for i, (s, e) in enumerate(sents) if e >= end), len(sents) - 1)
    lo, hi = min(lo, hi), max(lo, hi)

    while True:
        options = []
        if lo > 0 and sents[hi][1] - sents[lo - 1][0] <= target:
            options.append("left")
        if hi + 1 < len(sents) and sents[hi + 1][1] - sents[lo][0] <= target:
            options.append("right")
        if not options:
            break
        if rng.choice(options) == "left":
            lo -= 1
        else:
            hi += 1
    return min(sents[lo][0], start), max(sents[hi][1], end)
