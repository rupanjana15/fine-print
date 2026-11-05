import random

from fineprint.chunk import chunk_text, split_sentences, window_around

TEXT = " ".join(f"Sentence number {i} says something about the terms." for i in range(60))


def test_sentences_cover_text():
    spans = split_sentences(TEXT, max_chars=800)
    assert len(spans) == 60
    assert TEXT[spans[0][0]:spans[0][1]] == "Sentence number 0 says something about the terms."


def test_long_sentence_is_split():
    long = "word " * 500
    assert all(e - s <= 800 for s, e in split_sentences(long.strip(), max_chars=800))


def test_chunks_respect_size_and_overlap():
    chunks = chunk_text(TEXT, target_tokens=100, overlap_tokens=15)
    assert all(e - s <= 400 for s, e in chunks)
    assert chunks[0][0] == 0 and chunks[-1][1] == len(TEXT)
    assert all(b[0] < a[1] for a, b in zip(chunks, chunks[1:]))  # consecutive chunks overlap


def test_window_contains_quote_and_varies():
    q = "Sentence number 30 says something about the terms."
    start = TEXT.index(q)
    windows = {window_around(TEXT, start, start + len(q), 100, random.Random(seed)) for seed in range(10)}
    for s, e in windows:
        assert s <= start and e >= start + len(q) and e - s <= 400
    assert len(windows) > 1  # quote position inside the window is randomised
