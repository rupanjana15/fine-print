from fineprint.data.build_dataset import find_spans, flags_inside, overlaps, split_of


def test_split_is_stable_and_roughly_proportional():
    groups = [f"service:{i}" for i in range(5000)]
    splits = [split_of(g, 0.1, 0.1) for g in groups]
    assert splits == [split_of(g, 0.1, 0.1) for g in groups]  # deterministic
    assert 0.07 < splits.count("test") / len(splits) < 0.13
    assert 0.07 < splits.count("val") / len(splits) < 0.13


def test_overlaps():
    assert overlaps((5, 10), [(8, 20)])
    assert not overlaps((5, 10), [(10, 20)])


def test_find_spans():
    assert find_spans("abc def ghi", ["def", "xyz"]) == [(4, 7), None]


def test_flags_inside_only_full_quotes_in_order_deduped():
    f = {"category": "c", "severity": "bad", "explanation": "e"}
    labeled = [{**f, "quote": "B", "span": (50, 60)}, {**f, "quote": "A", "span": (10, 20)},
               {**f, "quote": "A", "span": (10, 20)}, {**f, "quote": "C", "span": (95, 120)}]
    flags = flags_inside((0, 100), labeled)
    assert [x["quote"] for x in flags] == ["A", "B"]
    assert "span" not in flags[0]


def test_max_quote_similarity_flags_repeated_clause():
    from fineprint.data.build_dataset import max_quote_similarity
    quotes = ["These terms are governed by the laws of Germany.",
              "We may share your data with advertising partners.",
              "You can delete your account at any time."]
    texts = ["Welcome to our site. These terms are governed by the laws of Germany.",
             "Our office is open on weekdays. Contact support by email."]
    sims = max_quote_similarity(texts, quotes * 2)  # min_df=2 needs each term twice
    assert sims[0] > 0.6 > sims[1]
