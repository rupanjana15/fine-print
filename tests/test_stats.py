from fineprint.data.stats import looks_english, source_domain


def test_source_domain():
    assert source_domain("https://www.facebook.com/policy/cookies") == "facebook.com"
    assert source_domain("forum.example.de/privacy") == "forum.example.de"
    assert source_domain(None) is None


def test_looks_english():
    assert looks_english("We may share your information with third parties for advertising.")
    assert not looks_english("Bei den gesetzten Cookies handelt es sich um persistente Cookies")
    assert not looks_english("ok")


def test_source_domain_handles_missing_and_malformed():
    assert source_domain(float("nan")) is None
    assert source_domain("http://[bad") is None
