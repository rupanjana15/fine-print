from fineprint.data.clean import clean_text, is_english


def test_strips_tags_and_entities():
    raw = "<p>We will NEVER&nbsp;sell your data.</p>\n<ul><li>Not ever.</li></ul>"
    assert clean_text(raw) == "We will NEVER sell your data. Not ever."


def test_strips_partial_tags_at_edges():
    assert clean_text("whether or not you are logged in.</li>\n</") == "whether or not you are logged in."
    assert clean_text('class="x">We may share data') == "We may share data"


def test_keeps_plain_text_untouched():
    assert clean_text("You agree to arbitration.") == "You agree to arbitration."


def test_language():
    assert is_english("We may share your personal information with advertising partners and affiliates.")
    assert not is_english("Nous pouvons partager vos informations personnelles avec nos partenaires.")
