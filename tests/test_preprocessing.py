from src.data.preprocessing import TextPreprocessor


def test_lowercase():
    p = TextPreprocessor()
    assert p.clean("Hello World") == "hello world"


def test_html_strip():
    p = TextPreprocessor()
    assert "<b>" not in p.clean("<b>Revenue</b> grew")
    assert "revenue" in p.clean("<b>Revenue</b> grew")


def test_billion_normalization():
    p = TextPreprocessor()
    assert "billion" in p.clean("$1.2B revenue")


def test_million_normalization():
    p = TextPreprocessor()
    assert "million" in p.clean("profit of $500M")


def test_quarter_normalization():
    p = TextPreprocessor()
    assert "third quarter" in p.clean("Q3 earnings beat estimates")


def test_percent_normalization():
    p = TextPreprocessor()
    assert "percent" in p.clean("Growth of 5.2%")


def test_whitespace():
    p = TextPreprocessor()
    assert p.clean("  too   many   spaces  ") == "too many spaces"


def test_batch_clean():
    p = TextPreprocessor()
    results = p.batch_clean(["Hello", "WORLD"])
    assert results == ["hello", "world"]


def test_non_ascii_removed():
    p = TextPreprocessor()
    result = p.clean("caf\u00e9 revenues")
    assert "\u00e9" not in result
