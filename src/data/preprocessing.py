import re
import html


class TextPreprocessor:
    """Custom preprocessing for financial text."""

    _BILLION = re.compile(r"\$?([\d,.]+)\s*[Bb](?:illion)?")
    _MILLION = re.compile(r"\$?([\d,.]+)\s*[Mm](?:illion)?")
    _PERCENT = re.compile(r"([\d,.]+)\s*%")
    _QUARTER = re.compile(r"\bQ([1-4])\b")
    _FISCAL_YEAR = re.compile(r"\bFY\s?(\d{2,4})\b")
    _HTML_TAG = re.compile(r"<[^>]+>")
    _NON_ASCII = re.compile(r"[^\x00-\x7F]+")
    _WHITESPACE = re.compile(r"\s+")

    _QUARTER_MAP = {
        "1": "first quarter",
        "2": "second quarter",
        "3": "third quarter",
        "4": "fourth quarter",
    }

    def clean(self, text: str) -> str:
        text = html.unescape(text)
        text = self._HTML_TAG.sub(" ", text)
        text = self._NON_ASCII.sub(" ", text)
        text = self._normalize_financials(text)
        text = text.lower()
        text = self._WHITESPACE.sub(" ", text).strip()
        return text

    def _normalize_financials(self, text: str) -> str:
        text = self._BILLION.sub(lambda m: f"{m.group(1)} billion", text)
        text = self._MILLION.sub(lambda m: f"{m.group(1)} million", text)
        text = self._PERCENT.sub(lambda m: f"{m.group(1)} percent", text)
        text = self._QUARTER.sub(lambda m: self._QUARTER_MAP[m.group(1)], text)
        text = self._FISCAL_YEAR.sub(lambda m: f"fiscal year {m.group(1)}", text)
        return text

    def batch_clean(self, texts: list) -> list:
        return [self.clean(t) for t in texts]
