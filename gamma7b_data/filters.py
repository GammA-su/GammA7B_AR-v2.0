import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


@dataclass
class FilterResult:
    keep: bool
    reason: Optional[str] = None


class BaseFilter:
    def check(self, text: str) -> FilterResult:
        return FilterResult(True, None)


class LengthFilter(BaseFilter):
    def __init__(self, min_chars: int = 200, max_chars: int = 1_000_000):
        self.min_chars = min_chars
        self.max_chars = max_chars

    def check(self, text: str) -> FilterResult:
        size = len(text)
        if size < self.min_chars:
            return FilterResult(False, "too_short")
        if size > self.max_chars:
            return FilterResult(False, "too_long")
        return FilterResult(True, None)


class AsciiRatioFilter(BaseFilter):
    def __init__(self, min_ratio: float = 0.8):
        self.min_ratio = min_ratio

    def check(self, text: str) -> FilterResult:
        if not text:
            return FilterResult(False, "empty")
        ascii_count = sum(1 for ch in text if ord(ch) < 128)
        ratio = ascii_count / max(1, len(text))
        if ratio < self.min_ratio:
            return FilterResult(False, "low_ascii_ratio")
        return FilterResult(True, None)


class SymbolRatioFilter(BaseFilter):
    def __init__(self, max_symbol_ratio: float = 0.45):
        self.max_symbol_ratio = max_symbol_ratio

    def check(self, text: str) -> FilterResult:
        if not text:
            return FilterResult(False, "empty")
        symbol_count = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
        ratio = symbol_count / max(1, len(text))
        if ratio > self.max_symbol_ratio:
            return FilterResult(False, "symbol_ratio")
        return FilterResult(True, None)


class RepetitionFilter(BaseFilter):
    def __init__(self, max_repeat_ratio: float = 0.3, min_lines: int = 6):
        self.max_repeat_ratio = max_repeat_ratio
        self.min_lines = min_lines

    def check(self, text: str) -> FilterResult:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) < self.min_lines:
            return FilterResult(True, None)
        counts = {}
        for ln in lines:
            counts[ln] = counts.get(ln, 0) + 1
        most_common = max(counts.values())
        if most_common / len(lines) > self.max_repeat_ratio:
            return FilterResult(False, "boilerplate_repeat")
        return FilterResult(True, None)


class BoilerplateFilter(BaseFilter):
    def __init__(self):
        self.patterns = [
            "all rights reserved",
            "terms of service",
            "cookie policy",
            "subscribe to our newsletter",
            "privacy policy",
            "navigation",
            "footer",
        ]

    def check(self, text: str) -> FilterResult:
        lowered = text.lower()
        for pat in self.patterns:
            if pat in lowered:
                return FilterResult(False, "boilerplate_phrase")
        return FilterResult(True, None)


class ToxicityFilter(BaseFilter):
    def __init__(self):
        self.pattern = re.compile(r"\b(?:fuck|shit|bitch|cunt|nigger|faggot)\b", re.IGNORECASE)

    def check(self, text: str) -> FilterResult:
        if self.pattern.search(text):
            return FilterResult(False, "toxicity")
        return FilterResult(True, None)


def build_default_filters(use_language_heuristic: bool = True) -> List[BaseFilter]:
    filters: List[BaseFilter] = [LengthFilter()]
    if use_language_heuristic:
        filters.append(AsciiRatioFilter())
    filters.extend([SymbolRatioFilter(), RepetitionFilter(), BoilerplateFilter(), ToxicityFilter()])
    return filters


def apply_filters(text: str, filters: List[BaseFilter]) -> Tuple[bool, Optional[str]]:
    for flt in filters:
        result = flt.check(text)
        if not result.keep:
            return False, result.reason
    return True, None

