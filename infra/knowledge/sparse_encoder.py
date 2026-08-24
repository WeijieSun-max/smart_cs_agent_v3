from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter

from qdrant_client import models


class LocalSparseEncoder:
    def encode(self, text: str) -> models.SparseVector:
        counts = Counter(_terms(text))
        by_index: Counter[int] = Counter()
        for term, count in counts.items():
            by_index[_term_index(term)] += float(count) * _term_weight(term)
        indices = sorted(by_index)
        values = [float(by_index[index]) for index in indices]
        return models.SparseVector(indices=indices, values=values)


def _terms(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    chinese_runs = re.findall(r"[㐀-鿿]+", normalized)
    ascii_words = re.findall(r"[a-z0-9]+", normalized)
    terms: list[str] = list(ascii_words)
    for run in chinese_runs:
        terms.extend(run)
        terms.extend(run[index:index + 2] for index in range(len(run) - 1))
    return terms


def _term_weight(term: str) -> float:
    if re.fullmatch(r"[㐀-鿿]{2}", term):
        return 2.0
    if re.fullmatch(r"[㐀-鿿]", term):
        return 0.5
    return 1.5


def _term_index(term: str) -> int:
    return int.from_bytes(hashlib.blake2b(term.encode("utf-8"), digest_size=4).digest(), "big")
