from __future__ import annotations

import math
import re
import statistics
from collections.abc import Iterable

WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)
SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def word_count(text: str) -> int:
    return len(words(text))


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _document_metrics(text: str) -> dict[str, float]:
    token_count = max(word_count(text), 1)
    sentences = [match.group(0).strip() for match in SENTENCE_RE.finditer(text)]
    sentence_lengths = [
        float(word_count(sentence)) for sentence in sentences if word_count(sentence)
    ]
    paragraphs = [paragraph for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    paragraph_lengths = [float(word_count(paragraph)) for paragraph in paragraphs]

    def per_thousand(character: str) -> float:
        return round(text.count(character) * 1000 / token_count, 3)

    return {
        "words": float(token_count),
        "sentence_words_mean": round(statistics.fmean(sentence_lengths), 3)
        if sentence_lengths
        else 0.0,
        "sentence_words_median": round(statistics.median(sentence_lengths), 3)
        if sentence_lengths
        else 0.0,
        "sentence_words_p25": round(_percentile(sentence_lengths, 0.25), 3),
        "sentence_words_p75": round(_percentile(sentence_lengths, 0.75), 3),
        "sentence_words_stdev": round(statistics.pstdev(sentence_lengths), 3)
        if len(sentence_lengths) > 1
        else 0.0,
        "paragraph_words_median": round(statistics.median(paragraph_lengths), 3)
        if paragraph_lengths
        else 0.0,
        "paragraph_count": float(len(paragraphs)),
        "question_marks_per_1000_words": per_thousand("?"),
        "exclamation_marks_per_1000_words": per_thousand("!"),
        "colons_per_1000_words": per_thousand(":"),
        "semicolons_per_1000_words": per_thousand(";"),
        "parentheses_per_1000_words": round(
            (text.count("(") + text.count(")")) * 500 / token_count, 3
        ),
        "dashes_per_1000_words": round(
            (text.count("—") + text.count("–") + text.count(" -- ")) * 1000 / token_count,
            3,
        ),
        "ellipsis_per_1000_words": per_thousand("…")
        + round(text.count("...") * 1000 / token_count, 3),
        "list_lines_share": round(
            sum(1 for line in text.splitlines() if re.match(r"^\s*(?:[-*+] |\d+[.)] )", line))
            / max(len(text.splitlines()), 1),
            3,
        ),
        "heading_lines_share": round(
            sum(1 for line in text.splitlines() if re.match(r"^\s{0,3}#{1,6}\s+", line))
            / max(len(text.splitlines()), 1),
            3,
        ),
    }


def analyze_documents(documents: Iterable[str]) -> dict[str, object]:
    """Measure each document first so a long document cannot dominate the corpus."""
    per_document = [_document_metrics(document) for document in documents if document.strip()]
    if not per_document:
        return {"document_count": 0, "equal_document_weighting": True, "measures": {}}

    measure_names = tuple(per_document[0])
    measures: dict[str, dict[str, float]] = {}
    for name in measure_names:
        values = [document[name] for document in per_document]
        measures[name] = {
            "median": round(statistics.median(values), 3),
            "p25": round(_percentile(values, 0.25), 3),
            "p75": round(_percentile(values, 0.75), 3),
        }
    return {
        "document_count": len(per_document),
        "equal_document_weighting": True,
        "measures": measures,
    }
