from __future__ import annotations

import hashlib

import pytest

from idiolect.corpus import (
    MIN_CLEAN_WORDS,
    CorpusError,
    build_corpus,
    decode_text_file,
    make_sample,
    normalize_skill_name,
)
from idiolect.models import SampleMetadata, WritingSample


def _sample(name: str, text: str) -> WritingSample:
    return make_sample(name, text, SampleMetadata())


def _numbered_prose(prefix: str, count: int = 160) -> str:
    return " ".join(f"{prefix}{index}" for index in range(count))


def test_decode_text_file_accepts_utf8_bom_and_utf16() -> None:
    assert decode_text_file("Café".encode("utf-8-sig")) == "Café"
    assert decode_text_file("Grüße\nWelt".encode("utf-16")) == "Grüße\nWelt"


@pytest.mark.parametrize("data", [b"plain\x00binary", b"\xff\xff", b""])
def test_decode_text_file_rejects_non_text_input(data: bytes) -> None:
    with pytest.raises(CorpusError):
        decode_text_file(data)


def test_make_sample_normalizes_newlines_unicode_and_digest() -> None:
    sample = _sample("sample.txt", "Cafe\u0301\r\nFirst\rSecond")

    assert sample.text == "Café\nFirst\nSecond"
    assert sample.sha256 == hashlib.sha256(sample.text.encode("utf-8")).hexdigest()


def test_build_corpus_deduplicates_blocks_after_case_and_whitespace_normalization() -> None:
    original = " ".join("I" for _ in range(650))
    normalized_duplicate = "\t".join("i" for _ in range(650))

    corpus = build_corpus(
        [
            _sample("first.txt", original),
            _sample("duplicate.txt", normalized_duplicate),
        ]
    )

    assert corpus.stats.duplicate_blocks == 1
    assert corpus.stats.submitted_documents == 2
    assert corpus.stats.included_documents == 1
    assert len(corpus.train) == 1
    assert not corpus.holdout
    assert corpus.all_source_text == original


def test_build_corpus_enforces_the_clean_word_minimum_at_the_boundary() -> None:
    too_short = " ".join("x" for _ in range(MIN_CLEAN_WORDS - 1))
    sufficient = " ".join("x" for _ in range(MIN_CLEAN_WORDS))

    with pytest.raises(CorpusError, match=rf"at least {MIN_CLEAN_WORDS} clean words"):
        build_corpus([_sample("short.txt", too_short)])

    corpus = build_corpus([_sample("enough.txt", sufficient)])
    assert corpus.stats.clean_words == MIN_CLEAN_WORDS


def test_document_families_are_kept_wholly_out_of_the_training_split() -> None:
    related = _numbered_prose("amber")
    related_variant = related.rsplit(" ", 1)[0] + " changed"
    samples = [
        _sample("amber-one.txt", related),
        _sample("amber-two.txt", related_variant),
        _sample("birch.txt", _numbered_prose("birch")),
        _sample("cedar.txt", _numbered_prose("cedar")),
        _sample("dogwood.txt", _numbered_prose("dogwood")),
    ]

    first = build_corpus(samples)
    second = build_corpus(samples)
    family_by_document = {document.id: document.family_id for document in first.documents}
    train_families = {segment.family_id for segment in first.train}
    holdout_families = {segment.family_id for segment in first.holdout}

    assert first.stats.document_families == 4
    assert family_by_document["D001"] == family_by_document["D002"]
    assert holdout_families
    assert train_families.isdisjoint(holdout_families)
    assert len(first.train) + len(first.holdout) == 5
    assert [(item.id, item.family_id) for item in first.train] == [
        (item.id, item.family_id) for item in second.train
    ]
    assert [(item.id, item.family_id) for item in first.holdout] == [
        (item.id, item.family_id) for item in second.holdout
    ]


def test_small_corpus_uses_a_nonempty_block_holdout() -> None:
    paragraphs = [" ".join([f"marker{index}", *("x" for _ in range(119))]) for index in range(5)]

    corpus = build_corpus([_sample("single-document.txt", "\n\n".join(paragraphs))])

    assert len(corpus.train) == 4
    assert len(corpus.holdout) == 1
    assert corpus.train[0].document_id == corpus.holdout[0].document_id
    assert corpus.metrics["measures"]["words"] == {
        "median": 480.0,
        "p25": 480.0,
        "p75": 480.0,
    }
    assert corpus.holdout_metrics["measures"]["words"] == {
        "median": 120.0,
        "p25": 120.0,
        "p75": 120.0,
    }
    assert any("held-out blocks" in warning for warning in corpus.stats.warnings)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("  Café Déjà Vu!  ", "cafe-deja-vu"),
        ("My___VOICE...Profile", "my-voice-profile"),
        ("a" * 70, "a" * 63),
    ],
)
def test_normalize_skill_name_produces_bounded_ascii_slugs(value: str, expected: str) -> None:
    assert normalize_skill_name(value) == expected


def test_normalize_skill_name_rejects_values_without_ascii_letters_or_numbers() -> None:
    with pytest.raises(CorpusError, match="at least one letter or number"):
        normalize_skill_name("🚀 🌟")
