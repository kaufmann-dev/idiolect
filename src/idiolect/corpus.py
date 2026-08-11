from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from typing import Literal

from idiolect.models import (
    Authorship,
    CorpusBundle,
    CorpusSegment,
    CorpusStats,
    DocumentSummary,
    SampleMetadata,
    WritingSample,
)
from idiolect.stylometry import analyze_documents, word_count

ALLOWED_EXTENSIONS = {".txt", ".md", ".markdown"}
MAX_FILE_BYTES = 1_000_000
MAX_TOTAL_BYTES = 3_000_000
MAX_TRAIN_CHARACTERS = 90_000
MAX_HOLDOUT_CHARACTERS = 30_000
MIN_CLEAN_WORDS = 600
MAX_BLOCK_CHARACTERS = 1_800


class CorpusError(ValueError):
    pass


def normalize_skill_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    if not slug:
        raise CorpusError("Enter a skill name that contains at least one letter or number.")
    if len(slug) > 63:
        slug = slug[:63].rstrip("-")
    return slug


def decode_text_file(data: bytes) -> str:
    if len(data) > MAX_FILE_BYTES:
        raise CorpusError("Each file must be smaller than 1 MB.")
    if not data:
        raise CorpusError("Empty files do not contain writing evidence.")
    if b"\x00" in data and not (data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff")):
        raise CorpusError("Binary files are not supported. Upload UTF-8 text or Markdown.")

    encodings = ("utf-8-sig", "utf-16") if b"\x00" in data[:200] else ("utf-8-sig",)
    for encoding in encodings:
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise CorpusError("The file is not valid UTF-8 text.")

    disallowed = sum(
        1 for char in text if unicodedata.category(char) == "Cc" and char not in {"\n", "\r", "\t"}
    )
    if disallowed > max(3, len(text) // 500):
        raise CorpusError("The file contains too many binary control characters.")
    return text


def make_sample(name: str, text: str, metadata: SampleMetadata) -> WritingSample:
    normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return WritingSample(name=name, text=normalized, metadata=metadata, sha256=digest)


def _normalized_fingerprint(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _chunk_long_block(block: str) -> list[str]:
    if len(block) <= MAX_BLOCK_CHARACTERS:
        return [block]
    sentences = re.split(r"(?<=[.!?])\s+", block)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > MAX_BLOCK_CHARACTERS:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                sentence[index : index + MAX_BLOCK_CHARACTERS]
                for index in range(0, len(sentence), MAX_BLOCK_CHARACTERS)
            )
            continue
        candidate = f"{current} {sentence}".strip()
        if len(candidate) > MAX_BLOCK_CHARACTERS and current:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _clean_blocks(text: str) -> tuple[list[str], int]:
    blocks: list[str] = []
    current: list[str] = []
    excluded = 0
    in_code_fence = False
    signature_reached = False

    def flush() -> None:
        nonlocal current
        value = "\n".join(current).strip()
        current = []
        if word_count(value) >= 4:
            blocks.extend(_chunk_long_block(value))

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            flush()
            in_code_fence = not in_code_fence
            excluded += 1
            continue
        if in_code_fence or signature_reached:
            excluded += 1
            continue
        if stripped in {"--", "-- "}:
            flush()
            signature_reached = True
            excluded += 1
            continue
        if not stripped:
            flush()
            continue
        if stripped.startswith(">"):
            flush()
            excluded += 1
            continue
        if stripped.count("|") >= 2 and re.search(r"\S\s*\|\s*\S", stripped):
            flush()
            excluded += 1
            continue
        if re.match(r"^(?:from|sent|to|subject):\s", stripped, re.IGNORECASE):
            flush()
            excluded += 1
            continue
        current.append(line)
    flush()
    return blocks, excluded


def _shingles(text: str, size: int = 5) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[^\W_]+", text.casefold())
    if len(tokens) < size:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def _jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _assign_families(clean_documents: Sequence[str]) -> list[str]:
    representatives: list[set[tuple[str, ...]]] = []
    family_ids: list[str] = []
    for text in clean_documents:
        signature = _shingles(text)
        match: int | None = None
        for index, representative in enumerate(representatives):
            if _jaccard(signature, representative) >= 0.78:
                match = index
                break
        if match is None:
            representatives.append(signature)
            match = len(representatives) - 1
        family_ids.append(f"F{match + 1:03d}")
    return family_ids


def _balanced_limit(
    segments: Iterable[CorpusSegment], character_budget: int
) -> tuple[CorpusSegment, ...]:
    queues: dict[str, deque[CorpusSegment]] = defaultdict(deque)
    for segment in segments:
        queues[segment.document_id].append(segment)
    selected: list[CorpusSegment] = []
    used = 0
    document_ids = deque(sorted(queues))
    while document_ids:
        document_id = document_ids.popleft()
        queue = queues[document_id]
        segment = queue.popleft()
        cost = len(segment.text)
        if used + cost <= character_budget or not selected:
            selected.append(segment)
            used += cost
        if queue and used < character_budget:
            document_ids.append(document_id)
    return tuple(selected)


def _analyze_split(segments: Sequence[CorpusSegment]) -> dict[str, object]:
    document_blocks: dict[str, list[str]] = defaultdict(list)
    for segment in segments:
        document_blocks[segment.document_id].append(segment.text)
    return analyze_documents(
        "\n\n".join(document_blocks[document_id]) for document_id in sorted(document_blocks)
    )


def _quality(clean_words: int, families: int) -> Literal["provisional", "usable", "strong"]:
    if clean_words >= 8_000 and families >= 8:
        return "strong"
    if clean_words >= 3_000 and families >= 3:
        return "usable"
    return "provisional"


def build_corpus(samples: Sequence[WritingSample]) -> CorpusBundle:
    eligible = [
        sample
        for sample in samples
        if sample.metadata.authorship is Authorship.SELF and sample.metadata.current_voice
    ]
    excluded_documents = len(samples) - len(eligible)
    if not eligible:
        raise CorpusError("Mark at least one current, fully self-authored sample for analysis.")

    cleaned: list[tuple[WritingSample, list[str], int]] = []
    for sample in eligible:
        blocks, excluded = _clean_blocks(sample.text)
        if blocks:
            cleaned.append((sample, blocks, excluded))
        else:
            excluded_documents += 1
    if not cleaned:
        raise CorpusError(
            "No usable prose remained after quoted, tabular, and code blocks were removed."
        )

    clean_documents = ["\n\n".join(blocks) for _, blocks, _ in cleaned]
    family_ids = _assign_families(clean_documents)
    seen_blocks: set[str] = set()
    segments: list[CorpusSegment] = []
    summaries: list[DocumentSummary] = []
    duplicate_blocks = 0
    excluded_blocks = sum(item[2] for item in cleaned)

    for document_index, ((sample, blocks, _), family_id) in enumerate(
        zip(cleaned, family_ids, strict=True), start=1
    ):
        document_id = f"D{document_index:03d}"
        kept_for_document: list[str] = []
        block_number = 0
        for block in blocks:
            fingerprint = hashlib.sha256(_normalized_fingerprint(block).encode()).hexdigest()
            if fingerprint in seen_blocks:
                duplicate_blocks += 1
                continue
            seen_blocks.add(fingerprint)
            block_number += 1
            kept_for_document.append(block)
            segments.append(
                CorpusSegment(
                    id=f"{document_id}:B{block_number:03d}",
                    document_id=document_id,
                    family_id=family_id,
                    genre=sample.metadata.genre.value,
                    language=sample.metadata.language,
                    text=block,
                    word_count=word_count(block),
                )
            )
        if kept_for_document:
            summaries.append(
                DocumentSummary(
                    id=document_id,
                    family_id=family_id,
                    genre=sample.metadata.genre.value,
                    language=sample.metadata.language,
                    word_count=sum(word_count(block) for block in kept_for_document),
                    block_count=len(kept_for_document),
                )
            )

    total_words = sum(segment.word_count for segment in segments)
    if total_words < MIN_CLEAN_WORDS:
        raise CorpusError(
            f"Add more self-authored prose. Idiolect needs at least {MIN_CLEAN_WORDS} clean words."
        )

    unique_families = sorted({segment.family_id for segment in segments})
    if len(unique_families) >= 4:
        ranked_families = sorted(
            unique_families,
            key=lambda family: hashlib.sha256(family.encode()).hexdigest(),
        )
        holdout_count = min(max(1, round(len(ranked_families) * 0.2)), len(ranked_families) - 2)
        holdout_families = set(ranked_families[:holdout_count])
        train_source = [
            segment for segment in segments if segment.family_id not in holdout_families
        ]
        holdout_source = [segment for segment in segments if segment.family_id in holdout_families]
    else:
        train_source = [segment for index, segment in enumerate(segments) if index % 5 != 4]
        holdout_source = [segment for index, segment in enumerate(segments) if index % 5 == 4]
        if not holdout_source and len(segments) > 1:
            holdout_source = [train_source.pop()]

    train = _balanced_limit(train_source, MAX_TRAIN_CHARACTERS)
    holdout = _balanced_limit(holdout_source, MAX_HOLDOUT_CHARACTERS)
    warnings: list[str] = []
    corpus_quality = _quality(total_words, len(unique_families))
    if corpus_quality == "provisional":
        warnings.append(
            "The profile is provisional. Three independent documents and 3,000 clean words make recurring traits easier to distinguish."
        )
    if len(unique_families) < 4:
        warnings.append(
            "There were too few independent document families for a document-level holdout, so validation uses held-out blocks."
        )
    languages = {summary.language for summary in summaries if summary.language != "auto"}
    if len(languages) > 1:
        warnings.append(
            "The corpus contains multiple declared languages; the profile will keep separate language scopes."
        )
    if excluded_documents:
        warnings.append(
            f"{excluded_documents} sample(s) were excluded by authorship or current-voice settings."
        )
    sampled_words = sum(segment.word_count for segment in (*train, *holdout))
    if sampled_words < total_words:
        warnings.append(
            f"A balanced set of {sampled_words:,} words was selected from the larger corpus for Codex analysis."
        )

    stats = CorpusStats(
        submitted_documents=len(samples),
        included_documents=len(summaries),
        document_families=len(unique_families),
        clean_words=total_words,
        clean_characters=sum(len(segment.text) for segment in segments),
        excluded_documents=excluded_documents,
        excluded_blocks=excluded_blocks,
        duplicate_blocks=duplicate_blocks,
        train_words=sum(segment.word_count for segment in train),
        holdout_words=sum(segment.word_count for segment in holdout),
        quality=corpus_quality,
        warnings=tuple(warnings),
    )
    return CorpusBundle(
        documents=tuple(summaries),
        train=train,
        holdout=holdout,
        metrics=_analyze_split(train),
        holdout_metrics=_analyze_split(holdout),
        stats=stats,
        all_source_text="\n".join(segment.text for segment in segments),
    )
