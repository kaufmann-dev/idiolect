from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=240)]
SentenceText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)
]
Slug = Annotated[
    str, StringConstraints(strip_whitespace=True, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
]
EvidenceId = Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^D\d{3}:B\d{3,5}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Authorship(StrEnum):
    SELF = "self"
    COAUTHORED = "coauthored"
    EDITED = "edited"
    UNKNOWN = "unknown"


class Genre(StrEnum):
    AUTO = "auto"
    EMAIL = "email"
    ARTICLE = "article"
    SOCIAL = "social-post"
    TECHNICAL = "technical-note"
    ACADEMIC = "academic"
    OTHER = "other"


class SampleMetadata(StrictModel):
    authorship: Authorship = Authorship.SELF
    genre: Genre = Genre.AUTO
    language: Annotated[str, StringConstraints(strip_whitespace=True, max_length=24)] = "auto"
    current_voice: bool = True


@dataclass(frozen=True, slots=True)
class WritingSample:
    name: str
    text: str
    metadata: SampleMetadata
    sha256: str


@dataclass(frozen=True, slots=True)
class CorpusSegment:
    id: str
    document_id: str
    family_id: str
    genre: str
    language: str
    text: str
    word_count: int


@dataclass(frozen=True, slots=True)
class DocumentSummary:
    id: str
    family_id: str
    genre: str
    language: str
    word_count: int
    block_count: int


@dataclass(frozen=True, slots=True)
class CorpusStats:
    submitted_documents: int
    included_documents: int
    document_families: int
    clean_words: int
    clean_characters: int
    excluded_documents: int
    excluded_blocks: int
    duplicate_blocks: int
    train_words: int
    holdout_words: int
    quality: Literal["provisional", "usable", "strong"]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CorpusBundle:
    documents: tuple[DocumentSummary, ...]
    train: tuple[CorpusSegment, ...]
    holdout: tuple[CorpusSegment, ...]
    metrics: dict[str, object]
    holdout_metrics: dict[str, object]
    stats: CorpusStats
    all_source_text: str = field(repr=False)


class TraitScope(StrictModel):
    languages: Annotated[list[ShortText], Field(min_length=1, max_length=4)]
    genres: Annotated[list[ShortText], Field(min_length=1, max_length=6)]
    registers: Annotated[list[ShortText], Field(min_length=1, max_length=5)]


class CandidateTrait(StrictModel):
    id: Annotated[Slug, StringConstraints(max_length=48)]
    dimension: Literal[
        "voice",
        "rhythm",
        "syntax",
        "diction",
        "punctuation",
        "structure",
        "stance",
        "address",
        "formatting",
    ]
    observation: SentenceText
    instruction: SentenceText
    scope: TraitScope
    confidence: Annotated[float, Field(ge=0, le=1)]
    evidence_ids: Annotated[list[EvidenceId], Field(min_length=1, max_length=8)]
    counterevidence_ids: Annotated[list[EvidenceId], Field(max_length=6)] = []
    confounds_checked: Annotated[list[ShortText], Field(min_length=1, max_length=6)]


class CandidateMode(StrictModel):
    id: Annotated[Slug, StringConstraints(max_length=48)]
    label: ShortText
    when_to_use: SentenceText
    traits: Annotated[list[CandidateTrait], Field(min_length=2, max_length=8)]


class CandidateStyleProfile(StrictModel):
    schema_version: Literal["idiolect.candidate/v1"]
    summary: SentenceText
    detected_languages: Annotated[list[ShortText], Field(min_length=1, max_length=5)]
    core_traits: Annotated[list[CandidateTrait], Field(min_length=4, max_length=14)]
    modes: Annotated[list[CandidateMode], Field(max_length=4)] = []
    uncertainties: Annotated[list[SentenceText], Field(max_length=8)] = []


class CalibrationPair(StrictModel):
    neutral: SentenceText
    in_voice: SentenceText
    illustrates_trait_ids: Annotated[list[Slug], Field(min_length=1, max_length=4)]


class FinalTrait(StrictModel):
    id: Annotated[Slug, StringConstraints(max_length=48)]
    dimension: Literal[
        "voice",
        "rhythm",
        "syntax",
        "diction",
        "punctuation",
        "structure",
        "stance",
        "address",
        "formatting",
    ]
    instruction: SentenceText
    scope: TraitScope
    confidence: Annotated[float, Field(ge=0, le=1)]
    strength: Literal["preference", "provisional"]
    evidence_ids: Annotated[list[EvidenceId], Field(min_length=1, max_length=10)]
    counterevidence_ids: Annotated[list[EvidenceId], Field(max_length=8)] = []
    rationale: SentenceText


class FinalMode(StrictModel):
    id: Annotated[Slug, StringConstraints(max_length=48)]
    label: ShortText
    when_to_use: SentenceText
    traits: Annotated[list[FinalTrait], Field(min_length=2, max_length=8)]


class FinalStyleProfile(StrictModel):
    schema_version: Literal["idiolect.style-profile/v1"]
    summary: SentenceText
    detected_languages: Annotated[list[ShortText], Field(min_length=1, max_length=5)]
    regional_notes: Annotated[list[SentenceText], Field(max_length=5)] = []
    core_traits: Annotated[list[FinalTrait], Field(min_length=4, max_length=14)]
    modes: Annotated[list[FinalMode], Field(max_length=4)] = []
    calibration_pairs: Annotated[list[CalibrationPair], Field(min_length=2, max_length=6)]
    uncertainties: Annotated[list[SentenceText], Field(max_length=8)] = []


class CompileRequest(StrictModel):
    accepted_trait_ids: Annotated[list[Slug], Field(min_length=3, max_length=40)]


class JobStage(StrEnum):
    QUEUED = "queued"
    ANALYZING = "analyzing"
    VALIDATING = "validating"
    REVIEW = "review"
    PACKAGING = "packaging"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class JobRecord:
    id: str
    skill_name: str
    voice_label: str
    stage: JobStage = JobStage.QUEUED
    status_message: str = "Waiting for the local Codex worker."
    stats: CorpusStats | None = None
    profile: FinalStyleProfile | None = None
    selected_trait_ids: tuple[str, ...] = ()
    package: SkillPackage | None = None
    error: str | None = None
    task: asyncio.Task[None] | None = field(default=None, repr=False)
    created_at: float = 0
    updated_at: float = 0


@dataclass(frozen=True, slots=True)
class SkillPackage:
    skill_name: str
    files: dict[str, bytes]
    archive: bytes
    preview: str
