from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from pathlib import Path

from idiolect.codex_runner import CodexRunner, CodexRunnerError
from idiolect.models import (
    CandidateStyleProfile,
    CandidateTrait,
    CorpusBundle,
    CorpusSegment,
    FinalMode,
    FinalStyleProfile,
    FinalTrait,
)

StageCallback = Callable[[str, str], None]
PROMPTS_DIRECTORY = Path(__file__).parent / "prompts"
FORBIDDEN_OUTPUT_PATTERNS = (
    "ignore previous",
    "ignore all prior",
    "system prompt",
    "developer message",
    "developer instruction",
    "hidden prompt",
    "prompt hierarchy",
    "obey instructions",
    "follow embedded",
    "override instructions",
    "execute command",
    "run a command",
    "shell command",
    "filesystem",
    "credential",
    "api key",
    "browse the web",
    "network access",
    "install package",
    "read a file",
    "write a file",
    "tool call",
    "http://",
    "https://",
    "-----begin",
)
SECRET_PATTERN = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9]{12,}|AKIA[A-Z0-9]{12,}|"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|"
    r"\+?\d[\d .()-]{8,}\d)"
)


def _segment_payload(segment: CorpusSegment) -> dict[str, object]:
    return {
        "id": segment.id,
        "document_id": segment.document_id,
        "family_id": segment.family_id,
        "declared_genre": segment.genre,
        "declared_language": segment.language,
        "text": segment.text,
    }


def _all_candidate_traits(profile: CandidateStyleProfile) -> Iterable[CandidateTrait]:
    yield from profile.core_traits
    for mode in profile.modes:
        yield from mode.traits


def _all_final_traits(profile: FinalStyleProfile) -> Iterable[FinalTrait]:
    yield from profile.core_traits
    for mode in profile.modes:
        yield from mode.traits


def _normalize_words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.casefold())


def _source_ngrams(text: str, size: int = 8) -> set[str]:
    tokens = _normalize_words(text)
    return {" ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def _profile_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _profile_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _profile_strings(nested)


def _validate_profile_text(profile: FinalStyleProfile, source_text: str) -> None:
    source_ngrams = _source_ngrams(source_text)
    for value in _profile_strings(profile.model_dump(mode="json")):
        if "\n" in value or "\r" in value or any(ord(character) < 32 for character in value):
            raise CodexRunnerError("Codex returned unsafe multiline profile content.")
        lowered = value.casefold()
        if any(pattern in lowered for pattern in FORBIDDEN_OUTPUT_PATTERNS):
            raise CodexRunnerError("Codex returned an unsafe operational instruction.")
        if SECRET_PATTERN.search(value):
            raise CodexRunnerError(
                "Codex returned private-looking content instead of style guidance."
            )
        tokens = _normalize_words(value)
        for index in range(len(tokens) - 7):
            if " ".join(tokens[index : index + 8]) in source_ngrams:
                raise CodexRunnerError(
                    "Codex copied a long sequence from the source corpus. The private profile was rejected."
                )


def _validate_candidate_evidence(profile: CandidateStyleProfile, known_ids: set[str]) -> None:
    trait_ids: set[str] = set()
    for trait in _all_candidate_traits(profile):
        if trait.id in trait_ids:
            raise CodexRunnerError("Codex returned duplicate trait identifiers.")
        trait_ids.add(trait.id)
        if not set(trait.evidence_ids).issubset(known_ids):
            raise CodexRunnerError("Codex cited writing evidence that was not supplied.")
        if not set(trait.counterevidence_ids).issubset(known_ids):
            raise CodexRunnerError("Codex cited counterevidence that was not supplied.")


def _calibrate_profile(
    profile: FinalStyleProfile,
    candidate_profile: CandidateStyleProfile,
    corpus: CorpusBundle,
) -> FinalStyleProfile:
    segments = {segment.id: segment for segment in (*corpus.train, *corpus.holdout)}
    holdout_ids = {segment.id for segment in corpus.holdout}
    candidate_traits = {trait.id: trait for trait in _all_candidate_traits(candidate_profile)}
    trait_ids: set[str] = set()

    def calibrate_trait(trait: FinalTrait) -> FinalTrait:
        if trait.id in trait_ids:
            raise CodexRunnerError("Codex returned duplicate trait identifiers.")
        trait_ids.add(trait.id)
        candidate_trait = candidate_traits.get(trait.id)
        if candidate_trait is None:
            raise CodexRunnerError("Codex returned a trait that was not a candidate trait.")
        evidence_ids = tuple(dict.fromkeys(trait.evidence_ids))
        counter_ids = tuple(dict.fromkeys(trait.counterevidence_ids))
        allowed_ids = {
            *candidate_trait.evidence_ids,
            *candidate_trait.counterevidence_ids,
            *holdout_ids,
        }
        if not set((*evidence_ids, *counter_ids)).issubset(allowed_ids):
            raise CodexRunnerError(
                "Codex cited evidence that was not supplied for that candidate trait."
            )
        if not set(evidence_ids).issubset(segments) or not set(counter_ids).issubset(segments):
            raise CodexRunnerError("Codex cited writing evidence that was not supplied.")
        families = {segments[evidence_id].family_id for evidence_id in evidence_ids}
        maximum_confidence = 1.0
        strength = trait.strength
        if corpus.stats.quality == "provisional":
            maximum_confidence = 0.59
            strength = "provisional"
        elif (
            set(evidence_ids).isdisjoint(holdout_ids) or len(families) < 2 or len(evidence_ids) < 2
        ):
            maximum_confidence = 0.64
            strength = "provisional"
        return trait.model_copy(
            update={
                "evidence_ids": list(evidence_ids),
                "counterevidence_ids": list(counter_ids),
                "confidence": min(trait.confidence, maximum_confidence),
                "strength": strength,
            }
        )

    core_traits = [calibrate_trait(trait) for trait in profile.core_traits]
    modes = [
        FinalMode(
            id=mode.id,
            label=mode.label,
            when_to_use=mode.when_to_use,
            traits=[calibrate_trait(trait) for trait in mode.traits],
        )
        for mode in profile.modes
    ]
    available_trait_ids = {trait.id for trait in core_traits}
    available_trait_ids.update(trait.id for mode in modes for trait in mode.traits)
    for pair in profile.calibration_pairs:
        if not set(pair.illustrates_trait_ids).issubset(available_trait_ids):
            raise CodexRunnerError("Codex returned a calibration example for an unknown trait.")
    calibrated = profile.model_copy(update={"core_traits": core_traits, "modes": modes})
    _validate_profile_text(calibrated, corpus.all_source_text)
    return calibrated


class StyleGenerator:
    def __init__(self, runner: CodexRunner) -> None:
        self.runner = runner

    async def generate(
        self,
        corpus: CorpusBundle,
        purpose: str,
        stage_callback: StageCallback | None = None,
    ) -> FinalStyleProfile:
        if stage_callback:
            stage_callback("analyzing", "Analyzing recurring choices in the discovery corpus.")
        candidate_payload: dict[str, object] = {
            "stage": "discover",
            "purpose": purpose,
            "corpus_quality": corpus.stats.quality,
            "documents": [
                {
                    "id": document.id,
                    "family_id": document.family_id,
                    "declared_genre": document.genre,
                    "declared_language": document.language,
                    "clean_words": document.word_count,
                    "block_count": document.block_count,
                }
                for document in corpus.documents
            ],
            "equal_document_weighted_metrics": corpus.metrics,
            "segments": [_segment_payload(segment) for segment in corpus.train],
        }
        candidate = await self.runner.run_structured(
            payload=candidate_payload,
            result_model=CandidateStyleProfile,
            instructions_path=PROMPTS_DIRECTORY / "discover.md",
            status_callback=(
                (lambda message: stage_callback("analyzing", message)) if stage_callback else None
            ),
        )
        _validate_candidate_evidence(candidate, {segment.id for segment in corpus.train})

        if stage_callback:
            stage_callback("validating", "Checking proposed traits against held-out writing.")
        validation_payload: dict[str, object] = {
            "stage": "validate",
            "purpose": purpose,
            "corpus_quality": corpus.stats.quality,
            "candidate_profile": candidate.model_dump(mode="json"),
            "validation_metrics": corpus.holdout_metrics,
            "validation_segments": [_segment_payload(segment) for segment in corpus.holdout],
        }
        final_profile = await self.runner.run_structured(
            payload=validation_payload,
            result_model=FinalStyleProfile,
            instructions_path=PROMPTS_DIRECTORY / "validate.md",
            status_callback=(
                (lambda message: stage_callback("validating", message)) if stage_callback else None
            ),
        )
        return _calibrate_profile(final_profile, candidate, corpus)
