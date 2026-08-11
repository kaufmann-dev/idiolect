from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from idiolect.codex_runner import CodexRunnerError
from idiolect.corpus import build_corpus, make_sample
from idiolect.generator import StyleGenerator, _validate_profile_text
from idiolect.models import (
    CalibrationPair,
    CandidateStyleProfile,
    CandidateTrait,
    FinalStyleProfile,
    FinalTrait,
    SampleMetadata,
    TraitScope,
)


class _CapturingRunner:
    def __init__(
        self,
        *,
        first_trait_id: str = "trait-1",
        first_evidence_source: str = "matching",
    ) -> None:
        self.payloads: list[dict[str, object]] = []
        self.train_ids: list[str] = []
        self.first_trait_id = first_trait_id
        self.first_evidence_source = first_evidence_source

    async def run_structured(
        self,
        *,
        payload: dict[str, object],
        result_model: type[Any],
        instructions_path: Path,
        status_callback: Any = None,
    ) -> Any:
        self.payloads.append(payload)
        scope = TraitScope(languages=["English"], genres=["prose"], registers=["neutral"])
        if result_model is CandidateStyleProfile:
            segments = payload["segments"]
            assert isinstance(segments, list)
            self.train_ids = [str(segment["id"]) for segment in segments]
            traits = [
                CandidateTrait(
                    id=f"trait-{index}",
                    dimension="syntax",
                    observation=f"Observed preference {index}.",
                    instruction=f"Apply preference {index}.",
                    scope=scope,
                    confidence=0.8,
                    evidence_ids=(
                        self.train_ids[:2]
                        if index == 1 and self.first_evidence_source == "two-train"
                        else [self.train_ids[index - 1]]
                    ),
                    confounds_checked=["genre"],
                )
                for index in range(1, 5)
            ]
            return CandidateStyleProfile(
                schema_version="idiolect.candidate/v1",
                summary="A controlled and direct writing style.",
                detected_languages=["English"],
                core_traits=traits,
            )

        validation_segments = payload["validation_segments"]
        assert isinstance(validation_segments, list)
        holdout_ids = [str(segment["id"]) for segment in validation_segments]
        first_evidence_ids = [self.train_ids[0]]
        if self.first_evidence_source == "other-trait":
            first_evidence_ids = [self.train_ids[1]]
        elif self.first_evidence_source == "matching-and-holdout":
            first_evidence_ids.append(holdout_ids[0])
        elif self.first_evidence_source == "two-train":
            first_evidence_ids = self.train_ids[:2]
        final_traits = [
            FinalTrait(
                id=self.first_trait_id if index == 1 else f"trait-{index}",
                dimension="syntax",
                instruction=f"Apply preference {index}.",
                scope=scope,
                confidence=0.8,
                strength="preference",
                evidence_ids=(first_evidence_ids if index == 1 else [self.train_ids[index - 1]]),
                rationale=f"The supplied evidence supports preference {index}.",
            )
            for index in range(1, 5)
        ]
        return FinalStyleProfile(
            schema_version="idiolect.style-profile/v1",
            summary="A controlled and direct writing style.",
            detected_languages=["English"],
            core_traits=final_traits,
            calibration_pairs=[
                CalibrationPair(
                    neutral="The first neutral example.",
                    in_voice="The first styled example.",
                    illustrates_trait_ids=[final_traits[0].id],
                ),
                CalibrationPair(
                    neutral="The second neutral example.",
                    in_voice="The second styled example.",
                    illustrates_trait_ids=["trait-2"],
                ),
            ],
        )


def test_profile_rejects_an_eight_word_sequence_copied_from_the_raw_corpus(
    style_profile: FinalStyleProfile,
) -> None:
    copied = "private alpha beta gamma delta epsilon zeta theta"
    source_text = f"The source contains this passage: {copied} and then continues."
    leaking_profile = style_profile.model_copy(update={"summary": copied})

    with pytest.raises(CodexRunnerError, match="copied a long sequence"):
        _validate_profile_text(leaking_profile, source_text)


async def test_discovery_and_validation_prompts_keep_fallback_blocks_disjoint() -> None:
    paragraphs = [" ".join([f"marker{index}", *("x" for _ in range(119))]) for index in range(5)]
    sample = make_sample(
        "single-document.txt",
        "\n\n".join(paragraphs),
        SampleMetadata(),
    )
    corpus = build_corpus([sample])
    runner = _CapturingRunner()

    await StyleGenerator(runner).generate(corpus, purpose="")  # type: ignore[arg-type]

    discovery, validation = runner.payloads
    discovery_segments = discovery["segments"]
    validation_segments = validation["validation_segments"]
    assert isinstance(discovery_segments, list)
    assert isinstance(validation_segments, list)
    assert [segment["text"] for segment in discovery_segments] == [
        segment.text for segment in corpus.train
    ]
    assert [segment["text"] for segment in validation_segments] == [
        segment.text for segment in corpus.holdout
    ]
    assert {segment["id"] for segment in discovery_segments}.isdisjoint(
        segment["id"] for segment in validation_segments
    )
    assert discovery["equal_document_weighted_metrics"] == corpus.metrics
    assert validation["validation_metrics"] == corpus.holdout_metrics
    assert paragraphs[-1] not in {segment["text"] for segment in discovery_segments}
    assert not set(paragraphs[:-1]).intersection(segment["text"] for segment in validation_segments)


def _fallback_holdout_corpus():
    paragraphs = [" ".join([f"marker{index}", *("x" for _ in range(119))]) for index in range(5)]
    sample = make_sample(
        "single-document.txt",
        "\n\n".join(paragraphs),
        SampleMetadata(),
    )
    return build_corpus([sample])


def _usable_family_holdout_corpus():
    samples = [
        make_sample(
            f"document-{document}.txt",
            " ".join(f"document{document}word{word}" for word in range(750)),
            SampleMetadata(),
        )
        for document in range(4)
    ]
    return build_corpus(samples)


async def test_validation_rejects_trait_id_not_proposed_by_discovery() -> None:
    runner = _CapturingRunner(first_trait_id="invented-trait")

    with pytest.raises(CodexRunnerError, match="candidate trait"):
        await StyleGenerator(runner).generate(  # type: ignore[arg-type]
            _fallback_holdout_corpus(), purpose=""
        )


async def test_validation_rejects_discovery_evidence_borrowed_from_another_trait() -> None:
    runner = _CapturingRunner(first_evidence_source="other-trait")

    with pytest.raises(CodexRunnerError, match="candidate trait"):
        await StyleGenerator(runner).generate(  # type: ignore[arg-type]
            _fallback_holdout_corpus(), purpose=""
        )


async def test_validation_accepts_same_trait_discovery_and_supplied_holdout_evidence() -> None:
    runner = _CapturingRunner(first_evidence_source="matching-and-holdout")

    profile = await StyleGenerator(runner).generate(  # type: ignore[arg-type]
        _fallback_holdout_corpus(), purpose=""
    )

    assert profile.core_traits[0].evidence_ids == ["D001:B001", "D001:B005"]


async def test_validation_without_holdout_support_stays_provisional() -> None:
    corpus = _usable_family_holdout_corpus()
    runner = _CapturingRunner(first_evidence_source="two-train")

    profile = await StyleGenerator(runner).generate(corpus, purpose="")  # type: ignore[arg-type]

    assert corpus.stats.quality == "usable"
    assert profile.core_traits[0].strength == "provisional"
    assert profile.core_traits[0].confidence == 0.64
