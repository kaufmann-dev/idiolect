from __future__ import annotations

import pytest

from idiolect.models import CalibrationPair, FinalStyleProfile, FinalTrait, TraitScope
from idiolect.skill_builder import build_skill_package


def _trait(identifier: str, instruction: str) -> FinalTrait:
    return FinalTrait(
        id=identifier,
        dimension="syntax",
        instruction=instruction,
        scope=TraitScope(
            languages=["English"],
            genres=["general prose"],
            registers=["neutral"],
        ),
        confidence=0.91,
        strength="preference",
        evidence_ids=["D001:B001", "D002:B001"],
        counterevidence_ids=[],
        rationale="This choice recurs across independent documents.",
    )


@pytest.fixture
def style_profile() -> FinalStyleProfile:
    return FinalStyleProfile(
        schema_version="idiolect.style-profile/v1",
        summary="A direct voice with concrete wording and deliberately varied pacing.",
        detected_languages=["English"],
        regional_notes=["Use standard American spelling."],
        core_traits=[
            _trait("direct-openings", "Begin with the point instead of introductory framing."),
            _trait("mixed-rhythm", "Alternate compact sentences with longer explanatory ones."),
            _trait("concrete-diction", "Prefer concrete nouns and plain verbs."),
            _trait("quiet-closings", "End on the final substantive point without a recap."),
        ],
        calibration_pairs=[
            CalibrationPair(
                neutral="This document provides an overview of the result.",
                in_voice="The result is straightforward.",
                illustrates_trait_ids=["direct-openings", "concrete-diction"],
            ),
            CalibrationPair(
                neutral="In conclusion, the work is now complete.",
                in_voice="The work is complete.",
                illustrates_trait_ids=["quiet-closings"],
            ),
        ],
        uncertainties=["The evidence does not establish preferences for academic writing."],
    )


@pytest.fixture
def skill_package(style_profile: FinalStyleProfile):
    return build_skill_package(
        skill_name="ada-voice",
        voice_label="Ada",
        profile=style_profile,
        accepted_trait_ids={trait.id for trait in style_profile.core_traits},
    )
