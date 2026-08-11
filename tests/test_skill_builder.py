from __future__ import annotations

import io
import stat
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

import idiolect.skill_builder as skill_builder
from idiolect.models import FinalStyleProfile, SkillPackage
from idiolect.skill_builder import SkillBuildError, build_skill_package, install_skill


def test_skill_rendering_and_zip_archive_are_deterministic(
    style_profile: FinalStyleProfile,
) -> None:
    accepted = {trait.id for trait in style_profile.core_traits}

    first = build_skill_package("ada-voice", "Ada", style_profile, accepted)
    second = build_skill_package("ada-voice", "Ada", style_profile, accepted)

    assert first.files == second.files
    assert first.archive == second.archive
    assert "===== SKILL.md =====" in first.preview
    assert "===== agents/openai.yaml =====" in first.preview
    assert "===== references/voice-profile.md =====" in first.preview

    with zipfile.ZipFile(io.BytesIO(first.archive)) as archive:
        entries = archive.infolist()
        assert [entry.filename for entry in entries] == [
            "ada-voice/SKILL.md",
            "ada-voice/agents/openai.yaml",
            "ada-voice/references/voice-profile.md",
        ]
        for entry in entries:
            relative_path = entry.filename.removeprefix("ada-voice/")
            assert entry.date_time == (1980, 1, 1, 0, 0, 0)
            assert stat.S_IMODE(entry.external_attr >> 16) == 0o600
            assert archive.read(entry) == first.files[relative_path]


def test_skill_package_contains_only_derived_files_and_no_evidence_text(
    skill_package: SkillPackage,
) -> None:
    rendered = b"\n".join(skill_package.files.values())

    assert set(skill_package.files) == {
        "SKILL.md",
        "agents/openai.yaml",
        "references/voice-profile.md",
    }
    assert b"D001:B001" not in rendered
    assert b"D002:B001" not in rendered
    assert not any(path.endswith((".txt", ".md.bak")) for path in skill_package.files)


def test_skill_package_contains_only_model_text_the_user_accepted(
    style_profile: FinalStyleProfile,
) -> None:
    accepted = {trait.id for trait in style_profile.core_traits[:3]}

    package = build_skill_package("ada-voice", "Ada", style_profile, accepted)
    rendered = package.files["references/voice-profile.md"].decode("utf-8")

    for trait in style_profile.core_traits[:3]:
        assert trait.instruction in rendered
        assert trait.scope.languages[0] in rendered
    assert style_profile.core_traits[3].instruction not in rendered
    assert style_profile.summary not in rendered
    assert style_profile.regional_notes[0] not in rendered
    assert style_profile.uncertainties[0] not in rendered
    assert style_profile.calibration_pairs[0].in_voice not in rendered
    assert style_profile.core_traits[0].rationale not in rendered


def test_skill_builder_requires_three_accepted_evidenced_traits(
    style_profile: FinalStyleProfile,
) -> None:
    accepted = {trait.id for trait in style_profile.core_traits[:2]}

    with pytest.raises(SkillBuildError, match="at least three evidenced traits"):
        build_skill_package("ada-voice", "Ada", style_profile, accepted)


def test_install_writes_the_complete_package_and_refuses_to_overwrite(
    tmp_path: Path,
    skill_package: SkillPackage,
) -> None:
    root = tmp_path / "skills"

    target = install_skill(skill_package, root)

    assert target == root / skill_package.skill_name
    for relative_path, content in skill_package.files.items():
        installed = target / relative_path
        assert installed.read_bytes() == content
        assert stat.S_IMODE(installed.stat().st_mode) == 0o600
    assert not list(root.glob(".idiolect-*.tmp"))

    marker = b"user-owned replacement"
    (target / "SKILL.md").write_bytes(marker)
    with pytest.raises(SkillBuildError, match="will not overwrite"):
        install_skill(skill_package, root)
    assert (target / "SKILL.md").read_bytes() == marker


def test_install_refuses_a_symlinked_skill_root(
    tmp_path: Path,
    skill_package: SkillPackage,
) -> None:
    real_root = tmp_path / "real-skills"
    real_root.mkdir()
    linked_root = tmp_path / "linked-skills"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(SkillBuildError, match="symlinked skill directory"):
        install_skill(skill_package, linked_root)

    assert not any(real_root.iterdir())


def test_install_refuses_a_skill_name_that_escapes_the_root(
    tmp_path: Path,
    skill_package: SkillPackage,
) -> None:
    unsafe_package = replace(skill_package, skill_name="../escaped")

    with pytest.raises(SkillBuildError):
        install_skill(unsafe_package, tmp_path / "skills")

    assert not (tmp_path / "escaped").exists()


def test_failed_publish_leaves_neither_a_partial_target_nor_staging_directory(
    tmp_path: Path,
    skill_package: SkillPackage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "skills"

    def fail_publish(_: Path, __: Path) -> None:
        raise OSError("simulated atomic publish failure")

    monkeypatch.setattr(skill_builder.os, "replace", fail_publish)

    with pytest.raises(OSError, match="simulated atomic publish failure"):
        install_skill(skill_package, root)

    assert not (root / skill_package.skill_name).exists()
    assert not list(root.glob(".idiolect-*.tmp"))
