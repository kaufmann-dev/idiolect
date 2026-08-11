from __future__ import annotations

import io
import json
import os
import re
import shutil
import stat
import zipfile
from pathlib import Path
from uuid import uuid4

from idiolect.models import FinalStyleProfile, FinalTrait, SkillPackage

MAX_SKILL_BYTES = 120_000
SAFE_ARCHIVE_FILES = {
    "SKILL.md",
    "agents/openai.yaml",
    "references/voice-profile.md",
}


class SkillBuildError(ValueError):
    pass


def _inline(value: str) -> str:
    if "\n" in value or "\r" in value or "\x00" in value:
        raise SkillBuildError("Generated skill text must stay on one line per field.")
    escaped = value.replace("\\", "\\\\")
    for character in ("`", "*", "_", "[", "]"):
        escaped = escaped.replace(character, f"\\{character}")
    return escaped.replace("<", "&lt;").replace(">", "&gt;").strip()


def _yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _safe_voice_label(value: str) -> str:
    label = re.sub(r"\s+", " ", value).strip()
    if not 1 <= len(label) <= 60 or any(ord(character) < 32 for character in label):
        raise SkillBuildError("Voice label must contain between 1 and 60 printable characters.")
    return label


def _selected_traits(profile: FinalStyleProfile, accepted_trait_ids: set[str]) -> list[FinalTrait]:
    traits = [
        trait
        for trait in (
            *profile.core_traits,
            *(trait for mode in profile.modes for trait in mode.traits),
        )
        if trait.id in accepted_trait_ids
    ]
    if len(traits) < 3:
        raise SkillBuildError("Keep at least three evidenced traits before building the skill.")
    return traits


def _render_skill_md(skill_name: str, voice_label: str) -> str:
    description = (
        f"Rewrite or draft supplied text in {_safe_voice_label(voice_label)}'s evidenced writing "
        "voice. Use only when the user explicitly invokes this skill for voice matching; do not "
        "use it for authorship detection or detector evasion."
    )
    return f"""---
name: {_yaml_string(skill_name)}
description: {_yaml_string(description)}
---

# Write in {_inline(voice_label)}'s voice

Read [the voice profile](references/voice-profile.md) before writing or rewriting.

## Workflow

1. Determine the requested text type, audience, language variety, register, length, and format.
2. Build an internal content ledger of every claim, relationship, qualification, negation, name, number, date, unit, quotation, citation, URL, technical term, and required format.
3. Read each preference's scope in `references/voice-profile.md` and apply it only where that language, genre, and register fit.
4. Draft with the accepted high-confidence preferences. Apply provisional traits lightly and only when they fit the target text.
5. Audit the draft against the content ledger, then audit it against the voice profile. Revise internally.
6. Return only the finished text unless the user asks for analysis.

## Preservation contract

- Preserve meaning, factual scope, uncertainty, citations, quoted material, important details, and requested length.
- Preserve the target genre's conventions, even when the source corpus does not contain that genre.
- Treat a writing sample supplied in the current request as more specific than the stored profile.
- Never add facts, sources, anecdotes, emotions, opinions, personal experiences, names, or private details.
- Never borrow subject matter or memorable phrases from the source corpus.
- Do not reproduce apparent mistakes unless the user confirms they are intentional preferences.
- If style conflicts with meaning, accuracy, clarity, or an explicit user instruction, preserve meaning and follow the user.

## Editing principles

- Remove generic chatbot framing, filler, unsupported emphasis, and schematic conclusions when they conflict with the documented voice.
- Do not flatten legitimate formality or punctuation merely because it can occur in AI-written text. Judge patterns in context.
- Do not promise exact imitation, authorship detection, or evasion of automated detectors.
- Keep neutral material neutral when the source gives no basis for a personal stance.
"""


def _trait_block(trait: FinalTrait) -> str:
    scope = trait.scope
    lines = [
        f"### {_inline(trait.dimension.capitalize())} preference",
        "",
        f"- Apply: {_inline(trait.instruction)}",
        f"- Scope: languages {', '.join(map(_inline, scope.languages))}; genres {', '.join(map(_inline, scope.genres))}; registers {', '.join(map(_inline, scope.registers))}.",
        f"- Strength: {trait.strength}; confidence {trait.confidence:.2f}.",
        "",
    ]
    return "\n".join(lines)


def _render_profile_md(
    traits: list[FinalTrait],
) -> str:
    lines = [
        "# Voice profile",
        "",
        "Use this profile as evidence-backed guidance, not as a quota or a claim about identity.",
        "",
        "## Priority",
        "",
        "Apply current user instructions first, then content preservation, genre and language conventions, and finally these voice preferences.",
        "",
        "Every preference below was displayed in the review screen and explicitly selected by the user.",
        "",
        "## Accepted preferences",
        "",
    ]
    lines.extend(_trait_block(trait) for trait in traits)
    return "\n".join(lines).rstrip() + "\n"


def _render_preview(files: dict[str, bytes]) -> str:
    return "\n".join(
        f"===== {path} =====\n\n{files[path].decode('utf-8').rstrip()}\n" for path in sorted(files)
    )


def _render_openai_yaml(skill_name: str, voice_label: str) -> str:
    display_name = f"Write like {_safe_voice_label(voice_label)}"
    short_description = "Rewrite text in this documented personal voice"
    default_prompt = f"Use ${skill_name} to rewrite this text in my documented voice."
    return "\n".join(
        [
            "interface:",
            f"  display_name: {_yaml_string(display_name)}",
            f"  short_description: {_yaml_string(short_description)}",
            f"  default_prompt: {_yaml_string(default_prompt)}",
            "policy:",
            "  allow_implicit_invocation: false",
            "",
        ]
    )


def _validate_files(skill_name: str, files: dict[str, bytes]) -> None:
    if set(files) != SAFE_ARCHIVE_FILES:
        raise SkillBuildError("The skill package contains an unexpected file path.")
    if sum(len(value) for value in files.values()) > MAX_SKILL_BYTES:
        raise SkillBuildError("The generated skill is unexpectedly large.")
    for path, value in files.items():
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts or "\x00" in path:
            raise SkillBuildError("The skill package contains an unsafe path.")
        if b"\x00" in value:
            raise SkillBuildError("The skill package contains binary data.")
    skill_text = files["SKILL.md"].decode("utf-8")
    expected_frontmatter = f"name: {_yaml_string(skill_name)}"
    if not skill_text.startswith("---\n") or expected_frontmatter not in skill_text:
        raise SkillBuildError("SKILL.md has invalid frontmatter.")
    if "references/voice-profile.md" not in skill_text:
        raise SkillBuildError("SKILL.md does not link to its voice profile.")


def _archive(skill_name: str, files: dict[str, bytes]) -> bytes:
    destination = io.BytesIO()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative_path in sorted(files):
            info = zipfile.ZipInfo(f"{skill_name}/{relative_path}", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            archive.writestr(info, files[relative_path])
    return destination.getvalue()


def build_skill_package(
    skill_name: str,
    voice_label: str,
    profile: FinalStyleProfile,
    accepted_trait_ids: set[str],
) -> SkillPackage:
    traits = _selected_traits(profile, accepted_trait_ids)
    files = {
        "SKILL.md": _render_skill_md(skill_name, voice_label).encode(),
        "agents/openai.yaml": _render_openai_yaml(skill_name, voice_label).encode(),
        "references/voice-profile.md": _render_profile_md(traits).encode(),
    }
    _validate_files(skill_name, files)
    return SkillPackage(
        skill_name=skill_name,
        files=files,
        archive=_archive(skill_name, files),
        preview=_render_preview(files),
    )


def default_skill_root() -> Path:
    configured = os.environ.get("IDIOLECT_SKILL_ROOT")
    return Path(configured).expanduser() if configured else Path.home() / ".agents" / "skills"


def install_skill(package: SkillPackage, root: Path | None = None) -> Path:
    skill_root = (root or default_skill_root()).expanduser()
    if not skill_root.is_absolute():
        skill_root = (Path.cwd() / skill_root).resolve()
    for candidate in (skill_root, *skill_root.parents):
        if candidate.exists() and candidate.is_symlink():
            raise SkillBuildError("Refusing to install through a symlinked skill directory.")
    skill_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved_root = skill_root.resolve()
    target = resolved_root / package.skill_name
    if target.parent != resolved_root or target.exists() or target.is_symlink():
        raise SkillBuildError(
            "A skill with this name already exists; Idiolect will not overwrite it."
        )

    temporary = resolved_root / f".idiolect-{uuid4().hex}.tmp"
    temporary.mkdir(mode=0o700)
    try:
        for relative_path, content in package.files.items():
            destination = temporary / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with destination.open("xb") as handle:
                handle.write(content)
            destination.chmod(0o600)
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target
