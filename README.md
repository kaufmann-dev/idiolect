# Idiolect

Compile your writing samples into an evidence-based personal voice skill for Codex.

Upload self-authored text, review the voice traits Idiolect finds, then install the skill locally. The skill rewrites or drafts supplied text in your documented voice — only when you explicitly invoke it.

## How it works

1. **Build the corpus** — upload `.txt` or `.md` files or paste text.
2. **Analyze** — Codex runs twice: once to propose recurring traits, once to check them against held-out writing.
3. **Review** — keep at least three traits; each approval covers exactly the instruction and scope shown.
4. **Use** — download the ZIP or install it straight into your local skills.

The generated skill contains fixed instructions plus a declarative voice profile. It contains none of your uploaded documents.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Codex CLI, signed in via `codex login`

## Quickstart

```sh
uv sync
uv run idiolect doctor   # confirm Codex is signed in
uv run idiolect serve    # start the local app
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). The app only listens on localhost. Use `uv run idiolect serve --port 9000` to pick another port.

## What to upload

- UTF-8 `.txt` or `.md` files (up to 24 at once, 1 MB each, 3 MB total), or pasted text.
- Only fully self-authored writing in your current voice is analyzed; mark anything coauthored or edited accurately.
- Minimum 600 clean words. Three independent documents and 3,000+ words give a more reliable profile.

Quoted, tabular, and code blocks are stripped before analysis, and near-duplicate documents are grouped so repeated boilerplate can't pose as style.

## What you get

A skill package with three files:

- `SKILL.md` — when and how to apply the voice, with a meaning-preservation contract
- `references/voice-profile.md` — the accepted traits with language, genre, and register scopes
- `agents/openai.yaml` — display metadata (implicit invocation disabled)

Installing writes to `~/.agents/skills/<skill-name>/` and never overwrites an existing skill. Set `IDIOLECT_SKILL_ROOT` to change the destination, or `IDIOLECT_CODEX_PATH` to point at a specific Codex executable.

## Privacy

Raw sample text is not saved into the skill and is not stored by the app. Selected excerpts are sent to OpenAI for the two Codex analysis passes under your ChatGPT workspace's data controls.

## Development

```sh
uv run pytest
uv run ruff check .
uv run mypy
```
