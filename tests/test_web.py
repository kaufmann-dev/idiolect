from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from idiolect.models import (
    CalibrationPair,
    CandidateStyleProfile,
    CandidateTrait,
    FinalStyleProfile,
    FinalTrait,
    TraitScope,
)
from idiolect.web import create_app


class FakeRunner:
    def __init__(self) -> None:
        self.train_ids: list[str] = []

    async def readiness(self) -> dict[str, str | bool]:
        return {"ready": True, "version": "codex-cli test", "message": "Ready."}

    async def run_structured(
        self,
        *,
        payload: dict[str, object],
        result_model: type[Any],
        instructions_path: Path,
        status_callback: Any = None,
    ) -> Any:
        assert instructions_path.is_file()
        if status_callback:
            status_callback("Fake Codex is working.")
        scope = TraitScope(languages=["English"], genres=["all"], registers=["all"])
        if result_model is CandidateStyleProfile:
            segments = payload["segments"]
            assert isinstance(segments, list)
            self.train_ids = [str(segment["id"]) for segment in segments]  # type: ignore[index]
            evidence = self.train_ids[:2] or self.train_ids
            traits = [
                CandidateTrait(
                    id=f"trait-{index}",
                    dimension=dimension,
                    observation=f"Observation {index} recurs.",
                    instruction=f"Apply writing preference {index} with restraint.",
                    scope=scope,
                    confidence=0.8,
                    evidence_ids=evidence,
                    confounds_checked=["topic", "template"],
                )
                for index, dimension in enumerate(
                    ["rhythm", "syntax", "diction", "structure"], start=1
                )
            ]
            return CandidateStyleProfile(
                schema_version="idiolect.candidate/v1",
                summary="A direct voice with controlled variation.",
                detected_languages=["English"],
                core_traits=traits,
            )

        evidence = self.train_ids[:2] or self.train_ids
        final_traits = [
            FinalTrait(
                id=f"trait-{index}",
                dimension=dimension,
                instruction=f"Apply writing preference {index} with restraint.",
                scope=scope,
                confidence=0.8,
                strength="preference",
                evidence_ids=evidence,
                rationale=f"Multiple passages support preference {index}.",
            )
            for index, dimension in enumerate(["rhythm", "syntax", "diction", "structure"], start=1)
        ]
        return FinalStyleProfile(
            schema_version="idiolect.style-profile/v1",
            summary="A direct voice with controlled variation.",
            detected_languages=["English"],
            core_traits=final_traits,
            calibration_pairs=[
                CalibrationPair(
                    neutral="The meeting begins at ten.",
                    in_voice="We start at ten.",
                    illustrates_trait_ids=["trait-1"],
                ),
                CalibrationPair(
                    neutral="Please review the attached proposal.",
                    in_voice="Take a look at the proposal.",
                    illustrates_trait_ids=["trait-2"],
                ),
            ],
        )


def _sample_text() -> str:
    paragraphs = []
    for index in range(12):
        words = [f"term{index}_{word}" for word in range(65)]
        paragraphs.append(" ".join(words) + ".")
    return "\n\n".join(paragraphs)


def _wait_for_stage(client: TestClient, job_id: str, token: str, stage: str) -> dict[str, Any]:
    for _ in range(100):
        response = client.get(f"/api/jobs/{job_id}", headers={"X-Idiolect-Token": token})
        assert response.status_code == 200
        payload = response.json()
        if payload["stage"] == stage:
            return payload
        if payload["stage"] == "failed":
            raise AssertionError(payload)
        time.sleep(0.01)
    raise AssertionError(f"Job did not reach {stage}")


def test_local_security_and_complete_review_flow(tmp_path: Path) -> None:
    token = "test-token"
    app = create_app(runner=FakeRunner(), skill_root=tmp_path / "skills", csrf_token=token)  # type: ignore[arg-type]
    with TestClient(app) as client:
        root = client.get("/")
        assert root.status_code == 200
        assert token in root.text
        assert "frame-ancestors 'none'" in root.headers["content-security-policy"]
        assert root.headers["cache-control"] == "no-store"

        assert client.get("/api/status").status_code == 403
        cross_origin = client.get(
            "/api/status",
            headers={"X-Idiolect-Token": token, "Origin": "https://hostile.example"},
        )
        assert cross_origin.status_code == 403

        metadata = '[{"authorship":"self","genre":"article","language":"en","current_voice":true}]'
        response = client.post(
            "/api/jobs",
            headers={"X-Idiolect-Token": token, "Idempotency-Key": "job-key-123"},
            data={
                "skill_name": "Write Like Alex",
                "voice_label": "Alex",
                "purpose": "Articles and emails",
                "rights_confirmed": "true",
                "sample_metadata": metadata,
            },
            files={"files": ("sample.md", _sample_text(), "text/markdown")},
        )
        assert response.status_code == 202, response.text
        job_id = response.json()["id"]
        review = _wait_for_stage(client, job_id, token, "review")
        assert len(review["profile"]["traits"]) == 4
        assert review["stats"]["clean_words"] >= 600

        compiled = client.post(
            f"/api/jobs/{job_id}/compile",
            headers={"X-Idiolect-Token": token},
            json={"accepted_trait_ids": ["trait-1", "trait-2", "trait-3"]},
        )
        assert compiled.status_code == 200, compiled.text
        assert compiled.json()["stage"] == "complete"
        assert "Write in Alex's voice" in compiled.json()["preview"]

        download = client.get(f"/api/jobs/{job_id}/download", headers={"X-Idiolect-Token": token})
        assert download.status_code == 200
        assert download.headers["content-type"] == "application/zip"

        installed = client.post(f"/api/jobs/{job_id}/install", headers={"X-Idiolect-Token": token})
        assert installed.status_code == 200
        assert (tmp_path / "skills" / "write-like-alex" / "SKILL.md").is_file()


def test_job_rejects_too_little_writing() -> None:
    token = "test-token"
    app = create_app(runner=FakeRunner(), csrf_token=token)  # type: ignore[arg-type]
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            headers={"X-Idiolect-Token": token, "Idempotency-Key": "job-key-small"},
            data={
                "skill_name": "small",
                "voice_label": "Alex",
                "sample_metadata": "[]",
                "pasted_text": "Only a few words.",
                "pasted_self_authored": "true",
                "rights_confirmed": "true",
            },
        )
        assert response.status_code == 400
        assert "at least 600 clean words" in response.json()["detail"]


def test_job_rejects_unconfirmed_pasted_authorship() -> None:
    token = "test-token"
    app = create_app(runner=FakeRunner(), csrf_token=token)  # type: ignore[arg-type]
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            headers={"X-Idiolect-Token": token, "Idempotency-Key": "job-key-paste"},
            data={
                "skill_name": "pasted",
                "voice_label": "Alex",
                "sample_metadata": "[]",
                "pasted_text": _sample_text(),
                "rights_confirmed": "true",
            },
        )

        assert response.status_code == 400
        assert "fully self-authored" in response.json()["detail"]
