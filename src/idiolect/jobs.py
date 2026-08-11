from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from idiolect.codex_runner import CodexRunner, CodexRunnerError
from idiolect.generator import StyleGenerator
from idiolect.models import (
    CompileRequest,
    CorpusBundle,
    FinalStyleProfile,
    FinalTrait,
    JobRecord,
    JobStage,
)
from idiolect.skill_builder import (
    SkillBuildError,
    build_skill_package,
    install_skill,
)

JOB_TTL_SECONDS = 3_600


class JobError(ValueError):
    pass


class JobManager:
    def __init__(self, runner: CodexRunner, skill_root: Path | None = None) -> None:
        self.runner = runner
        self.generator = StyleGenerator(runner)
        self.skill_root = skill_root
        self._jobs: dict[str, JobRecord] = {}
        self._idempotency: dict[str, str] = {}
        self._worker = asyncio.Semaphore(1)
        self._lock = asyncio.Lock()

    async def create(
        self,
        *,
        corpus: CorpusBundle,
        skill_name: str,
        voice_label: str,
        purpose: str,
        idempotency_key: str,
    ) -> JobRecord:
        async with self._lock:
            self._prune()
            if existing_id := self._idempotency.get(idempotency_key):
                return self._jobs[existing_id]
            token_jobs = {
                JobStage.QUEUED,
                JobStage.ANALYZING,
                JobStage.VALIDATING,
            }
            if any(job.stage in token_jobs for job in self._jobs.values()):
                raise JobError("Another Codex analysis is already queued or running.")
            now = time.time()
            record = JobRecord(
                id=uuid4().hex,
                skill_name=skill_name,
                voice_label=voice_label,
                stats=corpus.stats,
                created_at=now,
                updated_at=now,
            )
            task = asyncio.create_task(self._run(record, corpus, purpose))
            record.task = task
            self._jobs[record.id] = record
            self._idempotency[idempotency_key] = record.id
            return record

    async def _run(self, record: JobRecord, corpus: CorpusBundle, purpose: str) -> None:
        try:
            readiness = await self.runner.readiness()
            if not readiness.get("ready"):
                raise CodexRunnerError(str(readiness.get("message", "Codex is not ready.")))
            async with self._worker:
                self._update(
                    record,
                    JobStage.ANALYZING,
                    "Analyzing recurring choices across the discovery corpus.",
                )

                def stage_callback(stage: str, message: str) -> None:
                    job_stage = JobStage.VALIDATING if stage == "validating" else JobStage.ANALYZING
                    self._update(record, job_stage, message)

                profile = await self.generator.generate(corpus, purpose, stage_callback)
                record.profile = profile
                record.selected_trait_ids = tuple(trait.id for trait in self._traits(profile))
                self._update(
                    record,
                    JobStage.REVIEW,
                    "Review the evidenced traits before building the skill.",
                )
        except asyncio.CancelledError:
            self._update(
                record, JobStage.CANCELLED, "Cancelled. No skill was created or installed."
            )
        except (CodexRunnerError, SkillBuildError, JobError) as error:
            record.error = str(error)
            self._update(record, JobStage.FAILED, str(error))
        except Exception:
            record.error = "Unexpected local error. No skill was created or installed."
            self._update(record, JobStage.FAILED, record.error)

    @staticmethod
    def _traits(profile: FinalStyleProfile) -> list[FinalTrait]:
        return [
            *profile.core_traits,
            *(trait for mode in profile.modes for trait in mode.traits),
        ]

    def _update(self, record: JobRecord, stage: JobStage, message: str) -> None:
        record.stage = stage
        record.status_message = message
        record.updated_at = time.time()

    async def cancel(self, job_id: str) -> JobRecord:
        record = self.get(job_id)
        if record.stage not in {JobStage.QUEUED, JobStage.ANALYZING, JobStage.VALIDATING}:
            raise JobError("This job is no longer running.")
        task = record.task
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._update(record, JobStage.CANCELLED, "Cancelled. No skill was created or installed.")
        return record

    async def shutdown(self) -> None:
        tasks = [job.task for job in self._jobs.values() if job.task and not job.task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def compile(self, job_id: str, request: CompileRequest) -> JobRecord:
        record = self.get(job_id)
        if record.stage is not JobStage.REVIEW or record.profile is None:
            raise JobError("This job is not ready to build.")
        available = {trait.id for trait in self._traits(record.profile)}
        selected = set(request.accepted_trait_ids)
        if not selected.issubset(available):
            raise JobError("The review contains an unknown trait.")
        self._update(record, JobStage.PACKAGING, "Building and validating the skill package.")
        try:
            record.package = build_skill_package(
                record.skill_name,
                record.voice_label,
                record.profile,
                selected,
            )
        except SkillBuildError:
            self._update(record, JobStage.REVIEW, "Adjust the selected traits and try again.")
            raise
        record.selected_trait_ids = tuple(sorted(selected))
        self._update(record, JobStage.COMPLETE, "Skill built. Download it or install it locally.")
        return record

    def install(self, job_id: str) -> Path:
        record = self.get(job_id)
        if record.stage is not JobStage.COMPLETE or record.package is None:
            raise JobError("Build the skill before installing it.")
        return install_skill(record.package, self.skill_root)

    def get(self, job_id: str) -> JobRecord:
        if record := self._jobs.get(job_id):
            return record
        raise JobError("Job not found or expired.")

    def public(self, record: JobRecord) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": record.id,
            "skill_name": record.skill_name,
            "stage": record.stage.value,
            "message": record.status_message,
            "error": record.error,
            "stats": asdict(record.stats) if record.stats else None,
        }
        if record.profile is not None:
            mode_by_trait = {
                trait.id: mode.label for mode in record.profile.modes for trait in mode.traits
            }
            result["profile"] = {
                "summary": record.profile.summary,
                "languages": record.profile.detected_languages,
                "regional_notes": record.profile.regional_notes,
                "uncertainties": record.profile.uncertainties,
                "traits": [
                    {
                        "id": trait.id,
                        "dimension": trait.dimension,
                        "instruction": trait.instruction,
                        "confidence": trait.confidence,
                        "strength": trait.strength,
                        "evidence_count": len(trait.evidence_ids),
                        "mode": mode_by_trait.get(trait.id, "Core voice"),
                        "scope": trait.scope.model_dump(),
                    }
                    for trait in self._traits(record.profile)
                ],
            }
        if record.package is not None:
            result["preview"] = record.package.preview
        return result

    def _prune(self) -> None:
        cutoff = time.time() - JOB_TTL_SECONDS
        expired = [job_id for job_id, job in self._jobs.items() if job.updated_at < cutoff]
        for job_id in expired:
            del self._jobs[job_id]
        if expired:
            expired_set = set(expired)
            self._idempotency = {
                key: job_id
                for key, job_id in self._idempotency.items()
                if job_id not in expired_set
            }
