from __future__ import annotations

import json
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import RequestResponseEndpoint
from starlette.middleware.trustedhost import TrustedHostMiddleware

from idiolect.codex_runner import CodexRunner
from idiolect.corpus import (
    ALLOWED_EXTENSIONS,
    MAX_TOTAL_BYTES,
    CorpusError,
    build_corpus,
    decode_text_file,
    make_sample,
    normalize_skill_name,
)
from idiolect.jobs import JobError, JobManager
from idiolect.models import CompileRequest, SampleMetadata, WritingSample
from idiolect.skill_builder import SkillBuildError

PACKAGE_DIRECTORY = Path(__file__).parent
STATIC_DIRECTORY = PACKAGE_DIRECTORY / "static"


def _loopback_origin(origin: str) -> bool:
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    return parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}


def create_app(
    *,
    runner: CodexRunner | None = None,
    skill_root: Path | None = None,
    csrf_token: str | None = None,
) -> FastAPI:
    token = csrf_token or secrets.token_urlsafe(32)
    codex_runner = runner or CodexRunner()
    manager = JobManager(codex_runner, skill_root)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await manager.shutdown()

    app = FastAPI(
        title="Idiolect",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.manager = manager
    app.state.csrf_token = token
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"],
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")

    @app.middleware("http")
    async def security_headers(request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; "
            "frame-ancestors 'none'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        if request.url.path == "/" or request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    async def protect_api(
        request: Request,
        x_idiolect_token: Annotated[str | None, Header()] = None,
    ) -> None:
        if not x_idiolect_token or not secrets.compare_digest(x_idiolect_token, token):
            raise HTTPException(status_code=403, detail="Invalid local session token.")
        origin = request.headers.get("origin")
        if origin and not _loopback_origin(origin):
            raise HTTPException(status_code=403, detail="Cross-origin requests are not allowed.")

    @app.exception_handler(CorpusError)
    @app.exception_handler(JobError)
    @app.exception_handler(SkillBuildError)
    async def expected_error(_: Request, error: Exception) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(error)})

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        template = (STATIC_DIRECTORY / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(template.replace("__IDIOLECT_TOKEN__", token))

    @app.get("/api/status", dependencies=[Depends(protect_api)])
    async def status() -> dict[str, str | bool]:
        return await codex_runner.readiness()

    @app.post("/api/jobs", status_code=202, dependencies=[Depends(protect_api)])
    async def create_job(
        skill_name: Annotated[str, Form(max_length=80)],
        voice_label: Annotated[str, Form(max_length=60)],
        purpose: Annotated[str, Form(max_length=600)] = "",
        pasted_text: Annotated[str, Form(max_length=1_000_000)] = "",
        pasted_self_authored: Annotated[bool, Form()] = False,
        rights_confirmed: Annotated[bool, Form()] = False,
        sample_metadata: Annotated[str, Form(max_length=20_000)] = "[]",
        files: Annotated[list[UploadFile] | None, File()] = None,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> dict[str, object]:
        if not rights_confirmed:
            raise HTTPException(
                status_code=400,
                detail="Confirm that you wrote the samples or have permission to use them.",
            )
        if not idempotency_key or not 8 <= len(idempotency_key) <= 100:
            raise HTTPException(status_code=400, detail="Missing idempotency key.")
        try:
            metadata_values = json.loads(sample_metadata)
            if not isinstance(metadata_values, list):
                raise ValueError
            metadata = [SampleMetadata.model_validate(value) for value in metadata_values]
        except (json.JSONDecodeError, ValueError) as error:
            raise HTTPException(status_code=400, detail="Invalid sample metadata.") from error

        uploads = files or []
        if len(uploads) > 24:
            raise HTTPException(status_code=400, detail="Upload no more than 24 files at once.")
        if len(metadata) != len(uploads):
            raise HTTPException(status_code=400, detail="Sample metadata does not match the files.")
        samples: list[WritingSample] = []
        total_bytes = len(pasted_text.encode("utf-8"))
        for index, upload in enumerate(uploads):
            filename = Path(upload.filename or f"sample-{index + 1}.txt").name
            if Path(filename).suffix.casefold() not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"{filename}: upload UTF-8 .txt or .md files only.",
                )
            chunks: list[bytes] = []
            size = 0
            while chunk := await upload.read(65_536):
                size += len(chunk)
                total_bytes += len(chunk)
                if size > 1_000_000 or total_bytes > MAX_TOTAL_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Uploads exceed the 1 MB per-file or 3 MB total limit.",
                    )
                chunks.append(chunk)
            await upload.close()
            samples.append(
                make_sample(filename, decode_text_file(b"".join(chunks)), metadata[index])
            )
        if pasted_text.strip() and not pasted_self_authored:
            raise HTTPException(
                status_code=400,
                detail="Confirm that pasted text is fully self-authored and represents your current voice.",
            )
        if pasted_text.strip():
            samples.append(
                make_sample(
                    "pasted-text.txt",
                    pasted_text,
                    SampleMetadata(),
                )
            )
        if not samples:
            raise HTTPException(status_code=400, detail="Add at least one text sample.")
        if total_bytes > MAX_TOTAL_BYTES:
            raise HTTPException(status_code=413, detail="Uploads exceed the 3 MB total limit.")
        label = " ".join(voice_label.split())
        if not label:
            raise HTTPException(status_code=400, detail="Enter a voice label.")
        corpus = build_corpus(samples)
        record = await manager.create(
            corpus=corpus,
            skill_name=normalize_skill_name(skill_name),
            voice_label=label,
            purpose=purpose.strip(),
            idempotency_key=idempotency_key,
        )
        return manager.public(record)

    @app.get("/api/jobs/{job_id}", dependencies=[Depends(protect_api)])
    async def get_job(job_id: str) -> dict[str, object]:
        return manager.public(manager.get(job_id))

    @app.delete("/api/jobs/{job_id}", dependencies=[Depends(protect_api)])
    async def cancel_job(job_id: str) -> dict[str, object]:
        return manager.public(await manager.cancel(job_id))

    @app.post("/api/jobs/{job_id}/compile", dependencies=[Depends(protect_api)])
    async def compile_job(job_id: str, request: CompileRequest) -> dict[str, object]:
        return manager.public(manager.compile(job_id, request))

    @app.get("/api/jobs/{job_id}/download", dependencies=[Depends(protect_api)])
    async def download_job(job_id: str) -> Response:
        record = manager.get(job_id)
        if record.package is None:
            raise JobError("Build the skill before downloading it.")
        return Response(
            content=record.package.archive,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{record.skill_name}.zip"',
                "Cache-Control": "no-store",
            },
        )

    @app.post("/api/jobs/{job_id}/install", dependencies=[Depends(protect_api)])
    async def install_job(job_id: str) -> dict[str, str]:
        destination = manager.install(job_id)
        return {"message": "Skill installed locally.", "path": str(destination)}

    return app
