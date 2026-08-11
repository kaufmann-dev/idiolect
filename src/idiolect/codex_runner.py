from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

ResultModel = TypeVar("ResultModel", bound=BaseModel)
StatusCallback = Callable[[str], None]

MAX_STDOUT_BYTES = 4_000_000
MAX_STDERR_BYTES = 500_000
MAX_RESULT_BYTES = 500_000
MAX_STATUS_BYTES = 16_000
DEFAULT_TIMEOUT_SECONDS = 900


class CodexRunnerError(RuntimeError):
    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


class UnexpectedToolUse(CodexRunnerError):
    pass


def strict_output_schema(model: type[BaseModel]) -> dict[str, object]:
    """Return the subset-compatible strict schema required by Codex structured output."""

    def normalize(value: object) -> object:
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if not isinstance(value, dict):
            return value
        normalized = {key: normalize(nested) for key, nested in value.items() if key != "default"}
        properties = normalized.get("properties")
        if isinstance(properties, dict):
            normalized["required"] = list(properties)
            normalized["additionalProperties"] = False
        return normalized

    schema = normalize(model.model_json_schema())
    if not isinstance(schema, dict):
        raise CodexRunnerError("Could not create the strict Codex output schema.")
    return schema


class CodexRunner:
    def __init__(
        self, executable: str | None = None, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    ):
        configured = executable or os.environ.get("IDIOLECT_CODEX_PATH") or shutil.which("codex")
        if not configured:
            raise CodexRunnerError("Codex CLI was not found. Install Codex and run `codex login`.")
        resolved = Path(configured).expanduser().resolve()
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise CodexRunnerError("The configured Codex executable is not an executable file.")
        self.executable = resolved
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _minimal_environment() -> dict[str, str]:
        allowed = {
            "PATH",
            "HOME",
            "CODEX_HOME",
            "USER",
            "LOGNAME",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "TMPDIR",
            "XDG_RUNTIME_DIR",
            "XDG_CONFIG_HOME",
            "XDG_CACHE_HOME",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "REQUESTS_CA_BUNDLE",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "no_proxy",
        }
        environment = {key: value for key, value in os.environ.items() if key in allowed}
        environment["NO_COLOR"] = "1"
        environment["TERM"] = "dumb"
        return environment

    async def readiness(self) -> dict[str, str | bool]:
        try:
            version_code, version_output = await self._readiness_command("--version")
            login_code, login_output = await self._readiness_command("login", "status")
        except TimeoutError:
            return {"ready": False, "message": "Codex readiness check timed out."}
        except (CodexRunnerError, OSError):
            return {"ready": False, "message": "Codex CLI could not be started."}

        version = version_output.decode("utf-8", "replace").strip()[:120]
        login = login_output.decode("utf-8", "replace").strip()
        if version_code != 0 or login_code != 0 or "using chatgpt" not in login.casefold():
            return {
                "ready": False,
                "version": version,
                "message": "Sign in with ChatGPT by running `codex login` in a terminal.",
            }
        return {
            "ready": True,
            "version": version,
            "message": "Codex is signed in with ChatGPT.",
        }

    async def _readiness_command(self, *arguments: str) -> tuple[int, bytes]:
        process = await asyncio.create_subprocess_exec(
            str(self.executable),
            *arguments,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=self._minimal_environment(),
            start_new_session=os.name == "posix",
        )
        if process.stdout is None:
            await self._terminate(process)
            raise CodexRunnerError("Codex CLI opened without a readable process stream.")
        try:
            return_code, output = await asyncio.wait_for(
                asyncio.gather(
                    process.wait(),
                    self._capture_limited(process.stdout, MAX_STATUS_BYTES),
                ),
                timeout=10,
            )
        except BaseException:
            await self._terminate(process)
            raise
        return int(return_code), output

    def command_for(
        self,
        *,
        schema_path: Path,
        result_path: Path,
        instructions_path: Path,
        working_directory: Path,
    ) -> list[str]:
        quoted_instructions = json.dumps(str(instructions_path.resolve()))
        return [
            str(self.executable),
            "-a",
            "never",
            "exec",
            "--strict-config",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--json",
            "--color",
            "never",
            "--disable",
            "shell_tool",
            "--disable",
            "unified_exec",
            "--disable",
            "apps",
            "--disable",
            "plugins",
            "--disable",
            "remote_plugin",
            "--disable",
            "skill_mcp_dependency_install",
            "--disable",
            "skill_search",
            "--disable",
            "browser_use",
            "--disable",
            "computer_use",
            "--disable",
            "image_generation",
            "--disable",
            "view_image",
            "--disable",
            "multi_agent",
            "--disable",
            "multi_agent_v2",
            "--disable",
            "hooks",
            "-c",
            "agents.enabled=false",
            "-c",
            'web_search="disabled"',
            "-c",
            f"model_instructions_file={quoted_instructions}",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(result_path),
            "--cd",
            str(working_directory),
            "-",
        ]

    async def run_structured(
        self,
        *,
        payload: dict[str, object],
        result_model: type[ResultModel],
        instructions_path: Path,
        status_callback: StatusCallback | None = None,
    ) -> ResultModel:
        if not instructions_path.is_file():
            raise CodexRunnerError("The trusted Codex instruction file is missing.")
        encoded_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        with tempfile.TemporaryDirectory(prefix="idiolect-") as temporary_name:
            temporary_directory = Path(temporary_name)
            temporary_directory.chmod(stat.S_IRWXU)
            schema_path = temporary_directory / "output-schema.json"
            result_path = temporary_directory / "result.json"
            schema_path.write_text(
                json.dumps(strict_output_schema(result_model), ensure_ascii=False), encoding="utf-8"
            )
            schema_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
            command = self.command_for(
                schema_path=schema_path,
                result_path=result_path,
                instructions_path=instructions_path,
                working_directory=temporary_directory,
            )
            if status_callback:
                status_callback("Codex is analyzing bounded evidence without tools.")
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=self._minimal_environment(),
                    start_new_session=os.name == "posix",
                )
            except OSError as error:
                raise CodexRunnerError("Codex CLI could not be started.") from error

            try:
                if process.stdin is None or process.stdout is None or process.stderr is None:
                    raise CodexRunnerError("Codex CLI opened without the required process streams.")
                process.stdin.write(encoded_payload)
                await process.stdin.drain()
                process.stdin.close()
                await process.stdin.wait_closed()

                _, stdout_result, _ = await asyncio.wait_for(
                    asyncio.gather(
                        process.wait(),
                        self._read_json_events(process.stdout),
                        self._discard_limited(process.stderr, MAX_STDERR_BYTES),
                    ),
                    timeout=self.timeout_seconds,
                )
            except TimeoutError as error:
                await self._terminate(process)
                raise CodexRunnerError(
                    "Codex timed out. No skill was created or installed."
                ) from error
            except asyncio.CancelledError:
                await self._terminate(process)
                raise
            except Exception:
                await self._terminate(process)
                raise

            if process.returncode != 0:
                raise CodexRunnerError(
                    "Codex could not complete this analysis. No skill was created."
                )
            if not stdout_result:
                raise CodexRunnerError("Codex completed without a structured final response.")
            if not result_path.is_file() or result_path.stat().st_size > MAX_RESULT_BYTES:
                raise CodexRunnerError("Codex returned an invalid or oversized profile.")
            try:
                return result_model.model_validate_json(result_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, ValidationError) as error:
                raise CodexRunnerError(
                    "Codex returned a profile that failed strict validation."
                ) from error

    @staticmethod
    async def _read_json_events(stream: asyncio.StreamReader) -> bool:
        total = 0
        received_final = False
        allowed_item_types = {"agent_message", "reasoning"}
        while raw_line := await stream.readline():
            total += len(raw_line)
            if total > MAX_STDOUT_BYTES:
                raise CodexRunnerError("Codex produced too much process output.")
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise CodexRunnerError("Codex produced an invalid event stream.") from error
            if not isinstance(event, dict):
                raise CodexRunnerError("Codex produced an invalid event stream.")
            event_type = event.get("type")
            if isinstance(event_type, str) and event_type.startswith("item."):
                item = event.get("item")
                item_type = item.get("type") if isinstance(item, dict) else None
                if item_type not in allowed_item_types:
                    raise UnexpectedToolUse(
                        "Codex attempted to use a tool while reading untrusted samples. The run was stopped."
                    )
                if event_type == "item.completed" and item_type == "agent_message":
                    received_final = True
            if event_type in {"turn.failed", "error"}:
                raise CodexRunnerError("Codex reported an error during analysis.")
        return received_final

    @staticmethod
    async def _discard_limited(stream: asyncio.StreamReader, limit: int) -> None:
        total = 0
        while chunk := await stream.read(16_384):
            total += len(chunk)
            if total > limit:
                raise CodexRunnerError("Codex produced too much diagnostic output.")

    @staticmethod
    async def _capture_limited(stream: asyncio.StreamReader, limit: int) -> bytes:
        output = bytearray()
        while chunk := await stream.read(4096):
            output.extend(chunk)
            if len(output) > limit:
                raise CodexRunnerError("Codex produced too much readiness output.")
        return bytes(output)

    @staticmethod
    async def _terminate(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            await asyncio.wait_for(process.wait(), timeout=3)
        except (ProcessLookupError, TimeoutError):
            if process.returncode is None:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
