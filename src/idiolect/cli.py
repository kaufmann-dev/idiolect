from __future__ import annotations

import argparse
import asyncio

import uvicorn

from idiolect.codex_runner import CodexRunner, CodexRunnerError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="idiolect")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve", help="start the local Idiolect app")
    serve.add_argument("--port", type=int, default=8765)
    subparsers.add_parser("doctor", help="check the local Codex login")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "doctor":
        try:
            result = asyncio.run(CodexRunner().readiness())
        except CodexRunnerError as error:
            raise SystemExit(error.user_message) from error
        print(result.get("message", "Codex readiness is unknown."))
        if version := result.get("version"):
            print(version)
        if not result.get("ready"):
            raise SystemExit(1)
        return
    if not 1024 <= args.port <= 65_535:
        raise SystemExit("Port must be between 1024 and 65535.")
    uvicorn.run(
        "idiolect.web:create_app",
        factory=True,
        host="127.0.0.1",
        port=args.port,
        access_log=False,
    )
