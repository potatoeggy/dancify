"""Actionable terminal-client failures with stable exit codes."""

from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    OK = 0
    USAGE = 2
    CONFIGURATION = 3
    CONNECTION = 4
    API = 5
    RUNTIME = 6
    INTERRUPTED = 130


class ClientError(Exception):
    exit_code = ExitCode.RUNTIME

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def display(self) -> str:
        return self.message if self.hint is None else f"{self.message}\nHint: {self.hint}"


class ConfigurationError(ClientError):
    exit_code = ExitCode.CONFIGURATION


class ConnectionFailure(ClientError):
    exit_code = ExitCode.CONNECTION


class ProtocolError(ClientError):
    exit_code = ExitCode.API


class APIError(ClientError):
    exit_code = ExitCode.API

    def __init__(self, status_code: int, code: str, message: str) -> None:
        hint = {
            404: "Check that the ID belongs to this running backend process.",
            409: "Run `session show` and perform the next valid state transition.",
        }.get(status_code)
        super().__init__(f"backend {code} ({status_code}): {message}", hint=hint)
        self.status_code = status_code
        self.code = code


class GameplayAborted(ClientError):
    """The operator cancelled a live workflow before normal completion."""


class PlaybackError(ClientError):
    pass
