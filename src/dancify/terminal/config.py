"""Validated terminal client configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit

from dancify.environment import load_environment
from dancify.terminal.capture import DSUCaptureConfig
from dancify.terminal.errors import ConfigurationError

DEFAULT_BASE_URL = "http://127.0.0.1:5000"


@dataclass(frozen=True, slots=True)
class ClientConfig:
    """Settings shared by REST, Socket.IO, playback, and Cemuhook capture."""

    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = 10.0
    motion_batch_size: int = 200
    motion_queue_size: int = 4
    progress_hz: int = 10
    dsu_host: str = "127.0.0.1"
    dsu_port: int = 26760
    dsu_left_slot: int | None = None
    dsu_right_slot: int = 1
    dsu_stale_after: float = 1.0

    def __post_init__(self) -> None:
        normalized = self.base_url.rstrip("/")
        parsed = urlsplit(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ConfigurationError("server URL must be an absolute http:// or https:// URL")
        if parsed.query or parsed.fragment:
            raise ConfigurationError("server URL cannot contain a query or fragment")
        if self.timeout_seconds <= 0:
            raise ConfigurationError("timeout must be positive")
        if not 1 <= self.motion_batch_size <= 1000:
            raise ConfigurationError("motion batch size must be between 1 and 1000")
        if not 1 <= self.motion_queue_size <= 100:
            raise ConfigurationError("motion queue size must be between 1 and 100")
        if self.progress_hz != 10:
            raise ConfigurationError("Dancify progress must run at exactly 10 Hz")
        try:
            _ = self.capture_config
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        object.__setattr__(self, "base_url", normalized)

    @property
    def api_url(self) -> str:
        return f"{self.base_url}/api/v1"

    @property
    def capture_config(self) -> DSUCaptureConfig:
        return DSUCaptureConfig(
            host=self.dsu_host,
            port=self.dsu_port,
            left_slot=self.dsu_left_slot,
            right_slot=self.dsu_right_slot,
            stale_after=self.dsu_stale_after,
        )

    @classmethod
    def from_env(
        cls,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        dsu_host: str | None = None,
        dsu_port: int | None = None,
        dsu_left_slot: int | None = None,
        dsu_right_slot: int | None = None,
    ) -> ClientConfig:
        """Load explicit, process-environment, and .env overrides."""

        load_environment()
        raw_url = base_url if base_url is not None else os.getenv("DANCIFY_URL", DEFAULT_BASE_URL)
        raw_timeout: object = os.getenv("DANCIFY_TIMEOUT", "10") if timeout_seconds is None else timeout_seconds
        raw_dsu_host = dsu_host if dsu_host is not None else os.getenv("DANCIFY_DSU_HOST", "127.0.0.1")
        try:
            timeout = float(raw_timeout)  # type: ignore[arg-type]
            batch_size = int(os.getenv("DANCIFY_MOTION_BATCH_SIZE", "200"))
            queue_size = int(os.getenv("DANCIFY_MOTION_QUEUE_SIZE", "4"))
            port = int(os.getenv("DANCIFY_DSU_PORT", "26760")) if dsu_port is None else dsu_port
            raw_left = os.getenv("DANCIFY_DSU_LEFT_SLOT") if dsu_left_slot is None else dsu_left_slot
            left = None if raw_left is None or raw_left == "" else int(raw_left)
            right = int(os.getenv("DANCIFY_DSU_RIGHT_SLOT", "1")) if dsu_right_slot is None else dsu_right_slot
            stale = float(os.getenv("DANCIFY_DSU_STALE_AFTER", "1"))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                "DANCIFY_TIMEOUT, motion, DSU port/slots, and stale settings must be numeric"
            ) from exc
        return cls(
            base_url=raw_url,
            timeout_seconds=timeout,
            motion_batch_size=batch_size,
            motion_queue_size=queue_size,
            dsu_host=raw_dsu_host,
            dsu_port=port,
            dsu_left_slot=left,
            dsu_right_slot=right,
            dsu_stale_after=stale,
        )
