"""Async REST adapter for the versioned Dancify API."""

from __future__ import annotations

from types import TracebackType
from typing import Any, Self, cast

import httpx

from dancify.terminal.config import ClientConfig
from dancify.terminal.dto import (
    CalibrationResult,
    JsonObject,
    RawUploadResult,
    Routine,
    RoutineWindow,
    Score,
    Session,
    object_value,
)
from dancify.terminal.errors import APIError, ConnectionFailure, ProtocolError


class DancifyAPI:
    def __init__(self, config: ClientConfig, client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self._owned = client is None
        self._client = client or httpx.AsyncClient(base_url=config.api_url, timeout=config.timeout_seconds)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owned:
            await self._client.aclose()

    async def health(self) -> bool:
        return (await self._request("GET", "/health")).get("status") == "ok"

    async def import_routine(self, payload: JsonObject) -> Routine:
        return Routine.from_dict(await self._request("POST", "/routines", payload))

    async def get_routine(self, routine_id: str) -> Routine:
        return Routine.from_dict(await self._request("GET", f"/routines/{routine_id}"))

    async def get_windows(self, routine_id: str) -> tuple[RoutineWindow, ...]:
        data = await self._request("GET", f"/routines/{routine_id}/windows")
        return tuple(RoutineWindow.from_dict(item) for item in _object_list(data, "windows", "window"))

    async def create_session(self, routine_id: str, player_id: str, scoring_algorithm: str = "weighted_dtw") -> Session:
        return Session.from_dict(
            await self._request(
                "POST",
                "/sessions",
                {"routineID": routine_id, "playerID": player_id, "scoringAlgorithm": scoring_algorithm},
            )
        )

    async def get_session(self, session_id: str) -> Session:
        return Session.from_dict(await self._request("GET", f"/sessions/{session_id}"))

    async def calibrate(self, session_id: str, payload: JsonObject) -> CalibrationResult:
        return CalibrationResult.from_dict(await self._request("POST", f"/sessions/{session_id}/calibration", payload))

    async def start(self, session_id: str, delay_seconds: float) -> float:
        data = await self._request("POST", f"/sessions/{session_id}/start", {"delaySeconds": delay_seconds})
        value = data.get("startAt")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ProtocolError("startAt must be numeric")
        return float(value)

    async def upload_motion(self, session_id: str, features: list[dict[str, object]]) -> int:
        data = await self._request("POST", f"/sessions/{session_id}/motion", {"features": features})
        accepted = data.get("accepted")
        if isinstance(accepted, bool) or not isinstance(accepted, int):
            raise ProtocolError("accepted must be an integer")
        if accepted != len(features):
            raise ProtocolError(f"backend accepted {accepted} of {len(features)} motion features")
        return accepted

    async def upload_raw_motion(self, session_id: str, samples: list[JsonObject]) -> RawUploadResult:
        if not samples:
            raise ValueError("raw motion upload requires at least one sample")
        data = await self._request("POST", f"/sessions/{session_id}/motion/raw", {"samples": samples})
        return RawUploadResult.from_dict(data, len(samples))

    async def progress(self, session_id: str, video_time: float, server_time: float | None = None) -> tuple[Score, ...]:
        payload: dict[str, object] = {"videoTime": video_time}
        if server_time is not None:
            payload["serverTime"] = server_time
        data = await self._request("POST", f"/sessions/{session_id}/progress", payload)
        return tuple(Score.from_dict(item) for item in _object_list(data, "scores", "score"))

    async def retry(self, session_id: str) -> Session:
        return Session.from_dict(await self._request("POST", f"/sessions/{session_id}/retry"))

    async def abort(self, session_id: str) -> Session:
        return Session.from_dict(await self._request("POST", f"/sessions/{session_id}/abort"))

    async def _request(self, method: str, path: str, payload: JsonObject | None = None) -> JsonObject:
        try:
            response = await self._client.request(method, path, json=payload)
        except httpx.TimeoutException as exc:
            raise ConnectionFailure(
                f"request to {self.config.base_url} timed out",
                hint="Check the server and increase --timeout if it is intentionally slow.",
            ) from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure(
                f"cannot reach Dancify at {self.config.base_url}",
                hint="Start the backend with `uv run python -m dancify` or correct --url.",
            ) from exc
        try:
            body = object_value(response.json())
        except ValueError as exc:
            raise ProtocolError(f"backend returned non-JSON HTTP {response.status_code}") from exc
        if response.is_error:
            error = body.get("error")
            details = object_value(error, "error") if isinstance(error, dict) else {}
            raise APIError(
                response.status_code,
                str(details.get("code", "http_error")),
                str(details.get("message", response.reason_phrase)),
            )
        return body


def _object_list(data: JsonObject, key: str, label: str) -> list[JsonObject]:
    raw = data.get(key)
    if not isinstance(raw, list):
        raise ProtocolError(f"{key} must be a JSON list")
    return [object_value(item, label) for item in cast(list[Any], raw)]  # type: ignore[redundant-cast]
