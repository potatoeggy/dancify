"""Bounded deterministic-feature and live raw-motion uploaders."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from dancify.domain import RawImuSample, WristSide
from dancify.terminal.capture import CaptureError, DSUCapture
from dancify.terminal.demo_data import motion_features
from dancify.terminal.dto import JsonObject, MotionHealth, RawUploadResult, object_value
from dancify.terminal.errors import ConfigurationError

Feature = dict[str, object]


class MotionAPI(Protocol):
    async def upload_motion(self, session_id: str, features: list[Feature]) -> int: ...


class RawMotionAPI(Protocol):
    async def upload_raw_motion(self, session_id: str, samples: list[JsonObject]) -> RawUploadResult: ...


class MotionSource(Protocol):
    def __aiter__(self) -> AsyncIterator[Feature]: ...


class ListMotionSource:
    def __init__(self, features: list[Feature]) -> None:
        self._features = features

    def __aiter__(self) -> AsyncIterator[Feature]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Feature]:
        for feature in self._features:
            yield feature

    @classmethod
    def from_file(cls, path: Path) -> ListMotionSource:
        try:
            raw: Any = json.loads(path.read_text())
        except OSError as exc:
            raise ConfigurationError(f"cannot read motion file {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"motion file {path} is not valid JSON: {exc}") from exc
        if isinstance(raw, dict):
            raw = cast(dict[str, Any], raw).get("features")
        if not isinstance(raw, list):
            raise ConfigurationError("motion JSON must be a list or an object containing `features`")
        return cls([cast(Feature, object_value(item, "motion feature")) for item in cast(list[Any], raw)])  # type: ignore[redundant-cast]


class GeneratedMotionSource(ListMotionSource):
    def __init__(self, duration: float) -> None:
        super().__init__(motion_features(duration))


class BoundedMotionUploader:
    """Upload fixed-size feature batches through a bounded queue."""

    def __init__(self, api: MotionAPI, *, batch_size: int = 200, queue_size: int = 4) -> None:
        if batch_size <= 0 or queue_size <= 0:
            raise ValueError("batch and queue sizes must be positive")
        self._api = api
        self._batch_size = batch_size
        self._queue_size = queue_size

    async def upload(self, session_id: str, source: MotionSource) -> int:
        queue: asyncio.Queue[list[Feature] | None] = asyncio.Queue(maxsize=self._queue_size)
        accepted = 0

        async def produce() -> None:
            batch: list[Feature] = []
            async for feature in source:
                batch.append(feature)
                if len(batch) == self._batch_size:
                    await queue.put(batch)
                    batch = []
            if batch:
                await queue.put(batch)
            await queue.put(None)

        async def consume() -> None:
            nonlocal accepted
            while (batch := await queue.get()) is not None:
                accepted += await self._api.upload_motion(session_id, batch)

        async with asyncio.TaskGroup() as group:
            group.create_task(produce())
            group.create_task(consume())
        return accepted


@dataclass(frozen=True, slots=True)
class RawStreamResult:
    accepted: int
    dropped: int
    batches: int
    motion_health: MotionHealth | None


type RawResultCallback = Callable[[RawUploadResult], Awaitable[None] | None]


class RawCaptureUploader:
    """Stream capture samples in bounded batches with explicit wrist/clock mapping."""

    def __init__(self, api: RawMotionAPI, *, batch_size: int = 200, flush_seconds: float = 0.1) -> None:
        if not 1 <= batch_size <= 1000 or flush_seconds <= 0:
            raise ValueError("raw batch size or flush interval is invalid")
        self._api = api
        self._batch_size = batch_size
        self._flush_seconds = flush_seconds

    async def stream(
        self,
        session_id: str,
        capture: DSUCapture,
        stop: asyncio.Event,
        *,
        on_result: RawResultCallback | None = None,
    ) -> RawStreamResult:
        accepted = dropped = batches = 0
        latest: MotionHealth | None = None
        batch: list[JsonObject] = []
        while not stop.is_set():
            try:
                sample = await capture.receive(self._flush_seconds)
            except TimeoutError:
                sample = None
            except StopAsyncIteration:
                if not stop.is_set():
                    raise CaptureError("DSU capture stopped during gameplay") from None
                break
            if sample is not None:
                batch.append(raw_sample_payload(capture, sample))
            if batch and (len(batch) >= self._batch_size or sample is None):
                result = await self._api.upload_raw_motion(session_id, batch)
                accepted += result.accepted
                dropped += result.dropped
                batches += 1
                latest = result.motion_health
                batch = []
                if on_result is not None:
                    callback = on_result(result)
                    if inspect.isawaitable(callback):
                        await callback
        if batch:
            result = await self._api.upload_raw_motion(session_id, batch)
            accepted += result.accepted
            dropped += result.dropped
            batches += 1
            latest = result.motion_health
            if on_result is not None:
                callback = on_result(result)
                if inspect.isawaitable(callback):
                    await callback
        return RawStreamResult(accepted, dropped, batches, latest)


def raw_sample_payload(capture: DSUCapture, sample: RawImuSample) -> JsonObject:
    wrist = next(
        (side for side in WristSide if sample.device_id.lower().startswith(f"{side.value}:")),
        None,
    )
    if wrist is None:
        raise CaptureError(f"captured sample has no assigned wrist: {sample.device_id}")
    estimate = capture.clock_estimate(wrist)
    if estimate is None:
        raise CaptureError(f"{wrist.value} capture clock is not initialized")
    return {
        "wrist": wrist.value,
        "packetNumber": sample.packet_number,
        "captureTimestampUs": sample.device_timestamp_us,
        "clientTimestamp": estimate.to_monotonic_time(sample.device_timestamp_us),
        "accelerationG": sample.acceleration_g.to_list(),
        "angularVelocityDps": sample.angular_velocity_dps.to_list(),
    }
