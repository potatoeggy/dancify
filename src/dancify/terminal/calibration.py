"""Measured guided calibration for configured Cemuhook wrist capture."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from math import sqrt
from typing import Protocol

from dancify.domain import RawImuSample, Vector3, WristSide
from dancify.terminal.capture import CaptureError, DSUCapture
from dancify.terminal.dto import CalibrationResult, ClockObservation, JsonObject
from dancify.terminal.errors import GameplayAborted


class CalibrationAPI(Protocol):
    async def calibrate(self, session_id: str, payload: JsonObject) -> CalibrationResult: ...


class CalibrationSocket(Protocol):
    async def observe_clock(self) -> ClockObservation: ...


@dataclass(frozen=True, slots=True)
class CalibrationStatus:
    stage: str
    message: str
    attempt: int = 1
    countdown: int = 0
    left_samples: int = 0
    right_samples: int = 0
    quality: float | None = None


type StatusCallback = Callable[[CalibrationStatus], Awaitable[None] | None]


class GuidedCalibrator:
    """Collect stable measured poses; synthetic fixtures are intentionally absent."""

    def __init__(
        self,
        api: CalibrationAPI,
        socket: CalibrationSocket,
        capture: DSUCapture,
        *,
        samples_per_wrist: int = 20,
        countdown_seconds: int = 3,
        stage_timeout: float = 8.0,
        clock_observations: int = 5,
        max_retries: int = 2,
        cancel: asyncio.Event | None = None,
        status: StatusCallback | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if samples_per_wrist < 2 or countdown_seconds < 0 or stage_timeout <= 0:
            raise ValueError("calibration sample count, countdown, and timeout are invalid")
        if clock_observations < 2 or max_retries < 0:
            raise ValueError("calibration requires at least two clock observations and non-negative retries")
        self._api = api
        self._socket = socket
        self._capture = capture
        self._samples = samples_per_wrist
        self._countdown = countdown_seconds
        self._timeout = stage_timeout
        self._clock_count = clock_observations
        self._max_retries = max_retries
        self._cancel = cancel or asyncio.Event()
        self._status = status
        self._sleep = sleep

    async def calibrate(self, session_id: str) -> CalibrationResult:
        self._check_cancel()
        wrists = tuple(self._capture.config.slots)
        started_here = not self._capture.running
        try:
            if started_here:
                await self._await_cancelable(self._capture.start())
            await self._emit("clock", "Measuring backend clock offset")
            observations: list[ClockObservation] = []
            for _ in range(self._clock_count):
                observations.append(await self._await_cancelable(self._socket.observe_clock()))
            poses: dict[str, dict[WristSide, list[Vector3]]] = {}
            for stage, instruction in _instructions(wrists):
                self._check_cancel()
                poses[stage] = await self._measure_with_retry(stage, instruction, poses, wrists)
            payload: JsonObject = {
                "schemaVersion": 2,
                "clockObservations": [item.to_payload() for item in observations],
                "wrists": {
                    wrist.value: {
                        stage: [sample.to_list() for sample in poses[stage][wrist]]
                        for stage in ("neutral", "upward", "outward")
                    }
                    for wrist in wrists
                },
            }
            self._check_cancel()
            await self._emit("submit", "Submitting measured per-wrist calibration")
            result = await self._await_cancelable(self._api.calibrate(session_id, payload))
            self._check_cancel()
            await self._emit(
                "complete",
                f"Calibration complete (confidence {result.horizontal_confidence:.2f})",
                quality=result.horizontal_confidence,
            )
            return result
        finally:
            if started_here:
                await self._capture.stop()

    async def _measure_with_retry(
        self,
        stage: str,
        instruction: str,
        poses: dict[str, dict[WristSide, list[Vector3]]],
        wrists: tuple[WristSide, ...],
    ) -> dict[WristSide, list[Vector3]]:
        for attempt in range(1, self._max_retries + 2):
            for remaining in range(self._countdown, 0, -1):
                await self._emit(
                    stage, f"{instruction} — measuring in {remaining}", attempt=attempt, countdown=remaining
                )
                await self._await_cancelable(self._sleep(1.0))
            self._check_cancel()
            await self._drain_pending()
            await self._emit(stage, f"{instruction} — measuring now", attempt=attempt)
            measured = await self._collect(stage, attempt, wrists)
            quality = self._quality(stage, measured, poses, wrists)
            if quality >= 0.5:
                await self._emit(stage, f"{stage.title()} quality {quality:.0%}", attempt=attempt, quality=quality)
                return measured
            if attempt <= self._max_retries:
                await self._emit(
                    stage, f"Quality {quality:.0%} too low; retrying {stage}", attempt=attempt, quality=quality
                )
        raise CaptureError(
            f"could not obtain a stable {stage} calibration pose",
            hint="Hold each pose still, keep every configured controller streaming, and retry.",
        )

    async def _collect(self, stage: str, attempt: int, wrists: tuple[WristSide, ...]) -> dict[WristSide, list[Vector3]]:
        values: dict[WristSide, list[Vector3]] = {side: [] for side in wrists}
        deadline = asyncio.get_running_loop().time() + self._timeout
        while any(len(values[side]) < self._samples for side in wrists):
            self._check_cancel()
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                missing = ", ".join(f"{side.value} {len(values[side])}/{self._samples}" for side in wrists)
                raise CaptureError(f"calibration sample timeout ({missing})")
            try:
                sample = await self._await_cancelable(self._capture.receive(min(0.25, remaining)))
            except TimeoutError:
                self._ensure_healthy()
                continue
            except StopAsyncIteration:
                self._check_cancel()
                raise CaptureError("DSU capture stopped during calibration") from None
            wrist = _sample_wrist(sample)
            if wrist not in values:
                continue
            if len(values[wrist]) < self._samples:
                values[wrist].append(sample.acceleration_g)
                await self._emit(
                    stage,
                    "Collecting stable samples",
                    attempt=attempt,
                    left_samples=len(values.get(WristSide.LEFT, ())),
                    right_samples=len(values.get(WristSide.RIGHT, ())),
                )
        return values

    def _ensure_healthy(self) -> None:
        health = self._capture.health
        unhealthy = [slot for slot in health.slots if not slot.connected or slot.stale]
        if not health.running or unhealthy:
            labels = ", ".join(f"{slot.wrist.value} slot {slot.slot}" for slot in unhealthy) or "capture"
            raise CaptureError(f"DSU stream became unavailable during calibration: {labels}")

    async def _drain_pending(self) -> None:
        for _ in range(self._capture.health.queue_depth):
            self._check_cancel()
            try:
                await self._await_cancelable(self._capture.receive(0.001))
            except TimeoutError:
                break
            except StopAsyncIteration:
                self._check_cancel()
                raise CaptureError("DSU capture stopped during calibration") from None

    def _check_cancel(self) -> None:
        if self._cancel.is_set():
            raise GameplayAborted("calibration aborted by operator")

    async def _await_cancelable[T](self, awaitable: Awaitable[T]) -> T:
        self._check_cancel()
        operation = asyncio.ensure_future(awaitable)
        cancelled = asyncio.create_task(self._cancel.wait())
        try:
            done, _ = await asyncio.wait((operation, cancelled), return_when=asyncio.FIRST_COMPLETED)
            if cancelled in done and self._cancel.is_set():
                raise GameplayAborted("calibration aborted by operator")
            return await operation
        finally:
            if not operation.done():
                operation.cancel()
            cancelled.cancel()
            await asyncio.gather(operation, cancelled, return_exceptions=True)

    def _quality(
        self,
        stage: str,
        measured: dict[WristSide, list[Vector3]],
        poses: dict[str, dict[WristSide, list[Vector3]]],
        wrists: tuple[WristSide, ...],
    ) -> float:
        consistency = min(_consistency(measured[side]) for side in wrists)
        if stage == "neutral":
            return consistency
        neutral = poses["neutral"]
        displacements = {side: _mean(measured[side]) - _mean(neutral[side]) for side in wrists}
        movement = min(min(1.0, displacements[side].norm() / 0.2) for side in wrists)
        geometry = 1.0
        if stage == "outward" and "upward" in poses:
            upward = {side: (_mean(poses["upward"][side]) - _mean(neutral[side])).normalized() for side in wrists}
            geometry = min(1.0 - abs(displacements[side].normalized().dot(upward[side])) for side in wrists)
        return max(0.0, min(consistency, movement, geometry))

    async def _emit(self, stage: str, message: str, **values: int | float | None) -> None:
        if self._status is None:
            return
        raw_quality = values.get("quality")
        quality = None if raw_quality is None else float(raw_quality)
        status = CalibrationStatus(
            stage,
            message,
            int(values.get("attempt") or 1),
            int(values.get("countdown") or 0),
            int(values.get("left_samples") or 0),
            int(values.get("right_samples") or 0),
            quality,
        )
        result = self._status(status)
        if inspect.isawaitable(result):
            await result


def _sample_wrist(sample: RawImuSample) -> WristSide:
    for wrist in WristSide:
        if sample.device_id.lower().startswith(f"{wrist.value}:"):
            return wrist
    raise CaptureError(f"captured sample has no assigned wrist: {sample.device_id}")


def _instructions(wrists: tuple[WristSide, ...]) -> tuple[tuple[str, str], ...]:
    if wrists == (WristSide.RIGHT,):
        return (
            ("neutral", "Hold your right wrist still in a neutral pose"),
            ("upward", "Point/move your right wrist upward and hold"),
            ("outward", "Point/move your right wrist toward camera-right and hold"),
        )
    return (
        ("neutral", "Hold both wrists still in a neutral pose"),
        ("upward", "Point/move both wrists upward and hold"),
        ("outward", "Point/move both wrists toward camera-right and hold"),
    )


def _mean(values: list[Vector3]) -> Vector3:
    count = float(len(values))
    return Vector3(
        sum(value.x for value in values) / count,
        sum(value.y for value in values) / count,
        sum(value.z for value in values) / count,
    )


def _consistency(values: list[Vector3]) -> float:
    mean = _mean(values)
    rms = sqrt(sum((value - mean).norm() ** 2 for value in values) / len(values))
    return max(0.0, 1.0 - rms / 0.2)
