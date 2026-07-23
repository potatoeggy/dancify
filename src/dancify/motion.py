"""Motion-capture adapter contract, deterministic simulator, and rolling buffers."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import cos, sin
from random import Random
from threading import Lock
from typing import Protocol

from dancify.domain import RawImuSample, Vector3


class MotionCapturePort(Protocol):
    def start(self, consumer: Callable[[RawImuSample], None]) -> None: ...
    def stop(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    seed: int = 7
    sample_rate_hz: int = 100
    duration_seconds: float = 5.0
    packet_loss: float = 0.0
    jitter_us: int = 0
    reorder_every: int = 0

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0 or self.duration_seconds <= 0:
            raise ValueError("sample rate and duration must be positive")
        if not 0 <= self.packet_loss < 1 or self.jitter_us < 0 or self.reorder_every < 0:
            raise ValueError("invalid simulation quality configuration")


class DeterministicMotionSimulator:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self._config = config or SimulationConfig()
        self._running = False

    def samples(self) -> tuple[RawImuSample, ...]:
        random = Random(self._config.seed)
        count = int(self._config.sample_rate_hz * self._config.duration_seconds)
        interval = 1_000_000 // self._config.sample_rate_hz
        output: list[RawImuSample] = []
        for index in range(count):
            for device_offset, device in enumerate(("left", "right")):
                if random.random() < self._config.packet_loss:
                    continue
                phase = index / self._config.sample_rate_hz * 2.0 * 3.141592653589793
                timestamp = max(0, index * interval + random.randint(-self._config.jitter_us, self._config.jitter_us))
                sign = -1.0 if device_offset == 0 else 1.0
                output.append(
                    RawImuSample(
                        device,
                        timestamp,
                        index,
                        Vector3(sign * sin(phase), cos(phase), 1.0),
                        Vector3(0.0, 0.0, sign * 20.0 * sin(phase)),
                    )
                )
        if self._config.reorder_every:
            step = self._config.reorder_every
            for start in range(step - 1, len(output) - 1, step):
                output[start], output[start + 1] = output[start + 1], output[start]
        return tuple(output)

    def start(self, consumer: Callable[[RawImuSample], None]) -> None:
        self._running = True
        for sample in self.samples():
            if not self._running:
                break
            consumer(sample)

    def stop(self) -> None:
        self._running = False


@dataclass(frozen=True, slots=True)
class StreamHealth:
    accepted: int
    duplicates: int
    out_of_order: int
    estimated_loss: int
    quality: float


class CircularMotionBuffer:
    def __init__(self, retention_seconds: float = 5.0) -> None:
        if retention_seconds <= 0:
            raise ValueError("retention must be positive")
        self._retention_us = int(retention_seconds * 1_000_000)
        self._samples: deque[RawImuSample] = deque()
        self._last_packet: dict[str, int] = {}
        self._accepted = 0
        self._duplicates = 0
        self._out_of_order = 0
        self._loss = 0
        self._lock = Lock()

    def add(self, sample: RawImuSample) -> bool:
        with self._lock:
            previous = self._last_packet.get(sample.device_id)
            if previous is not None:
                if sample.packet_number == previous:
                    self._duplicates += 1
                    return False
                if sample.packet_number < previous:
                    self._out_of_order += 1
                    return False
                self._loss += max(0, sample.packet_number - previous - 1)
            self._last_packet[sample.device_id] = sample.packet_number
            self._samples.append(sample)
            self._accepted += 1
            cutoff = sample.device_timestamp_us - self._retention_us
            while self._samples and self._samples[0].device_timestamp_us < cutoff:
                self._samples.popleft()
            return True

    def between(self, start_us: int, end_us: int, device_id: str | None = None) -> tuple[RawImuSample, ...]:
        with self._lock:
            return tuple(
                sample
                for sample in self._samples
                if start_us <= sample.device_timestamp_us < end_us
                and (device_id is None or sample.device_id == device_id)
            )

    def extend(self, samples: Iterable[RawImuSample]) -> int:
        return sum(self.add(sample) for sample in samples)

    @property
    def health(self) -> StreamHealth:
        attempted = self._accepted + self._duplicates + self._out_of_order + self._loss
        quality = self._accepted / attempted if attempted else 1.0
        return StreamHealth(self._accepted, self._duplicates, self._out_of_order, self._loss, quality)

    def __len__(self) -> int:
        with self._lock:
            return len(self._samples)
