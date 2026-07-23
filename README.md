# Dancify Synchronization & Scoring

Typed Flask/Socket.IO modular monolith for importing video-derived wrist motion, calibrating two wearable IMUs, scoring fixed choreography windows, and publishing real-time gameplay events.

## Implemented architecture

- **Routine repository:** validates the video-ingestion `motion_signal` schema, stores metadata and versioned motion JSON in PostgreSQL, and generates one-second windows.
- **Timing calibration:** estimates browser/server offset from low-RTT exchanges and retains an affine device-clock mapping.
- **Spatial calibration:** uses neutral, upward, and outward gestures to remove gravity and project each 3D device signal into player-relative horizontal/vertical features. Horizontal confidence decays because a 6-axis IMU has no stable absolute yaw reference.
- **Motion capture boundary:** `MotionCapturePort` accepts raw timestamped IMU samples. A seeded 100 Hz two-device simulator and bounded, thread-safe five-second buffer implement the boundary; production Cemuhook parsing is deferred.
- **Scoring:** independent `WindowingStrategy` and `ScoringAlgorithm` protocols. The default uses non-overlapping one-second windows and Sakoe-Chiba-constrained DTW with `0.5 direction + 0.3 magnitude + 0.2 timing`.
- **Gameplay:** active sessions are in memory; routines and terminal summaries are durable. State is `created → calibrating → ready → scheduled → playing ↔ paused → completed`, with abort paths.
- **Frontend:** versioned REST resources plus the Socket.IO `/gameplay` namespace. Events carry schema version, session ID, monotonic sequence, and server timestamp.

## Setup

Requirements: Python 3.12–3.14, [`uv`](https://docs.astral.sh/uv/), and PostgreSQL for normal operation.

```bash
uv sync --frozen --group dev
docker compose up -d db
uv run alembic upgrade head
uv run python -m dancify
```

For a disposable local database:

```bash
export DATABASE_URL=sqlite:///dancify.db
uv run alembic upgrade head
uv run python -m dancify
```

Production-style local serving uses the same one-worker threaded topology as the container:

```bash
gunicorn -w 1 --threads 100 -b 0.0.0.0:5000 dancify.__main__:app
```

Multiple web workers require sticky routing and a Socket.IO message broker; that deployment is intentionally out of scope.

## Validation

```bash
uv run pytest
uv run ruff check .
uv run mypy
uv run pyright
DATABASE_URL=sqlite:////tmp/dancify-migration.db uv run alembic upgrade head
uv run python -m dancify.acceptance
```

The acceptance command creates deterministic 50 Hz two-wrist data, translates it through guided calibration, verifies `good > reversed > missing`, and fails if the local weighted-DTW p95 reaches 20 ms.

## Routine API

All application resources are under `/api/v1`; `GET /health` is also available for container probes.

- `POST /api/v1/routines`
- `GET /api/v1/routines/{routineID}`
- `GET /api/v1/routines/{routineID}/windows`
- `POST /api/v1/sessions` with `{routineID, playerID, scoringAlgorithm?}`
- `GET /api/v1/sessions/{sessionID}`
- `POST /api/v1/sessions/{sessionID}/calibration`
- `POST /api/v1/sessions/{sessionID}/start`
- `POST /api/v1/sessions/{sessionID}/motion` (canonical-feature integration/testing endpoint)
- `POST /api/v1/sessions/{sessionID}/progress`
- `POST /api/v1/sessions/{sessionID}/abort`

The routine importer accepts the ingestion repository's shape:

```json
{
  "title": "Demo",
  "metadata": {
    "source_video": "https://example.invalid/demo.mp4",
    "fps": 29.97,
    "duration_seconds": 120.0
  },
  "motion_signal": [
    {
      "timestamp": 0.0,
      "left_wrist": {"x": 0.5, "y": 0.5, "vx": 0.1, "vy": 0.0, "ax": 0.2, "ay": 0.0},
      "right_wrist": null
    }
  ]
}
```

Nullable wrist values are accepted; timestamps must be finite, non-negative, and strictly increasing.

Calibration submits several clock exchanges plus gesture accelerometer vectors:

```json
{
  "clockObservations": [
    {"clientSend": 0.0, "serverReceive": 0.01, "serverSend": 0.02, "clientReceive": 0.03}
  ],
  "neutral": [[0, 0, 1]],
  "upward": [[0, 1, 1]],
  "outward": [[1, 0, 1]]
}
```

## Socket.IO `/gameplay`

Client events:

- `session.join`
- `calibration.observation`
- `playback.ready`
- `playback.progress`
- `session.abort`

Server events:

- `session.snapshot`
- `calibration.result`
- `playback.scheduled`
- `score.update`
- `session.paused`
- `session.completed`
- `session.error`

A client should join the room after reconnecting, use the returned snapshot as authoritative state, and ignore events with sequence numbers older than that snapshot.

## Quality and synchronization rules

Both streams are resampled at 50 Hz. Brief gaps within 50 ms use the nearest valid feature; moderate incompleteness reduces window quality and therefore the score. Coverage below 50% invalidates the window and contributes zero to the cumulative mean. Playback drift above 500 ms pauses scoring; a later in-tolerance heartbeat resumes it. Completed window indices are retained so repeated heartbeats cannot score a window twice.

Reference camera X/Y acceleration and calibrated wearable horizontal/vertical linear acceleration are compared as normalized features. Signed horizontal scoring is used only while confidence is sufficient; otherwise the scorer automatically relies on vertical direction, intensity, and timing.

## Future Cemuhook adapter

The report's simplified 18-byte packet is not the standard DSU wire format. A production adapter should validate the 16-byte `DSUS`/`DSUC` header, version 1001, payload length, CRC32, event type `0x100002`, controller slot, packet number, 64-bit microsecond timestamp, acceleration floats in g, and gyro floats in degrees/second. It then emits `RawImuSample` through `MotionCapturePort`; no scoring or session code should parse datagrams.

Deferred work also includes labeled human-score tuning, magnetometer/external heading correction, move-defined windows, authentication, Redis/multiworker deployment, and video object storage.
