# Dancify Synchronization & Scoring

Typed Flask/Socket.IO modular monolith for importing video-derived wrist motion, calibrating two wearable IMUs, scoring fixed choreography windows, and publishing real-time gameplay events.

## Implemented architecture

- **Routine repository:** validates the video-ingestion `motion_signal` schema, stores metadata and versioned motion JSON in PostgreSQL, and generates one-second windows.
- **Timing calibration:** estimates browser/server offset from low-RTT exchanges and retains an affine device-clock mapping.
- **Spatial calibration:** uses neutral, upward, and outward gestures to remove gravity and project each 3D device signal into player-relative horizontal/vertical features. Horizontal confidence decays because a 6-axis IMU has no stable absolute yaw reference.
- **Motion capture boundary:** a strict Cemuhook/DSU v1001 UDP adapter discovers two explicit controller slots, validates complete CRC-protected packets, maps each device clock to local monotonic time, and exposes bounded per-wrist stream health. The seeded simulator remains available only for deterministic backend tests.
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
- `POST /api/v1/sessions/{sessionID}/motion` (canonical-feature deterministic/testing endpoint)
- `POST /api/v1/sessions/{sessionID}/motion/raw` (item-tolerant live configured-wrist batches)
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

Live calibration uses schema v2: the client records real four-timestamp Socket.IO clock exchanges and measured accelerometer poses independently for every configured wrist. The older unversioned shared-gesture shape remains accepted for deterministic compatibility.

```json
{
  "schemaVersion": 2,
  "clockObservations": [
    {"clientSend": 10.0, "serverReceive": 10.01, "serverSend": 10.02, "clientReceive": 10.03}
  ],
  "wrists": {
    "left": {
      "neutral": [[0, 0, 1]],
      "upward": [[0, 1, 1]],
      "outward": [[1, 0, 1]]
    },
    "right": {
      "neutral": [[0, 0, 1]],
      "upward": [[0, 1, 1]],
      "outward": [[1, 0, 1]]
    }
  }
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
- `motion.health`
- `session.paused`
- `session.completed`
- `session.error`

A client should join the room after reconnecting, use the returned snapshot as authoritative state, and ignore events with sequence numbers older than that snapshot.

## Quality and synchronization rules

Each active wrist stream is resampled at 50 Hz. Brief gaps within 50 ms use the nearest valid feature; moderate incompleteness reduces window quality and therefore the score. Coverage below 50% invalidates the window and contributes zero to the cumulative mean. Playback drift above 500 ms pauses scoring; a later in-tolerance heartbeat resumes it. Completed window indices are retained so repeated heartbeats cannot score a window twice.

Reference camera X/Y acceleration and calibrated wearable horizontal/vertical linear acceleration are compared as normalized features. Signed horizontal scoring is used only while confidence is sufficient; otherwise the scorer automatically relies on vertical direction, intensity, and timing.

## Cemuhook/DSU capture

The live adapter implements standard DSU v1001 rather than the report's simplified 18-byte sketch. It validates `DSUS`/`DSUC` headers, declared lengths, CRC32, server identity, explicit slots, packet serials, 64-bit microsecond timestamps, acceleration in g, and gyro in degrees/second. Left/right assignment is never inferred from response order. Queues are bounded; duplicate, out-of-order, loss, stale, and transport counters are visible to the operator.

Deferred work includes labeled human-score tuning, magnetometer/external heading correction, move-defined windows, authentication, Redis/multiworker deployment, and video object storage.

## Terminal client (no web frontend required)

The pinned environment installs `dancify-client`, a scriptable CLI and complete Textual operator. Global backend/DSU options precede the command:

```bash
uv run dancify-client \
  --url http://127.0.0.1:5000 \
  --dsu-host 127.0.0.1 --dsu-port 26760 \
  --right-slot 1 \
  doctor --capture --player mpv

# Right-handed mode is the default: omit --left-slot.
uv run dancify-client --right-slot 1 discover

# Explicit two-wrist compatibility mode.
uv run dancify-client --left-slot 0 --right-slot 1 discover
```

Equivalent environment variables are `DANCIFY_URL`, `DANCIFY_TIMEOUT`, `DANCIFY_DSU_HOST`, `DANCIFY_DSU_PORT`, `DANCIFY_DSU_LEFT_SLOT`, `DANCIFY_DSU_RIGHT_SLOT`, and `DANCIFY_DSU_STALE_AFTER`. `DANCIFY_DSU_RIGHT_SLOT` defaults to `1` and is always required by capture. Leave `DANCIFY_DSU_LEFT_SLOT` unset or blank for right-only mode; set it to a distinct numeric slot for explicit two-wrist mode. Configured slots must be in 0–3.

### Real operator workflow

Start the backend and Cemuhook server, then run:

```bash
uv run dancify-client tui
```

The keyboard-usable workflow is: **backend → routine/session → controller detection and explicit wrist assignment → guided calibration → media ready → live gameplay → results**. A routine may be selected by existing ID or imported from JSON; an existing session can also be resumed at its valid next step. Calibration visibly counts down through neutral, upward, and camera-right poses, measures only the configured wrist(s), reports sample quality, and retries unstable poses. Right-only mode never waits for or displays a fake left controller. Live mode never substitutes generated calibration or motion fixtures.

During gameplay, the TUI displays actual mpv position, authoritative session state and scores, capture/backend device health, accepted/dropped samples, and completion. `Ctrl+X` (or `x` outside a text field) and the Abort button cancel work, abort a non-terminal backend session, and idempotently disconnects Socket.IO, stops UDP capture, and terminates mpv. A stale/disconnected assigned controller performs the same safe abort; there is no unsupported pause/seek placeholder.

Scriptable workflow:

```bash
uv run dancify-client routine import routine.json
uv run dancify-client routine show ROUTINE_ID
uv run dancify-client routine windows ROUTINE_ID
uv run dancify-client session create ROUTINE_ID --player developer

# No --input: collect real configured-wrist poses and measured clock observations.
# With no global --left-slot (the default), this submits right-only calibration-v2.
uv run dancify-client session calibrate SESSION_ID

# Honest live mode: concurrent DSU raw upload plus actual mpv time-pos heartbeats.
uv run dancify-client run SESSION_ID --duration 120 --player mpv \
  --media /path/to/dance.mp4 --delay 2

# Keep audio but suppress mpv's native video window.
uv run dancify-client run SESSION_ID --duration 120 --player mpv \
  --media /path/to/dance.mp4 --audio-only

uv run dancify-client session show SESSION_ID
uv run dancify-client session abort SESSION_ID
```

mpv is honest and mandatory for live playback: it is preloaded paused through an isolated Unix JSON-IPC socket, launched with `shell=False`, and queried for real `time-pos`. The client does not silently fall back to a clock. Raw capture, bounded partial-acceptance uploads, progress heartbeats, Socket.IO score/health events, and cancellation run concurrently. Before the final completion heartbeat, pending raw samples are flushed; the final REST snapshot reconciles scores and state. Cleanup uses quit/terminate/kill fallbacks and is safe to call repeatedly.

### Explicit deterministic demo

Synthetic routines, calibration vectors, motion, and server timestamps are isolated to explicit demo/testing commands:

```bash
# Fully deterministic end-to-end demo: no sleeps, devices, audio, or video.
uv run dancify-client demo --deterministic --duration 2

# Run an already calibrated session with deterministic feature input.
uv run dancify-client run SESSION_ID --duration 2 --player clock --deterministic
uv run dancify-client run SESSION_ID --duration 2 --player clock \
  --deterministic --motion features.json
```

The deterministic path is retained for CI/repeatability. Live clock mode without mpv is rejected, so simulated timing cannot be mistaken for real playback.
