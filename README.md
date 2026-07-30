# Dancify Synchronization & Scoring

Typed Flask/Socket.IO modular monolith for importing video-derived wrist motion, calibrating two wearable IMUs, scoring fixed choreography windows, and publishing real-time gameplay events.

## Implemented architecture

- **Routine repository:** validates the video-ingestion `motion_signal` schema, stores metadata and versioned motion JSON in PostgreSQL, and generates one-second windows.
- **Timing calibration:** estimates browser/server offset from low-RTT exchanges and retains an affine device-clock mapping.
- **Spatial calibration:** uses neutral, upward, and outward gestures to remove gravity and project each 3D device signal into player-relative horizontal/vertical features. Horizontal confidence decays because a 6-axis IMU has no stable absolute yaw reference.
- **Motion capture boundary:** a strict Cemuhook/DSU v1001 UDP adapter discovers two explicit controller slots, validates complete CRC-protected packets, maps each device clock to local monotonic time, and exposes bounded per-wrist stream health. The seeded simulator remains available only for deterministic backend tests.
- **Scoring:** independent `WindowingStrategy` and `ScoringAlgorithm` protocols. The app defaults to the bounded `generous` profile: 50 Hz resampling with a 100 ms gap allowance and Sakoe-Chiba-constrained DTW using `0.45 direction + 0.25 magnitude + 0.30 timing`; direct no-argument scorer/evaluator construction remains legacy-strict compatible.
- **Gameplay:** active sessions are in memory; routines and terminal summaries are durable. State is `created → calibrating → ready → scheduled → playing ↔ paused → completed`, with abort paths.
- **Frontend:** versioned REST resources plus the Socket.IO `/gameplay` namespace. Events carry schema version, session ID, monotonic sequence, and server timestamp.

## Setup

Requirements: Python 3.12–3.14, [`uv`](https://docs.astral.sh/uv/), and PostgreSQL for normal operation.

Dancify automatically loads the nearest `.env` file found from the current working directory upward for the backend, Alembic, and terminal client. Copy the tracked template and edit it locally; `.env` itself is gitignored:

```bash
cp .env.example .env
```

Precedence is: explicit CLI or Flask configuration, existing process/container environment variables, `.env`, then built-in defaults. Dotenv loading never replaces a variable already exported by the shell or supplied by the container.

### Scoring difficulty profiles and configuration

`create_app()` defaults to `DANCIFY_SCORING_PROFILE=generous`. Profiles are immutable validated baselines; scalar overrides create a new immutable active configuration at startup. Restart the backend after changing an environment value. Explicit Flask config using the same `DANCIFY_SCORING_*` key overrides process environment and `.env`. The development diagnostics page reports the exact active values read-only; MOCK POST controls cannot modify them.

| Profile | Direction / magnitude / timing | Radius | Timing grace → zero | Min → full coverage | Coverage / sample-quality floors | Max resample gap | Timing path cost |
| --- | --- | ---: | --- | --- | --- | ---: | ---: |
| `generous` (app default) | `.45 / .25 / .30` | 18 | 150 → 450 ms | 20 → 65% | 70 / 85% | 100 ms | .35 |
| `balanced` | `.475 / .275 / .25` | 14 | 75 → 300 ms | 35 → 80% | 55 / 65% | 75 ms | .20 |
| `strict` | `.50 / .30 / .20` | 10 | legacy linear index timing, zero grace | 50 → 100% | exact coverage / 0% | 50 ms | 0 |

`strict` reproduces the previous scorer/evaluator behavior: normalized-index linear timing, unchanged DTW path cost, 50% validity, quality exactly `coverage × mean sample quality`, and target-grid resampling timestamps in `(timestamp, wrist)` order. `WeightedDtwScoringAlgorithm()` and `WindowScoringEvaluator()` with no arguments intentionally remain strict-compatible for direct-constructor and acceptance compatibility; only the Flask app default changed. All profiles default to 50 Hz, but `DANCIFY_SCORING_SAMPLE_RATE_HZ` configures the active rate from 1–240 Hz. Non-strict synchronized-time profiles retain nearest-source timestamps for timing while preserving target-grid sequence order rather than regrouping equal source timestamps.

Supported environment/Flask keys and inclusive startup bounds are:

- `DANCIFY_SCORING_PROFILE`: exactly `generous`, `balanced`, or `strict`.
- `DANCIFY_SCORING_TIMING_GRACE_SECONDS`: 0–1; `DANCIFY_SCORING_TIMING_FALLOFF_SECONDS`: 0.001–2 and greater than grace for synchronized-time profiles.
- `DANCIFY_SCORING_MIN_COVERAGE` and `DANCIFY_SCORING_FULL_COVERAGE`: 0–1 with minimum strictly below full.
- `DANCIFY_SCORING_RESAMPLE_MAX_GAP_SECONDS`: 0.001–0.5.
- `DANCIFY_SCORING_DIRECTION_WEIGHT`, `DANCIFY_SCORING_MAGNITUDE_WEIGHT`, and `DANCIFY_SCORING_TIMING_WEIGHT`: each 0–1 and together exactly 1.
- `DANCIFY_SCORING_SAKOE_CHIBA_RADIUS`: integer 0–100; `DANCIFY_SCORING_SAMPLE_RATE_HZ`: integer 1–240.
- `DANCIFY_SCORING_COVERAGE_QUALITY_FLOOR`, `DANCIFY_SCORING_SAMPLE_QUALITY_FLOOR`, and `DANCIFY_SCORING_TIMING_PATH_COST_WEIGHT`: 0–1.

Values must be finite numbers (integers where stated); booleans, NaN, infinity, malformed values, and inconsistent bounds fail startup with a `RuntimeError` naming the offending key(s).

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
- `POST /api/v1/sessions/{sessionID}/retry`

Retry requires an aborted source session and creates a new session with a different ID. Valid calibration is reused when available, but all gameplay state, scores, and motion are reset. Use the returned session ID—not the aborted source ID—for the next `dancify-client run` command.

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


## Development-only MOCK scoring diagnostics

The scoring diagnostics page is **disabled by default** and its routes do not exist unless it is explicitly enabled in a development process. Start a disposable local instance with both gates set:

```bash
export DANCIFY_ENVIRONMENT=development
export DANCIFY_ENABLE_DEBUG_UI=true
export DATABASE_URL=sqlite:///dancify.db
uv run python -m dancify
# Open http://127.0.0.1:5000/_dev/scoring/
```

Removing `DANCIFY_ENABLE_DEBUG_UI=true` restores the default 404 behavior. Values must be exactly `true` or `false`; enabling the page outside `development` fails startup. The page and its JSON endpoints additionally reject non-loopback clients, send no-store and restrictive browser security headers, cap request bodies, and accept only bounded scalar perturbation controls. **Do not expose or reverse-proxy `/_dev/scoring`**: a local proxy can make a remote client appear to originate from loopback. This is a developer tool, not an authenticated production feature.

Import routines through the normal routine API first, then use the page to select a fixed scoring window. It plots honest camera-plane wrist acceleration-derived horizontal/vertical components and intensity. It does not reconstruct a person, pose, animation, or video. Each run is labeled **MOCK** and deterministically perturbs canonical post-calibration features (direction, intensity, timing, coverage, sample quality, and horizontal confidence), then invokes the same profile-configured resampling rate and gap, coverage/quality validity rule, and registered weighted-DTW scorer used by gameplay. Results include validity, coverage, quality, and direction/intensity/timing breakdowns. Attempts are stateless: the server creates no session or database row, while the browser retains at most 20 responses in the current tab.

This page does not read physical controllers and is not live controller practice. DSU capture currently belongs to the terminal process, which performs bounded UDP ingestion, clock mapping, wrist assignment, and calibration before backend upload. The preferred next real-practice slice is a terminal `practice ROUTINE_ID --window N` mode reusing that pipeline and the shared window evaluator. A browser workflow would instead require a separately designed authenticated, random-token, loopback-only bridge with origin checks, lifecycle control, clock synchronization, and backpressure; bypassing the existing capture/calibration boundary would not be valid scoring.
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

Each active wrist stream is resampled at the profile's configured rate and maximum gap. The default `generous` profile uses 50 Hz and 100 ms, accepts non-empty coverage from 20%, smoothly removes its coverage penalty by 65%, floors the coverage factor at 70%, and limits sample-quality influence with an 85% floor. Coverage, mean sample quality, and the resulting quality adjustment remain visible in diagnostics; no data or coverage below the configured minimum is invalid and scores zero. The `strict` profile retains the prior 50 ms gap, 50% validity threshold, and exact `coverage × mean sample quality` adjustment. Playback drift above 500 ms pauses scoring; a later in-tolerance heartbeat resumes it. Completed window indices are retained so repeated heartbeats cannot score a window twice.

Reference camera X/Y acceleration and calibrated wearable horizontal/vertical linear acceleration are compared as normalized features. Non-strict profiles score timing from actual synchronized sample-time deltas: similarity is 1 through the configured grace, then follows a smooth bounded falloff to 0 at the configured falloff time. A bounded timing term is also present in local DTW path cost so phase cannot be traded away arbitrarily. `strict` retains the prior normalized-index linear timing formula and path selection exactly. Signed horizontal scoring is used only while confidence is sufficient; otherwise the scorer automatically relies on vertical direction, intensity, and timing.

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

# Safe recovery: the source must already be aborted. This creates a new session,
# reuses valid calibration when available, and clears gameplay state, scores, and motion.
uv run dancify-client session retry ABORTED_SESSION_ID
uv run dancify-client run RETURNED_NEW_SESSION_ID --duration 120 --player mpv \
  --media /path/to/dance.mp4 --delay 2
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
