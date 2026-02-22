# OSRS Flipping AI

Python + MongoDB backend for OSRS Grand Exchange flip scoring, personalization, alerts, and dashboard APIs.

## Local setup

1. Create virtual environment and install dependencies.
2. Copy `.env.example` to `.env` and fill values.
3. Start API service:
   `RUN_MODE=api uvicorn backend.app:app --host 0.0.0.0 --port 8001`
4. Start worker service in a second terminal:
   `RUN_MODE=worker python -m backend.worker`

## Core endpoints

- `GET /flips/top`
- `GET /flips/filtered`
- `GET /flips/top5`
- `POST /portfolio/optimize`
- `POST /api/user/risk_profile`
- `POST /api/user/flips/import`
- `GET /api/user/insights`
- `PATCH /api/user/alerts/settings`
- `GET /api/user/alerts/test`
- `GET /health`
- `GET /status`

`/flips/top5` is cache-only and served from precomputed worker results.  
`/flips/top` is cache-first; use `?fresh=1` for live recompute (rate-limited).

## Observability

- `/health` includes DB connectivity, cache backend, last poll timestamp, scored item count, cache hit rate, alert sent count, and errors in the last hour.
- `/status` is UI-friendly runtime state: `worker_ok`, `last_poll_ts`, `cache_age_seconds`, `items_scored_count`, `profile_counts`.
- API logs include structured request data: `request_id`, `method`, `path`, `status_code`, `latency_ms`, `cache_hit`, `profile`.

## Plugin auth and limits

- `/flips/top5` requires `X-API-Key` by default (unless `ALLOW_ANON=true`).
- API keys are validated against `api_keys.key_hash` (or `users.api_key_hash` fallback).
- Rate limits:
  - `/flips/top5`: `TOP5_RATE_LIMIT_PER_MINUTE` (default `60`)
  - `/flips/top`: `TOP_RATE_LIMIT_PER_MINUTE` (default `20`)

## Explainability fields

`/flips/top` and `/flips/top5` include:
- `reasons`: short human-readable rationale strings
- `badges`: compact tags like `SAFE`, `FAST`, `VOLATILE`, `HIGH_ROI`
- `confidence_pct`: always normalized to 0–100

## Data guardrails

- Worker uses a circuit breaker on repeated upstream fetch failures (`WORKER_CIRCUIT_FAILURE_THRESHOLD`, `WORKER_CIRCUIT_OPEN_SECONDS`).
- Scoring excludes stale snapshots (`SCORE_STALE_MAX_MINUTES`) and enforces stricter veto/penalty rules for:
  - non-positive margin after tax
  - low fill probability by profile
  - wide spreads
  - spread-compression spikes

## Load test

Run:
`k6 run loadtests/k6_flips.js -e BASE_URL=http://localhost:8001`

## QA checklist

See `docs/QA.md`.

## Railway deployment (2 services)

Deploy two Railway services from the same repo:

1. Web/API service:
   - Start command: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
   - Env: `RUN_MODE=api`
2. Worker service:
   - Start command: `python -m backend.worker`
   - Env: `RUN_MODE=worker`

Both services should share the same `MONGODB_URL` and `DATABASE_NAME`.
If `REDIS_URL` is set, cache is shared across instances/restarts; otherwise an in-memory fallback is used.

## Cache keys

- `flips:top5:{profile}`
- `flips:top100:{profile}`
- `flips:stats:{profile}`
- `flips:last_updated_ts`

